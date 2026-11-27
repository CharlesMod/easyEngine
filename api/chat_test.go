package api_test

import (
	"bufio"
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/web"
)

// chatServer is a ready engine.
func chatServer(t *testing.T) *httptest.Server {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	srv := httptest.NewServer((&api.API{S: s}).Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return srv
}

func TestChatServesWrapperAndStripsThePrefix(t *testing.T) {
	srv := chatServer(t)
	c, body, h := do(t, "GET", srv.URL+"/chat/", "", nil)
	if c != 200 || !strings.Contains(body, `<iframe`) || h.Get("Cache-Control") != "no-store" {
		t.Fatalf("wrapper: %d %s %v", c, body, h)
	}
	// The tab is ours, not llama's: the wrapper is the top page, so its title and icon show.
	if !strings.Contains(body, "<title>easyEngine</title>") || !strings.Contains(body, `rel="icon"`) {
		t.Errorf("wrapper has no easyEngine title and icon: %s", body)
	}
	if c, icon, h := do(t, "GET", srv.URL+"/chat/icon.png", "", nil); c != 200 || h.Get("Content-Type") != "image/png" || icon != string(web.Icon) || !strings.HasPrefix(icon, "\x89PNG") {
		t.Errorf("icon: %d %q", c, h.Get("Content-Type"))
	}
	// The UI and its assets come from llama-server with /chat/ui stripped.
	if c, body, _ := do(t, "GET", srv.URL+"/chat/ui/", "", nil); c != 200 || body != "fake llama web ui path=/" {
		t.Fatalf("ui: %d %q", c, body)
	}
	if c, body, _ := do(t, "GET", srv.URL+"/chat/ui/_app/x.js", "", nil); c != 200 || body != "fake llama web ui path=/_app/x.js" {
		t.Fatalf("asset: %d %q", c, body)
	}
	// The UI's own API calls are relative to /chat/ui/, so they reach llama-server's routes.
	if c, body, _ := do(t, "GET", srv.URL+"/chat/ui/props", "", nil); c != 200 || !strings.Contains(body, "total_slots") {
		t.Fatalf("props: %d %q", c, body)
	}
	// Bare prefixes redirect to the slash form, so relative addresses resolve under it.
	for from, to := range map[string]string{"/chat": "/chat/", "/chat/ui": "/chat/ui/"} {
		req, _ := http.NewRequest("GET", srv.URL+from, nil)
		resp, err := (&http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}).Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 308 || resp.Header.Get("Location") != to {
			t.Errorf("%s: %d -> %q", from, resp.StatusCode, resp.Header.Get("Location"))
		}
	}
	if c, _, _ := do(t, "GET", srv.URL+"/chat/other", "", nil); c != 404 {
		t.Errorf("unknown chat path: %d", c)
	}
}

// The wrapper is cross-origin isolated like the UI it frames; the not-ready page is not.
func TestChatWrapperIsCrossOriginIsolated(t *testing.T) {
	srv := chatServer(t)
	_, _, h := do(t, "GET", srv.URL+"/chat/", "", nil)
	if h.Get("Cross-Origin-Embedder-Policy") != "require-corp" || h.Get("Cross-Origin-Opener-Policy") != "same-origin" {
		t.Fatalf("headers %v", h)
	}
}

// The proxy hands the client's Accept-Encoding to llama-server and the gzip body back with its
// Content-Encoding untouched, as a browser needs (the real UI answers 415 without gzip).
func TestChatUIPassesGzipThrough(t *testing.T) {
	srv := chatServer(t)
	cl := &http.Client{Transport: &http.Transport{DisableCompression: true}}
	req, _ := http.NewRequest("GET", srv.URL+"/chat/ui/", nil)
	req.Header.Set("Accept-Encoding", "gzip")
	resp, err := cl.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("Content-Encoding") != "gzip" {
		t.Fatalf("%d %v", resp.StatusCode, resp.Header)
	}
	if resp.Header.Get("Cross-Origin-Embedder-Policy") != "require-corp" {
		t.Errorf("the UI's own isolation headers must pass: %v", resp.Header)
	}
	zr, err := gzip.NewReader(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(zr)
	if string(b) != "fake llama web ui path=/" {
		t.Fatalf("body %q", b)
	}
}

func TestChatStreamsUnbuffered(t *testing.T) {
	srv := chatServer(t)
	req, _ := http.NewRequest("POST", srv.URL+"/chat/ui/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	t0 := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	line, _ := br.ReadString('\n')
	first := time.Since(t0)
	if !strings.HasPrefix(line, "data:") {
		t.Fatalf("first line %q", line)
	}
	io.Copy(io.Discard, br)
	if total := time.Since(t0); first > total/2 {
		t.Fatalf("stream was buffered: first chunk %v of %v", first, total)
	}
}

func TestChatWhenNoModelIsReady(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "1500ms"
	s, srv := serve(t, e, e.Config())
	testkit.WaitFor(t, 10*time.Second, "loading phase", func() bool { return s.Phase() == "loading" })
	c, body, h := do(t, "GET", srv.URL+"/chat/", "", nil)
	if h.Get("Cross-Origin-Embedder-Policy") != "" {
		t.Errorf("the not-ready page carries isolation headers")
	}
	if c != 200 || !strings.Contains(body, "No model is ready") || strings.Contains(body, "<iframe") || strings.Contains(h.Get("Content-Type"), "json") {
		t.Fatalf("want a plain not-ready page, got %d %s", c, body)
	}
	if !strings.Contains(body, "<title>easyEngine</title>") || !strings.Contains(body, `href="/chat/icon.png"`) {
		t.Errorf("not-ready page lacks the easyEngine title and icon: %s", body)
	}
	if c, body, _ := do(t, "GET", srv.URL+"/chat/ui/", "", nil); c != 503 || !strings.Contains(body, "No model is ready") {
		t.Fatalf("ui while loading: %d %s", c, body)
	}
}
