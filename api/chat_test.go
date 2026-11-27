package api_test

import (
	"bufio"
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// chatServer is a ready engine whose "open the browser" is recorded instead of done.
func chatServer(t *testing.T) (srv *httptest.Server, opened func() []string) {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	var mu sync.Mutex
	var got []string
	a := &api.API{S: s, OpenURL: func(u string) error { mu.Lock(); got = append(got, u); mu.Unlock(); return nil }}
	srv = httptest.NewServer(a.Handler())
	a.ChatAddr = strings.TrimPrefix(srv.URL, "http://") // as app.Serve sets it from its own listener
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return srv, func() []string { mu.Lock(); defer mu.Unlock(); return append([]string(nil), got...) }
}

func TestChatServesWrapperAndStripsThePrefix(t *testing.T) {
	srv, _ := chatServer(t)
	c, body, h := do(t, "GET", srv.URL+"/chat/", "", nil)
	if c != 200 || !strings.Contains(body, `<iframe`) || !strings.Contains(body, "mini=1") || h.Get("Cache-Control") != "no-store" {
		t.Fatalf("wrapper: %d %s %v", c, body, h)
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
	srv, _ := chatServer(t)
	_, _, h := do(t, "GET", srv.URL+"/chat/", "", nil)
	if h.Get("Cross-Origin-Embedder-Policy") != "require-corp" || h.Get("Cross-Origin-Opener-Policy") != "same-origin" {
		t.Fatalf("headers %v", h)
	}
}

// The proxy hands the client's Accept-Encoding to llama-server and the gzip body back with its
// Content-Encoding untouched, as a browser needs (the real UI answers 415 without gzip).
func TestChatUIPassesGzipThrough(t *testing.T) {
	srv, _ := chatServer(t)
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
	srv, _ := chatServer(t)
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
	if c, body, _ := do(t, "GET", srv.URL+"/chat/ui/", "", nil); c != 503 || !strings.Contains(body, "No model is ready") {
		t.Fatalf("ui while loading: %d %s", c, body)
	}
}

func TestChatOpenOpensTheFullPage(t *testing.T) {
	srv, opened := chatServer(t)
	post := func(body string, hdr map[string]string) int {
		c, _, _ := do(t, "POST", srv.URL+"/engine/v1/chat/open", body, hdr)
		return c
	}
	if c := post(`{}`, nil); c != 403 {
		t.Fatalf("without X-Easy-Engine: %d", c)
	}
	if got := opened(); len(got) != 0 {
		t.Fatalf("opened without the header: %v", got)
	}
	if c := post(`{"hash":"#/chat/abc-123"}`, csrf); c != 200 {
		t.Fatalf("open: %d", c)
	}
	if c := post(`{}`, csrf); c != 200 {
		t.Fatalf("open with no conversation: %d", c)
	}
	for _, bad := range []string{`{"hash":"javascript:alert(1)"}`, `{"hash":"#/chat/a b"}`, `{"hash":"//evil"}`, `nonsense`} {
		if c := post(bad, csrf); c != 400 {
			t.Errorf("%s: %d, want 400", bad, c)
		}
	}
	host := strings.TrimPrefix(srv.URL, "http://")
	want := []string{"http://" + host + "/chat/#/chat/abc-123", "http://" + host + "/chat/"}
	got := opened()
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("opened %v, want %v", got, want)
	}
}

// A forged Host header cannot steer the engine's browser to another site: the address opened is
// the engine's own, not the request's.
func TestChatOpenIgnoresAForgedHost(t *testing.T) {
	srv, opened := chatServer(t)
	hdr := map[string]string{"X-Easy-Engine": "1", "Host": "evil.example"}
	c, body, _ := do(t, "POST", srv.URL+"/engine/v1/chat/open", `{"hash":"#/chat/abc"}`, hdr)
	// The gate refuses a non-loopback Host name outright; either way nothing foreign is opened.
	if c == 200 {
		t.Fatalf("a forged Host was served: %s", body)
	}
	hdr["Host"] = "localhost:1" // a loopback name the gate accepts, but not the engine's own port
	if c, body, _ = do(t, "POST", srv.URL+"/engine/v1/chat/open", `{"hash":"#/chat/abc"}`, hdr); c != 200 {
		t.Fatalf("%d %s", c, body)
	}
	want := "http://" + strings.TrimPrefix(srv.URL, "http://") + "/chat/#/chat/abc"
	if got := opened(); len(got) != 1 || got[0] != want {
		t.Fatalf("opened %v, want [%s]", got, want)
	}
}

func TestChatOpenRefusedOnANetworkListener(t *testing.T) {
	a := &api.API{AnyHost: true, Open: true, OpenURL: func(string) error { t.Error("opened"); return nil }}
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/chat/open", `{}`, csrf); c != 403 {
		t.Fatalf("%d", c)
	}
}
