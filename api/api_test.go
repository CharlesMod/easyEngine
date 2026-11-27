package api_test

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

func serve(t *testing.T, e *testkit.Env, cfg run.Config) (*run.Supervisor, *httptest.Server) {
	t.Helper()
	s := e.StartSup(cfg)
	a := &api.API{S: s}
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	return s, srv
}

func do(t *testing.T, method, url, body string, hdr map[string]string) (int, string, http.Header) {
	t.Helper()
	req, _ := http.NewRequest(method, url, strings.NewReader(body))
	for k, v := range hdr {
		if k == "Host" {
			req.Host = v
			continue
		}
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), resp.Header
}

var csrf = map[string]string{"X-Easy-Engine": "1"}

func readyEnv(t *testing.T) (*testkit.Env, *run.Supervisor, *httptest.Server) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv
}

func TestCSRFAndHost(t *testing.T) {
	_, _, srv := readyEnv(t)
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/sleep", "", nil); c != 403 {
		t.Fatalf("POST without the header: %d", c)
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/cancel", "", map[string]string{"X-Easy-Engine": "1", "Origin": "http://evil.example"}); c != 403 {
		t.Fatalf("foreign origin: %d", c)
	}
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", map[string]string{"Host": "evil.example"}); c != 403 {
		t.Fatalf("foreign host (DNS rebinding): %d", c)
	}
	c, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
	if c != 200 {
		t.Fatalf("status %d", c)
	}
	var st map[string]any
	json.Unmarshal([]byte(body), &st)
	for _, k := range []string{"phase", "agent", "parents", "tok_s", "shape", "model"} {
		if _, ok := st[k]; !ok {
			t.Errorf("status lacks %q: %s", k, body)
		}
	}
	if st["phase"] != "ready" {
		t.Errorf("phase %v", st["phase"])
	}
	if c, _, _ := do(t, "GET", srv.URL+"/", "", nil); c != 200 {
		t.Fatalf("setup page %d", c)
	}
}

func TestProxyAndStreaming(t *testing.T) {
	e, _, srv := readyEnv(t)
	_ = e
	c, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"stream":false}`, map[string]string{"Content-Type": "application/json"})
	if c != 200 || !strings.Contains(body, "ready") {
		t.Fatalf("proxy chat: %d %s", c, body)
	}
	// SSE: the first token must arrive long before the stream ends (no buffering)
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true}`))
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
	total := time.Since(t0)
	if first > total/2 {
		t.Fatalf("stream was buffered: first chunk %v of %v", first, total)
	}
}

func TestNoShrinkWhileProxyRequestInFlight(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_CHAT_DELAY"] = "1500ms"
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	done := make(chan struct{})
	go func() {
		defer close(done)
		do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, nil)
	}()
	testkit.WaitFor(t, 10*time.Second, "the request in flight", func() bool { return s.InFlight() == 1 })
	from := testkit.Mark(t, s, "the room shrinks under a request")
	e.SetUsed(16000)
	e.SetOthers(8000)
	<-done
	testkit.WaitFor(t, 10*time.Second, "the request's end", func() bool { return s.InFlight() == 0 })
	// nothing may restart the server between the room's change and the request's end
	var end uint64
	for _, ev := range testkit.Said(t, s, from, 0, "request") {
		var b struct {
			What     string `json:"what"`
			Inflight int    `json:"inflight"`
		}
		json.Unmarshal(ev.Body, &b)
		if b.What == "end" && b.Inflight == 0 {
			end = ev.Seq
			break
		}
	}
	if end == 0 {
		t.Fatal("no request end said")
	}
	// (a reshape may say "resizing" and drain: new requests are held then; the server is not
	// restarted under the one in flight)
	for _, p := range testkit.Phases(t, s, from, end) {
		if p == "starting" || p == "loading" {
			t.Fatalf("restarted with a request in flight: %v", testkit.Phases(t, s, from, end))
		}
	}
	testkit.WaitFor(t, 15*time.Second, "shrink after the request ended", func() bool { return len(e.Starts()) >= 2 })
}

func TestExternalMode(t *testing.T) {
	testkit.SkipWindows(t)
	var gotAuth, gotPath atomic.Value
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth.Store(r.Header.Get("Authorization"))
		gotPath.Store(r.URL.Path)
		switch r.URL.Path {
		case "/v1/models":
			io.WriteString(w, `{"data":[{"id":"gpt-x"}]}`)
		case "/v1/chat/completions":
			io.WriteString(w, `{"choices":[{"message":{"content":"ready"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer up.Close()
	e := testkit.New(t)
	s, srv := serve(t, e, e.Config())
	set := `{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":"gpt-x","api_key":"sekret"},"linger_s":0,"kv_type":"q4_0"}`
	if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", set, csrf); c != 200 {
		t.Fatalf("settings %d %s", c, b)
	}
	testkit.WaitPhase(t, s, "external", 5*time.Second)
	c, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, nil)
	if c != 200 || gotAuth.Load() != "Bearer sekret" || gotPath.Load() != "/v1/chat/completions" {
		t.Fatalf("forward: %d %s auth=%v path=%v", c, body, gotAuth.Load(), gotPath.Load())
	}
	if c, _, _ := do(t, "GET", srv.URL+"/props", "", nil); c != 404 {
		t.Fatalf("non-/v1 path forwarded in external mode: %d", c)
	}
	// the key is never sent back
	_, st, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
	if strings.Contains(st, "sekret") {
		t.Fatalf("status leaks the key: %s", st)
	}
	_, ss, _ := do(t, "GET", srv.URL+"/engine/v1/settings", "", nil)
	if strings.Contains(ss, "sekret") {
		t.Fatalf("settings: %s", ss)
	}
	// POST test goes through the probe rules
	c, body, _ = do(t, "POST", srv.URL+"/engine/v1/test", "", csrf)
	var r api.TestResult
	json.Unmarshal([]byte(body), &r)
	if c != 200 || r.Light != "green" || r.Code != "ok" {
		t.Fatalf("test: %d %s", c, body)
	}
}

func TestPostTestLocal(t *testing.T) {
	_, _, srv := readyEnv(t)
	c, body, _ := do(t, "POST", srv.URL+"/engine/v1/test", "", csrf)
	var r api.TestResult
	json.Unmarshal([]byte(body), &r)
	if c != 200 || r.Light != "green" {
		t.Fatalf("test: %d %s", c, body)
	}
}

func TestInstallErrors(t *testing.T) {
	_, _, srv := readyEnv(t)
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/install", `{}`, csrf); c != 400 {
		t.Fatalf("empty body: %d", c)
	}
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/install", "", nil); c != 405 {
		t.Fatalf("GET install: %d", c)
	}
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/nope", "", nil); c != 404 {
		t.Fatalf("unknown: %d", c)
	}
}

func TestListenFallback(t *testing.T) {
	l1, p1, err := api.Listen("127.0.0.1", 47700)
	if err != nil {
		t.Skip("no port")
	}
	defer l1.Close()
	l2, p2, err := api.Listen("127.0.0.1", p1)
	if err != nil {
		t.Fatal(err)
	}
	defer l2.Close()
	if p2 <= p1 || p2 > p1+9 {
		t.Fatalf("fallback %d -> %d", p1, p2)
	}
}

func TestStorageRoutes(t *testing.T) {
	e, _, srv := readyEnv(t)
	c, body, _ := do(t, "GET", srv.URL+"/engine/v1/storage?refresh=1", "", nil)
	if c != 200 || !strings.Contains(body, `"data_dir"`) || !strings.Contains(body, `"parts"`) || !strings.Contains(body, `"models"`) {
		t.Fatalf("storage: %d %s", c, body)
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/remove-downloads", `{}`, csrf); c != 400 {
		t.Fatalf("no confirm: %d", c)
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/remove-downloads", `{"confirm":true}`, nil); c != 403 {
		t.Fatalf("no csrf: %d", c)
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/delete-model", `{}`, csrf); c != 400 {
		t.Fatalf("empty delete: %d", c)
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/delete-model", `{"file":"nope.gguf"}`, csrf); c != 404 {
		t.Fatalf("missing delete: %d", c)
	}
	// the installed fake model is serving: 409
	if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/delete-model", `{"id":"m1"}`, csrf); c != 409 {
		t.Fatalf("serving delete: %d %s", c, b)
	}
	if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/remove-downloads", `{"confirm":true}`, csrf); c != 202 || !strings.Contains(b, "started") {
		t.Fatalf("remove: %d %s", c, b)
	}
	testkit.WaitFor(t, 15*time.Second, "not_installed", func() bool {
		_, b, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
		return strings.Contains(b, `"not_installed"`) && strings.Contains(b, "Removed downloaded files")
	})
	_ = e
}
