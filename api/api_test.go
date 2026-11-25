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

	"hermesengine/api"
	"hermesengine/internal/testkit"
	"hermesengine/run"
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

var csrf = map[string]string{"X-Hermes-Engine": "1"}

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
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/cancel", "", map[string]string{"X-Hermes-Engine": "1", "Origin": "http://evil.example"}); c != 403 {
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

func TestUnavailableWhileLoading(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "1500ms"
	s, srv := serve(t, e, e.Config())
	testkit.WaitFor(t, 10*time.Second, "loading phase", func() bool { return s.Phase() == "loading" })
	c, body, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, nil)
	if c != 503 || h.Get("Retry-After") == "" {
		t.Fatalf("want 503 with Retry-After, got %d %s", c, body)
	}
	var j struct {
		Error struct {
			Code    int
			Type    string
			Message string
		}
	}
	if json.Unmarshal([]byte(body), &j) != nil || j.Error.Code != 503 || j.Error.Message == "" {
		t.Fatalf("not llama-server's error shape: %s", body)
	}
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
}

func TestNoShrinkWhileProxyRequestInFlight(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_CHAT_DELAY"] = "1500ms"
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	var done int32
	go func() {
		do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, nil)
		atomic.StoreInt32(&done, 1)
	}()
	time.Sleep(200 * time.Millisecond)
	e.SetUsed(16000)
	e.SetOthers(8000)
	time.Sleep(900 * time.Millisecond)
	if atomic.LoadInt32(&done) == 1 {
		t.Skip("request ended too early to judge")
	}
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("resized with a request in flight: %v", e.Starts())
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
