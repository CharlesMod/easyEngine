package api_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/run"
)

func hasCORS(h http.Header) bool {
	for k := range h {
		if strings.HasPrefix(strings.ToLower(k), "access-control-") {
			return true
		}
	}
	return false
}

// The gate: every path (engine, setup page, proxy) on a loopback listener.
func TestGateLoopback(t *testing.T) {
	_, _, srv := readyEnv(t)
	own := srv.URL // http://127.0.0.1:port
	paths := []struct{ method, path, body string }{
		{"POST", "/v1/chat/completions", `{"stream":false}`},
		{"GET", "/slots", ""},
		{"GET", "/engine/v1/status", ""},
		{"GET", "/", ""},
	}
	cases := []struct {
		name string
		hdr  map[string]string
		want int // 0: the path's normal success
	}{
		{"plain", nil, 0},
		{"localhost host", map[string]string{"Host": "localhost:47480"}, 0},
		{"ipv6 host", map[string]string{"Host": "[::1]:47480"}, 0},
		{"own origin", map[string]string{"Origin": own}, 0},
		{"same-origin fetch", map[string]string{"Sec-Fetch-Site": "same-origin"}, 0},
		{"navigation", map[string]string{"Sec-Fetch-Site": "none"}, 0},
		{"wrong host", map[string]string{"Host": "evil.example"}, 403},
		{"wrong host with port", map[string]string{"Host": "evil.example:47480"}, 403},
		{"lan ip host", map[string]string{"Host": "192.168.1.5:47480"}, 403},
		{"foreign origin", map[string]string{"Origin": "http://evil.example"}, 403},
		{"foreign origin, loopback name", map[string]string{"Origin": "http://localhost:9999"}, 403},
		{"https origin", map[string]string{"Origin": "https://" + strings.TrimPrefix(own, "http://")}, 403},
		{"null origin", map[string]string{"Origin": "null"}, 403},
		{"cross-site fetch", map[string]string{"Sec-Fetch-Site": "cross-site"}, 403},
		{"same-site fetch", map[string]string{"Sec-Fetch-Site": "same-site"}, 403},
	}
	for _, p := range paths {
		for _, c := range cases {
			h := map[string]string{"Content-Type": "application/json", "X-Easy-Engine": "1"}
			for k, v := range c.hdr {
				h[k] = v
			}
			code, _, rh := do(t, p.method, srv.URL+p.path, p.body, h)
			if c.want == 403 && code != 403 || c.want == 0 && code != 200 {
				t.Errorf("%s %s [%s]: got %d", p.method, p.path, c.name, code)
			}
			if hasCORS(rh) {
				t.Errorf("%s %s [%s]: CORS header reached the client: %v", p.method, p.path, c.name, rh)
			}
		}
		code, _, rh := do(t, "OPTIONS", srv.URL+p.path, "", map[string]string{"Origin": own, "Access-Control-Request-Method": "POST"})
		if code != 403 || hasCORS(rh) {
			t.Errorf("OPTIONS %s: %d %v", p.path, code, rh)
		}
	}
}

// The local model server reflects Origin and the proxy must not pass that on; a client's
// Authorization must not reach it.
func TestProxyDropsCORSAndAuthorizationLocal(t *testing.T) {
	_, _, srv := readyEnv(t)
	_, _, h := do(t, "GET", srv.URL+"/props", "", map[string]string{"Origin": srv.URL, "Authorization": "Bearer secret"})
	if hasCORS(h) {
		t.Fatalf("CORS from upstream reached the client: %v", h)
	}
	if v := h.Values("X-Fake-Saw-Authorization"); len(v) != 1 || v[0] != "" {
		t.Fatalf("Authorization reached the local model server: %q", v)
	}
}

func TestSetupPageNeedsNoTokenOnLoopback(t *testing.T) {
	_, _, srv := readyEnv(t)
	if c, _, _ := do(t, "GET", srv.URL+"/", "", map[string]string{"Authorization": "Bearer junk"}); c != 200 {
		t.Fatalf("loopback with a stray bearer: %d", c)
	}
}

// lanServer is an engine on a non-loopback listener (the test server's address stands in for it).
func lanServer(t *testing.T, mut func(*api.API)) (*testkit.Env, *run.Supervisor, *httptest.Server) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	a := &api.API{S: s, AnyHost: true}
	mut(a)
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv
}

func bearer(tok string) map[string]string { return map[string]string{"Authorization": "Bearer " + tok} }

func TestTokenOnNonLoopback(t *testing.T) {
	st := tokens.Open(t.TempDir())
	tok, err := st.Master()
	if err != nil {
		t.Fatal(err)
	}
	_, _, srv := lanServer(t, func(a *api.API) { a.Tokens = st })
	paths := []string{"/engine/v1/status", "/slots", "/props"}
	// the setup page is a fixed file that reads the token from the address: it loads without one
	for _, h := range []map[string]string{nil, bearer("nope"), bearer(tok)} {
		if c, _, _ := do(t, "GET", srv.URL+"/", "", h); c != 200 {
			t.Errorf("setup page: %d", c)
		}
	}
	if c, _, _ := do(t, "POST", srv.URL+"/", "{}", nil); c != 401 {
		t.Errorf("POST / without a token: %d", c)
	}
	for _, p := range paths {
		for name, h := range map[string]map[string]string{
			"missing":    nil,
			"wrong":      bearer("nope"),
			"empty":      {"Authorization": "Bearer "},
			"other kind": {"Authorization": "Basic " + tok},
			"truncated":  bearer(tok[:len(tok)-1]),
		} {
			c, _, rh := do(t, "GET", srv.URL+p, "", h)
			if c != 401 || rh.Get("WWW-Authenticate") != "Bearer" {
				t.Errorf("%s [%s]: %d %q", p, name, c, rh.Get("WWW-Authenticate"))
			}
		}
		if c, _, _ := do(t, "GET", srv.URL+p, "", bearer(tok)); c != 200 {
			t.Errorf("%s with the token: %d", p, c)
		}
		// with a token the Host header is not restricted
		h := bearer(tok)
		h["Host"] = "lan-box.local:47480"
		if c, _, _ := do(t, "GET", srv.URL+p, "", h); c != 200 {
			t.Errorf("%s from a LAN host name: %d", p, c)
		}
	}
	if c, _, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, bearer(tok)); c != 200 {
		t.Errorf("proxy POST with the token: %d", c)
	}
	// cross-site is refused even with the token
	h := bearer(tok)
	h["Origin"] = "http://evil.example"
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", h); c != 403 {
		t.Errorf("foreign origin with the token: %d", c)
	}
	// the token is never in status
	_, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", bearer(tok))
	if strings.Contains(body, tok) {
		t.Fatal("token in status")
	}
	// a new master takes effect at once
	n, _ := st.NewMaster()
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", bearer(tok)); c != 401 {
		t.Errorf("old token after --new: %d", c)
	}
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", bearer(n)); c != 200 {
		t.Errorf("new token: %d", c)
	}
}

func TestTokenNeverRequiredOnLoopback(t *testing.T) {
	st := tokens.Open(t.TempDir())
	st.Master()
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	srv := httptest.NewServer((&api.API{S: s, Tokens: st}).Handler()) // AnyHost false: loopback
	defer srv.Close()
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil); c != 200 {
		t.Fatalf("loopback asked for a token: %d", c)
	}
}

func TestOpenListenerNeedsNoToken(t *testing.T) {
	_, _, srv := lanServer(t, func(a *api.API) { a.Open = true })
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", map[string]string{"Host": "lan-box.local"}); c != 200 {
		t.Fatalf("--open: %d", c)
	}
}

func TestNonLoopbackWithNeitherTokenNorOpenRefuses(t *testing.T) {
	_, _, srv := lanServer(t, func(a *api.API) {})
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil); c != 401 {
		t.Fatalf("fail-closed: %d", c)
	}
}

// External mode: the engine's own bearer is consumed; the player's key goes upstream.
func TestExternalConsumesEngineToken(t *testing.T) {
	var auth atomic.Value
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth.Store(r.Header.Get("Authorization"))
		w.Header().Set("Access-Control-Allow-Origin", "*")
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer up.Close()
	st := tokens.Open(t.TempDir())
	tok, _ := st.Master()
	_, s, srv := lanServer(t, func(a *api.API) { a.Tokens = st })
	h := bearer(tok)
	h["X-Easy-Engine"] = "1"
	put := func(body string) {
		if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", body, h); c != 200 {
			t.Fatalf("settings %d %s", c, b)
		}
	}
	put(`{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":"m","api_key":"player-key"}}`)
	testkit.WaitPhase(t, s, "external", 5*time.Second)
	c, _, rh := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, h)
	if c != 200 || auth.Load() != "Bearer player-key" {
		t.Fatalf("with a key: %d upstream saw %q", c, auth.Load())
	}
	if hasCORS(rh) {
		t.Fatalf("CORS from the external endpoint reached the client: %v", rh)
	}
	put(`{"clear_api_key":true}`)
	if c, _, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, h); c != 200 || auth.Load() != "" {
		t.Fatalf("no key: %d upstream saw %q (the engine token must not leak)", c, auth.Load())
	}
}

// A partial settings body changes only what it names.
func TestSettingsPartialUpdate(t *testing.T) {
	_, _, srv := readyEnv(t)
	put := func(body string) {
		t.Helper()
		if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", body, csrf); c != 200 {
			t.Fatalf("settings %q: %d %s", body, c, b)
		}
	}
	status := func() (ext struct {
		BaseURL   string `json:"base_url"`
		Model     string `json:"model"`
		APIKeySet bool   `json:"api_key_set"`
	}, linger int, kv string) {
		_, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
		var st struct {
			External struct {
				BaseURL   string `json:"base_url"`
				Model     string `json:"model"`
				APIKeySet bool   `json:"api_key_set"`
			}
			LingerS int    `json:"linger_s"`
			KVType  string `json:"kv_type"`
		}
		json.Unmarshal([]byte(body), &st)
		return st.External, st.LingerS, st.KVType
	}
	put(`{"mode":"local","external":{"base_url":"http://x.example/v1","model":"m1","api_key":"k"},"linger_s":5,"kv_type":"q8_0"}`)
	put(`{"linger_s":60}`)
	ext, linger, kv := status()
	if ext.BaseURL != "http://x.example/v1" || ext.Model != "m1" || !ext.APIKeySet || linger != 60 || kv != "q8_0" {
		t.Fatalf("a partial body changed other fields: %+v linger=%d kv=%s", ext, linger, kv)
	}
	put(`{"external":{"model":"m2"}}`)
	if ext, linger, _ = status(); ext.BaseURL != "http://x.example/v1" || ext.Model != "m2" || !ext.APIKeySet || linger != 60 {
		t.Fatalf("external sub-field: %+v linger=%d", ext, linger)
	}
	put(`{"external":{"base_url":""}}`)
	if ext, _, _ = status(); ext.BaseURL != "" || ext.Model != "m2" {
		t.Fatalf("explicit empty base_url: %+v", ext)
	}
	put(`{"clear_api_key":true}`)
	if ext, _, _ = status(); ext.APIKeySet || ext.Model != "m2" {
		t.Fatalf("clear_api_key: %+v", ext)
	}
	put(`{}`)
	if _, linger, _ = status(); linger != 60 {
		t.Fatalf("empty body changed linger: %d", linger)
	}
}

func TestStoppingRefusesAttachAndShowsPhase(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	var attached atomic.Int32
	a := &api.API{S: s, Attach: func(int) error { attached.Add(1); return nil }}
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/attach", `{"pid":1}`, csrf); c != 200 {
		t.Fatalf("attach before stopping: %d", c)
	}
	a.SetStopping()
	c, body, _ := do(t, "POST", srv.URL+"/engine/v1/attach", `{"pid":1}`, csrf)
	if c != 409 || !strings.Contains(body, "stopping") || attached.Load() != 1 {
		t.Fatalf("attach while stopping: %d %s (attached %d)", c, body, attached.Load())
	}
	_, sb, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
	var st struct{ Phase string }
	json.Unmarshal([]byte(sb), &st)
	if st.Phase != "stopping" {
		t.Fatalf("phase %q", st.Phase)
	}
}

type badBody struct{}

func (badBody) Read([]byte) (int, error) { return 0, errors.New("boom") }

func TestUnreadableBodyIs400AndAppliedTellsTheTruth(t *testing.T) {
	_, s, _ := samplerEnv(t)
	h := (&api.API{S: s}).Handler()
	req := httptest.NewRequest("POST", "/v1/chat/completions", badBody{})
	req.Host = "127.0.0.1:1"
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("unreadable body: %d %s", rec.Code, rec.Body)
	}
	if v := rec.Header().Get("X-Easy-Sampler-Applied"); strings.Contains(v, "enforce") && !strings.HasPrefix(v, "none") {
		t.Fatalf("Applied claims a rewrite that did not happen: %q", v)
	}
	req = httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("not json"))
	req.Host = "127.0.0.1:1"
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if v := rec.Header().Get("X-Easy-Sampler-Applied"); !strings.HasPrefix(v, "none;") {
		t.Fatalf("a body that was not rewritten: Applied=%q", v)
	}
}
