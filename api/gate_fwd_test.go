package api_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/tokens"
)

// D7: a request on the loopback listener that carries a forwarding header is another machine's,
// passed on by something on this machine: remote everywhere. It needs a token as on a
// non-loopback listener (none: 401; a paired token: scoped, the reduced status; the master key:
// everything, as master@<ip>), never the identity "loopback". A plain loopback request is
// unchanged, and the setup page still loads in a local browser.
func TestForwardedOnLoopbackIsRemote(t *testing.T) {
	_, s, _ := readyEnv(t)
	st := tokens.Open(t.TempDir())
	master, err := st.Master()
	if err != nil {
		t.Fatal(err)
	}
	_, paired, err := st.Issue("Kitchen PC")
	if err != nil {
		t.Fatal(err)
	}
	loop := httptest.NewServer((&api.API{S: s, Tokens: st}).Handler()) // AnyHost false: the loopback listener
	t.Cleanup(loop.Close)
	bare := httptest.NewServer((&api.API{S: s}).Handler()) // a loopback listener with no token store
	t.Cleanup(bare.Close)
	full := func(b string) int { var m map[string]any; json.Unmarshal([]byte(b), &m); return len(m) }
	_, plainStatus, _ := do(t, "GET", loop.URL+"/engine/v1/status", "", csrf)
	for _, h := range []map[string]string{{"X-Forwarded-For": "10.1.2.3"}, {"Forwarded": "for=10.1.2.3"}, {"X-Real-IP": "10.1.2.3"}} {
		var name string
		for k := range h {
			name = k
		}
		// no token: 401 on every control route but the keyless ones, and on the proxy
		for _, url := range []string{loop.URL, bare.URL} {
			for _, r := range scopeRoutes {
				if r.stream || r.path == "hello" || r.path == "pair" || strings.HasPrefix(r.path, "pair/") {
					continue
				}
				if c, b, _ := do(t, r.method, url+"/engine/v1/"+r.path, "{}", with(csrf, h)); c != 401 {
					t.Errorf("%s, no token, %s %s: %d %.200s", name, r.method, r.path, c, b)
				}
			}
			for _, p := range []string{"/v1/models", "/slots", "/props"} {
				if c, b, _ := do(t, "GET", url+p, "", h); c != 401 {
					t.Errorf("%s, no token, GET %s: %d %.200s", name, p, c, b)
				}
			}
		}
		// a paired token: scoped as a paired token is, the reduced status
		ph := with(csrf, h, bearer(paired))
		for _, r := range scopeRoutes {
			c, b, _ := do(t, r.method, loop.URL+"/engine/v1/"+r.path, "{}", ph)
			if r.allowed && notAllowed(c, b) {
				t.Errorf("%s, paired, %s %s refused: %d %s", name, r.method, r.path, c, b)
			}
			if !r.allowed && !notAllowed(c, b) {
				t.Errorf("%s, paired, %s %s: %d %.200s, want 403 not_allowed", name, r.method, r.path, c, b)
			}
		}
		if c, b, _ := do(t, "GET", loop.URL+"/engine/v1/status", "", ph); c != 200 || full(b) >= full(plainStatus) {
			t.Errorf("%s, paired status: %d, %d fields (plain %d)", name, c, full(b), full(plainStatus))
		}
		if c, b, _ := do(t, "GET", loop.URL+"/slots", "", ph); !notAllowed(c, b) {
			t.Errorf("%s, paired GET /slots: %d %.200s", name, c, b)
		}
		// the master key: everything, as master@<ip>, never "loopback"
		mh := with(csrf, h, bearer(master))
		for _, r := range scopeRoutes {
			if r.stream || r.path == "shutdown" || strings.HasPrefix(r.path, "pairing") {
				continue
			}
			if c, b, _ := do(t, r.method, loop.URL+"/engine/v1/"+r.path, "{}", mh); notAllowed(c, b) || c == 401 {
				t.Errorf("%s, master, %s %s: %d %s", name, r.method, r.path, c, b)
			}
		}
		ah := openHold(t, loop, `{"hold":true,"name":"f","profile":{"slots":1}}`, with(h, bearer(master)))
		if id, _ := ah.first["client_id"].(string); ah.status != 200 || !strings.HasPrefix(id, "master@") || ah.first["local"] != false {
			t.Errorf("%s, master attach: %d %v", name, ah.status, ah.first)
		}
		ah.cancel()
	}
	// plain loopback: unchanged
	ph := openHold(t, loop, `{"hold":true,"name":"p","profile":{"slots":1}}`, nil)
	if ph.status != 200 || ph.first["client_id"] != "loopback" {
		t.Errorf("plain loopback attach: %d %v", ph.status, ph.first)
	}
	ph.cancel()
	for _, r := range scopeRoutes {
		if r.stream || r.path == "shutdown" {
			continue
		}
		if c, b, _ := do(t, r.method, loop.URL+"/engine/v1/"+r.path, "{}", csrf); c == 401 || (notAllowed(c, b) && r.path != "pairing/answer") {
			t.Errorf("plain loopback %s %s: %d %s", r.method, r.path, c, b)
		}
	}
	if c, b, _ := do(t, "GET", loop.URL+"/", "", nil); c != 200 || !strings.Contains(b, "<html") {
		t.Errorf("the setup page in a local browser: %d %.100s", c, b)
	}
}
