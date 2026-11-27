package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

type stubChecker map[string]string // token -> client id

func (s stubChecker) Check(b string) (string, bool) { id, ok := s[b]; return id, ok }

// The gate alone, with a stub behind it: the client id it attaches, and whether the engine's
// own bearer is marked consumed.
func TestGateClientIDs(t *testing.T) {
	cases := []struct {
		name     string
		a        *API
		host     string
		auth     string
		code     int
		id       string
		consumed bool
	}{
		{"loopback", &API{}, "127.0.0.1:47480", "", 200, ClientLoopback, false},
		{"loopback ignores a token", &API{Tokens: stubChecker{"t": "master"}}, "localhost", "", 200, ClientLoopback, false},
		{"loopback wrong host", &API{}, "evil.example", "", 403, "", false},
		{"master", &API{AnyHost: true, Tokens: stubChecker{"t": "master"}}, "lan.local", "Bearer t", 200, "master", true},
		{"issued", &API{AnyHost: true, Tokens: stubChecker{"t": "master", "u": "c1"}}, "lan.local", "bearer u", 200, "c1", true},
		{"no bearer", &API{AnyHost: true, Tokens: stubChecker{"t": "master"}}, "lan.local", "", 401, "", false},
		{"open", &API{AnyHost: true, Open: true}, "lan.local", "", 200, ClientOpen, false},
		{"neither", &API{AnyHost: true}, "lan.local", "", 401, "", false},
	}
	for _, c := range cases {
		var id string
		var consumed bool
		h := c.a.gate(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id, consumed = ClientID(r.Context()), bearerConsumed(r.Context())
		}))
		req := httptest.NewRequest("GET", "/v1/models", nil)
		req.Host = c.host
		if c.auth != "" {
			req.Header.Set("Authorization", c.auth)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != c.code || id != c.id || consumed != c.consumed {
			t.Errorf("%s: code=%d id=%q consumed=%v", c.name, rec.Code, id, consumed)
		}
	}
}
