package api

import (
	"context"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// Checker decides whether a bearer token is accepted and names the client it belongs to.
// internal/tokens.Store is the implementation; the gate depends on nothing else about it.
type Checker interface {
	Check(bearer string) (clientID string, ok bool)
}

type ctxKey int

const (
	clientKey ctxKey = iota
	bearerKey
)

// Client ids the gate attaches when no token named one.
const (
	ClientLoopback = "loopback" // a loopback listener: no token is asked for
	ClientOpen     = "open"     // a non-loopback listener started with --open
	ClientPage     = "page"     // the setup page on a tokened listener (no token is sent with a navigation)
)

// ClientID is the id of the client the gate accepted for this request ("master" for the
// master token, an issued token's id, ClientLoopback or ClientOpen), or "" outside the gate.
func ClientID(ctx context.Context) string {
	s, _ := ctx.Value(clientKey).(string)
	return s
}

// bearerConsumed reports whether the request's Authorization header was the engine's own token.
func bearerConsumed(ctx context.Context) bool {
	b, _ := ctx.Value(bearerKey).(bool)
	return b
}

func loopbackHost(hostport string) bool {
	h := hostport
	if hh, _, err := net.SplitHostPort(h); err == nil {
		h = hh
	}
	h = strings.Trim(h, "[]")
	return h == "127.0.0.1" || h == "localhost" || h == "::1"
}

// ownOrigin reports whether an Origin header names this engine: the scheme the request came in
// on, and the request's own Host.
func ownOrigin(r *http.Request, origin string) bool {
	u, err := url.Parse(origin)
	if err != nil || u.Host == "" {
		return false
	}
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return u.Scheme == scheme && strings.EqualFold(u.Host, r.Host)
}

func setupPage(r *http.Request) bool {
	return (r.Method == http.MethodGet || r.Method == http.MethodHead) && (r.URL.Path == "/" || r.URL.Path == "/index.html")
}

func bearerOf(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if len(h) > 7 && strings.EqualFold(h[:7], "bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}

// gate is the one door every path passes (engine, setup page and proxy):
//
//  1. a loopback listener accepts only a loopback Host name;
//  2. a request from another site is refused, GET included: an Origin that is not the engine's
//     own, a Sec-Fetch-Site other than same-origin or none, and every OPTIONS preflight;
//  3. a non-loopback listener needs a bearer token on every path but the setup page itself
//     (GET / and /index.html, a fixed file that reads the token from the address), unless it
//     was started with --open. With neither a token store nor Open it refuses everything.
//
// A loopback listener never asks for a token. The accepted client id goes on the request context.
func (a *API) gate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !a.AnyHost && !loopbackHost(r.Host) {
			fail(w, http.StatusForbidden, "bad host")
			return
		}
		if o := r.Header.Get("Origin"); o != "" && !ownOrigin(r, o) {
			fail(w, http.StatusForbidden, "cross-site request refused")
			return
		}
		if s := r.Header.Get("Sec-Fetch-Site"); s != "" && s != "same-origin" && s != "none" {
			fail(w, http.StatusForbidden, "cross-site request refused")
			return
		}
		if r.Method == http.MethodOptions {
			fail(w, http.StatusForbidden, "preflight refused")
			return
		}
		client, used := ClientLoopback, false
		switch {
		case !a.AnyHost:
		case a.Tokens != nil && setupPage(r):
			// The page itself is a fixed file with nothing secret in it; the browser cannot send
			// the token with a navigation, and the page's script needs to load to read it from
			// the address. Every call the page makes goes through the token.
			client = ClientPage
		case a.Tokens != nil:
			id, ok := a.Tokens.Check(bearerOf(r))
			if !ok {
				w.Header().Set("WWW-Authenticate", "Bearer")
				fail(w, http.StatusUnauthorized, "bearer token required (easy-engine token prints it)")
				return
			}
			client, used = id, true
		case a.Open:
			client = ClientOpen
		default:
			w.Header().Set("WWW-Authenticate", "Bearer")
			fail(w, http.StatusUnauthorized, "no token configured")
			return
		}
		ctx := context.WithValue(r.Context(), clientKey, client)
		ctx = context.WithValue(ctx, bearerKey, used)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
