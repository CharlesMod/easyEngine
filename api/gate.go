package api

import (
	"context"
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/CharlesMod/easyEngine/internal/tokens"
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
	verdictKey
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
// A loopback listener never asks for a token of a request from this machine. A request on it that
// carries a forwarding header (Forwarded, X-Forwarded-*, X-Real-IP, Via, Tailscale-*) is something
// on this machine passing on another machine's request: it is remote, and asked for a token as on
// a non-loopback listener (the master key, or a paired token scoped as everywhere); without one it
// is 401. A reverse proxy in front of the engine must pass a token. The accepted client id goes on
// the request context.
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
		client, used, verdict := ClientLoopback, false, tokenNone
		fwd := forwarded(r)
		switch {
		case !a.AnyHost && !fwd:
		case !a.AnyHost && a.Tokens == nil && !keyless(r):
			// forwarded to a loopback listener that has no token store: a token is required and
			// none can be accepted
			unauthorized(w, CodeTokenRequired, "bearer token required (easy-engine token prints it)")
			return
		case a.Tokens != nil && setupPage(r):
			// The page itself is a fixed file with nothing secret in it; the browser cannot send
			// the token with a navigation, and the page's script needs to load to read it from
			// the address. Every call the page makes goes through the token.
			client = ClientPage
		case a.Tokens != nil:
			bearer := bearerOf(r)
			id, v := a.verify(bearer)
			verdict = v
			if v != tokenValid && keyless(r) {
				// hello and the client's side of pairing need no token and never answer 401;
				// hello says what it made of the token it was sent.
				client = ""
				break
			}
			switch v {
			case tokenValid:
			case tokenRevoked:
				unauthorized(w, CodeTokenRevoked, "bearer token required (easy-engine token prints it)")
				return
			case tokenUnknown:
				unauthorized(w, CodeTokenInvalid, "bearer token required (easy-engine token prints it)")
				return
			default:
				unauthorized(w, CodeTokenRequired, "bearer token required (easy-engine token prints it)")
				return
			}
			if !pairedMay(id, r) {
				failCode(w, http.StatusForbidden, CodeNotAllowed, scopeRefusal)
				return
			}
			client, used = id, true
			a.sawClient(id)
		case a.Open:
			client = ClientOpen
			if !peerLocal(r) && (r.URL.Path == "/engine" || strings.HasPrefix(r.URL.Path, "/engine/")) && !pairedMay(ClientOpen, r) {
				// an --open caller from another machine may do on the control API exactly what a
				// paired token may (scope.go); administration from another machine needs the
				// master key. A caller of this machine is unchanged.
				failCode(w, http.StatusForbidden, CodeNotAllowed, scopeRefusal)
				return
			}
		case keyless(r):
			client = ""
		default:
			unauthorized(w, CodeTokenRequired, "no token configured")
			return
		}
		ctx := context.WithValue(r.Context(), clientKey, client)
		ctx = context.WithValue(ctx, bearerKey, used)
		ctx = context.WithValue(ctx, verdictKey, verdict)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// The codes a 401 body carries beside its message: {"error": "<message>", "code": "<code>"}.
const (
	CodeTokenRequired = "token_required" // no bearer was sent
	CodeTokenRevoked  = "token_revoked"  // the bearer was issued here and revoked: pair again
	CodeTokenInvalid  = "token_invalid"  // the bearer is not one this engine knows
)

func unauthorized(w http.ResponseWriter, code, msg string) {
	w.Header().Set("WWW-Authenticate", "Bearer")
	writeJSON(w, http.StatusUnauthorized, map[string]string{"error": msg, "code": code})
}

// What the gate made of a request's bearer.
type tokenVerdict int

const (
	tokenNone    tokenVerdict = iota // none sent
	tokenValid                       // accepted
	tokenRevoked                     // issued here, then revoked
	tokenUnknown                     // sent, not known
)

// Verifier is a Checker that also tells a revoked token from an unknown one
// (internal/tokens.Store). A Checker without it never says revoked.
type Verifier interface {
	Verify(bearer string) (clientID string, v tokens.Verdict)
}

func (a *API) verify(bearer string) (string, tokenVerdict) {
	if bearer == "" {
		return "", tokenNone
	}
	if v, ok := a.Tokens.(Verifier); ok {
		id, verdict := v.Verify(bearer)
		switch verdict {
		case tokens.Valid:
			return id, tokenValid
		case tokens.Revoked:
			return id, tokenRevoked
		}
		return "", tokenUnknown
	}
	if id, ok := a.Tokens.Check(bearer); ok {
		return id, tokenValid
	}
	return "", tokenUnknown
}

func verdictOf(ctx context.Context) tokenVerdict {
	v, _ := ctx.Value(verdictKey).(tokenVerdict)
	return v
}

// keyless: the paths a client on the network may call before it holds a token: hello, and its
// own side of pairing (ask, collect, withdraw). The host, origin and preflight rules still apply.
func keyless(r *http.Request) bool {
	switch p := r.URL.Path; {
	case p == "/engine/v1/hello":
		return r.Method == http.MethodGet || r.Method == http.MethodHead
	case p == "/engine/v1/pair":
		return r.Method == http.MethodPost
	case strings.HasPrefix(p, "/engine/v1/pair/"):
		rest := strings.TrimPrefix(p, "/engine/v1/pair/")
		return rest != "" && !strings.Contains(rest, "/") && (r.Method == http.MethodGet || r.Method == http.MethodDelete)
	}
	return false
}
