package app

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"strconv"

	"github.com/CharlesMod/easyEngine/internal/legacy"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/lifecycle"
)

// ErrStopping is what attaching to an engine that has begun shutting down returns (HTTP 409).
var ErrStopping = errors.New("engine is stopping")

// IsLoopbackHost reports whether a --host value (or a lock's host) is a loopback address.
// Empty means the default, 127.0.0.1.
func IsLoopbackHost(host string) bool {
	switch host {
	case "", "127.0.0.1", "localhost", "::1", "[::1]":
		return true
	}
	return false
}

// Target is how to reach a running engine: its base URL and, when it listens beyond loopback
// with a token, the token to send as `Authorization: Bearer`.
type Target struct {
	Host  string
	Port  int
	Token string
}

// Base is the engine's base URL, e.g. http://127.0.0.1:47480.
func (t Target) Base() string {
	return "http://" + net.JoinHostPort(t.dialHost(), strconv.Itoa(t.Port))
}

// dialHost is the address to call: a wildcard bind is reached through loopback.
func (t Target) dialHost() string {
	switch t.Host {
	case "", "0.0.0.0", "::", "[::]":
		return "127.0.0.1"
	}
	return t.Host
}

// NewRequest builds a request to the engine's control API (path like "/engine/v1/status") with
// the anti-CSRF header and the token, when there is one.
func (t Target) NewRequest(method, path string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequest(method, t.Base()+path, body)
	if err != nil {
		return nil, err
	}
	legacy.MarkRequest(req.Header, "X-Easy-Engine")
	if t.Token != "" {
		req.Header.Set("Authorization", "Bearer "+t.Token)
	}
	return req, nil
}

// targetFor builds the Target for a lock. A lock that says the engine listens beyond loopback
// gets the token from the data dir's token file (never created here); an old lock has no host
// and means loopback.
func targetFor(dir string, d lifecycle.LockData) Target {
	t := Target{Host: d.Host, Port: d.Port}
	if !IsLoopbackHost(d.Host) {
		t.Token, _ = tokens.Open(dir).Existing()
	}
	return t
}

// Find returns how to reach a healthy engine in dir, if any.
func Find(dir string) (Target, bool) {
	d, err := lifecycle.ReadLock(filepath.Join(dir, "engine.lock"))
	if err != nil || !alive(d) {
		return Target{}, false
	}
	t := targetFor(dir, d)
	if !statusOK(t) {
		return Target{}, false
	}
	return t, true
}

// SetupURL is the setup page's address for an engine: the token, if the engine needs one,
// rides in the fragment (the page takes it from there and clears the address bar). Callers
// print URL without the token (Base + "/") and open the one this returns.
func SetupURL(dir string, t Target) string {
	u := t.Base() + "/"
	if !IsLoopbackHost(t.Host) {
		if tok, ok := tokens.Open(dir).Existing(); ok {
			u += "#token=" + tok
		}
	}
	return u
}

// EngineTarget is the Target for an engine this process just started or attached to.
func EngineTarget(dir, host string, port int) Target {
	return targetFor(dir, lifecycle.LockData{Host: host, Port: port})
}

func statusErr(resp *http.Response) error {
	m, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	if resp.StatusCode == http.StatusConflict {
		return fmt.Errorf("attach: %w", ErrStopping)
	}
	return fmt.Errorf("attach: %s %s", resp.Status, m)
}
