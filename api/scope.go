package api

import (
	"net/http"
	"path"
	"strings"

	"github.com/CharlesMod/easyEngine/internal/tokens"
)

// Scope of a paired token (docs/ENGINE.md section 7f). A paired token is a client on another
// machine, such as a game mod: it may use the model and read how the engine is doing, and
// nothing else. The list below is an allow list; whatever it does not name is refused with 403
// not_allowed. The master token, the loopback listener, the setup page and an open listener
// are not scoped. The list can grow when a client needs one more call.

const scopeRefusal = "this call needs the engine owner's own key or the engine's own machine; a paired client may use the model and read the engine's status"

// proxyPost and proxyGet are the model server's own paths a paired client may call (compared on
// the cleaned path).
var (
	proxyPost = map[string]bool{
		"/v1/chat/completions": true, "/chat/completions": true,
		"/v1/completions": true, "/completions": true, "/completion": true,
		"/tokenize": true, "/detokenize": true, "/apply-template": true,
	}
	proxyGet = map[string]bool{
		"/v1/models": true, "/models": true, "/health": true, "/v1/health": true,
	}
)

// pairedMay reports whether the caller with this client id may make this request. Only a paired
// token (any id but the master's) is limited. It is called once, from the gate.
func pairedMay(id string, r *http.Request) bool {
	if id == tokens.MasterID {
		return true
	}
	p := r.URL.Path
	if strings.HasPrefix(p, "/engine/") || p == "/engine" {
		// The control API answers on the exact path only, so the exact path is what is judged.
		if keyless(r) {
			return true
		}
		switch {
		case p == "/engine/v1/status":
			return r.Method == http.MethodGet
		case p == "/engine/v1/attach":
			return r.Method == http.MethodPost
		case strings.HasPrefix(p, "/engine/v1/attach/"):
			rest := strings.TrimPrefix(p, "/engine/v1/attach/")
			return rest != "" && !strings.Contains(rest, "/") && r.Method == http.MethodDelete
		}
		return false
	}
	// The model server's paths. A dot segment is never needed by a client; refuse it so the
	// path the engine judged is the path the model server gets.
	for _, seg := range strings.Split(p, "/") {
		if seg == "." || seg == ".." {
			return false
		}
	}
	c := path.Clean("/" + p)
	switch r.Method {
	case http.MethodPost:
		return proxyPost[c]
	case http.MethodGet:
		return proxyGet[c]
	}
	return false
}
