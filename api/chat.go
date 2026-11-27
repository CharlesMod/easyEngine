package api

import (
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/CharlesMod/easyEngine/openfile"
	"github.com/CharlesMod/easyEngine/web"
)

// The chat window. llama-server's own web UI ("Llama Chat") keeps its conversations in browser
// storage keyed by origin, and llama-server's port changes on every launch, so the engine serves
// the UI at a fixed address:
//
//	/chat/      a thin page (web/chat.html): a slim bar and a same-origin frame
//	/chat/ui/*  llama-server itself, the /chat/ui prefix stripped, through the engine's proxy
//
// The UI builds every asset and API address relative to its own location (its base is
// new URL('.', location)) and routes by the address hash (#/chat/<id>), so a prefix-stripping
// proxy is all it needs; llama-server's routes (health, metrics, slots) are not moved.
//
// Like the rest of the proxy this is for a loopback listener; on a listener that needs a token
// a browser navigation cannot send it, so the page is not reachable there.

const (
	chatPath = "/chat/"
	uiPrefix = "/chat/ui"
)

// notReadyPage is what /chat/ shows while there is no local model to talk to.
const notReadyPage = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chat</title>
<style>body{font:16px system-ui,sans-serif;margin:0;display:grid;place-items:center;min-height:100vh;color:#444;background:#fafafa}
@media(prefers-color-scheme:dark){body{color:#ccc;background:#1b1b1b}}p{max-width:24em;text-align:center;padding:1em}</style>
<p>No model is ready to chat yet.<br>Reload this page once one is. <a href="">Reload</a></p>
`

func (a *API) chatReady() bool {
	if a.S.Status().Phase != "ready" {
		return false
	}
	return a.S.Route().Kind == "local"
}

func (a *API) chat(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	switch {
	case path == "/chat" || path == uiPrefix:
		u := *r.URL
		u.Path += "/"
		http.Redirect(w, r, u.String(), http.StatusPermanentRedirect)
	case path == chatPath:
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		if !a.chatReady() {
			w.Write([]byte(notReadyPage))
			return
		}
		w.Write(web.Chat)
	case strings.HasPrefix(path, uiPrefix+"/"):
		if !a.chatReady() {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-store")
			w.Header().Set("Retry-After", "2")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(notReadyPage))
			return
		}
		r2 := r.Clone(r.Context())
		r2.URL.Path = strings.TrimPrefix(path, uiPrefix)
		r2.URL.RawPath = ""
		a.proxy(w, r2)
	default:
		http.NotFound(w, r)
	}
}

// chatHash is the part of the address that names a conversation, as llama's UI writes it.
var chatHash = regexp.MustCompile(`^#/[A-Za-z0-9/_.~%=&?-]{0,300}$`)

// chatOpen is POST /engine/v1/chat/open {"hash": "#/chat/<id>"}: open the full (not popup) chat
// page in the user's default browser, at that conversation. Only a loopback listener does it:
// the browser opens on this machine, at this machine's own address.
func (a *API) chatOpen(w http.ResponseWriter, r *http.Request) {
	if a.AnyHost {
		fail(w, http.StatusForbidden, "chat opens only from a loopback listener")
		return
	}
	var in struct {
		Hash string `json:"hash"`
	}
	if r.ContentLength != 0 {
		if err := decode(r, &in); err != nil {
			fail(w, http.StatusBadRequest, err.Error())
			return
		}
	}
	if in.Hash != "" && !chatHash.MatchString(in.Hash) {
		fail(w, http.StatusBadRequest, "bad hash")
		return
	}
	if a.ChatAddr == "" {
		fail(w, http.StatusServiceUnavailable, "the engine does not know its own address")
		return
	}
	u := (&url.URL{Scheme: "http", Host: a.ChatAddr, Path: chatPath}).String() + in.Hash
	open := a.OpenURL
	if open == nil {
		open = openfile.OpenURL
	}
	if err := open(u); err != nil {
		fail(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, 200, map[string]string{"opened": u})
}
