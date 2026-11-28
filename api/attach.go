package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"

	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/run"
)

// Clients on the wire (docs/ENGINE.md section 7g): POST attach with a profile, a held attach
// (hold: true) whose connection is the attachment's presence, DELETE attach/{id}, and the
// X-Easy-Client header on proxied requests (proxy.go).

// ClientHeader names the attachment a proxied request is for.
const ClientHeader = "X-Easy-Client"

const (
	maxAttachLocal  = 1 << 20  // a local client's attach body
	maxAttachRemote = 64 << 10 // a remote client's: input from the network
)

// forwardHeaders say a request came through a proxy on someone's behalf: such a request is never
// local, whatever its peer address (a header can only take privilege away, never give it).
var forwardHeaders = []string{"Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Real-Ip", "Via"}

func forwarded(r *http.Request) bool {
	for _, k := range forwardHeaders {
		if _, ok := r.Header[k]; ok {
			return true
		}
	}
	for k := range r.Header {
		if strings.HasPrefix(k, "Tailscale-") {
			return true
		}
	}
	return false
}

// localCaller is a request from this machine: a loopback peer whose gate client is the loopback
// listener, the setup page, an open listener or the master token, and that carries no forwarding
// header (Forwarded, X-Forwarded-*, X-Real-IP, Via, Tailscale-*). Every paired token is remote (it
// names a client on another machine), whatever address it comes from. A forwarder on this machine
// that adds no such header (ssh -L, socat) makes its callers local: docs/ENGINE.md section 7g.
func localCaller(r *http.Request) bool {
	if forwarded(r) {
		return false
	}
	h, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(h)
	if ip == nil {
		return false
	}
	if !ip.IsLoopback() {
		// a peer on one of this machine's own addresses is this machine (a listener bound to a
		// network address is reached from here on that address, not on loopback)
		la, ok := r.Context().Value(http.LocalAddrContextKey).(net.Addr)
		ta, ok2 := la.(*net.TCPAddr)
		if !ok || !ok2 || !ta.IP.Equal(ip) {
			return false
		}
	}
	switch ClientID(r.Context()) {
	case ClientLoopback, ClientPage, ClientOpen, tokens.MasterID:
		return true
	}
	return false
}

// identity is who an attach or a proxied request is from: "loopback" for a local caller, else
// the gate's client id: a paired token's id, or, for the callers with no id of their own, the
// address they come from ("open@<ip>" on an --open listener, "master@<ip>" for the master token
// from off the machine), so two of them are two clients.
func identity(r *http.Request) string {
	if localCaller(r) {
		return "loopback"
	}
	id := ClientID(r.Context())
	switch id {
	case "":
		return "unknown"
	case ClientOpen, tokens.MasterID:
		ip, _, err := net.SplitHostPort(r.RemoteAddr)
		if err != nil {
			ip = r.RemoteAddr
		}
		return id + "@" + ip
	}
	return id
}

func clientFail(w http.ResponseWriter, e *run.ClientError) {
	body := map[string]any{"error": e.Msg, "code": e.Code}
	for k, v := range e.Extra {
		body[k] = v
	}
	writeJSON(w, e.Status, body)
}

// attach is POST attach: the old {pid} wire and the profile (docs/ENGINE.md section 7g).
func (a *API) attach(w http.ResponseWriter, r *http.Request) {
	if a.stopping.Load() {
		fail(w, 409, "engine is stopping")
		return
	}
	local := localCaller(r)
	limit := int64(maxAttachRemote)
	if local {
		limit = maxAttachLocal
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			clientFail(w, &run.ClientError{Status: 413, Code: "bad_request", Msg: "the attach body is too large"})
			return
		}
		fail(w, 400, "body must be {\"pid\": n}")
		return
	}
	var in struct {
		Pid     int          `json:"pid"`
		Name    string       `json:"name"`
		Hold    bool         `json:"hold"`
		Profile *run.Profile `json:"profile"`
	}
	if err := json.NewDecoder(bytes.NewReader(raw)).Decode(&in); err != nil {
		fail(w, 400, "body must be {\"pid\": n}")
		return
	}
	if in.Pid <= 0 && !in.Hold {
		fail(w, 400, "body must be {\"pid\": n}") // the old wire's answer, word for word
		return
	}
	if len(in.Name) > 256 {
		clientFail(w, &run.ClientError{Status: 400, Code: "bad_request", Msg: "name is longer than 256 bytes"})
		return
	}
	if in.Pid > 0 && a.Attach == nil && in.Profile == nil && !in.Hold { // the old wire, as before
		fail(w, 501, "attach not available")
		return
	}
	if in.Pid > 0 && local && a.Attach != nil {
		// the pid is a parent as before (serve ends with its parents; the app counts it). It is
		// made a parent first, so a failure here touches no attachment (a standing one of the same
		// key stays as it was); a pid the attach then refuses stays a parent, as the old wire's.
		if err := a.Attach(in.Pid); err != nil {
			fail(w, 400, err.Error())
			return
		}
	}
	v, hold, ce := a.S.AttachClient(run.AttachRequest{Pid: in.Pid, Name: in.Name, Hold: in.Hold, Profile: in.Profile,
		Identity: identity(r), Local: local})
	if ce != nil {
		clientFail(w, ce)
		return
	}
	if a.Present != nil {
		a.Present()
	}
	a.S.Nudge("attach")
	a.sayClientOf(r)
	if hold == nil {
		writeJSON(w, 200, v)
		return
	}
	a.holdAttach(w, r, v, hold)
}

// sayClientOf says a paired client's attachments in its client fact.
func (a *API) sayClientOf(r *http.Request) {
	if id := ClientID(r.Context()); id != "" && id != tokens.MasterID && !localCaller(r) {
		a.sawClient(id)
		a.S.SayClientAttachments(id)
	}
}

// holdAttach serves a held attachment: one NDJSON line now, one each time what it can rely on
// changes (its slots, the context per slot, its options), and a last {"end": ...} line. The
// attachment leaves when this connection ends; a dead peer is found by TCP keepalive on the
// listener (the network timeout, ListenKeepAlive) and by the next write that fails.
func (a *API) holdAttach(w http.ResponseWriter, r *http.Request, v run.AttachView, h *run.AttachHold) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(200)
	out := http.NewResponseController(w)
	enc := json.NewEncoder(w)
	last, _ := json.Marshal(v)
	if enc.Encode(v) != nil || out.Flush() != nil {
		a.S.LeaveHold(h.ID, h.Gen)
		return
	}
	hub := a.S.Hub()
	for {
		select {
		case word := <-h.End:
			enc.Encode(map[string]string{"end": word})
			out.Flush()
			if word == "engine_ending" {
				a.S.LeaveHold(h.ID, h.Gen)
			}
			return // replaced: the attachment stands on its new connection; detached: it left
		case <-h.Notify:
			nv, ok := a.S.ClientView(h.ID)
			if !ok {
				return
			}
			nv.Status = "attached"
			b, _ := json.Marshal(nv)
			if bytes.Equal(b, last) {
				continue
			}
			last = b
			if _, err := w.Write(append(b, '\n')); err != nil || out.Flush() != nil {
				a.S.LeaveHold(h.ID, h.Gen)
				return
			}
		case <-r.Context().Done():
			a.S.LeaveHold(h.ID, h.Gen)
			return
		case <-hub.Done():
			enc.Encode(map[string]string{"end": "engine_ending"})
			out.Flush()
			a.S.LeaveHold(h.ID, h.Gen)
			return
		}
	}
}

// detach is DELETE attach/{id}.
func (a *API) detach(w http.ResponseWriter, r *http.Request, id string) {
	if id == "" || len(id) > 64 || strings.ContainsAny(id, "/?#") {
		clientFail(w, &run.ClientError{Status: 404, Code: "attachment_unknown", Msg: "no such attachment"})
		return
	}
	if ce := a.S.Detach(id, identity(r)); ce != nil {
		clientFail(w, ce)
		return
	}
	a.sayClientOf(r)
	w.WriteHeader(204)
}
