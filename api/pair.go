package api

import (
	"errors"
	"log"
	"net"
	"net/http"
	"strings"

	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/pair"
	"github.com/CharlesMod/easyEngine/run"
)

// Hello is GET /engine/v1/hello: who this engine is, for a client that has found it on the
// network and holds no token yet. It carries nothing secret: no paths, no user name, no tokens,
// no client list.
type Hello struct {
	Name          string `json:"name"`
	Engine        string `json:"engine"` // always "easy-engine"
	API           int    `json:"api"`
	Version       string `json:"version"`
	EngineID      string `json:"engine_id"`
	ModelName     string `json:"model_name"`
	ModelID       string `json:"model_id"` // the selected model's catalog id; "" when none (or a custom file)
	Phase         string `json:"phase"`
	Card          string `json:"card"`
	Pairing       bool   `json:"pairing"`
	TokenRequired bool   `json:"token_required"`
	// Vision is true while the running model server has its vision projector loaded.
	Vision bool `json:"vision"`
	// Said only when the request sent a bearer that this engine checks.
	Paired   *bool  `json:"paired,omitempty"`
	ClientID string `json:"client_id,omitempty"`
	Revoked  bool   `json:"revoked,omitempty"`
}

// The codes the pairing paths answer with beside their message.
const (
	CodePairingOff  = "pairing_off"  // POST pair where pairing is not offered
	CodePairDenied  = "pair_denied"  // 403: the person pressed Deny
	CodePairGone    = "pair_gone"    // 410: withdrawn, pushed out, collected, or another life
	CodeNotAllowed  = "not_allowed"  // 403: answering needs loopback or the master token
	CodeBadRequest  = "bad_request"  // 400 on a keyless path: the body was not {name} JSON
	CodePairFailed  = "pair_failed"  // 500: the engine could not open the request (detail in its log)
	CodeIssueFailed = "issue_failed" // 500: the engine could not issue the token (detail in its log); the Allow stands
	CodePairChanged = "pair_changed" // 409: the key now shows another code (its source asked again): look again
)

func failCode(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]string{"error": msg, "code": code})
}

// pairRoute serves hello, the client's side of pairing (pair, pair/{id}) and the person's side
// (pairing, pairing/answer). It reports whether name was one of them.
func (a *API) pairRoute(w http.ResponseWriter, r *http.Request, name string, need func(string) bool) bool {
	switch {
	case name == "hello":
		if r.Method != http.MethodHead && !need(http.MethodGet) {
			return true
		}
		writeJSON(w, http.StatusOK, a.hello(r))
	case name == "pair":
		if need(http.MethodPost) {
			a.pairAsk(w, r)
		}
	case strings.HasPrefix(name, "pair/"):
		id := strings.TrimPrefix(name, "pair/")
		switch r.Method {
		case http.MethodGet:
			a.pairCollect(w, r, id)
		case http.MethodDelete:
			if a.Pair == nil || !a.Pair.Withdraw(id) {
				failCode(w, http.StatusGone, CodePairGone, "no such pairing request")
				return true
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			w.Header().Set("Allow", "GET, DELETE")
			fail(w, http.StatusMethodNotAllowed, "use GET or DELETE")
		}
	case name == "pairing":
		if !need(http.MethodGet) || !a.answerer(w, r) {
			return true
		}
		list := []pair.Request{}
		if a.Pair != nil {
			list = a.Pair.Waiting()
		}
		writeJSON(w, http.StatusOK, map[string]any{"offered": a.Pair != nil, "requests": list})
	case name == "pairing/answer":
		if !need(http.MethodPost) || !a.answerer(w, r) {
			return true
		}
		a.pairAnswer(w, r)
	default:
		return false
	}
	return true
}

func (a *API) hello(r *http.Request) Hello {
	h := Hello{Engine: "easy-engine", API: run.APIVersion, EngineID: a.EngineID,
		Pairing: a.Pair != nil, TokenRequired: a.AnyHost && a.Tokens != nil}
	if a.Name != nil {
		h.Name = a.Name()
	}
	if a.S != nil {
		st := a.S.Status()
		h.Version, h.Phase, h.ModelName, h.Vision = st.Version, st.Phase, st.ModelName, st.Vision
		if h.ModelName == "" {
			h.ModelName = st.Model
		}
		if !strings.HasPrefix(st.Model, "custom:") { // a custom file has no catalog id
			h.ModelID = st.Model
		}
		if g := st.GPU; g != nil && g.EngineAdapter >= 0 && g.EngineAdapter < len(g.Adapters) {
			h.Card = g.Adapters[g.EngineAdapter].Name
		}
		if a.stopping.Load() {
			h.Phase = "stopping"
		}
	}
	switch verdictOf(r.Context()) {
	case tokenValid:
		yes := true
		h.Paired, h.ClientID = &yes, ClientID(r.Context())
	case tokenRevoked:
		no := false
		h.Paired, h.Revoked = &no, true
	case tokenUnknown:
		no := false
		h.Paired = &no
	}
	return h
}

func (a *API) pairAsk(w http.ResponseWriter, r *http.Request) {
	if a.Pair == nil {
		failCode(w, http.StatusNotFound, CodePairingOff, "pairing is not offered here (a loopback listener, or --open)")
		return
	}
	var in struct {
		Name string `json:"name"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, 4<<10)
	if err := decode(r, &in); err != nil {
		// a keyless path says no error text of its own: only what was wrong with the call
		failCode(w, http.StatusBadRequest, CodeBadRequest, `send {"name": "<your name>"} as JSON, at most 4 KiB`)
		return
	}
	from := r.RemoteAddr
	if h, _, err := net.SplitHostPort(from); err == nil {
		from = h
	}
	req, id, err := a.Pair.Ask(in.Name, from)
	if err != nil {
		if errors.Is(err, pair.ErrClosed) {
			failCode(w, http.StatusGone, CodePairGone, "the engine is ending")
			return
		}
		log.Printf("easy-engine: pairing request: %v", err) // the detail stays on the engine's side
		failCode(w, http.StatusInternalServerError, CodePairFailed, "the pairing request could not be opened")
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, map[string]string{"id": id, "code": req.Code})
}

// pairCollect holds GET pair/{id} until the person answers. A client that hangs up leaves its
// request waiting: it may ask again with the same id.
func (a *API) pairCollect(w http.ResponseWriter, r *http.Request, id string) {
	if a.Pair == nil {
		failCode(w, http.StatusGone, CodePairGone, "no such pairing request")
		return
	}
	res, err := a.Pair.Wait(r.Context(), id)
	if err != nil {
		if r.Context().Err() != nil {
			return // the client went away
		}
		// The detail (a token file error names the data dir, and with it the user) stays in the
		// engine's log; the client on the network hears only that it failed. The Allow stands.
		log.Printf("easy-engine: pairing: could not issue the token: %v", err)
		failCode(w, http.StatusInternalServerError, CodeIssueFailed, "the token could not be issued; ask again")
		return
	}
	switch res.Outcome {
	case pair.Allowed:
		w.Header().Set("Cache-Control", "no-store")
		writeJSON(w, http.StatusOK, map[string]string{"token": res.Token, "client_id": res.ClientID, "engine_id": a.EngineID})
	case pair.Denied:
		failCode(w, http.StatusForbidden, CodePairDenied, "the request was denied")
	default:
		failCode(w, http.StatusGone, CodePairGone, "no such pairing request (withdrawn, replaced, collected, or the engine restarted)")
	}
}

// answerer: only the engine's own side answers a pairing request: a loopback listener, or the
// master token. A paired client's token never does, nor anything from the network without one.
func (a *API) answerer(w http.ResponseWriter, r *http.Request) bool {
	switch ClientID(r.Context()) {
	case ClientLoopback, tokens.MasterID:
		return true
	}
	failCode(w, http.StatusForbidden, CodeNotAllowed, "only the engine's own side answers pairing (loopback or the master token)")
	return false
}

func (a *API) pairAnswer(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Key   string `json:"key"`
		Code  string `json:"code"`
		Allow *bool  `json:"allow"`
	}
	if err := decode(r, &in); err != nil {
		fail(w, http.StatusBadRequest, err.Error())
		return
	}
	if in.Allow == nil || in.Code == "" {
		fail(w, http.StatusBadRequest, "send {code, allow: true|false}, with the request's key beside the code when you have it")
		return
	}
	if a.Pair == nil {
		fail(w, http.StatusNotFound, pair.ErrNoRequest.Error())
		return
	}
	key := in.Key
	if key == "" { // by code alone: the code is the thing matched
		req, err := a.Pair.ByCode(in.Code)
		switch {
		case errors.Is(err, pair.ErrAmbiguous):
			fail(w, http.StatusConflict, err.Error())
			return
		case err != nil:
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		key = req.Key
	}
	req, err := a.Pair.Answer(key, in.Code, *in.Allow)
	switch {
	case errors.Is(err, pair.ErrChanged):
		failCode(w, http.StatusConflict, CodePairChanged, err.Error())
		return
	case err != nil:
		fail(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"key": req.Key, "name": req.Name, "code": req.Code, "allowed": *in.Allow})
}

// namer gives an issued client id's name (internal/tokens.Store).
type namer interface {
	NameOf(id string) string
}

// ForgetRevoked says the client fact gone for every client seen in this life whose token no
// longer stands (revoked, or the file replaced). The app calls it when tokens.json changes (the
// kernel's word, host.Changes), so a revoke is shown at once, like the gate's refusal.
func (a *API) ForgetRevoked() {
	if a.S == nil || a.S.Hub() == nil || a.Tokens == nil {
		return
	}
	// a revoked token's attachments end at once (and a replaced master's): held streams end with
	// {"end":"revoked"}, slots go vacant, the facts say them gone
	a.S.DetachRevoked(func(id, cred string) bool { return a.credValid(id, cred) })
	v, ok := a.Tokens.(interface{ Valid(id string) bool })
	if !ok {
		return
	}
	a.seen.Range(func(k, _ any) bool {
		id := k.(string)
		if !v.Valid(id) {
			a.seen.Delete(id)
			_, _ = a.S.Hub().Say(event.Event{Name: "client", Key: id, Gone: true}, nil)
		}
		return true
	})
}

// credValid says whether the credential an attachment of identity came with still stands: a
// paired token not revoked, the master token not replaced (cred: its hash). Callers with no
// credential of their own (loopback, --open) always stand.
func (a *API) credValid(id, cred string) bool {
	switch {
	case strings.HasPrefix(id, tokens.MasterID+"@"):
		m, ok := a.Tokens.(interface{ MasterIs(sha string) bool })
		return !ok || m.MasterIs(cred)
	case id == "loopback" || id == "unknown" || id == "engine" || strings.Contains(id, "@"):
		return true
	}
	v, ok := a.Tokens.(interface{ Valid(id string) bool })
	return !ok || v.Valid(id)
}

// sawClient says a paired client as a client fact the first time the gate accepts its token in
// this life: its id and name, never its token.
func (a *API) sawClient(id string) {
	if id == "" || id == tokens.MasterID || a.S == nil || a.S.Hub() == nil {
		return
	}
	if _, was := a.seen.LoadOrStore(id, true); was {
		return
	}
	name := ""
	if n, ok := a.Tokens.(namer); ok {
		name = n.NameOf(id)
	}
	_, _ = a.S.Hub().Say(event.Event{Name: "client", Key: id}, map[string]string{"name": name})
}
