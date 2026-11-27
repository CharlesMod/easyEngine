package api

import (
	"errors"
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
	Phase         string `json:"phase"`
	Card          string `json:"card"`
	Pairing       bool   `json:"pairing"`
	TokenRequired bool   `json:"token_required"`
	// Said only when the request sent a bearer that this engine checks.
	Paired   *bool  `json:"paired,omitempty"`
	ClientID string `json:"client_id,omitempty"`
	Revoked  bool   `json:"revoked,omitempty"`
}

// The codes the pairing paths answer with beside their message.
const (
	CodePairingOff = "pairing_off" // POST pair where pairing is not offered
	CodePairDenied = "pair_denied" // 403: the person pressed Deny
	CodePairGone   = "pair_gone"   // 410: withdrawn, pushed out, collected, or another life
	CodeNotAllowed = "not_allowed" // 403: answering needs loopback or the master token
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
		h.Version, h.Phase, h.ModelName = st.Version, st.Phase, st.ModelName
		if h.ModelName == "" {
			h.ModelName = st.Model
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
		fail(w, http.StatusBadRequest, err.Error())
		return
	}
	from := r.RemoteAddr
	if h, _, err := net.SplitHostPort(from); err == nil {
		from = h
	}
	req, id, err := a.Pair.Ask(in.Name, from)
	if err != nil {
		failCode(w, http.StatusGone, CodePairGone, err.Error())
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
		fail(w, http.StatusInternalServerError, "could not issue the token: "+err.Error())
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
	if in.Allow == nil || (in.Key == "") == (in.Code == "") {
		fail(w, http.StatusBadRequest, "send {key or code, allow: true|false}")
		return
	}
	if a.Pair == nil {
		fail(w, http.StatusNotFound, pair.ErrNoRequest.Error())
		return
	}
	key := in.Key
	if in.Code != "" {
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
	req, err := a.Pair.Answer(key, *in.Allow)
	if err != nil {
		fail(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"key": req.Key, "name": req.Name, "code": req.Code, "allowed": *in.Allow})
}

// namer gives an issued client id's name (internal/tokens.Store).
type namer interface {
	NameOf(id string) string
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
