package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/run"
)

var transport = &http.Transport{
	Proxy:                 nil,
	MaxIdleConnsPerHost:   16,
	IdleConnTimeout:       60 * time.Second,
	ResponseHeaderTimeout: 0, // prefill of a long prompt can take minutes
}

func unavailable(w http.ResponseWriter, code int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	if code == 503 {
		w.Header().Set("Retry-After", "2")
	}
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": code, "type": typ, "message": msg}})
}

// yieldedReply answers while the card is yielded to another program: 503, a string code
// "yielded" (clients tell it from "loading" by that), and a longer Retry-After.
func yieldedReply(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "10")
	w.WriteHeader(503)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": "yielded", "type": "unavailable_error", "message": msg}})
}

// proxy forwards every non-/engine path to llama-server (local) or, in external
// mode, /v1/* to the player's endpoint with its key.
func (a *API) proxy(w http.ResponseWriter, r *http.Request) {
	if slotFiles(r) {
		fail(w, http.StatusForbidden, "slot save and restore are the engine's own (keep_context); a client may erase a slot")
		return
	}
	id := requestID(r)
	w.Header().Set(RequestIDHeader, id)
	rt, done := a.S.Enter(id)
	var sw *streamWriter // a streamed request's stream, opened before its hold
	if rt.Hold {
		if a.S.Streamer() && r.Method == http.MethodPost && completionPath(r.URL.Path) {
			want, ok := wantsStream(w, r)
			if !ok {
				return
			}
			if want {
				// The stream is opened as for a slot the upstream has not answered (docs/STREAMING.md
				// section 4): after 2 s of hold the client gets 200 and the waiting word, and the
				// reply follows in the same stream when the model is up.
				sw = newStreamWriter(w, a.fwd(false, id))
				defer sw.finish(r.Context().Err)
			}
		}
		var ok bool
		if rt, done, ok = a.hold(r, id); !ok {
			return // the client went away while held
		}
	}
	var sum *run.RequestEnd // what the streamer read of the reply, said in the request's end
	defer func() { done(sum) }()
	if sw != nil {
		w = sw // from here every answer goes through the stream opened for the hold
	}
	switch rt.Kind {
	case "unavailable":
		if sw != nil {
			sw.refuse(rt) // a JSON 503 as below, or the terminal event if the stream is open
			return
		}
		if rt.Code == "yielded" {
			yieldedReply(w, rt.Message)
			return
		}
		unavailable(w, 503, "unavailable_error", rt.Message)
		return
	case "external":
		if rt.URL == "" {
			unavailable(w, 503, "unavailable_error", "No external endpoint configured")
			return
		}
		if r.URL.Path != "/v1" && !strings.HasPrefix(r.URL.Path, "/v1/") {
			unavailable(w, 404, "not_found_error", "Only /v1 is forwarded in external mode")
			return
		}
		base, err := url.Parse(rt.URL)
		if err != nil {
			unavailable(w, 502, "bad_gateway", "bad external URL")
			return
		}
		if !a.applySampler(w, r) {
			return
		}
		rest := strings.TrimPrefix(r.URL.Path, "/v1")
		consumed := bearerConsumed(r.Context())
		rp := &httputil.ReverseProxy{
			Transport:     transport,
			FlushInterval: -1,
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.Out.URL.Scheme, pr.Out.URL.Host = base.Scheme, base.Host
				pr.Out.URL.Path = strings.TrimRight(base.Path, "/") + rest
				pr.Out.URL.RawPath = ""
				pr.Out.Host = base.Host
				if rt.Key != "" {
					pr.Out.Header.Set("Authorization", "Bearer "+rt.Key)
				} else if consumed {
					pr.Out.Header.Del("Authorization") // the engine's own token is not for the upstream
				}
			},
			ModifyResponse: dropCORS,
			ErrorHandler:   badGateway,
		}
		forward(w, r, rp, a.fwd(true, id).with(sw).ending(&sum))
	default:
		target, err := url.Parse(rt.URL)
		if err != nil {
			unavailable(w, 502, "bad_gateway", "bad server URL")
			return
		}
		if !a.applySampler(w, r) {
			return
		}
		rp := &httputil.ReverseProxy{
			Transport:     transport,
			FlushInterval: -1, // flush every write: SSE tokens must not sit in a buffer
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.SetURL(target)
				pr.Out.Host = target.Host
				pr.Out.Header.Del("Authorization") // a client's credentials never go to the local model server
			},
			ModifyResponse: dropCORS,
			ErrorHandler:   badGateway,
		}
		forward(w, r, rp, a.fwd(false, id).with(sw).ending(&sum))
	}
}

// slotFiles is true for llama-server's slot save and restore (POST /slots/{id}?action=save|restore).
// They write and read files in the engine's own contexts directory (--slot-save-path,
// run/context.go) and are the engine's alone: a client that could call them could fill the disk
// with files the engine does not know of, or lay another slot's saved prompt into its own. A
// client keeps GET /slots and action=erase. Anything on /slots/ that names an action other than
// erase is refused, whatever the method, so a spelling the model server would take cannot pass.
func slotFiles(r *http.Request) bool {
	if !strings.HasPrefix(path.Clean("/"+r.URL.Path), "/slots/") {
		return false
	}
	for _, act := range r.URL.Query()["action"] {
		if act != "erase" {
			return true
		}
	}
	return false
}

// holdOn are the names whose change can end a hold: the phase, the model and mode (a switch to
// external), the yield, and the upgrade (the swap to the full model).
var holdOn = []string{"phase", "model", "yield", "upgrade"}

// hold is a request the door holds (docs/ENGINE.md, "The door holds"): Enter said Hold (the
// model is on its way up). The request waits on the hub, never on a clock: listen first, then
// look at what stands (Enter again), then wait for a word on holdOn, the client's going (its
// context) or the engine's end. The held request is counted (Supervisor.Hold: the door fact, the
// resident app's linger), never in flight. It answers the route to forward by and the end of its
// count in flight; ok is false when the client went away while held.
func (a *API) hold(r *http.Request, id string) (rt run.Route, end func(*run.RequestEnd), ok bool) {
	hub := a.S.Hub()
	wk, err := hub.Wake(holdOn...)
	if err != nil { // a book without the names: answer as before the hold
		return run.Route{Kind: "unavailable", Message: "Loading model"}, func(*run.RequestEnd) {}, true
	}
	defer wk.Close()
	release := a.S.Hold()
	defer release()
	for {
		if rt, end = a.S.Enter(id); !rt.Hold {
			return rt, end, true
		}
		select {
		case <-wk.C():
		case <-r.Context().Done():
			return rt, nil, false
		case <-hub.Done():
			return run.Route{Kind: "unavailable", Message: "The engine is stopping"}, func(*run.RequestEnd) {}, true
		}
	}
}

// heardWriter tells heard on every Write: the model server's reply bytes are flowing, so the
// watchdog for silence on the requests in flight is put back. It stays a Flusher (a stream must not
// sit in a buffer), and Unwrap lets http.ResponseController reach the real writer (flush, hijack).
type heardWriter struct {
	http.ResponseWriter
	heard func()
}

func (h *heardWriter) Write(b []byte) (int, error) {
	h.heard()
	return h.ResponseWriter.Write(b)
}

func (h *heardWriter) Flush() { http.NewResponseController(h.ResponseWriter).Flush() }

func (h *heardWriter) Unwrap() http.ResponseWriter { return h.ResponseWriter }

// hearing wraps w so each reply write counts as the model server being heard from.
func (a *API) hearing(w http.ResponseWriter) http.ResponseWriter {
	heard := a.heard
	if heard == nil {
		heard = a.S.Heard
	}
	return &heardWriter{ResponseWriter: w, heard: heard}
}

// dropCORS removes every Access-Control-* header the upstream sent: the model server reflects
// Origin, and the engine allows no cross-site use at all (the gate refuses it).
func dropCORS(resp *http.Response) error {
	for k := range resp.Header {
		if strings.HasPrefix(strings.ToLower(k), "access-control-") {
			delete(resp.Header, k)
		}
	}
	return nil
}

func badGateway(w http.ResponseWriter, r *http.Request, err error) {
	if r.Context().Err() != nil {
		return // the client went away
	}
	unavailable(w, 502, "bad_gateway", fmt.Sprintf("upstream: %v", err))
}
