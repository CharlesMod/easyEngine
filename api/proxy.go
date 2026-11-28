package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strconv"
	"strings"
	"sync"
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
	if classify(r.Method, r.URL.Path) == pathSlotID {
		// a slot action is never passed as sent: exactly one action=erase in the URL, and the
		// request upstream is the engine's own (no body, no form a model server would read)
		if acts := r.URL.Query()["action"]; len(acts) != 1 || acts[0] != "erase" || formBody(r) {
			badRequest(w, "a slot action here is exactly one action=erase in the URL, with no form body")
			return
		}
		r.URL.RawQuery = "action=erase"
		r.Header.Del("Content-Type")
		setBody(r, nil)
	}
	if (streamPath(r.URL.Path) || bufferedBody(r)) && formBody(r) {
		badRequest(w, "a request body here is JSON (Content-Type application/json): a form-encoded or multipart body is not read as the engine reads it")
		return
	}
	id := requestID(r)
	w.Header().Set(RequestIDHeader, id)
	att := strings.TrimSpace(r.Header.Get(ClientHeader))
	r.Header.Del(ClientHeader) // the engine's word, not the model server's
	if len(att) > 64 {
		att = "-" // no attachment has such an id: answered as unknown
	}
	local := localCaller(r)
	if !local && ClientID(r.Context()) == ClientOpen && !proxyMay(r) {
		// an --open caller from another machine reaches the model server only as a paired token
		// does (api/scope.go: the one allow list)
		failCode(w, http.StatusForbidden, "not_allowed", scopeRefusal)
		return
	}
	if !local && streamPath(r.URL.Path) {
		clientFail(w, &run.ClientError{Status: 403, Code: "path_not_offered", Msg: "resumable streams are offered to this machine's callers only"})
		return
	}
	// a body the engine must look at (a JSON POST the door confines or rewrites) is read whole
	// before the door: a client that sends slowly holds no slot, no count in flight and no drain.
	// Every other body streams through as it comes.
	if bufferedBody(r) {
		give, ok := a.readBody(w, r, local)
		if !ok {
			return
		}
		defer give() // the kept bytes count until the request ends
	}
	q := run.DoorAsk{ReqID: id, Att: att, Identity: identity(r), Local: local, Touch: touchOf(r)}
	rt, door, done := a.S.EnterAs(q)
	if door.Err != nil && !rt.Hold {
		clientFail(w, door.Err)
		return
	}
	var sw *streamWriter // a streamed request's stream, opened before its hold
	var unheld func()    // a remote request's count among those held
	if rt.Hold {
		if !local {
			// held requests of other machines are bounded by count (each a goroutine, a
			// connection and its kept body)
			var ok bool
			if unheld, ok = a.holdRemote(w, r); !ok {
				return
			}
			defer unheld()
		}
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
		rt, door, done, ok = a.hold(r, q)
		if unheld != nil {
			unheld() // no longer held: the count is the door's waiting ones only
		}
		if !ok {
			return // the client went away while held
		}
		if door.Err != nil {
			if sw != nil {
				sw.refuse(run.Route{Kind: "unavailable", Message: door.Err.Msg})
				return
			}
			clientFail(w, door.Err)
			return
		}
	}
	// a request of an attachment: its slot numbers are its own (mapped below), said in the end
	var clientSlot, engineSlot *int
	var engineSlots []int
	if door.ID != "" {
		end := done
		done = func(re *run.RequestEnd) {
			if re == nil {
				re = &run.RequestEnd{}
			}
			re.Attachment = door.ID
			if re.ClientSlot == nil {
				re.ClientSlot = clientSlot
			}
			if re.Slot == nil {
				re.Slot = engineSlot // the slot the door sent it to (a streamed reply says its own)
			}
			end(re)
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
		if !a.applySampler(w, r, door.ID) {
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
		if !a.applySampler(w, r, door.ID) {
			return
		}
		// the conversation ids of resumable streams are the engine's name space: each carries its
		// owner, so one client can neither replay nor cancel another's stream (convid.go)
		lookup, ok := a.ownConversations(w, r, door)
		if !ok {
			return
		}
		if door.Slots != nil { // confined (run.ClientDoor): one slot of its range, no other
			// a slot that holds another owner's prompt is erased before the request runs there
			// (llama-server clears its tokens and KV, server-context.cpp:3307-3332, :393-399)
			erase := func(es int) bool {
				u := *target
				u.Path, u.RawPath, u.RawQuery = "/slots/"+strconv.Itoa(es), "", "action=erase"
				// not ended by the client's going: the erase must be answered before the slot is
				// anyone's (it waits in the model server only while the slot finishes a task)
				req, err := http.NewRequestWithContext(context.WithoutCancel(r.Context()), http.MethodPost, u.String(), nil)
				if err != nil {
					return false
				}
				resp, err := transport.RoundTrip(req)
				if err != nil {
					return false
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				return resp.StatusCode == http.StatusOK
			}
			cs, es, release, ok := a.confine(w, r, door, erase)
			if !ok {
				return
			}
			defer release()
			if es >= 0 {
				engineSlot = &es
				if cs >= 0 {
					clientSlot = &cs
				}
			}
			if door.Own {
				engineSlots = door.Slots        // its numbers in the reply are mapped back
				r.Header.Del("Accept-Encoding") // a compressed reply could not be mapped
			}
		}
		modify := dropCORS
		if (r.Method == http.MethodGet || r.Method == http.MethodHead) && path.Clean("/"+r.URL.Path) == "/slots" {
			// every caller reads only its own slots' detail: another owner's slot keeps id,
			// is_processing and n_ctx (no prompt, no generated text, no token counts)
			own := a.S.SlotsShownTo(identity(r), local)
			r.Header.Del("Accept-Encoding") // a compressed reply could not be read
			modify = func(resp *http.Response) error {
				dropCORS(resp)
				return slotsShown(resp, own)
			}
		} else if engineSlots != nil || lookup != nil {
			modify = func(resp *http.Response) error {
				dropCORS(resp)
				if lookup != nil {
					return conversationsBack(resp, lookup)
				}
				return slotsBack(resp, engineSlots)
			}
		}
		var rt http.RoundTripper = transport
		if door.Slots != nil {
			rt = queuedFirst{transport} // a confined request's task is in the model server before a cancel ends it
		}
		rp := &httputil.ReverseProxy{
			Transport:     rt,
			FlushInterval: -1, // flush every write: SSE tokens must not sit in a buffer
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.SetURL(target)
				pr.Out.Host = target.Host
				pr.Out.Header.Del("Authorization") // a client's credentials never go to the local model server
			},
			ModifyResponse: modify,
			ErrorHandler:   badGateway,
		}
		forward(w, r, rp, a.fwd(false, id).with(sw).slots(engineSlots).ending(&sum))
	}
}

// queuedFirst sends a confined request upstream so that the client's going cannot end it before
// the model server has answered its headers. llama-server answers headers only after it posted
// the request's task (a stream: once the task is queued; otherwise with the whole reply), and a
// closed connection then cancels that task (server-queue.cpp cleanup_pending_task, a slot's
// release). Ended earlier, the task could be queued after the request's slot mark was released and
// after the next owner's erase: it would run on that slot after the erase, before the next owner's
// request, and leave its prompt there. So the request's mark (the slot's count in flight, which a
// handover waits for) ends only after the model server has the task, never before.
type queuedFirst struct{ rt http.RoundTripper }

func (q queuedFirst) RoundTrip(req *http.Request) (*http.Response, error) {
	client := req.Context()
	up, cancel := context.WithCancel(context.WithoutCancel(client))
	resp, err := q.rt.RoundTrip(req.WithContext(up))
	if err != nil {
		cancel()
		return nil, err
	}
	stop := context.AfterFunc(client, cancel) // from here the client's going ends it
	resp.Body = &endsBody{ReadCloser: resp.Body, end: func() { stop(); cancel() }}
	return resp, nil
}

// endsBody runs end once when the body is closed.
type endsBody struct {
	io.ReadCloser
	end  func()
	once sync.Once
}

func (b *endsBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.end)
	return err
}

// slotsShown filters a GET /slots reply (llama-server's to_json per slot,
// server-context.cpp:835-866: prompt, generated, params, token counts) to what the caller may
// read: another owner's slot keeps id, is_processing and n_ctx. A reply that is not a JSON array
// of objects is replaced by an error: nothing passes unread.
func slotsShown(resp *http.Response, own func(e int) bool) error {
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxSamplerBody+1))
	resp.Body.Close()
	if err != nil || len(b) > maxSamplerBody {
		return fmt.Errorf("the slots reply could not be read whole")
	}
	var slots []map[string]json.RawMessage
	if err := json.Unmarshal(b, &slots); err != nil {
		return fmt.Errorf("the slots reply is not a list of slots")
	}
	for i, sl := range slots {
		var id int
		if json.Unmarshal(sl["id"], &id) != nil || !own(id) {
			slots[i] = map[string]json.RawMessage{"id": sl["id"], "is_processing": sl["is_processing"], "n_ctx": sl["n_ctx"]}
		}
	}
	out, err := json.Marshal(slots)
	if err != nil {
		return err
	}
	resp.Body = io.NopCloser(bytes.NewReader(out))
	resp.ContentLength = int64(len(out))
	resp.Header.Set("Content-Length", strconv.Itoa(len(out)))
	resp.Header.Del("Content-Encoding")
	return nil
}

// touchOf is what a proxied request does with slots, from the path table (paths.go).
func touchOf(r *http.Request) run.Touch {
	if streamPath(r.URL.Path) {
		// a resumable stream's replay, cancel or lookup runs nothing in a slot, and its ids are
		// in the caller's own name space (convid.go); a remote caller is refused them above
		return run.TouchNone
	}
	switch classify(r.Method, r.URL.Path) {
	case pathFree:
		return run.TouchNone
	case pathSlot, pathSlotID:
		return run.TouchSlot
	}
	return run.TouchUnconfinable
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
var holdOn = []string{"phase", "model", "yield", "upgrade", "attachment", "wants"}

// hold is a request the door holds (docs/ENGINE.md, "The door holds"): Enter said Hold (the
// model is on its way up). The request waits on the hub, never on a clock: listen first, then
// look at what stands (Enter again), then wait for a word on holdOn, the client's going (its
// context) or the engine's end. The held request is counted (Supervisor.Hold: the door fact, the
// resident app's linger), never in flight. It answers the route to forward by and the end of its
// count in flight; ok is false when the client went away while held.
func (a *API) hold(r *http.Request, q run.DoorAsk) (rt run.Route, door run.ClientDoor, end func(*run.RequestEnd), ok bool) {
	hub := a.S.Hub()
	wk, err := hub.Wake(holdOn...)
	if err != nil { // a book without the names: answer as before the hold
		return run.Route{Kind: "unavailable", Message: "Loading model"}, run.ClientDoor{}, func(*run.RequestEnd) {}, true
	}
	defer wk.Close()
	release := a.S.Hold()
	defer release()
	for {
		if rt, door, end = a.S.EnterAs(q); !rt.Hold {
			if end == nil {
				end = func(*run.RequestEnd) {}
			}
			return rt, door, end, true
		}
		select {
		case <-wk.C():
		case <-r.Context().Done():
			return rt, door, nil, false
		case <-hub.Done():
			return run.Route{Kind: "unavailable", Message: "The engine is stopping"}, run.ClientDoor{}, func(*run.RequestEnd) {}, true
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
	var big replyTooLarge
	if errors.As(err, &big) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(502)
		json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": "reply_too_large", "type": "bad_gateway", "message": big.Error()}})
		return
	}
	unavailable(w, 502, "bad_gateway", fmt.Sprintf("upstream: %v", err))
}
