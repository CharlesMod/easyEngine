package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/run"
)

// A request the door confines (run.ClientDoor, docs/ENGINE.md section 7g "Whose slots") goes to
// one slot of its range and no other. An attachment numbers its own slots 0..n-1 (a game mod: 0
// its narrator, 1..n its actors): the door maps them to the engine slots it holds in the running
// mind and maps the slot numbers in the reply back. A request confined to the commons (the base's
// range, or the slots no attachment owns) keeps the engine's numbers and may name only those.
// What cannot be confined exactly is refused (fail closed): a body that is not one JSON object,
// a slot path that is not digits, a number outside the range.

// maxJSONDepth is the deepest nesting a confined request's body may have. llama-server's parser
// has no limit; the door's must hold every body it passes.
const maxJSONDepth = 64

var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

func badRequest(w http.ResponseWriter, msg string) {
	clientFail(w, &run.ClientError{Status: 400, Code: "bad_request", Msg: msg})
}

// jsonDepth is the deepest nesting of objects and arrays in b (strings skipped).
func jsonDepth(b []byte) int {
	depth, most, inStr, esc := 0, 0, false, false
	for _, c := range b {
		switch {
		case inStr && esc:
			esc = false
		case inStr && c == '\\':
			esc = true
		case inStr && c == '"':
			inStr = false
		case inStr:
		case c == '"':
			inStr = true
		case c == '{' || c == '[':
			depth++
			most = max(most, depth)
		case c == '}' || c == ']':
			depth--
		}
	}
	return most
}

// oneObject is a confined request's body as exactly one JSON object, or why it is not one. A
// byte-order mark is refused outright (llama-server skips it; the door does not guess).
func oneObject(body []byte) (map[string]json.RawMessage, string) {
	if bytes.HasPrefix(body, utf8BOM) {
		return nil, "the request body starts with a byte-order mark: send plain JSON"
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return nil, "the request body is empty: it must be one JSON object"
	}
	if jsonDepth(body) > maxJSONDepth {
		return nil, fmt.Sprintf("the request body nests deeper than %d", maxJSONDepth)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil || obj == nil {
		return nil, "the request body must be exactly one JSON object"
	}
	if k, dup := duplicateKey(body); dup {
		// two readers may keep different ones of the two (the door one, the model server the
		// other): refused, never guessed
		return nil, fmt.Sprintf("the request body gives the key %q twice", k)
	}
	return obj, ""
}

// duplicateKey is a key the body's top-level object gives more than once (keys compared as
// decoded, so "n" and "\u006e" are one key, as llama-server's parser reads them).
func duplicateKey(body []byte) (string, bool) {
	dec := json.NewDecoder(bytes.NewReader(body))
	if t, err := dec.Token(); err != nil || t != json.Delim('{') {
		return "", false
	}
	seen := map[string]bool{}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return "", false
		}
		k, _ := t.(string)
		if seen[k] {
			return k, true
		}
		seen[k] = true
		var skip json.RawMessage
		if dec.Decode(&skip) != nil {
			return "", false
		}
	}
	return "", false
}

// oneCompletion: a confined body's n / n_cmpl is absent, null or exactly the literal 1. The model
// server reads it as int32 by a plain cast (server-schema.cpp:621-634, nlohmann json.hpp:5203-5236):
// -4294967294 becomes 2, 1.9 becomes 1, true becomes 1; so the raw token is judged, never a number
// decoded from it.
func oneCompletion(raw json.RawMessage) bool {
	switch string(bytes.TrimSpace(raw)) {
	case "1", "null":
		return true
	}
	return false
}

// slotDigits is a /slots/{id} id when confined: ASCII digits only (llama-server reads " 1", "+1"
// and "1x" as slot 1; the door reads none of them).
func slotDigits(s string) (int, bool) {
	if s == "" || len(s) > 6 {
		return 0, false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, false
		}
	}
	n, err := strconv.Atoi(s)
	return n, err == nil
}

// confine sends a confined request to one slot of door's range: a completion body's id_slot
// (absent, null or -1: the least recently used free slot of the range, taken atomically), or the
// slot of /slots/{n}?action=erase. It answers the client's number (cs, -1 for the commons), the
// engine slot, the release of the slot's mark, and ok false after it has answered the client.
func (a *API) confine(w http.ResponseWriter, r *http.Request, door run.ClientDoor) (cs, es int, release func(), ok bool) {
	slots := door.Slots
	cs = -1
	toEngine := func(n int) (int, *run.ClientError) {
		if door.Own {
			if n < 0 || n >= len(slots) {
				return 0, &run.ClientError{Status: 400, Code: "slot_out_of_range",
					Msg: fmt.Sprintf("id_slot %d is outside this client's %d slot(s) (0..%d)", n, len(slots), len(slots)-1)}
			}
			return slots[n], nil
		}
		for _, e := range slots {
			if e == n {
				return e, nil
			}
		}
		return 0, &run.ClientError{Status: 400, Code: "slot_not_offered",
			Msg: fmt.Sprintf("slot %d belongs to an attached client; this request may use slots %v", n, slots)}
	}
	clientOf := func(e int) int {
		if !door.Own {
			return -1
		}
		for i, x := range slots {
			if x == e {
				return i
			}
		}
		return -1
	}
	switch classify(r.Method, r.URL.Path) {
	case pathSlotID:
		n, digits := slotDigits(strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/slots/"))
		if !digits {
			badRequest(w, "a slot path is /slots/{n} with n in digits")
			return 0, -1, nil, false
		}
		e, ce := toEngine(n)
		if ce != nil {
			clientFail(w, ce)
			return 0, -1, nil, false
		}
		es, release = a.S.TakeSlot(slots, e)
		r.URL.Path, r.URL.RawPath = "/slots/"+strconv.Itoa(es), ""
		return clientOf(es), es, release, true
	case pathSlot:
	default:
		return -1, -1, func() {}, true // touches no slot the door must name
	}
	obj, why := oneObject(bodyOf(r))
	if why != "" {
		badRequest(w, why)
		return 0, -1, nil, false
	}
	// n / n_cmpl above 1 makes the server run child tasks on any free slots, copying the state
	// over them (server-schema.cpp:64-67, server-task.h:329 "child tasks cannot specify slot",
	// server-context.cpp:3008-3016, :4726-4745): never in a confined request
	for _, k := range []string{"n", "n_cmpl"} {
		if raw, has := obj[k]; has && !oneCompletion(raw) {
			clientFail(w, &run.ClientError{Status: 400, Code: "n_not_offered",
				Msg: k + " must be absent, null or exactly 1 here: more runs completions on slots that are not this request's; ask for one completion per request"})
			return 0, -1, nil, false
		}
	}
	want := -1
	if raw, has := obj["id_slot"]; has && string(bytes.TrimSpace(raw)) != "null" {
		n, isInt := slotInt(raw)
		if !isInt || n < -1 {
			badRequest(w, "id_slot must be an integer, or -1 for any of this client's slots")
			return 0, -1, nil, false
		}
		if n >= 0 {
			e, ce := toEngine(n)
			if ce != nil {
				clientFail(w, ce)
				return 0, -1, nil, false
			}
			want = e
		}
	}
	es, release = a.S.TakeSlot(slots, want)
	obj["id_slot"] = json.RawMessage(strconv.Itoa(es))
	out, err := marshalObject(obj)
	if err != nil {
		release()
		badRequest(w, "the request body must be exactly one JSON object")
		return 0, -1, nil, false
	}
	setBody(r, out)
	return clientOf(es), es, release, true
}

// marshalObject writes a body object back: every value as the client sent it (numbers, unicode
// and the order of keys inside each value kept; no HTML escaping), the top-level keys sorted.
func marshalObject(obj map[string]json.RawMessage) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(obj); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// heldBody is a body read whole before the door: its bytes are taken as they are, not copied.
type heldBody struct {
	*bytes.Reader
	b []byte
}

func (heldBody) Close() error { return nil }

// bodyOf is r's body bytes (a held body's own; else read now).
func bodyOf(r *http.Request) []byte {
	if hb, ok := r.Body.(heldBody); ok {
		return hb.b
	}
	if r.Body == nil {
		return nil
	}
	b, _ := io.ReadAll(r.Body)
	r.Body.Close()
	return b
}

func setBody(r *http.Request, b []byte) {
	r.Body = heldBody{bytes.NewReader(b), b}
	r.ContentLength = int64(len(b))
	r.Header.Set("Content-Length", strconv.Itoa(len(b)))
	r.TransferEncoding = nil
}

// silentBody arms the read deadline before every read of a request body.
type silentBody struct {
	r   io.Reader
	arm func()
}

func (b *silentBody) Read(p []byte) (int, error) {
	b.arm()
	return b.r.Read(p)
}

// bodySilence is how long a request body may go silent between reads; bodyTotal how long one
// body may take in all.
func (a *API) bodySilence() time.Duration {
	if a.BodySilence > 0 {
		return a.BodySilence
	}
	return 30 * time.Second
}

func (a *API) bodyTotal() time.Duration {
	if a.BodyTotal > 0 {
		return a.BodyTotal
	}
	return 5 * time.Minute
}

// What the engine keeps for callers on other machines is bounded whatever they send and however
// long they are held (docs/ENGINE.md §7g): at most maxRemoteReads bodies being read, each at most
// maxRemoteBody, and at most maxRemoteKept bytes of bodies read and not yet ended (held at the
// door or in flight), so at most 8 x 16 + 64 = 192 MiB; at most maxRemoteHeldPer requests of one
// remote client (every --open address one) and maxRemoteHeld of all held at the door at once.
// Over any of them: 429 busy.
const (
	maxRemoteBody    = 16 << 20 // a remote caller's buffered body
	maxRemoteReads   = 8        // remote bodies being read at once (a count: the 9th is 429 busy)
	maxRemoteKept    = 64 << 20 // bytes of remote bodies read whose requests have not ended
	maxRemoteHeldPer = 8        // one remote client's requests held at the door at once
	maxRemoteHeld    = 32       // all remote requests held at the door at once
)

// busy is the 429 every remote bound answers.
func busy(w http.ResponseWriter, msg string) {
	w.Header().Set("Connection", "close")
	clientFail(w, &run.ClientError{Status: 429, Code: "busy", Msg: msg})
}

// holdRemote counts a remote request held at the door against its client's share and the remote
// total; false (answered 429 busy) over either. The result ends the count.
func (a *API) holdRemote(w http.ResponseWriter, r *http.Request) (func(), bool) {
	who := identity(r)
	if strings.HasPrefix(who, "open@") {
		who = "open" // every --open address is one client (one budget, one share)
	}
	a.heldMu.Lock()
	defer a.heldMu.Unlock()
	if a.heldBy == nil {
		a.heldBy = map[string]int{}
	}
	if a.heldBy[who] >= maxRemoteHeldPer || a.heldAll >= maxRemoteHeld {
		busy(w, fmt.Sprintf("too many requests from other machines are waiting (at most %d of one client, %d in all)", maxRemoteHeldPer, maxRemoteHeld))
		return nil, false
	}
	a.heldBy[who]++
	a.heldAll++
	var once sync.Once
	return func() {
		once.Do(func() {
			a.heldMu.Lock()
			if a.heldBy[who]--; a.heldBy[who] == 0 {
				delete(a.heldBy, who)
			}
			a.heldAll--
			a.heldMu.Unlock()
		})
	}, true
}

// bufferedBody: the engine must look at r's body before the door (a JSON POST the table confines,
// or the stream lookup it rewrites). Every other body streams through as on main.
func bufferedBody(r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}
	return classify(r.Method, r.URL.Path) == pathSlot || path.Clean("/"+r.URL.Path) == "/v1/streams/lookup"
}

// readBody reads a proxied request's whole body before the request reaches the door: a client
// that sends slowly holds no slot, no count in flight and no drain. Each read is bounded by a
// network timeout on silence (BodySilence), the whole read by a second one (BodyTotal), the body
// by 64 MiB from a local caller and 16 MiB from a remote one (413 body_too_large); at most 8
// remote bodies are read at once, and a remote body read is kept only while all those kept stay
// within maxRemoteKept (429 busy). The result gives the kept bytes back when the request ends;
// ok is false after it has answered the client.
func (a *API) readBody(w http.ResponseWriter, r *http.Request, local bool) (func(), bool) {
	if r.Body == nil || r.Body == http.NoBody {
		return func() {}, true
	}
	limit := int64(maxSamplerBody)
	if !local {
		limit = maxRemoteBody
		if n := a.remoteReads.Add(1); n > maxRemoteReads {
			a.remoteReads.Add(-1)
			busy(w, "too many request bodies from other machines are arriving at once")
			return nil, false
		} else if a.readsSaid != nil {
			a.readsSaid(n)
		}
		defer func() {
			if n := a.remoteReads.Add(-1); a.readsSaid != nil {
				a.readsSaid(n)
			}
		}()
	}
	rc := http.NewResponseController(w)
	d, end := a.bodySilence(), time.Now().Add(a.bodyTotal())
	arm := func() {
		// network timeout: a request body that sends nothing for d between two reads is cut;
		// network timeout: and one that is still arriving at end (a drip) is cut too
		dl := time.Now().Add(d)
		if end.Before(dl) {
			dl = end
		}
		_ = rc.SetReadDeadline(dl)
	}
	body, err := io.ReadAll(&silentBody{r: http.MaxBytesReader(w, r.Body, limit), arm: arm})
	if err != nil {
		// the deadline stays passed: the server must not wait for the rest of the body before it
		// answers (net/http reads an unread body before a reply), and the connection ends after
		w.Header().Set("Connection", "close")
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			clientFail(w, &run.ClientError{Status: 413, Code: "body_too_large", Msg: fmt.Sprintf("the request body is larger than %d MiB", limit>>20)})
			return nil, false
		}
		clientFail(w, &run.ClientError{Status: 400, Code: "body_unread", Msg: "the request body did not arrive: " + err.Error()})
		return nil, false
	}
	give, ok := a.keepRemote(w, len(body), local)
	if !ok {
		return nil, false
	}
	_ = rc.SetReadDeadline(time.Time{})
	r.Body.Close()
	setBody(r, body)
	return give, true
}

// keepRemote counts n bytes a remote request keeps against maxRemoteKept (429 busy over it, and
// false); the result gives them back, once. A local request keeps nothing counted.
func (a *API) keepRemote(w http.ResponseWriter, n int, local bool) (func(), bool) {
	if local {
		return func() {}, true
	}
	k := int64(n)
	if a.keptAdd(k) > maxRemoteKept {
		a.keptAdd(-k)
		busy(w, fmt.Sprintf("the request bodies from other machines kept now are at the engine's bound (%d MiB)", maxRemoteKept>>20))
		return nil, false
	}
	var once sync.Once
	return func() { once.Do(func() { a.keptAdd(-k) }) }, true
}

// keptAdd changes remoteKept by k and says the new count to a test listening (keptSaid).
func (a *API) keptAdd(k int64) int64 {
	n := a.remoteKept.Add(k)
	if f := a.keptSaid.Load(); f != nil {
		(*f)(n)
	}
	return n
}

// replyTooLarge is a JSON reply over the cap the door maps slot numbers in: answered 502 with a
// code (badGateway), never cut short in silence.
type replyTooLarge struct{}

func (replyTooLarge) Error() string {
	return fmt.Sprintf("the model server's reply is larger than %d MiB", maxSamplerBody>>20)
}

// slotsBack maps the engine's slot numbers in a reply back to the client's: an event stream line
// by line as it flows, a JSON reply whole.
func slotsBack(resp *http.Response, slots []int) error {
	toClient := func(e int) (int, bool) {
		for i, x := range slots {
			if x == e {
				return i, true
			}
		}
		return 0, false
	}
	if resp.Header.Get("Content-Encoding") != "" {
		return nil // compressed: passed as it is
	}
	mt, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	switch mt {
	case "text/event-stream":
		resp.Body = newSlotStream(resp.Body, toClient)
	case "application/json":
		b, err := io.ReadAll(io.LimitReader(resp.Body, maxSamplerBody+1))
		resp.Body.Close()
		if err != nil {
			return err
		}
		if len(b) > maxSamplerBody {
			return replyTooLarge{}
		}
		if out, changed := mapReplySlots(b, toClient); changed {
			b = out
		}
		resp.Body = io.NopCloser(bytes.NewReader(b))
		resp.ContentLength = int64(len(b))
		resp.Header.Set("Content-Length", strconv.Itoa(len(b)))
	}
	return nil
}
