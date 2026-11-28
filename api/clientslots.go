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
	return obj, ""
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
	var body []byte
	if r.Body != nil {
		body, _ = io.ReadAll(r.Body) // read before the door (readBody): in memory
		r.Body.Close()
	}
	obj, why := oneObject(body)
	if why != "" {
		badRequest(w, why)
		return 0, -1, nil, false
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
	out, err := json.Marshal(obj)
	if err != nil {
		release()
		badRequest(w, "the request body must be exactly one JSON object")
		return 0, -1, nil, false
	}
	setBody(r, out)
	return clientOf(es), es, release, true
}

func setBody(r *http.Request, b []byte) {
	r.Body = io.NopCloser(bytes.NewReader(b))
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

// bodySilence is how long a request body may go silent between reads.
func (a *API) bodySilence() time.Duration {
	if a.BodySilence > 0 {
		return a.BodySilence
	}
	return 30 * time.Second
}

// readBody reads a proxied request's whole body before the request reaches the door: a client
// that sends slowly holds no slot, no count in flight and no drain. Each read is bounded by a
// network timeout on silence (BodySilence); the body by maxSamplerBody (413 body_too_large).
// False after it has answered the client.
func (a *API) readBody(w http.ResponseWriter, r *http.Request) bool {
	if r.Body == nil || r.Body == http.NoBody {
		return true
	}
	rc := http.NewResponseController(w)
	d := a.bodySilence()
	// network timeout: a request body that sends nothing for d between two reads is cut
	arm := func() { _ = rc.SetReadDeadline(time.Now().Add(d)) }
	body, err := io.ReadAll(&silentBody{r: http.MaxBytesReader(w, r.Body, maxSamplerBody), arm: arm})
	if err != nil {
		// the deadline stays passed: the server must not wait for the rest of the body before it
		// answers (net/http reads an unread body before a reply), and the connection ends after
		w.Header().Set("Connection", "close")
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			clientFail(w, &run.ClientError{Status: 413, Code: "body_too_large", Msg: fmt.Sprintf("the request body is larger than %d MiB", maxSamplerBody>>20)})
			return false
		}
		clientFail(w, &run.ClientError{Status: 400, Code: "body_unread", Msg: "the request body did not arrive: " + err.Error()})
		return false
	}
	_ = rc.SetReadDeadline(time.Time{})
	r.Body.Close()
	setBody(r, body)
	return true
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
