package api

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"path"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/run"
)

// A client numbers its own slots 0..n-1 (a game mod: 0 its narrator, 1..n its actors);
// the door maps them to the engine slots its attachment holds in the running mind, and maps the
// slot numbers in the reply back. A number outside its own count is a 400, never another
// client's slot.

// mapClientRequest rewrites a request of an attachment that holds slots: a completion body's
// id_slot (absent or -1: its own free slot), or the slot of /slots/{n}?action=erase. It answers
// the client's number and the engine slot used (es -1: no slot was named or picked), and ok false
// after it has answered the client itself.
func (a *API) mapClientRequest(w http.ResponseWriter, r *http.Request, slots []int) (cs, es int, ok bool) {
	es = -1
	outOfRange := func(n int) *run.ClientError {
		return &run.ClientError{Status: 400, Code: "slot_out_of_range",
			Msg: fmt.Sprintf("id_slot %d is outside this client's %d slot(s) (0..%d)", n, len(slots), len(slots)-1)}
	}
	if p := path.Clean("/" + r.URL.Path); strings.HasPrefix(p, "/slots/") {
		n, err := strconv.Atoi(strings.TrimPrefix(p, "/slots/"))
		if err != nil {
			return 0, -1, true // not a slot number: llama-server answers it
		}
		if n < 0 || n >= len(slots) {
			clientFail(w, outOfRange(n))
			return 0, -1, false
		}
		r.URL.Path = "/slots/" + strconv.Itoa(slots[n])
		r.URL.RawPath = ""
		return n, slots[n], true
	}
	if r.Method != http.MethodPost || !completionPath(r.URL.Path) || r.Body == nil || r.Body == http.NoBody {
		return 0, -1, true
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxSamplerBody))
	r.Body.Close()
	if err != nil {
		fail(w, http.StatusBadRequest, "could not read the request body: "+err.Error())
		return 0, -1, false
	}
	toEngine := func(n int) (int, error) {
		if n < 0 || n >= len(slots) {
			return 0, outOfRange(n)
		}
		cs = n
		return slots[n], nil
	}
	pick := func() int {
		e := a.S.PickSlot(slots)
		for i, x := range slots {
			if x == e {
				cs = i
			}
		}
		return e
	}
	out, changed, err := mapRequestSlot(body, toEngine, pick)
	if err != nil {
		var ce *run.ClientError
		var se *SlotError
		switch {
		case errors.As(err, &ce):
			clientFail(w, ce)
		case errors.As(err, &se):
			clientFail(w, &run.ClientError{Status: 400, Code: "bad_request", Msg: se.Msg})
		default:
			fail(w, 400, err.Error())
		}
		return 0, -1, false
	}
	if changed {
		body = out
		es = slots[cs]
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))
	r.TransferEncoding = nil
	return cs, es, true
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
		if out, changed := mapReplySlots(b, toClient); changed {
			b = out
		}
		resp.Body = io.NopCloser(bytes.NewReader(b))
		resp.ContentLength = int64(len(b))
		resp.Header.Set("Content-Length", strconv.Itoa(len(b)))
	}
	return nil
}
