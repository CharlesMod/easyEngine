//go:build !windows

package api_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// ---- helpers ------------------------------------------------------------------------------

// clientsEnv is a ready engine with a base attachment (--profile given: 3 slots) whose fake
// says the slot it used in every reply. The base's ctx max is low, so a leave of a client
// gives the base nothing and a vacant range stays (no reshape on a leave).
func clientsEnv(t *testing.T, base bool, mut func(*testkit.Env)) (*testkit.Env, *run.Supervisor, *httptest.Server) {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_ID_SLOT"] = "1"
	e.Llama["FAKELLAMA_TIMINGS"] = "1"
	e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms" // named moment: the fake's pace between tokens
	e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	if mut != nil {
		mut(e)
	}
	cfg := e.Config()
	cfg.ProfileGiven = base
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv
}

// held is one held attachment: its NDJSON lines come on a channel as they are read.
type held struct {
	t      *testing.T
	id     string
	status int
	first  map[string]any
	lines  chan map[string]any // closed when the stream ends
	cancel context.CancelFunc
}

// openHold posts a held attach and reads its first line.
func openHold(t *testing.T, srv *httptest.Server, body string, hdr map[string]string) *held {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	req, _ := http.NewRequestWithContext(ctx, "POST", srv.URL+"/engine/v1/attach", strings.NewReader(body))
	req.Header.Set("X-Easy-Engine", "1")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	h := &held{t: t, status: resp.StatusCode, lines: make(chan map[string]any, 64), cancel: cancel}
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		h.first = map[string]any{}
		json.Unmarshal(b, &h.first)
		close(h.lines)
		return h
	}
	br := bufio.NewReader(resp.Body)
	line, err := br.ReadString('\n') // an event: the answer
	if err != nil {
		t.Fatalf("first line: %v", err)
	}
	if json.Unmarshal([]byte(line), &h.first) != nil {
		t.Fatalf("first line %q", line)
	}
	h.id, _ = h.first["attachment"].(string)
	go func() {
		defer resp.Body.Close()
		defer close(h.lines)
		for {
			l, err := br.ReadString('\n')
			if l != "" {
				var m map[string]any
				if json.Unmarshal([]byte(l), &m) == nil {
					h.lines <- m
				}
			}
			if err != nil {
				return
			}
		}
	}()
	return h
}

// next is the next line; ok false when the stream ended.
func (h *held) next(what string) (map[string]any, bool) {
	h.t.Helper()
	select {
	case m, ok := <-h.lines:
		return m, ok
	case <-time.After(15 * time.Second): // watchdog for silence: the engine said nothing
		h.t.Fatalf("no line on the held stream: %s", what)
	}
	return nil, false
}

func slotsOf(m map[string]any) (want, given, first int) {
	sl, _ := m["slots"].(map[string]any)
	f := func(k string) int { v, _ := sl[k].(float64); return int(v) }
	return f("want"), f("given"), f("first")
}

// untilGiven reads lines (the first line counts) until slots.given == n.
func (h *held) untilGiven(n int) map[string]any {
	h.t.Helper()
	m := h.first
	for {
		if _, g, _ := slotsOf(m); g == n {
			return m
		}
		var ok bool
		if m, ok = h.next("slots.given == " + strconv.Itoa(n)); !ok {
			h.t.Fatalf("the stream ended before slots.given == %d", n)
		}
	}
}

// growB attaches the client "b" (held, 2 slots) to a running engine with the 3-slot base, and
// waits until the engine has been reshaped to 5 slots and B holds 2.
func growB(t *testing.T, e *testkit.Env, s *run.Supervisor, srv *httptest.Server) *held {
	t.Helper()
	n0 := len(e.Starts())
	b := openHold(t, srv, `{"hold":true,"name":"b","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	if b.status != 200 {
		t.Fatalf("attach b: %d %v", b.status, b.first)
	}
	b.untilGiven(2)
	testkit.WaitFor(t, 15*time.Second, "the start for b", func() bool { return len(e.Starts()) > n0 && s.Phase() == "ready" })
	if _, g, first := slotsOf(b.first); g == 2 && first != 3 {
		t.Fatalf("b's first slot %d", first)
	}
	return b
}

func asClient(id string) map[string]string {
	return hdr(map[string]string{"X-Easy-Client": id})
}

func upstreamBytes(t *testing.T, e *testkit.Env) string {
	t.Helper()
	b, _ := os.ReadFile(filepath.Join(e.Ctl, "lastbody.json"))
	return string(b)
}

func codeOf(body string) string {
	var m struct{ Code string }
	json.Unmarshal([]byte(body), &m)
	return m.Code
}

// sseSlots is every id_slot a streamed reply says, and how many data lines it had.
func sseSlots(t *testing.T, srv *httptest.Server, body string, h map[string]string) (slots []int, lines int) {
	t.Helper()
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(body))
	for k, v := range h {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("stream: %d %s", resp.StatusCode, b)
	}
	br := bufio.NewReader(resp.Body)
	for {
		l, err := br.ReadString('\n')
		if strings.HasPrefix(l, "data:") && !strings.Contains(l, "[DONE]") {
			lines++
			var m struct {
				IDSlot *int `json:"id_slot"`
			}
			if json.Unmarshal([]byte(strings.TrimSpace(strings.TrimPrefix(l, "data:"))), &m) == nil && m.IDSlot != nil {
				slots = append(slots, *m.IDSlot)
			}
		}
		if err != nil {
			return
		}
	}
}

// requestEnd is the end of the request happening said after mark for an attachment.
func requestEnd(t *testing.T, s *run.Supervisor, from uint64, att string) map[string]any {
	t.Helper()
	var got map[string]any
	find := func() bool {
		for _, ev := range testkit.Said(t, s, from, 0, "request") {
			var m map[string]any
			json.Unmarshal(ev.Body, &m)
			if m["what"] == "end" && m["attachment"] == att {
				got = m
				return true
			}
		}
		return false
	}
	testkit.WaitFor(t, 15*time.Second, "the request end for the attachment", find)
	return got
}

func intField(m map[string]any, k string) (int, bool) {
	v, ok := m[k].(float64)
	return int(v), ok
}

func attachmentGone(t *testing.T, s *run.Supervisor, id string) {
	t.Helper()
	testkit.WaitFor(t, 15*time.Second, "the attachment fact gone", func() bool {
		_, ok := s.Hub().Fact("attachment", id)
		return !ok
	})
}

func setStreamer(t *testing.T, srv *httptest.Server, v string) {
	t.Helper()
	if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"streamer":"`+v+`"}`, csrf); c != 200 {
		t.Fatalf("streamer %s: %d %s", v, c, b)
	}
}

// ---- 1. the slot map, both ways -----------------------------------------------------------

func TestClientSlotMapBothWays(t *testing.T) {
	for _, streamer := range []string{"on", "off"} {
		t.Run("streamer_"+streamer, func(t *testing.T) {
			e, s, srv := clientsEnv(t, true, nil)
			b := growB(t, e, s, srv)
			setStreamer(t, srv, streamer)
			h := asClient(b.id)

			// not streamed
			from := testkit.Mark(t, s, "b's plain request")
			c, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[{"role":"user","content":"hello"}]}`, h)
			if c != 200 {
				t.Fatalf("plain: %d %s", c, body)
			}
			var up map[string]any
			json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
			if up["id_slot"] != float64(4) {
				t.Fatalf("the fake got id_slot %v, want 4: %s", up["id_slot"], upstreamBytes(t, e))
			}
			var rep map[string]any
			json.Unmarshal([]byte(body), &rep)
			if rep["id_slot"] != float64(1) {
				t.Fatalf("the reply's id_slot %v, want 1: %s", rep["id_slot"], body)
			}
			end := requestEnd(t, s, from, b.id)
			if cs, ok := intField(end, "client_slot"); !ok || cs != 1 {
				t.Fatalf("request end client_slot: %v", end)
			}

			// streamed
			from = testkit.Mark(t, s, "b's streamed request")
			slots, n := sseSlots(t, srv, `{"stream":true,"id_slot":1,"messages":[{"role":"user","content":"hello again"}]}`, h)
			if n == 0 || len(slots) == 0 {
				t.Fatalf("no data lines with an id_slot (%d lines)", n)
			}
			for _, x := range slots {
				if x != 1 {
					t.Fatalf("a streamed line says id_slot %d, want 1 (all: %v)", x, slots)
				}
			}
			if len(slots) != n {
				t.Fatalf("%d of %d data lines carry an id_slot: %v", len(slots), n, slots)
			}
			json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
			if up["id_slot"] != float64(4) {
				t.Fatalf("streamed: the fake got id_slot %v", up["id_slot"])
			}
			end = requestEnd(t, s, from, b.id)
			cs, ok1 := intField(end, "client_slot")
			if !ok1 || cs != 1 {
				t.Fatalf("streamed request end: client_slot %v (%v)", cs, end)
			}
			// the engine's slot comes from the streamer's summary: said only when the streamer ran
			if es, ok := intField(end, "slot"); streamer == "on" && (!ok || es != 4) || streamer == "off" && ok && es != 4 {
				t.Fatalf("streamed request end: slot %v (%v)", end["slot"], end)
			}
		})
	}
}

// ---- 2. no id_slot, and -1 -------------------------------------------------------------------

func TestClientAbsentAndMinusOneGoToItsOwnSlots(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	b := growB(t, e, s, srv)
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"one"}]}`,
		`{"id_slot":-1,"messages":[{"role":"user","content":"two"}]}`,
	} {
		c, rb, _ := do(t, "POST", srv.URL+"/v1/chat/completions", body, asClient(b.id))
		if c != 200 {
			t.Fatalf("%s: %d %s", body, c, rb)
		}
		var up, rep map[string]any
		json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
		json.Unmarshal([]byte(rb), &rep)
		es, _ := up["id_slot"].(float64)
		if es != 3 && es != 4 {
			t.Fatalf("%s: the fake got id_slot %v, want one of b's (3, 4)", body, up["id_slot"])
		}
		if cs, _ := rep["id_slot"].(float64); cs != es-3 {
			t.Fatalf("%s: engine %v came back as %v", body, es, cs)
		}
	}
}

// ---- 3. out of range ------------------------------------------------------------------------

func TestClientSlotOutOfRange(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	b := growB(t, e, s, srv)
	if c, _, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"x"}]}`, asClient(b.id)); c != 200 {
		t.Fatal("a slot in range was refused")
	}
	before := upstreamBytes(t, e)
	for _, n := range []string{"2", "3"} {
		c, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":`+n+`,"messages":[{"role":"user","content":"y"}]}`, asClient(b.id))
		if c != 400 || codeOf(body) != "slot_out_of_range" {
			t.Fatalf("id_slot %s: %d %s", n, c, body)
		}
	}
	if got := upstreamBytes(t, e); got != before {
		t.Fatalf("the fake got a request that was refused: %s", got)
	}
}

// ---- 4. unknown ids and anonymous callers ----------------------------------------------------

func TestClientUnknownAttachmentAndAnonymous(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	b := growB(t, e, s, srv)
	before := upstreamBytes(t, e)
	c, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[]}`, asClient("0000nosuch"))
	if c != 404 || codeOf(body) != "attachment_unknown" {
		t.Fatalf("unknown id: %d %s", c, body)
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("the fake got a request for an unknown attachment")
	}
	_ = b
	// anonymous: confined to the base's three slots (0..2)
	c, body, _ = do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[{"role":"user","content":"anon"}]}`, jsonCT)
	if c != 200 {
		t.Fatalf("anonymous id_slot 1: %d %s", c, body)
	}
	var up map[string]any
	json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
	if up["id_slot"] != float64(1) {
		t.Fatalf("anonymous id_slot 1 went to %v", up["id_slot"])
	}
	before = upstreamBytes(t, e)
	c, body, _ = do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":3,"messages":[{"role":"user","content":"anon"}]}`, jsonCT)
	if c != 400 || codeOf(body) != "slot_not_offered" {
		t.Fatalf("anonymous id_slot 3: %d %s", c, body)
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("the fake got an anonymous request that was refused")
	}
}

func TestClientAnonymousPassesThroughWithNoBase(t *testing.T) {
	e, s, srv := clientsEnv(t, false, nil)
	n0 := len(e.Starts())
	b := openHold(t, srv, `{"hold":true,"name":"b","profile":{"slots":4,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	if b.status != 200 {
		t.Fatalf("attach: %d %v", b.status, b.first)
	}
	b.untilGiven(4)
	testkit.WaitFor(t, 15*time.Second, "the start for b", func() bool { return len(e.Starts()) > n0 && s.Phase() == "ready" })
	c, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":3,"messages":[{"role":"user","content":"anon"}]}`, jsonCT)
	if c != 200 {
		t.Fatalf("anonymous id_slot 3 with no base: %d %s", c, body)
	}
	var up, rep map[string]any
	json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
	json.Unmarshal([]byte(body), &rep)
	if up["id_slot"] != float64(3) || rep["id_slot"] != float64(3) {
		t.Fatalf("passed through changed: upstream %v reply %v", up["id_slot"], rep["id_slot"])
	}
}

// ---- 5. held attachment: drop, resume, replace -----------------------------------------------

func TestAttachResumesAfterADrop(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	const body = `{"hold":true,"name":"b","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`
	b := growB(t, e, s, srv) // a lone client: its leave reshapes nothing, its range waits for it
	if c, rb, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"one two three"}]}`, asClient(b.id)); c != 200 {
		t.Fatalf("warm: %d %s", c, rb)
	}
	starts := len(e.Starts())
	b.cancel() // the connection drops
	attachmentGone(t, s, b.id)
	b2 := openHold(t, srv, body, nil)
	if b2.status != 200 {
		t.Fatalf("re-attach: %d %v", b2.status, b2.first)
	}
	if b2.first["resumed"] != true {
		t.Fatalf("not resumed: %v", b2.first)
	}
	if _, g, first := slotsOf(b2.first); g != 2 || first != 3 {
		t.Fatalf("resumed slots: given %d first %d", g, first)
	}
	// the same prompt grown: the slot's live cache answers it
	c, rb, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"one two three four five six seven eight"}]}`, asClient(b2.id))
	if c != 200 {
		t.Fatalf("after resume: %d %s", c, rb)
	}
	var rep struct {
		Timings struct {
			CacheN int `json:"cache_n"`
		}
	}
	json.Unmarshal([]byte(rb), &rep)
	if rep.Timings.CacheN <= 0 {
		t.Fatalf("the cache was not kept: %s", rb)
	}
	if got := len(e.Starts()); got != starts {
		t.Fatalf("a resume started the server again: %d -> %d", starts, got)
	}
}

func TestAttachReplacesTheFirstHold(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	const body = `{"hold":true,"name":"b","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`
	b := growB(t, e, s, srv)
	starts := len(e.Starts())
	b2 := openHold(t, srv, body, nil)
	if b2.status != 200 {
		t.Fatalf("second attach: %d %v", b2.status, b2.first)
	}
	if b2.first["replaced"] != true || b2.id != b.id {
		t.Fatalf("not a replace: replaced %v id %s vs %s", b2.first["replaced"], b2.id, b.id)
	}
	if _, g, first := slotsOf(b2.first); g != 2 || first != 3 {
		t.Fatalf("replace slots: given %d first %d", g, first)
	}
	for {
		m, ok := b.next("the first stream's end")
		if !ok {
			t.Fatal("the first stream ended with no {\"end\":\"replaced\"} line")
		}
		if m["end"] != nil {
			if m["end"] != "replaced" {
				t.Fatalf("end %v", m["end"])
			}
			break
		}
	}
	if _, ok := s.Hub().Fact("attachment", b.id); !ok {
		t.Fatal("the attachment left when its first connection was replaced")
	}
	if got := len(e.Starts()); got != starts {
		t.Fatalf("a replace started the server: %d -> %d", starts, got)
	}
}

// ---- 6. DELETE -------------------------------------------------------------------------------

func TestDeleteAttachment(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	b := growB(t, e, s, srv)
	if c, body, _ := do(t, "DELETE", srv.URL+"/engine/v1/attach/"+b.id, "", csrf); c != 204 {
		t.Fatalf("delete: %d %s", c, body)
	}
	for {
		m, ok := b.next("the detached end")
		if !ok {
			t.Fatal("the stream ended with no end line")
		}
		if m["end"] != nil {
			if m["end"] != "detached" {
				t.Fatalf("end %v", m["end"])
			}
			break
		}
	}
	attachmentGone(t, s, b.id)
	c, body, _ := do(t, "DELETE", srv.URL+"/engine/v1/attach/"+b.id, "", csrf)
	if c != 404 || codeOf(body) != "attachment_unknown" {
		t.Fatalf("second delete: %d %s", c, body)
	}
}

// ---- 7. served from another's vacant range ----------------------------------------------------

func TestAttachServedFromAVacantRange(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	b := growB(t, e, s, srv)
	starts := len(e.Starts())
	b.cancel()
	attachmentGone(t, s, b.id)
	c := openHold(t, srv, `{"hold":true,"name":"c","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	if c.status != 200 {
		t.Fatalf("attach c: %d %v", c.status, c.first)
	}
	if c.first["resumed"] == true {
		t.Fatalf("c is not b: %v", c.first)
	}
	if _, g, first := slotsOf(c.first); g != 2 || first != 3 {
		t.Fatalf("c's slots: given %d first %d (want b's old 3..4): %v", g, first, c.first)
	}
	// and it can use them
	if code, body, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[{"role":"user","content":"c"}]}`, asClient(c.id)); code != 200 {
		t.Fatalf("c's request: %d %s", code, body)
	}
	var up map[string]any
	json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
	if up["id_slot"] != float64(4) {
		t.Fatalf("c's slot 1 went to engine %v", up["id_slot"])
	}
	if got := len(e.Starts()); got != starts {
		t.Fatalf("c's attach started the server: %d -> %d", starts, got)
	}
}

// ---- 8. the old wire ---------------------------------------------------------------------------

func TestAttachOldWire(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	var asked []int
	srv := httptest.NewServer((&api.API{S: s, Attach: func(pid int) error { asked = append(asked, pid); return nil }}).Handler())
	t.Cleanup(srv.Close)
	nosrv := httptest.NewServer((&api.API{S: s}).Handler())
	t.Cleanup(nosrv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)

	cat := exec.Command("cat")
	in, _ := cat.StdinPipe()
	if err := cat.Start(); err != nil {
		t.Fatal(err)
	}
	defer cat.Wait()
	defer in.Close()
	pid := cat.Process.Pid

	c, body, _ := do(t, "POST", srv.URL+"/engine/v1/attach", `{"pid":`+strconv.Itoa(pid)+`}`, csrf)
	if c != 200 {
		t.Fatalf("attach pid: %d %s", c, body)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatalf("answer %q: %v", body, err)
	}
	if m["status"] != "attached" || m["pid"] != float64(pid) {
		t.Fatalf("old fields: %v", m)
	}
	for _, k := range []string{"attachment", "key", "presence", "slots"} {
		if _, ok := m[k]; !ok {
			t.Errorf("answer lacks the new field %q: %s", k, body)
		}
	}
	if len(asked) != 1 || asked[0] != pid {
		t.Fatalf("Attach was asked %v", asked)
	}
	if id, _ := m["attachment"].(string); id != "" {
		in.Close()
		attachmentGone(t, s, id)
	}

	c, body, _ = do(t, "POST", srv.URL+"/engine/v1/attach", `{}`, csrf)
	if c != 400 || strings.TrimSpace(body) != `{"error":"body must be {\"pid\": n}"}` {
		t.Fatalf("empty body: %d %q", c, body)
	}
	c, body, _ = do(t, "POST", nosrv.URL+"/engine/v1/attach", `{"pid":`+strconv.Itoa(pid)+`}`, csrf)
	if c != 501 {
		t.Fatalf("no Attach func: %d %s", c, body)
	}
}

// ---- 9. samplers per attachment ------------------------------------------------------------------

func TestClientSamplersPerAttachment(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Sampler = map[string]catalog.SamplerProfile{
		"official": {Temperature: fp(1.0)},
		"rp":       {Temperature: fp(0.9), TopK: ip(64)},
	}
	e.Cat.Models[0].SamplerMatch = []string{"modelone"}
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	b := openHold(t, srv, `{"hold":true,"name":"b","profile":{"samplers":{"cold":{"temperature":0.2}}}}`, nil)
	if b.status != 200 {
		t.Fatalf("attach: %d %v", b.status, b.first)
	}
	c, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}]}`,
		hdr(map[string]string{"X-Easy-Client": b.id, "X-Easy-Sampler": "cold"}))
	if c != 200 {
		t.Fatalf("with the client: %d", c)
	}
	if got := lastBody(t, e); got["temperature"] != 0.2 || !strings.HasPrefix(h.Get("X-Easy-Sampler-Applied"), "cold;") {
		t.Fatalf("b's cold: body %v applied %q", got, h.Get("X-Easy-Sampler-Applied"))
	}
	// without the header, cold is nobody's: unknown, falls back to rp
	c, _, h = do(t, "POST", srv.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}]}`,
		hdr(map[string]string{"X-Easy-Sampler": "cold"}))
	if c != 200 {
		t.Fatalf("anonymous: %d", c)
	}
	if got := lastBody(t, e); got["temperature"] != 0.9 || strings.HasPrefix(h.Get("X-Easy-Sampler-Applied"), "cold;") {
		t.Fatalf("anonymous cold should fall back to rp: body %v applied %q", got, h.Get("X-Easy-Sampler-Applied"))
	}
}

// ---- 10. keepalive -----------------------------------------------------------------------------------

func TestListenKeepAlive(t *testing.T) {
	if k := api.ListenKeepAlive; !k.Enable || k.Idle != 30*time.Second || k.Interval != 10*time.Second || k.Count != 3 {
		t.Fatalf("ListenKeepAlive %+v", k)
	}
	l, _, err := api.Listen("127.0.0.1", 0)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	acc := make(chan net.Conn, 1)
	go func() {
		c, err := l.Accept()
		if err != nil {
			close(acc)
			return
		}
		acc <- c
	}()
	d, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	var sc net.Conn
	select {
	case sc = <-acc:
	case <-time.After(15 * time.Second): // watchdog for silence: no accept
		t.Fatal("nothing accepted")
	}
	defer sc.Close()
	tc, ok := sc.(*net.TCPConn)
	if !ok {
		t.Fatalf("accepted a %T", sc)
	}
	raw, err := tc.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	var on int
	var gerr error
	if err := raw.Control(func(fd uintptr) { on, gerr = syscall.GetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_KEEPALIVE) }); err != nil || gerr != nil {
		t.Fatalf("getsockopt: %v %v", err, gerr)
	}
	if on == 0 {
		t.Fatal("the accepted connection has no TCP keepalive")
	}
}

// A held events stream stays open through requests on other connections.
func TestHeldEventsSurviveOtherRequests(t *testing.T) {
	_, s, _ := readyEnv(t)
	l, _, err := api.Listen("127.0.0.1", 0)
	if err != nil {
		t.Fatal(err)
	}
	hs := &http.Server{Handler: (&api.API{S: s}).Handler()}
	go hs.Serve(l)
	t.Cleanup(func() { hs.Close() })
	base := "http://" + l.Addr().String()
	c, body, _ := do(t, "GET", base+"/engine/v1/facts", "", nil)
	var f event.Standing
	if c != 200 || json.Unmarshal([]byte(body), &f) != nil {
		t.Fatalf("facts %d %s", c, body)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", base+"/engine/v1/events?life="+f.Life+"&since="+seqStr(f.Seq), nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("events %d", resp.StatusCode)
	}
	br := bufio.NewReader(resp.Body)
	// requests on other connections (no keep-alive: each is its own connection)
	other := &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}
	for i := 0; i < 3; i++ {
		r, err := other.Get(base + "/engine/v1/status")
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, r.Body)
		r.Body.Close()
	}
	// the held stream still carries what is said now
	mark := testkit.Mark(t, s, "after other requests")
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("the held events stream was cut: %v", err)
		}
		var ev event.Event
		if json.Unmarshal([]byte(line), &ev) != nil {
			t.Fatalf("line %q", line)
		}
		if ev.Seq == mark && ev.Name == "mark" {
			break
		}
	}
}

// ---- 11. a remote caller ----------------------------------------------------------------------------

type testChecker map[string]string

func (c testChecker) Check(b string) (string, bool) { id, ok := c[b]; return id, ok }

func TestRemoteCallerRefusedExtraArgs(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	srv := httptest.NewServer((&api.API{S: s, AnyHost: true, Tokens: testChecker{"tok-1": "client-1"}}).Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	auth := map[string]string{"Authorization": "Bearer tok-1"}

	r := openHold(t, srv, `{"hold":true,"name":"r","profile":{"server":{"extra_args":["--top-n-sigma","1.5"]}}}`, auth)
	if r.status != 400 || r.first["code"] != "extra_args_not_allowed" {
		t.Fatalf("remote extra_args: %d %v", r.status, r.first)
	}
	if n := s.Attached(); n != 0 {
		t.Fatalf("a refused attach was registered (%d)", n)
	}
	// the same remote client without extra_args is welcome (the refusal is about the field)
	ok := openHold(t, srv, `{"hold":true,"name":"r","profile":{"server":{"ubatch":256}}}`, auth)
	if ok.status != 200 {
		t.Fatalf("remote without extra_args: %d %v", ok.status, ok.first)
	}
	if ok.first["local"] != false || ok.first["client_id"] != "client-1" {
		t.Fatalf("not seen as the remote client-1: %v", ok.first)
	}
}
