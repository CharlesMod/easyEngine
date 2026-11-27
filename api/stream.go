package api

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"mime"
	"net/http"
	"net/http/httputil"
	"strings"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/stream"
)

// The token streamer (docs/STREAMING.md): a streamed completion reply is read event by event
// (package stream), a reply the model server cuts short ends with one terminal error event, and
// while the upstream says nothing the engine says a comment so the client knows the line is alive.

// RequestIDHeader names every proxied reply: the client's own id when it sent a usable one,
// else one the engine makes. The same id is in a terminal error event.
const RequestIDHeader = "X-Request-Id"

// upstreamIDHeader keeps the id an upstream (a player's own endpoint) gave its reply, so the
// reply carries one X-Request-Id and the provider's id is not lost.
const upstreamIDHeader = "X-Upstream-Request-Id"

const (
	// watchdog for silence: an upstream that has sent no response header this long after the
	// request was handed over (a request waiting for a free slot gets no byte at all) is
	// answered by the engine itself: 200, an event stream, and the waiting word.
	openAfter = 2 * time.Second
	// watchdog for silence: an open stream that has carried no byte this long gets a comment.
	// It is under the 30 s that many clients and llama-server's own ping use.
	quietEvery = 15 * time.Second
)

// The comments the engine says into an open stream. A comment line is ignored by every SSE
// client, so they can never be taken for tokens.
var (
	waitingWord = []byte(": easy waiting\n\n") // nothing from the upstream yet
	quietWord   = []byte(": easy quiet\n\n")   // the upstream has spoken and is silent now
)

// The terminal event's codes (docs/STREAMING.md).
const (
	codeYielded         = "yielded"
	codeModelServerGone = "model_server_gone"
	codeUpstreamEnded   = "upstream_ended"
	codeUpstreamError   = "upstream_error"
	// unavailable: a held request's hold ended in a phase answered at once (no model installed,
	// the model server's error, a person's stop, the engine stopping) after its stream was opened.
	codeUnavailable = "unavailable"
)

// maxErrBody is how much of an upstream's error body is kept to find its message.
const maxErrBody = 64 << 10

// requestID is the client's X-Request-Id when it is printable ASCII and at most 128 bytes, else
// 16 random hex digits.
func requestID(r *http.Request) string {
	if v := r.Header.Get(RequestIDHeader); v != "" && len(v) <= 128 {
		ok := true
		for i := 0; i < len(v); i++ {
			if v[i] < 0x20 || v[i] > 0x7e {
				ok = false
				break
			}
		}
		if ok {
			return v
		}
	}
	var b [8]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// termError is the value of "error" in a terminal event: the engine's error-body shape (code,
// type, message) with the HTTP status where one applied and the request id.
type termError struct {
	Code      string `json:"code"`
	Type      string `json:"type"`
	Message   string `json:"message"`
	Status    int    `json:"status,omitempty"`
	RequestID string `json:"request_id"`
}

func (e termError) json() []byte {
	b, _ := json.Marshal(e)
	return b
}

// termEvent is the whole terminal event, the same bytes package stream appends after a cut.
func termEvent(e termError) []byte {
	return []byte("data: {\"error\":" + string(e.json()) + "}\n\n")
}

// fwd is what forward needs to know of the engine; proxy fills it from the supervisor.
type fwd struct {
	on       bool   // the streamer setting
	external bool   // external mode: the upstream is the player's endpoint
	id       string // the request id
	// yielded reports, at the moment of a cut, whether the engine gave the card to a game, and
	// the message of the 503 for that state.
	yielded func() (bool, string)
	onEvent func() // per upstream event (nil in api/: every proxied write is heard already)
	// hear wraps the writer the reverse proxy writes to, so the upstream's bytes count as the
	// model server being heard from (API.hearing). nil: no wrapping (tests).
	hear func(http.ResponseWriter) http.ResponseWriter
	// open and quiet replace openAfter and quietEvery when set (tests shorten them).
	open, quiet time.Duration
	// sw is the stream a held request opened before its hold (proxy); forward goes on in it.
	sw *streamWriter
	// onEnd takes the streamer's summary of a streamed reply (stream.Options.OnEnd).
	onEnd func(stream.Summary)
	// slotsOf is the attachment's engine slots when its reply's slot numbers are its own (the
	// streamer reads the client's number; the end says the engine's slot and the client's).
	slotsOf []int
}

// slots is f for a request whose reply numbers slots in the client's numbering.
func (f fwd) slots(s []int) fwd {
	f.slotsOf = s
	return f
}

// ending is f with the streamer's summary kept in *to (said in the request's end).
func (f fwd) ending(to **run.RequestEnd) fwd {
	f.onEnd = func(s stream.Summary) {
		re := &run.RequestEnd{End: string(s.End), Events: s.Events, FinishReason: s.FinishReason,
			PromptTokens: s.PromptTokens, CachedTokens: s.CachedTokens, CompletionTokens: s.CompletionTokens,
			TokS: s.TokPerSec, PromptMs: s.PromptMs, PredictedMs: s.PredictedMs}
		if s.Slot >= 0 {
			slot := s.Slot
			re.Slot = &slot
			if f.slotsOf != nil {
				cs := s.Slot
				re.ClientSlot = &cs
				if cs < len(f.slotsOf) {
					e := f.slotsOf[cs]
					re.Slot = &e
				}
			}
		}
		*to = re
	}
	return f
}

// with is f going on in the stream a held request opened (nil: none).
func (f fwd) with(sw *streamWriter) fwd {
	f.sw = sw
	return f
}

// fwd is what the streamer needs of the supervisor for one request.
func (a *API) fwd(external bool, id string) fwd {
	return fwd{
		on: a.S.Streamer(), external: external, id: id, hear: a.hearing,
		yielded: func() (bool, string) {
			// The status is asked, not the phase and not Route. The yield stands in the status from
			// the moment its drain begins, while the phase says "yielded" only after the model
			// server has stopped, which is after the cut. Route would say it too, but a Route
			// asked of a sleeping engine wakes the model. The message is Route's for this state.
			if y := a.S.Status().Yield; y != nil {
				return true, "Paused while " + y.Reason + " runs"
			}
			return false, ""
		},
	}
}

func (f fwd) wrap(w http.ResponseWriter) http.ResponseWriter {
	if f.hear == nil {
		return w
	}
	return f.hear(w)
}

// cut is the terminal error for a stream the upstream ended early.
func (f fwd) cut() []byte {
	e := termError{RequestID: f.id}
	switch {
	case f.external:
		e.Code, e.Type, e.Message = codeUpstreamEnded, "upstream_error", "The endpoint ended the reply before it was finished"
	default:
		if f.yielded != nil {
			if y, msg := f.yielded(); y {
				e.Code, e.Type, e.Message, e.Status = codeYielded, "unavailable_error", msg, http.StatusServiceUnavailable
				return e.json()
			}
		}
		e.Code, e.Type, e.Message = codeModelServerGone, "server_error", "The model server stopped before the reply was finished"
	}
	return e.json()
}

// dialect is the stream dialect of a completion path.
func dialect(path string) stream.Dialect {
	switch path {
	case "/v1/chat/completions", "/chat/completions":
		return stream.Chat
	case "/v1/completions", "/completions":
		return stream.Completions
	case "/completion":
		return stream.Native
	}
	return stream.Unknown
}

// eventStream is true for a reply the streamer may read: an uncompressed text/event-stream.
func eventStream(h http.Header) bool {
	mt, _, _ := mime.ParseMediaType(h.Get("Content-Type"))
	return mt == "text/event-stream" && h.Get("Content-Encoding") == ""
}

// wantsStream reads a completion request's body for "stream": true and puts the body back as
// it was. ok is false after it answered 400 itself (the body could not be read).
func wantsStream(w http.ResponseWriter, r *http.Request) (want, ok bool) {
	if r.Body == nil || r.Body == http.NoBody {
		return false, true
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxSamplerBody))
	r.Body.Close()
	if err != nil {
		fail(w, http.StatusBadRequest, "could not read the request body: "+err.Error())
		return false, false
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var req struct {
		Stream bool `json:"stream"`
	}
	json.Unmarshal(body, &req) // a body that is not JSON is forwarded as it is, unstreamed
	return req.Stream, true
}

// forward hands r to rp. A completion request that asks for a stream, with the streamer on, is
// answered through the streamer; every other request goes through rp as it always has.
func forward(w http.ResponseWriter, r *http.Request, rp *httputil.ReverseProxy, f fwd) {
	base := rp.ModifyResponse
	modify := func(resp *http.Response) error {
		if base != nil {
			if err := base(resp); err != nil {
				return err
			}
		}
		if v := resp.Header.Values(RequestIDHeader); len(v) > 0 {
			resp.Header.Del(RequestIDHeader)
			resp.Header[upstreamIDHeader] = v
		}
		return nil
	}
	rp.ModifyResponse = modify
	if f.sw != nil { // a held streamed request: its stream is open (or ready to be) already
		f.sw.mu.Lock()
		f.sw.cutErr = f.cut // the route is known now (local or external)
		f.sw.mu.Unlock()
		streamThrough(w, r, rp, f, f.sw, modify)
		return
	}
	if !f.on || r.Method != http.MethodPost || !completionPath(r.URL.Path) {
		rp.ServeHTTP(f.wrap(w), r)
		return
	}
	want, ok := wantsStream(w, r)
	if !ok {
		return
	}
	if !want {
		rp.ServeHTTP(f.wrap(w), r)
		return
	}
	// The streamWriter sits nearest the client, inside the hearing wrapper: the upstream's bytes
	// pass the hearing wrapper and are heard; the engine's own comments and terminal events go
	// straight to the client and are not (were they heard, a hung model server would look alive
	// for as long as a client waits).
	sw := newStreamWriter(w, f)
	defer sw.finish(r.Context().Err)
	streamThrough(w, r, rp, f, sw, modify)
}

// streamThrough serves a streamed completion through sw: the upstream's event stream is read
// event by event (package stream) and every write goes to the client through sw.
func streamThrough(w http.ResponseWriter, r *http.Request, rp *httputil.ReverseProxy, f fwd, sw *streamWriter, modify func(*http.Response) error) {
	d := dialect(r.URL.Path)
	ctx := r.Context()
	rp.ModifyResponse = func(resp *http.Response) error {
		if err := modify(resp); err != nil {
			return err
		}
		if resp.StatusCode != http.StatusOK || !eventStream(resp.Header) {
			return nil // a JSON error, a whole reply or a compressed one: left alone
		}
		if resp.Header.Get("Cache-Control") == "" {
			resp.Header.Set("Cache-Control", "no-cache")
		}
		resp.Body = stream.New(resp.Body, stream.Options{
			Dialect:   d,
			OnEvent:   f.onEvent,
			OnEnd:     f.onEnd,
			Cut:       sw.cut,
			Gone:      func() bool { return ctx.Err() != nil },
			RequestID: f.id,
		})
		return nil
	}
	rp.ServeHTTP(f.wrap(sw), r)
}

// streamWriter stands between the reverse proxy and the client for a streamed request. It
// gives the proxy a header map of its own, so the watchdog (which may open the stream itself)
// and the proxy (which copies the upstream's headers) never touch one map, and it serialises
// every write to the client under one lock, so a comment only ever goes between two events.
type streamWriter struct {
	w  http.ResponseWriter
	rc *http.ResponseController
	id string

	mu     sync.Mutex
	gone   sync.Cond   // signalled when the watchdog's callback is no longer armed
	h      http.Header // the proxy's map, until its status line goes out
	passed bool        // the proxy's status line went to the client: Header is w's own now
	opened bool        // the engine opened the stream itself (the waiting word)
	sse    bool        // the client's reply is an event stream comments may go into
	body   bool        // an upstream body byte has gone out
	// convert: the stream was opened, then the upstream answered something that is not a
	// stream (an error); its body is kept and becomes the terminal event at the end.
	convert bool
	status  int
	term    []byte // the terminal event of a cut stream, written by finish (so it is not heard)
	cutErr  func() []byte
	ctype   string
	errBody []byte
	tail    []byte // the last bytes written, to tell an event boundary
	last    time.Time
	timer   *time.Timer
	quiet   time.Duration
	armed   bool // the watchdog's callback is scheduled or running
	done    bool // the handler is returning
}

func newStreamWriter(w http.ResponseWriter, f fwd) *streamWriter {
	sw := &streamWriter{w: w, rc: http.NewResponseController(w), id: f.id, h: w.Header().Clone(), quiet: quietEvery, cutErr: f.cut}
	open := openAfter
	if f.open > 0 {
		open = f.open
	}
	if f.quiet > 0 {
		sw.quiet = f.quiet
	}
	sw.gone.L = &sw.mu
	sw.mu.Lock()
	sw.last = time.Now()
	sw.armed = true
	sw.timer = time.AfterFunc(open, sw.fire) // watchdog for silence: no header yet
	sw.mu.Unlock()
	return sw
}

func (sw *streamWriter) Header() http.Header {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	if sw.passed {
		return sw.w.Header()
	}
	return sw.h
}

func (sw *streamWriter) WriteHeader(code int) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.writeHeaderLocked(code)
}

func (sw *streamWriter) writeHeaderLocked(code int) {
	if code < 200 {
		return // a 1xx is only a hint, and the stream may already be open: not passed on
	}
	if sw.passed || sw.convert {
		return
	}
	if sw.opened {
		if code == http.StatusOK && eventStream(sw.h) {
			return // the upstream's stream goes on in the one already open
		}
		sw.convert, sw.status, sw.ctype = true, code, sw.h.Get("Content-Type")
		return
	}
	real := sw.w.Header()
	clear(real)
	for k, v := range sw.h {
		real[k] = v
	}
	sw.passed = true
	// A reply that declared its length has no room for a comment: one more byte than promised
	// and the client's connection is failed. Such a stream is carried, never spoken into.
	sw.sse = code == http.StatusOK && eventStream(sw.h) && sw.h.Get("Content-Length") == ""
	sw.w.WriteHeader(code)
	sw.last = time.Now()
}

func (sw *streamWriter) Write(p []byte) (int, error) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	if !sw.passed && !sw.opened {
		sw.writeHeaderLocked(http.StatusOK)
	}
	if sw.convert {
		if room := maxErrBody - len(sw.errBody); room > 0 {
			sw.errBody = append(sw.errBody, p[:min(len(p), room)]...)
		}
		return len(p), nil
	}
	n, err := sw.w.Write(p)
	if n > 0 {
		sw.body = true
		sw.last = time.Now()
		sw.tail = append(sw.tail, p[max(0, n-4):n]...)
		sw.tail = sw.tail[max(0, len(sw.tail)-4):]
	}
	return n, err
}

// atBoundary is true when the client has been sent whole events only, so a comment may follow.
func (sw *streamWriter) atBoundary() bool {
	return len(sw.tail) == 0 || bytes.HasSuffix(sw.tail, []byte("\n\n")) || bytes.HasSuffix(sw.tail, []byte("\r\n\r\n"))
}

// FlushError is what http.ResponseController and the reverse proxy call after every write.
func (sw *streamWriter) FlushError() error {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	if sw.convert {
		return nil
	}
	if !sw.passed && !sw.opened {
		sw.writeHeaderLocked(http.StatusOK)
	}
	return sw.rc.Flush()
}

func (sw *streamWriter) Flush() { sw.FlushError() }

// Unwrap lets http.ResponseController reach the client's writer (deadlines and the like).
func (sw *streamWriter) Unwrap() http.ResponseWriter { return sw.w }

// say writes one comment and flushes it. Under mu.
func (sw *streamWriter) say(c []byte) {
	sw.w.Write(c)
	sw.rc.Flush()
	sw.last = time.Now()
	sw.tail = sw.tail[:0]
}

// fire is the watchdog for silence. It opens the stream when the upstream has sent no header,
// says a comment when the stream has been silent for quietEvery, and re-arms itself from the
// time of the last write (so a busy stream costs no timer call per token).
func (sw *streamWriter) fire() {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	if sw.done {
		sw.armed = false
		sw.gone.Broadcast()
		return
	}
	switch {
	case !sw.passed && !sw.opened:
		h := sw.w.Header()
		h.Del("Content-Length")
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no")
		sw.w.WriteHeader(http.StatusOK)
		sw.opened, sw.sse = true, true
		sw.say(waitingWord)
		sw.timer.Reset(sw.quiet) // watchdog for silence
	case sw.sse && !sw.convert:
		if idle := time.Since(sw.last); idle < sw.quiet {
			sw.timer.Reset(sw.quiet - idle) // watchdog for silence, from the last write
			return
		}
		if sw.atBoundary() {
			if sw.body {
				sw.say(quietWord)
			} else {
				sw.say(waitingWord)
			}
		}
		sw.timer.Reset(sw.quiet) // watchdog for silence
	default:
		// A whole reply (not a stream) or an error being kept: nothing more to watch.
		sw.armed = false
		sw.gone.Broadcast()
	}
}

// cut is the stream reader's Cut: it keeps the terminal event for finish to write and gives the
// reader nothing to append. The reader has dropped the cut event's tail and ends at a boundary,
// so the event still comes right after the last whole event, but straight to the client.
func (sw *streamWriter) cut() []byte {
	e := sw.cutErr()
	sw.mu.Lock()
	sw.term = []byte("data: {\"error\":" + string(e) + "}\n\n")
	sw.mu.Unlock()
	return nil
}

// refuse answers a held streamed request whose hold ended in a phase answered at once (yielded,
// no model installed, the model server's error, a person's stop, the engine stopping). If the
// stream is not open yet the client gets the same JSON 503 as an unstreamed request; if it is
// (the waiting word went out), the stream ends with the terminal event: code "yielded" or
// "unavailable", status 503, the route's message. finish writes it.
func (sw *streamWriter) refuse(rt run.Route) {
	e := termError{Code: codeUnavailable, Type: "unavailable_error", Message: rt.Message, Status: http.StatusServiceUnavailable, RequestID: sw.id}
	retry := "2"
	if rt.Code == codeYielded {
		e.Code, retry = codeYielded, "10"
	}
	sw.mu.Lock()
	defer sw.mu.Unlock()
	if sw.opened {
		sw.term = termEvent(e)
		return
	}
	sw.h.Set("Content-Type", "application/json")
	sw.h.Set("Retry-After", retry)
	sw.writeHeaderLocked(http.StatusServiceUnavailable)
	var code any = http.StatusServiceUnavailable
	if rt.Code == codeYielded {
		code = codeYielded
	}
	json.NewEncoder(sw.w).Encode(map[string]any{"error": map[string]any{"code": code, "type": "unavailable_error", "message": rt.Message}})
}

// finish runs as the handler returns: it stops the watchdog and waits until its callback can
// no longer run (no goroutine outlives the request), then, if the upstream answered an error
// into a stream the engine had opened, writes that error as the terminal event.
func (sw *streamWriter) finish(ctxErr func() error) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	sw.done = true
	if sw.armed && sw.timer.Stop() {
		sw.armed = false
	}
	for sw.armed {
		sw.gone.Wait()
	}
	if ctxErr() != nil {
		return
	}
	if sw.term != nil {
		sw.w.Write(sw.term)
		sw.rc.Flush()
		return
	}
	if !sw.convert {
		return
	}
	sw.w.Write(termEvent(termError{
		Code: codeUpstreamError, Type: "upstream_error", Message: upstreamMessage(sw.status, sw.ctype, sw.errBody),
		Status: sw.status, RequestID: sw.id,
	}))
	sw.rc.Flush()
}

// upstreamMessage finds the message in an upstream's error body: {"error":{"message"}},
// {"error":"..."} or {"message"}, else the start of the text, else the status text.
func upstreamMessage(status int, ctype string, body []byte) string {
	var b struct {
		Error   json.RawMessage `json:"error"`
		Message string          `json:"message"`
	}
	if json.Unmarshal(body, &b) == nil {
		var inner struct {
			Message string `json:"message"`
		}
		var s string
		switch {
		case json.Unmarshal(b.Error, &inner) == nil && inner.Message != "":
			return inner.Message
		case json.Unmarshal(b.Error, &s) == nil && s != "":
			return s
		case b.Message != "":
			return b.Message
		}
	}
	if status == http.StatusOK {
		return "the upstream answered " + ctype + ", not an event stream"
	}
	if t := strings.TrimSpace(string(body)); t != "" {
		if len(t) > 512 {
			t = t[:512]
		}
		return t
	}
	return http.StatusText(status)
}
