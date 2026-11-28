package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/lookimg"
	"github.com/CharlesMod/easyEngine/run"
)

// POST /engine/v1/look (docs/ENGINE.md section 7h, "The look call"): the one way a caller from
// another machine gets images to the model. The engine decodes each image itself (lookimg:
// PNG and JPEG only, bounded before any pixel is decoded), scales it down to the detail's long
// edge from the serving model's projector, re-encodes it, and builds the chat request for the
// model server from these parts alone: no caller byte and no caller JSON is forwarded. The
// request then goes through the proxy as a chat completion of the caller's own (the door, the
// hold, the caller's slots and id_slot mapping, the sampler policy), marked so the remote media
// checks do not refuse the engine's own image parts.

// The look call's bounds (docs/ENGINE.md lists them).
const (
	maxLookImages   = 4        // images in one look
	maxLookQuestion = 16 << 10 // bytes of the question
	maxLookLabel    = 256      // bytes of one image's label
	maxLookTokens   = 4096     // the most max_tokens a look may ask
	defLookTokens   = 512      // max_tokens when none is given
	maxLookSlot     = 1023     // the largest id_slot a look may name (the door judges the caller's range)
	maxLookDecoding = 2        // looks decoding images at once, all callers
	maxLookAnswer   = 4 << 20  // bytes of the model server's non-streamed answer the engine reads
	defLookDetail   = "high"   // detail when none is given
	lookObject      = "look.result"
)

// lookGate bounds what looks hold beyond their bodies: maxLookDecoding decode at once (each holds
// one decoded image at a time: at most 64 MiB at the default pixel limit), and a caller from
// another machine has one look at a time. A remote look's bytes (its body, then the request built
// from it) count against the proxy's one bound of remote bodies kept (maxRemoteKept, readBody).
// Over a bound the look is answered 429 busy at once: nothing waits for room.
type lookGate struct {
	mu       sync.Mutex
	decoding int
	remote   map[string]bool
}

func (g *lookGate) enter(identity string, local bool) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.decoding >= maxLookDecoding || !local && g.remote[identity] {
		return false
	}
	if g.remote == nil {
		g.remote = map[string]bool{}
	}
	g.decoding++
	if !local {
		g.remote[identity] = true
	}
	return true
}

// decoded: the look's images are built; its decoding share is free.
func (g *lookGate) decoded() {
	g.mu.Lock()
	g.decoding--
	g.mu.Unlock()
}

func (g *lookGate) release(identity string, local bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !local {
		delete(g.remote, identity)
	}
}

func lookBusy(why string) *run.ClientError {
	return &run.ClientError{Status: 429, Code: "busy", Msg: why}
}

func lookTooLarge(n int) *run.ClientError {
	return lookBad("look_too_large", "the look's images come to %d MiB once built; at most %d MiB from another machine (fewer images or a lower detail)", n>>20, maxRemoteBody>>20)
}

func lookBad(code, format string, args ...any) *run.ClientError {
	return &run.ClientError{Status: 400, Code: code, Msg: fmt.Sprintf(format, args...)}
}

// lookCtxKey marks the engine's own chat request of a look (proxy.go skips the remote media
// checks for it: its parts are the engine's, built from decoded images).
type lookCtxKey struct{}

func isLook(r *http.Request) bool { return r.Context().Value(lookCtxKey{}) != nil }

type lookImage struct {
	// data is the base64 text: a view into the look body, never a copy of it (a copy only when the
	// caller escaped a character in it, which base64 never needs)
	data  []byte
	label string
}

type lookReq struct {
	images   []lookImage
	question string
	detail   string
	maxTok   int
	idSlot   int // -1: none
	stream   bool
}

// parseLook reads a look request strictly: one JSON object, no unknown or repeated key at any
// level, numbers as integer tokens in range (never floats), strings as strings. The body is
// walked in place (lookWalk): each image's base64 is a view of it, so the body is the only copy
// of the caller's images held, until each is decoded.
func parseLook(body []byte) (lookReq, *run.ClientError) {
	q := lookReq{detail: defLookDetail, maxTok: defLookTokens, idSlot: -1}
	if jsonDepth(body) > 8 {
		return q, lookBad("bad_request", "a look body nests at most 8 deep")
	}
	if !json.Valid(body) {
		// refused bodies only: the decoder's own buffer is a second copy, for the one answer
		dec := json.NewDecoder(bytes.NewReader(body))
		if err := dec.Decode(&struct{}{}); err == nil {
			return q, lookBad("bad_request", "the look body has data after its object")
		}
		return q, lookBad("bad_request", "the look body is not one JSON object")
	}
	ce := lookMembers(body, "the look body", func(k string, v []byte) *run.ClientError {
		switch k {
		case "images":
			if !isJSONArray(v) {
				return lookBad("bad_request", "images must be an array")
			}
			n := 0
			if ce := lookElements(v, func(raw []byte) *run.ClientError {
				if n++; n > maxLookImages {
					return lookBad("too_many_images", "a look takes at most %d images", maxLookImages)
				}
				im, ce := parseLookImage(n-1, raw)
				if ce == nil {
					q.images = append(q.images, im)
				}
				return ce
			}); ce != nil {
				return ce
			}
			if n == 0 {
				return lookBad("bad_request", "images must hold at least one image")
			}
		case "question":
			if err := jsonString(v, &q.question); err != nil {
				return lookBad("bad_request", "question must be a string")
			}
		case "detail":
			if err := jsonString(v, &q.detail); err != nil || (q.detail != "low" && q.detail != "high" && q.detail != "max") {
				return lookBad("bad_request", "detail must be low, high or max")
			}
		case "max_tokens":
			n, ok := jsonInt(v, 1, maxLookTokens)
			if !ok {
				return lookBad("bad_request", "max_tokens must be an integer from 1 to %d", maxLookTokens)
			}
			q.maxTok = n
		case "id_slot":
			n, ok := jsonInt(v, 0, maxLookSlot)
			if !ok {
				return lookBad("bad_request", "id_slot must be an integer from 0 to %d", maxLookSlot)
			}
			q.idSlot = n
		case "stream":
			if !isJSONBool(v) {
				return lookBad("bad_request", "stream must be true or false")
			}
			q.stream = string(v) == "true"
		default:
			return lookBad("bad_request", "unknown field %q", k)
		}
		return nil
	})
	if ce != nil {
		return q, ce
	}
	if q.images == nil {
		return q, lookBad("bad_request", "images is required")
	}
	if strings.TrimSpace(q.question) == "" {
		return q, lookBad("bad_request", "question is required")
	}
	if len(q.question) > maxLookQuestion {
		return q, lookBad("bad_request", "question is longer than %d bytes", maxLookQuestion)
	}
	return q, nil
}

func parseLookImage(i int, raw []byte) (lookImage, *run.ClientError) {
	var im lookImage
	ce := lookMembers(raw, fmt.Sprintf("images[%d]", i), func(k string, v []byte) *run.ClientError {
		switch k {
		case "data":
			if len(v) < 2 || v[0] != '"' {
				return lookBad("bad_request", "images[%d].data must be a base64 string", i)
			}
			if in := v[1 : len(v)-1]; bytes.IndexByte(in, '\\') < 0 {
				im.data = in // a view of the body
			} else {
				var t string // an escaped character: decoded, the one case that copies
				if err := json.Unmarshal(v, &t); err != nil {
					return lookBad("bad_request", "images[%d].data must be a base64 string", i)
				}
				im.data = []byte(t)
			}
		case "label":
			if err := jsonString(v, &im.label); err != nil {
				return lookBad("bad_request", "images[%d].label must be a string", i)
			}
			if len(im.label) > maxLookLabel {
				return lookBad("bad_request", "images[%d].label is longer than %d bytes", i, maxLookLabel)
			}
		default:
			return lookBad("bad_request", "images[%d]: unknown field %q", i, k)
		}
		return nil
	})
	if ce != nil {
		return im, ce
	}
	if len(im.data) == 0 {
		return im, lookBad("bad_request", "images[%d].data is required", i)
	}
	return im, nil
}

// lookWalk walks JSON that json.Valid has passed, in place: every value it gives is a view of the
// walked bytes, never a copy.
type lookWalk struct {
	b []byte
	i int
}

func (w *lookWalk) ws() {
	for w.i < len(w.b) && (w.b[w.i] == ' ' || w.b[w.i] == '\t' || w.b[w.i] == '\n' || w.b[w.i] == '\r') {
		w.i++
	}
}

func (w *lookWalk) str() {
	for w.i++; w.b[w.i] != '"'; w.i++ {
		if w.b[w.i] == '\\' {
			w.i++
		}
	}
	w.i++
}

// value gives the extent of the next value.
func (w *lookWalk) value() []byte {
	w.ws()
	start := w.i
	switch w.b[w.i] {
	case '"':
		w.str()
	case '{', '[':
		for depth := 0; ; {
			switch w.b[w.i] {
			case '"':
				w.str()
				continue
			case '{', '[':
				depth++
			case '}', ']':
				if depth--; depth == 0 {
					w.i++
					return w.b[start:w.i]
				}
			}
			w.i++
		}
	default:
		for w.i < len(w.b) && !strings.ContainsRune(" \t\n\r,]}", rune(w.b[w.i])) {
			w.i++
		}
	}
	return w.b[start:w.i]
}

// lookMembers calls f with each member of the object v (valid JSON) in order; a key given twice,
// or v not an object, is refused.
func lookMembers(v []byte, what string, f func(k string, v []byte) *run.ClientError) *run.ClientError {
	w := &lookWalk{b: v}
	w.ws()
	if w.i >= len(v) || v[w.i] != '{' {
		return lookBad("bad_request", "%s must be an object", what)
	}
	w.i++
	seen := map[string]bool{}
	for {
		w.ws()
		if v[w.i] == '}' {
			return nil
		}
		raw := w.value()
		var k string
		if len(raw) > 2+256 || json.Unmarshal(raw, &k) != nil {
			k = "(a key too long to name)"
		}
		if seen[k] {
			return lookBad("bad_request", "%s: %q is given twice", what, k)
		}
		seen[k] = true
		w.ws()
		w.i++ // the colon
		if ce := f(k, w.value()); ce != nil {
			return ce
		}
		w.ws()
		if v[w.i] == ',' {
			w.i++
		}
	}
}

// lookElements calls f with each element of the array v (valid JSON) in order.
func lookElements(v []byte, f func(e []byte) *run.ClientError) *run.ClientError {
	w := &lookWalk{b: v}
	w.ws()
	w.i++ // the bracket
	for {
		w.ws()
		if v[w.i] == ']' {
			return nil
		}
		if ce := f(w.value()); ce != nil {
			return ce
		}
		w.ws()
		if v[w.i] == ',' {
			w.i++
		}
	}
}

func isJSONArray(v []byte) bool { return len(v) > 0 && v[0] == '[' }

func jsonString(v json.RawMessage, s *string) error {
	if len(v) == 0 || v[0] != '"' {
		return fmt.Errorf("not a string")
	}
	return json.Unmarshal(v, s)
}

func isJSONBool(v json.RawMessage) bool {
	t := strings.TrimSpace(string(v))
	return t == "true" || t == "false"
}

var intToken = regexp.MustCompile(`^-?(0|[1-9][0-9]{0,9})$`)

// jsonInt reads an integer token (no fraction, no exponent, no leading zero, at most 10 digits)
// in [lo, hi]: llama-server casts numbers to 32 bits, so a float or a huge number is never judged.
func jsonInt(v json.RawMessage, lo, hi int) (int, bool) {
	t := strings.TrimSpace(string(v))
	if !intToken.MatchString(t) {
		return 0, false
	}
	n, err := strconv.ParseInt(t, 10, 64)
	if err != nil || n < int64(lo) || n > int64(hi) {
		return 0, false
	}
	return int(n), true
}

// look is POST /engine/v1/look.
func (a *API) look(w http.ResponseWriter, r *http.Request) {
	local := localCaller(r)
	who := identity(r)
	// the body counts against the remote bodies kept (readBody) until the request built from it
	// takes its place, counted before the body's count is given back (never uncounted). The body
	// is read before the look's gate is taken: a slow sender holds only its read.
	give, ok := a.readBody(w, r, local)
	if !ok {
		return
	}
	defer give()
	if !a.looks.enter(who, local) {
		clientFail(w, lookBusy("looks are being decoded now, or this caller has a look running: ask again when it ends"))
		return
	}
	decoding := true
	defer func() {
		if decoding {
			a.looks.decoded()
		}
		a.looks.release(who, local)
	}()
	q, ce := parseLook(bodyOf(r))
	if ce != nil {
		clientFail(w, ce)
		return
	}
	setBody(r, nil) // the caller's bytes are let go: only the parts read from them go on
	spec, ok := a.S.LookSpec()
	if !ok {
		clientFail(w, &run.ClientError{Status: 409, Code: "vision_off", Msg: "the model server has no vision projector loaded (status vision_note says why)"})
		return
	}
	edge := spec.Detail.Edge(q.detail)
	if edge <= 0 {
		clientFail(w, &run.ClientError{Status: 409, Code: "vision_off", Msg: "the served model's projector has no detail table"})
		return
	}
	built, ce := buildLook(r.Context(), &q, spec, edge, local)
	if ce != nil {
		clientFail(w, ce)
		return
	}
	a.looks.decoded()
	decoding = false
	// the hand-over: the built request is counted, then the body's count is given back; the
	// proxy reads no body for a look (it is held already) and counts nothing more
	keepBuilt, ok := a.keepRemote(w, len(built), local)
	if !ok {
		return
	}
	defer keepBuilt()
	give()

	r2 := r.Clone(context.WithValue(r.Context(), lookCtxKey{}, true))
	r2.Method = http.MethodPost
	r2.URL.Path, r2.URL.RawPath, r2.URL.RawQuery = "/v1/chat/completions", "", ""
	r2.RequestURI = ""
	r2.Header.Del("Content-Encoding")
	r2.Header.Set("Content-Type", "application/json")
	setBody(r2, built)
	if q.stream {
		lw := &lookStream{w: w}
		a.proxy(lw, r2)
		lw.end()
		return
	}
	rec := &lookRecorder{w: w, h: http.Header{}}
	a.proxy(rec, r2)
	rec.answer()
}

// buildLook builds the child chat request into one buffer as each image is decoded: each image's
// base64 is let go once its part is written, the parts are never held apart from the buffer, and
// a remote look's buffer is bounded as it grows (maxRemoteBody, look_too_large). The request's
// context is asked between images (and by Prepare between its own steps).
func buildLook(ctx context.Context, q *lookReq, spec catalog.Projector, edge int, local bool) ([]byte, *run.ClientError) {
	var buf lookBuf
	str := func(v string) { b, _ := json.Marshal(v); buf.Write(b) }
	buf.WriteString(`{"messages":[{"role":"user","content":[`)
	for i := range q.images {
		if err := ctx.Err(); err != nil {
			return nil, &run.ClientError{Status: 400, Code: "image_cancelled", Msg: "the look was cancelled", Extra: map[string]any{"image": i}}
		}
		if ce := lookPart(ctx, &buf, i, &q.images[i], edge, local); ce != nil {
			return nil, ce
		}
	}
	buf.WriteString(`{"type":"text","text":`)
	str(q.question)
	fmt.Fprintf(&buf, `}]}],"max_tokens":%d,"stream":%t`, q.maxTok, q.stream)
	if q.stream {
		buf.WriteString(`,"stream_options":{"include_usage":true}`)
	}
	if q.idSlot >= 0 {
		fmt.Fprintf(&buf, `,"id_slot":%d`, q.idSlot)
	}
	if len(spec.ChatTemplateKwargs) > 0 {
		kw, err := json.Marshal(spec.ChatTemplateKwargs)
		if err != nil {
			return nil, &run.ClientError{Status: 500, Code: "look_failed", Msg: err.Error()}
		}
		buf.WriteString(`,"chat_template_kwargs":`)
		buf.Write(kw)
	}
	buf.WriteByte('}')
	if !local && buf.Len() > maxRemoteBody {
		return nil, lookTooLarge(buf.Len())
	}
	return buf.b, nil
}

// lookBuf is the built request's buffer. It grows by what is asked plus a little (never doubled
// like a bytes.Buffer), so the request kept is about its own length, and a moment's copy while
// it grows is at most the old bytes beside the new.
type lookBuf struct{ b []byte }

func (l *lookBuf) Len() int { return len(l.b) }

func (l *lookBuf) Grow(n int) {
	if cap(l.b)-len(l.b) < n {
		nb := make([]byte, len(l.b), len(l.b)+n+4<<10)
		copy(nb, l.b)
		l.b = nb
	}
}

func (l *lookBuf) Write(p []byte) (int, error) {
	l.Grow(len(p))
	l.b = append(l.b, p...)
	return len(p), nil
}

func (l *lookBuf) WriteString(s string) { l.Grow(len(s)); l.b = append(l.b, s...) }

func (l *lookBuf) WriteByte(c byte) error { l.Grow(1); l.b = append(l.b, c); return nil }

// lookPart decodes image i, writes its label and image parts into buf and lets the caller's
// base64 go.
func lookPart(ctx context.Context, buf *lookBuf, i int, im *lookImage, edge int, local bool) *run.ClientError {
	res, ie := lookimg.PrepareBytes(ctx, im.data, edge, lookimg.Default)
	im.data = nil // the caller's image is let go once it is built
	if ie != nil {
		return &run.ClientError{Status: 400, Code: ie.Code, Msg: fmt.Sprintf("images[%d]: %s", i, ie.Msg), Extra: map[string]any{"image": i}}
	}
	if im.label != "" {
		buf.WriteString(`{"type":"text","text":`)
		b, _ := json.Marshal(im.label)
		buf.Write(b)
		buf.WriteString(`},`)
	}
	// a remote look's built images are bounded as they are made, not only once all are built
	n := base64.StdEncoding.EncodedLen(len(res.Data))
	if !local && buf.Len()+n > maxRemoteBody {
		return lookTooLarge(buf.Len() + n)
	}
	buf.Grow(n + 64)
	buf.WriteString(`{"type":"image_url","image_url":{"url":"data:` + res.MIME + `;base64,`)
	enc := base64.NewEncoder(base64.StdEncoding, buf)
	enc.Write(res.Data)
	enc.Close()
	buf.WriteString(`"}},`)
	return nil
}

// lookTotals are the figures a look answer reports, read from the model server's own answer:
// usage.prompt_tokens and usage.completion_tokens, timings.prompt_ms and timings.predicted_ms.
// llama-server does not say how many tokens the images became: tokens.image is -1. A figure the
// answer does not carry is -1.
type lookTotals struct {
	Image      int `json:"image"`
	Prompt     int `json:"prompt"`
	Completion int `json:"completion"`
}

type lookMS struct {
	Prefill int `json:"prefill"`
	Total   int `json:"total"`
}

type lookAnswer struct {
	Object string     `json:"object,omitempty"`
	Text   string     `json:"text"`
	Tokens lookTotals `json:"tokens"`
	MS     lookMS     `json:"ms"`
}

// upstreamFigures is what is read of a chat answer or chunk.
type upstreamFigures struct {
	Choices []struct {
		Message *struct {
			Content *string `json:"content"`
		} `json:"message"`
		Delta *struct {
			Content *string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     *float64 `json:"prompt_tokens"`
		CompletionTokens *float64 `json:"completion_tokens"`
	} `json:"usage"`
	Timings *struct {
		PromptMS    *float64 `json:"prompt_ms"`
		PredictedMS *float64 `json:"predicted_ms"`
	} `json:"timings"`
}

func newLookAnswer() lookAnswer {
	return lookAnswer{Tokens: lookTotals{Image: -1, Prompt: -1, Completion: -1}, MS: lookMS{Prefill: -1, Total: -1}}
}

func roundFig(f *float64) int {
	if f == nil || math.IsNaN(*f) || *f < 0 || *f > math.MaxInt32 {
		return -1
	}
	return int(math.Round(*f))
}

// take lays the figures of one answer or chunk over the answer (the last that carries one wins).
func (la *lookAnswer) take(u upstreamFigures) {
	if u.Usage != nil {
		if v := roundFig(u.Usage.PromptTokens); v >= 0 {
			la.Tokens.Prompt = v
		}
		if v := roundFig(u.Usage.CompletionTokens); v >= 0 {
			la.Tokens.Completion = v
		}
	}
	if t := u.Timings; t != nil {
		p, d := roundFig(t.PromptMS), roundFig(t.PredictedMS)
		if p >= 0 {
			la.MS.Prefill = p
		}
		if p >= 0 && d >= 0 {
			la.MS.Total = int(math.Round(*t.PromptMS + *t.PredictedMS))
		}
	}
}

// lookRecorder takes the proxy's answer to a non-streamed look: a 200 chat completion becomes
// the look answer; anything else (a refusal of the door, the model server's error) goes to the
// caller as it is.
type lookRecorder struct {
	w      http.ResponseWriter
	h      http.Header
	status int
	buf    bytes.Buffer
	over   bool
}

func (l *lookRecorder) Header() http.Header { return l.h }
func (l *lookRecorder) WriteHeader(c int) {
	if l.status == 0 {
		l.status = c
	}
}
func (l *lookRecorder) Write(p []byte) (int, error) {
	if l.status == 0 {
		l.status = 200
	}
	if l.buf.Len()+len(p) > maxLookAnswer {
		l.over = true
		return len(p), nil
	}
	return l.buf.Write(p)
}
func (l *lookRecorder) Flush() {}

// Unwrap gives http.ResponseController the connection (read deadlines while a body is read).
func (l *lookRecorder) Unwrap() http.ResponseWriter { return l.w }

func (l *lookRecorder) answer() {
	for _, k := range []string{RequestIDHeader, AppliedHeader, "Retry-After"} {
		if v := l.h.Get(k); v != "" {
			l.w.Header().Set(k, v)
		}
	}
	if l.status == 0 {
		l.status = http.StatusBadGateway
	}
	if l.status != http.StatusOK {
		for k, vs := range l.h {
			if k == "Content-Length" {
				continue
			}
			l.w.Header()[k] = vs
		}
		l.w.WriteHeader(l.status)
		l.w.Write(l.buf.Bytes())
		return
	}
	if l.over {
		clientFail(l.w, &run.ClientError{Status: 502, Code: "bad_gateway", Msg: fmt.Sprintf("the model server's answer is larger than %d MiB", maxLookAnswer>>20)})
		return
	}
	var u upstreamFigures
	if err := json.Unmarshal(l.buf.Bytes(), &u); err != nil || len(u.Choices) == 0 || u.Choices[0].Message == nil {
		clientFail(l.w, &run.ClientError{Status: 502, Code: "bad_gateway", Msg: "the model server's answer is not a chat completion"})
		return
	}
	la := newLookAnswer()
	if c := u.Choices[0].Message.Content; c != nil {
		la.Text = *c
	}
	la.take(u)
	writeJSON(l.w, 200, la)
}

// lookStream passes a streamed look through as the proxy streams a chat completion: every
// event as the model server (or the door's hold) sent it. Before the final `data: [DONE]` it
// adds one event of its own, `data: {"object":"look.result","text":...,"tokens":...,"ms":...}`,
// with the whole text and the figures of the stream's last usage and timings. A stream that
// ends without [DONE] (an error event, the caller gone) gets no look.result.
type lookStream struct {
	w      http.ResponseWriter
	status int
	pend   []byte
	la     lookAnswer
	text   strings.Builder
	init   bool
}

func (l *lookStream) Header() http.Header { return l.w.Header() }
func (l *lookStream) WriteHeader(c int) {
	if l.status == 0 {
		l.status = c
	}
	l.w.WriteHeader(c)
}
func (l *lookStream) Flush() {
	if f, ok := l.w.(http.Flusher); ok {
		f.Flush()
	}
}
func (l *lookStream) Unwrap() http.ResponseWriter { return l.w }

func (l *lookStream) Write(p []byte) (int, error) {
	if l.status == 0 {
		l.status = 200
	}
	if l.status != http.StatusOK || !strings.HasPrefix(l.w.Header().Get("Content-Type"), "text/event-stream") {
		return l.w.Write(p)
	}
	if !l.init {
		l.la, l.init = newLookAnswer(), true
	}
	l.pend = append(l.pend, p...)
	for {
		i := bytes.Index(l.pend, []byte("\n\n"))
		if i < 0 {
			break
		}
		ev := l.pend[:i+2]
		if err := l.event(ev); err != nil {
			return len(p), err
		}
		l.pend = l.pend[i+2:]
	}
	if len(l.pend) > maxLookAnswer { // one event that never ends: passed on as it is
		if _, err := l.w.Write(l.pend); err != nil {
			return len(p), err
		}
		l.pend = nil
	}
	return len(p), nil
}

func (l *lookStream) event(ev []byte) error {
	sc := bufio.NewScanner(bytes.NewReader(ev))
	sc.Buffer(make([]byte, 0, 64<<10), maxLookAnswer)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			l.la.Object, l.la.Text = lookObject, l.text.String()
			b, _ := json.Marshal(l.la)
			if _, err := fmt.Fprintf(l.w, "data: %s\n\n", b); err != nil {
				return err
			}
			break
		}
		var u upstreamFigures
		if json.Unmarshal([]byte(data), &u) == nil {
			for _, c := range u.Choices {
				if c.Delta != nil && c.Delta.Content != nil {
					l.text.WriteString(*c.Delta.Content)
				}
			}
			l.la.take(u)
		}
	}
	_, err := l.w.Write(ev)
	return err
}

// end passes on what is left of a stream that did not end on an event boundary.
func (l *lookStream) end() {
	if len(l.pend) > 0 {
		l.w.Write(l.pend)
		l.pend = nil
		l.Flush()
	}
}
