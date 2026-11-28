package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"

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
	data, label string
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
// level, numbers as integer tokens in range (never floats), strings as strings.
func parseLook(body []byte) (lookReq, *run.ClientError) {
	q := lookReq{detail: defLookDetail, maxTok: defLookTokens, idSlot: -1}
	if jsonDepth(body) > 8 {
		return q, lookBad("bad_request", "a look body nests at most 8 deep")
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var root map[string]json.RawMessage
	keys, err := objectKeys(dec, &root)
	if err != nil {
		return q, lookBad("bad_request", "the look body is not one JSON object: %v", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return q, lookBad("bad_request", "the look body has data after its object")
	}
	seen := map[string]bool{}
	for _, k := range keys {
		if seen[k] {
			return q, lookBad("bad_request", "%q is given twice", k)
		}
		seen[k] = true
	}
	for k, v := range root {
		switch k {
		case "images":
			var raws []json.RawMessage
			if err := json.Unmarshal(v, &raws); err != nil || raws == nil {
				return q, lookBad("bad_request", "images must be an array")
			}
			if len(raws) == 0 {
				return q, lookBad("bad_request", "images must hold at least one image")
			}
			if len(raws) > maxLookImages {
				return q, lookBad("too_many_images", "a look takes at most %d images, not %d", maxLookImages, len(raws))
			}
			for i, raw := range raws {
				im, ce := parseLookImage(i, raw)
				if ce != nil {
					return q, ce
				}
				q.images = append(q.images, im)
			}
		case "question":
			if err := jsonString(v, &q.question); err != nil {
				return q, lookBad("bad_request", "question must be a string")
			}
		case "detail":
			if err := jsonString(v, &q.detail); err != nil || (q.detail != "low" && q.detail != "high" && q.detail != "max") {
				return q, lookBad("bad_request", "detail must be low, high or max")
			}
		case "max_tokens":
			n, ok := jsonInt(v, 1, maxLookTokens)
			if !ok {
				return q, lookBad("bad_request", "max_tokens must be an integer from 1 to %d", maxLookTokens)
			}
			q.maxTok = n
		case "id_slot":
			n, ok := jsonInt(v, 0, maxLookSlot)
			if !ok {
				return q, lookBad("bad_request", "id_slot must be an integer from 0 to %d", maxLookSlot)
			}
			q.idSlot = n
		case "stream":
			if err := json.Unmarshal(v, &q.stream); err != nil || !isJSONBool(v) {
				return q, lookBad("bad_request", "stream must be true or false")
			}
		default:
			return q, lookBad("bad_request", "unknown field %q", k)
		}
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

func parseLookImage(i int, raw json.RawMessage) (lookImage, *run.ClientError) {
	var im lookImage
	dec := json.NewDecoder(bytes.NewReader(raw))
	var obj map[string]json.RawMessage
	keys, err := objectKeys(dec, &obj)
	if err != nil {
		return im, lookBad("bad_request", "images[%d] must be an object", i)
	}
	seen := map[string]bool{}
	for _, k := range keys {
		if seen[k] {
			return im, lookBad("bad_request", "images[%d]: %q is given twice", i, k)
		}
		seen[k] = true
	}
	for k, v := range obj {
		switch k {
		case "data":
			if err := jsonString(v, &im.data); err != nil {
				return im, lookBad("bad_request", "images[%d].data must be a base64 string", i)
			}
		case "label":
			if err := jsonString(v, &im.label); err != nil {
				return im, lookBad("bad_request", "images[%d].label must be a string", i)
			}
			if len(im.label) > maxLookLabel {
				return im, lookBad("bad_request", "images[%d].label is longer than %d bytes", i, maxLookLabel)
			}
		default:
			return im, lookBad("bad_request", "images[%d]: unknown field %q", i, k)
		}
	}
	if im.data == "" {
		return im, lookBad("bad_request", "images[%d].data is required", i)
	}
	return im, nil
}

// objectKeys decodes one JSON object into into and returns its keys in order, repeated ones
// included (a map alone would keep only the last).
func objectKeys(dec *json.Decoder, into *map[string]json.RawMessage) ([]string, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := t.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("not an object")
	}
	m := map[string]json.RawMessage{}
	var keys []string
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, err
		}
		k, _ := kt.(string)
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, err
		}
		keys = append(keys, k)
		m[k] = v
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	*into = m
	return keys, nil
}

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
	// the body counts against the remote bodies kept (readBody) until the request built from it
	// takes its place: the proxy's own read of that request counts it
	give, ok := a.readBody(w, r, local)
	if !ok {
		return
	}
	defer give()
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
	content := make([]map[string]any, 0, 2*len(q.images)+1)
	builtSize := 0
	for i, im := range q.images {
		res, ie := lookimg.Prepare(r.Context(), im.data, edge, lookimg.Default)
		q.images[i].data = "" // each caller image is let go once it is built
		if ie != nil {
			clientFail(w, &run.ClientError{Status: 400, Code: ie.Code, Msg: fmt.Sprintf("images[%d]: %s", i, ie.Msg), Extra: map[string]any{"image": i}})
			return
		}
		if im.label != "" {
			content = append(content, map[string]any{"type": "text", "text": im.label})
		}
		// a remote look's built images are bounded as they are made, not only once all are built
		builtSize += base64.StdEncoding.EncodedLen(len(res.Data))
		if !local && builtSize > maxRemoteBody {
			clientFail(w, lookTooLarge(builtSize))
			return
		}
		uri := "data:" + res.MIME + ";base64," + base64.StdEncoding.EncodeToString(res.Data)
		content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": uri}})
	}
	content = append(content, map[string]any{"type": "text", "text": q.question})
	child := map[string]any{
		"messages":   []map[string]any{{"role": "user", "content": content}},
		"max_tokens": q.maxTok,
		"stream":     q.stream,
	}
	if q.stream {
		child["stream_options"] = map[string]any{"include_usage": true}
	}
	if q.idSlot >= 0 {
		child["id_slot"] = q.idSlot
	}
	if len(spec.ChatTemplateKwargs) > 0 {
		child["chat_template_kwargs"] = spec.ChatTemplateKwargs
	}
	built, err := json.Marshal(child)
	content, child = nil, nil
	if err != nil {
		clientFail(w, &run.ClientError{Status: 500, Code: "look_failed", Msg: err.Error()})
		return
	}
	if !local && len(built) > maxRemoteBody {
		clientFail(w, lookTooLarge(len(built)))
		return
	}
	a.looks.decoded()
	decoding = false
	give() // the caller's body is let go; the built request is counted by the proxy's read below

	r2 := r.Clone(context.WithValue(r.Context(), lookCtxKey{}, true))
	r2.Method = http.MethodPost
	r2.URL.Path, r2.URL.RawPath, r2.URL.RawQuery = "/v1/chat/completions", "", ""
	r2.RequestURI = ""
	r2.Header.Del("Content-Encoding")
	r2.Header.Set("Content-Type", "application/json")
	setBody(r2, built)
	built = nil
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
