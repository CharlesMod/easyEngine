// Package stream reads a model server's server-sent events (SSE) reply event by event as the
// proxy forwards it. It never rewrites a byte: every event goes out exactly as the upstream sent
// it, the moment its blank line arrives. Beside that it reads each event's data for the few facts
// the engine wants (the end of the reply, an upstream error, token counts, speed, the slot), and
// when the upstream stops before the reply is complete it ends the stream with one terminal error
// event, so a client hears "the reply was cut" instead of seeing a reply that merely stopped.
//
// It uses no timer and starts no goroutine: everything happens inside the caller's Read and Close.
package stream

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"sync"
)

// Dialect is which server's stream this is; it decides what "complete" means.
type Dialect int

const (
	// Unknown is a stream the reader only counts: nothing is ever dropped or added.
	Unknown Dialect = iota
	// Chat is OpenAI chat completions (/v1/chat/completions): ends with data: [DONE].
	Chat
	// Completions is OpenAI text completions (/v1/completions): ends with data: [DONE].
	Completions
	// Native is llama-server's own /completion: no [DONE]; its last event says "stop": true.
	Native
)

func (d Dialect) String() string {
	switch d {
	case Chat:
		return "chat"
	case Completions:
		return "completions"
	case Native:
		return "native"
	}
	return "unknown"
}

// End is how a stream ended.
type End string

const (
	// Done: the reply was complete.
	Done End = "done"
	// UpstreamError: the upstream sent its own error event (forwarded as it was).
	UpstreamError End = "upstream_error"
	// EndedEarly: the upstream stopped before the reply was complete, the client still there.
	EndedEarly End = "ended_early"
	// ClientGone: the client left (or the reader was closed) before the reply was complete.
	ClientGone End = "client_gone"
)

// Summary is what the reader learned of one reply. It is handed to OnEnd once.
type Summary struct {
	Dialect      Dialect
	End          End
	Events       int // upstream events forwarded (comments included)
	FinishReason string
	// Token counts come from the usage chunk when the upstream sent one, else from timings
	// (prompt = prompt_n + cache_n there, as usage counts it).
	PromptTokens, CachedTokens, CompletionTokens int
	TokPerSec                                    float64 // timings.predicted_per_second
	PromptMs, PredictedMs                        float64
	Slot                                         int // llama-server's id_slot when said, else -1
	RequestID                                    string
}

// Options are the reader's hooks. Every one may be nil.
type Options struct {
	Dialect Dialect
	// OnEvent is called once per upstream event, as it is released.
	OnEvent func()
	// OnEnd is called exactly once, on every path: at the end of the stream or at Close.
	OnEnd func(Summary)
	// Cut is called when the upstream ends early and the client is still there. It returns the
	// JSON object sent as the value of "error" in the terminal event; nil sends nothing.
	Cut func() []byte
	// Gone reports that the client has left: then no terminal event is written.
	Gone func() bool
	// MaxEvent is the longest event the reader frames (default 4 MiB). Past it the reader stops
	// framing, passes the rest through raw and adds nothing at the end.
	MaxEvent int
	// RequestID is copied into the Summary.
	RequestID string
}

// DefaultMaxEvent is Options.MaxEvent when it is zero.
const DefaultMaxEvent = 4 << 20

// readSize is how much the reader asks of the upstream at once.
const readSize = 32 << 10

type reader struct {
	body io.ReadCloser
	o    Options

	pend []byte // read from the upstream, not yet released: the start of an unfinished event
	out  []byte // released, not yet handed to the caller: whole events (or raw bytes)
	scan int    // how far pend has been searched for the blank line
	line int    // where the current line starts in pend
	tmp  []byte // the upstream read buffer

	raw   bool  // framing stopped (an event longer than MaxEvent, or the Unknown dialect)
	final error // returned once out is empty: io.EOF or the upstream's read error

	// mu guards what Close reads: Close may come from another goroutine than Read.
	mu       sync.Mutex
	complete bool // the reply is whole by the upstream's own word
	upErr    bool // the upstream sent its own error event
	sawFin   bool // a finish_reason was seen
	usage    bool // a usage chunk was seen: its counts win over timings
	sum      Summary
	fired    bool
}

// New wraps an upstream SSE body. Each Read returns whole events (or the next piece of an event
// longer than the caller's buffer), each exactly the bytes the upstream sent.
func New(body io.ReadCloser, o Options) io.ReadCloser {
	if o.MaxEvent <= 0 {
		o.MaxEvent = DefaultMaxEvent
	}
	r := &reader{body: body, o: o, raw: o.Dialect == Unknown}
	r.sum = Summary{Dialect: o.Dialect, Slot: -1, RequestID: o.RequestID}
	return r
}

func (r *reader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for {
		if len(r.out) > 0 {
			n := copy(p, r.out)
			r.out = r.out[n:]
			return n, nil
		}
		if r.final != nil {
			r.fire(r.endAs())
			return 0, r.final
		}
		if r.raw && len(r.pend) == 0 {
			// Raw: the caller's buffer is filled straight from the upstream, nothing in between.
			n, err := r.body.Read(p)
			if n > 0 && r.o.Dialect == Unknown {
				r.countRaw(p[:n])
			}
			if err != nil {
				r.final = err
				if n == 0 {
					r.fire(r.endAs())
				}
			}
			return n, err
		}
		if r.tmp == nil {
			r.tmp = make([]byte, readSize)
		}
		n, err := r.body.Read(r.tmp)
		r.pend = append(r.pend, r.tmp[:n]...)
		r.frame()
		if err != nil {
			r.finish(err)
		}
	}
}

// frame moves every whole event at the front of pend to out, reading each one as it goes.
func (r *reader) frame() {
	start := 0 // the first byte of pend not yet released
	b := r.pend
	i := r.scan
	for i < len(b) {
		c := b[i]
		if c != '\n' && c != '\r' {
			i++
			continue
		}
		end := i + 1
		if c == '\r' {
			if i+1 == len(b) {
				break // may be the first half of \r\n: wait for the next byte
			}
			if b[i+1] == '\n' {
				end = i + 2
			}
		}
		if i == r.line { // an empty line: the event ends here
			r.event(b[start:end])
			start = end
		}
		r.line, i = end, end
	}
	r.scan = i
	if start > 0 {
		r.out = b[:start:start] // capped, so a later append to pend never writes into it
		r.pend = b[start:]
		r.scan -= start
		r.line -= start
	}
	if len(r.pend) > r.o.MaxEvent {
		// An event this long is not a token stream any more: stop framing, pass it on raw.
		r.out = append(r.out, r.pend...)
		r.pend, r.scan, r.line, r.raw = nil, 0, 0, true
	}
	if len(r.pend) == 0 {
		r.pend = nil // the next read starts a fresh array: out may still hold the old one
	}
}

// finish is the upstream's end (EOF or a read error) once every whole event is released.
func (r *reader) finish(err error) {
	gone := r.o.Gone != nil && r.o.Gone()
	r.mu.Lock()
	if r.sawFin && !r.complete && !r.upErr && errors.Is(err, io.EOF) && r.o.Dialect != Native {
		r.complete = true // a server that sends no [DONE] after its finish_reason
	}
	whole := r.complete || r.upErr
	r.mu.Unlock()
	r.final = io.EOF
	switch {
	case r.raw:
		// Raw (Unknown dialect or an oversize event): nothing is dropped or added.
		r.out = append(r.out, r.pend...)
		r.final = err
	case whole:
		// The reply is whole (or the upstream said its own error): any trailing bytes go on as
		// they are, and a read error after the end is not the client's concern.
		r.out = append(r.out, r.pend...)
	case gone:
		r.final = err
	default:
		// Cut short: the unfinished bytes of the last event are dropped (half an event is worse
		// than none), and one terminal error event says why.
		if r.o.Cut != nil {
			if c := r.o.Cut(); c != nil {
				r.out = append(r.out, "data: {\"error\":"...)
				r.out = append(r.out, c...)
				r.out = append(r.out, "}\n\n"...)
			}
		}
	}
	r.pend = nil
}

// endAs is the End of a stream that has run to its end in Read.
func (r *reader) endAs() End {
	r.mu.Lock()
	upErr, complete := r.upErr, r.complete
	r.mu.Unlock()
	switch {
	case upErr:
		return UpstreamError
	case complete:
		return Done
	case r.raw && errors.Is(r.final, io.EOF):
		return Done // a raw stream that ended cleanly: the reader cannot tell more
	case r.o.Gone != nil && r.o.Gone():
		return ClientGone
	}
	return EndedEarly
}

// event reads one whole event (its exact bytes) and counts it.
func (r *reader) event(ev []byte) {
	r.mu.Lock()
	r.sum.Events++
	r.mu.Unlock()
	if r.o.OnEvent != nil {
		r.o.OnEvent()
	}
	data, ok := eventData(ev)
	if !ok {
		return
	}
	if bytes.Equal(data, []byte("[DONE]")) {
		if r.o.Dialect == Chat || r.o.Dialect == Completions {
			r.mu.Lock()
			r.complete = true
			r.mu.Unlock()
		}
		return
	}
	if len(data) == 0 || data[0] != '{' {
		return
	}
	var c chunk
	if err := json.Unmarshal(data, &c); err != nil {
		// A field of another type than this reader expects (another server's "stop" or "usage")
		// must not hide the rest of the chunk: Unmarshal has filled in what it could. Were the
		// finish_reason lost here, a whole reply with no [DONE] would be taken for a cut one.
		var te *json.UnmarshalTypeError
		if !errors.As(err, &te) {
			return
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(c.Error) > 0 && !bytes.Equal(c.Error, []byte("null")) {
		r.upErr = true
	}
	for _, ch := range c.Choices {
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			r.sawFin = true
			r.sum.FinishReason = *ch.FinishReason
		}
	}
	if r.o.Dialect == Native && c.Stop != nil && *c.Stop {
		r.complete = true
		if r.sum.FinishReason == "" {
			r.sum.FinishReason = c.StopType
		}
	}
	if c.IDSlot != nil {
		r.sum.Slot = *c.IDSlot
	}
	if t := c.Timings; t != nil {
		r.sum.TokPerSec, r.sum.PromptMs, r.sum.PredictedMs = t.PredictedPerSecond, t.PromptMS, t.PredictedMS
		if c.Usage == nil && !r.usage {
			r.sum.PromptTokens, r.sum.CachedTokens, r.sum.CompletionTokens = t.PromptN+t.CacheN, t.CacheN, t.PredictedN
		}
	}
	if u := c.Usage; u != nil {
		r.sum.PromptTokens, r.sum.CompletionTokens = u.PromptTokens, u.CompletionTokens
		if u.Details != nil {
			r.sum.CachedTokens = u.Details.CachedTokens
		} else if c.Timings != nil {
			r.sum.CachedTokens = c.Timings.CacheN
		}
		r.usage = true
	}
}

// countRaw counts the events of an Unknown stream (it is never framed): one per blank line
// "\n\n" seen, close enough for a count that nothing depends on.
func (r *reader) countRaw(b []byte) {
	n := bytes.Count(b, []byte("\n\n"))
	if n == 0 {
		return
	}
	r.mu.Lock()
	r.sum.Events += n
	r.mu.Unlock()
	if r.o.OnEvent != nil {
		for range n {
			r.o.OnEvent()
		}
	}
}

// Close closes the upstream body (this is what cancels the generation when the client leaves)
// and ends the summary if the stream had not ended yet.
func (r *reader) Close() error {
	err := r.body.Close()
	end := ClientGone
	r.mu.Lock()
	if r.upErr {
		end = UpstreamError
	} else if r.complete {
		end = Done
	}
	r.mu.Unlock()
	r.fire(end) // a no-op when the stream already ended in Read
	return err
}

// fire calls OnEnd once.
func (r *reader) fire(end End) {
	r.mu.Lock()
	if r.fired {
		r.mu.Unlock()
		return
	}
	r.fired = true
	r.sum.End = end
	s := r.sum
	r.mu.Unlock()
	if r.o.OnEnd != nil {
		r.o.OnEnd(s)
	}
}

// eventData joins an event's data: lines (SSE: "data:" and one optional space, lines joined by
// "\n"). ok is false for an event with no data line (a comment, a ping).
func eventData(ev []byte) (data []byte, ok bool) {
	for len(ev) > 0 {
		var line []byte
		i := bytes.IndexAny(ev, "\r\n")
		if i < 0 {
			line, ev = ev, nil
		} else {
			line = ev[:i]
			if ev[i] == '\r' && i+1 < len(ev) && ev[i+1] == '\n' {
				i++
			}
			ev = ev[i+1:]
		}
		v, found := bytes.CutPrefix(line, []byte("data:"))
		if !found {
			continue
		}
		v, _ = bytes.CutPrefix(v, []byte(" "))
		if ok {
			data = append(append(data, '\n'), v...)
		} else {
			data, ok = v, true
		}
	}
	return data, ok
}

// chunk is the little of a streamed JSON chunk the reader reads; the rest is ignored.
type chunk struct {
	Choices []struct {
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Stop     *bool           `json:"stop"`
	StopType string          `json:"stop_type"`
	Error    json.RawMessage `json:"error"`
	IDSlot   *int            `json:"id_slot"`
	Usage    *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		Details          *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	Timings *struct {
		CacheN             int     `json:"cache_n"`
		PromptN            int     `json:"prompt_n"`
		PromptMS           float64 `json:"prompt_ms"`
		PredictedN         int     `json:"predicted_n"`
		PredictedMS        float64 `json:"predicted_ms"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
	} `json:"timings"`
}
