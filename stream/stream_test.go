package stream

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---- helpers ----

// pieceBody hands out its pieces one per Read, then ends with err (io.EOF when nil).
type pieceBody struct {
	pieces [][]byte
	err    error
	mu     sync.Mutex
	closed int
}

func (b *pieceBody) Read(p []byte) (int, error) {
	if len(b.pieces) == 0 {
		if b.err != nil {
			return 0, b.err
		}
		return 0, io.EOF
	}
	n := copy(p, b.pieces[0])
	b.pieces[0] = b.pieces[0][n:]
	if len(b.pieces[0]) == 0 {
		b.pieces = b.pieces[1:]
	}
	return n, nil
}

func (b *pieceBody) Close() error {
	b.mu.Lock()
	b.closed++
	b.mu.Unlock()
	return nil
}

func (b *pieceBody) closes() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.closed
}

func whole(s string) *pieceBody { return &pieceBody{pieces: [][]byte{[]byte(s)}} }

// splitAt cuts s in two at k; an empty piece is left out.
func splitAt(s string, k int) *pieceBody {
	b := &pieceBody{}
	if k > 0 {
		b.pieces = append(b.pieces, []byte(s[:k]))
	}
	if k < len(s) {
		b.pieces = append(b.pieces, []byte(s[k:]))
	}
	return b
}

func perByte(s string) *pieceBody {
	b := &pieceBody{}
	for i := 0; i < len(s); i++ {
		b.pieces = append(b.pieces, []byte{s[i]})
	}
	return b
}

// rec collects what the hooks were told.
type rec struct {
	mu     sync.Mutex
	ends   []Summary
	events int
	cuts   int
}

func (c *rec) opts(d Dialect) Options {
	return Options{
		Dialect: d,
		OnEvent: func() { c.mu.Lock(); c.events++; c.mu.Unlock() },
		OnEnd:   func(s Summary) { c.mu.Lock(); c.ends = append(c.ends, s); c.mu.Unlock() },
		Cut: func() []byte {
			c.mu.Lock()
			c.cuts++
			c.mu.Unlock()
			return []byte(cutJSON)
		},
	}
}

func (c *rec) endCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.ends)
}

func (c *rec) cutCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cuts
}

func (c *rec) eventCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.events
}

// summary is the one Summary, failing the test when OnEnd did not fire exactly once.
func (c *rec) summary(t *testing.T) Summary {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.ends) != 1 {
		t.Fatalf("OnEnd fired %d times, want 1", len(c.ends))
	}
	return c.ends[0]
}

const cutJSON = `{"code":"cut","message":"upstream stopped"}`
const cutEvent = "data: {\"error\":" + cutJSON + "}\n\n"

// readChunks reads until the end with a buffer of the given size and returns what each Read gave.
// A final error other than io.EOF is returned.
func readChunks(r io.Reader, size int) ([][]byte, error) {
	var chunks [][]byte
	buf := make([]byte, size)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			chunks = append(chunks, append([]byte(nil), buf[:n]...))
		}
		if err == io.EOF {
			return chunks, nil
		}
		if err != nil {
			return chunks, err
		}
	}
}

func readAll(r io.Reader) ([]byte, error) {
	chunks, err := readChunks(r, 32<<10)
	return bytes.Join(chunks, nil), err
}

func endsAtBoundary(b []byte) bool {
	return bytes.HasSuffix(b, []byte("\n\n")) || bytes.HasSuffix(b, []byte("\r\n\r\n"))
}

func mustSame(t *testing.T, got []byte, want string) {
	t.Helper()
	if string(got) != want {
		t.Fatalf("output differs\n got: %q\nwant: %q", got, want)
	}
}

func timeoutCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second) // network timeout
	t.Cleanup(cancel)
	return ctx
}

// ---- fixtures ----

const chatHead = `"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"b1"`

// chatEvents is a recorded llama-server chat stream, event by event.
var chatEvents = []string{
	`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}],` + chatHead + "}\n\n",
	`data: {"prompt_progress":{"total":36,"cache":64,"processed":36,"time_ms":12.5},` + chatHead + "}\n\n",
	`data: {"choices":[{"index":0,"finish_reason":null,"delta":{"content":"Hel"}}],` + chatHead + "}\n\n",
	`data: {"choices":[{"index":0,"finish_reason":null,"delta":{"content":"lo"}}],` + chatHead + "}\n\n",
	`data: {"choices":[{"index":0,"finish_reason":null,"delta":{"content":" there"}}],` + chatHead + "}\n\n",
	":\n\n",
	`data: {"choices":[{"index":0,"finish_reason":null,"delta":{"content":"!"}}],` + chatHead + "}\r\n\r\n",
	`data: {"choices":[{"index":0,"finish_reason":"stop","delta":{}}],` + chatHead +
		`,"timings":{"cache_n":64,"prompt_n":36,"prompt_ms":12.5,"prompt_per_token_ms":0.3,"predicted_n":19,"predicted_ms":400.5,"predicted_per_second":47.4}}` + "\n\n",
	`data: {"choices":[],` + chatHead +
		`,"usage":{"completion_tokens":20,"prompt_tokens":100,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":64}}}` + "\n\n",
	"data: [DONE]\n\n",
}

var chatStream = strings.Join(chatEvents, "")

func checkChatSummary(t *testing.T, s Summary) {
	t.Helper()
	if s.End != Done {
		t.Errorf("End = %q, want done", s.End)
	}
	if s.Events != len(chatEvents) {
		t.Errorf("Events = %d, want %d", s.Events, len(chatEvents))
	}
	if s.FinishReason != "stop" {
		t.Errorf("FinishReason = %q", s.FinishReason)
	}
	// Usage wins over timings: the fixture's predicted_n (19) differs from usage (20) on purpose.
	if s.PromptTokens != 100 || s.CachedTokens != 64 || s.CompletionTokens != 20 {
		t.Errorf("tokens = %d/%d/%d, want 100/64/20", s.PromptTokens, s.CachedTokens, s.CompletionTokens)
	}
	if s.TokPerSec != 47.4 || s.PromptMs != 12.5 || s.PredictedMs != 400.5 {
		t.Errorf("speed = %v/%v/%v", s.TokPerSec, s.PromptMs, s.PredictedMs)
	}
	if s.Slot != -1 {
		t.Errorf("Slot = %d, want -1", s.Slot)
	}
	if s.Dialect != Chat {
		t.Errorf("Dialect = %v", s.Dialect)
	}
}

// ---- tests ----

func TestZeroDialectIsUnknown(t *testing.T) {
	var d Dialect
	if d != Unknown || d.String() != "unknown" {
		t.Fatalf("zero Dialect = %v", d)
	}
}

func TestChatForwardedByteForByte(t *testing.T) {
	var c rec
	o := c.opts(Chat)
	o.RequestID = "req-7"
	r := New(whole(chatStream), o)
	got, err := readAll(r)
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, chatStream)
	s := c.summary(t)
	checkChatSummary(t, s)
	if s.RequestID != "req-7" {
		t.Errorf("RequestID = %q", s.RequestID)
	}
	if c.cutCount() != 0 {
		t.Error("Cut called on a complete stream")
	}
}

func TestSplitAtEveryOffset(t *testing.T) {
	for k := 0; k <= len(chatStream); k++ {
		var c rec
		r := New(splitAt(chatStream, k), c.opts(Chat))
		chunks, err := readChunks(r, 1<<20)
		if err != nil {
			t.Fatalf("k=%d: %v", k, err)
		}
		for i, ch := range chunks {
			if !endsAtBoundary(ch) {
				t.Fatalf("k=%d: Read %d does not end at an event boundary: %q", k, i, ch)
			}
		}
		if got := string(bytes.Join(chunks, nil)); got != chatStream {
			t.Fatalf("k=%d: output differs\n got: %q", k, got)
		}
		checkChatSummary(t, c.summary(t))
	}
}

func TestOneBytePerRead(t *testing.T) {
	var c rec
	r := New(perByte(chatStream), c.opts(Chat))
	chunks, err := readChunks(r, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	for i, ch := range chunks {
		if !endsAtBoundary(ch) {
			t.Fatalf("Read %d does not end at an event boundary: %q", i, ch)
		}
	}
	mustSame(t, bytes.Join(chunks, nil), chatStream)
	if len(chunks) != len(chatEvents) {
		t.Errorf("got %d Reads with data, want one per event (%d)", len(chunks), len(chatEvents))
	}
	checkChatSummary(t, c.summary(t))
}

func TestEventReleasedAtOnce(t *testing.T) {
	ctx := timeoutCtx(t)
	pr, pw := io.Pipe()
	var c rec
	r := New(pr, c.opts(Chat))

	second := make(chan struct{})
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		defer pw.Close()
		if _, err := pw.Write([]byte(chatEvents[0])); err != nil {
			return
		}
		// The writer now blocks until the test lets the second event go.
		select {
		case <-second:
		case <-ctx.Done():
			return
		}
		pw.Write([]byte(chatEvents[1]))
	}()

	type res struct {
		b   []byte
		err error
	}
	first := make(chan res, 1)
	go func() {
		buf := make([]byte, 1<<20)
		n, err := r.Read(buf)
		first <- res{append([]byte(nil), buf[:n]...), err}
	}()
	select {
	case got := <-first:
		if got.err != nil {
			t.Fatal(got.err)
		}
		mustSame(t, got.b, chatEvents[0])
	case <-ctx.Done():
		t.Fatal("the first event was held back while the upstream was silent")
	}
	if c.eventCount() != 1 {
		t.Errorf("OnEvent count = %d after the first event", c.eventCount())
	}
	close(second)
	rest, err := readAll(r)
	if err != nil {
		t.Fatal(err)
	}
	// The pipe ends here with no [DONE]: the stream was cut, so the terminal event follows.
	mustSame(t, rest, chatEvents[1]+cutEvent)
	<-writerDone
}

const nativeStop = `data: {"index":0,"content":"","tokens":[],"stop":true,"stop_type":"eos","id_slot":0,` +
	`"timings":{"cache_n":10,"prompt_n":5,"prompt_ms":3.5,"predicted_n":7,"predicted_ms":140,"predicted_per_second":50}}` + "\n\n"

var nativeStream = `data: {"index":0,"content":"Hi","stop":false,"id_slot":0}` + "\n\n" +
	`data: {"index":0,"content":"!","stop":false,"id_slot":0}` + "\n\n" + nativeStop

func TestNativeEndsOnStop(t *testing.T) {
	var c rec
	got, err := readAll(New(whole(nativeStream), c.opts(Native)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, nativeStream)
	s := c.summary(t)
	if s.End != Done {
		t.Errorf("End = %q", s.End)
	}
	if s.Events != 3 {
		t.Errorf("Events = %d", s.Events)
	}
	if s.Slot != 0 {
		t.Errorf("Slot = %d, want 0", s.Slot)
	}
	if s.FinishReason != "eos" {
		t.Errorf("FinishReason = %q", s.FinishReason)
	}
	if s.PromptTokens != 15 || s.CachedTokens != 10 || s.CompletionTokens != 7 {
		t.Errorf("tokens = %d/%d/%d, want 15/10/7", s.PromptTokens, s.CachedTokens, s.CompletionTokens)
	}
	if s.TokPerSec != 50 || s.PromptMs != 3.5 || s.PredictedMs != 140 {
		t.Errorf("speed = %v/%v/%v", s.TokPerSec, s.PromptMs, s.PredictedMs)
	}
	if c.cutCount() != 0 {
		t.Error("Cut called")
	}
}

func TestNativeWithoutStopIsCut(t *testing.T) {
	var c rec
	in := `data: {"content":"Hi","stop":false}` + "\n\n"
	got, err := readAll(New(whole(in), c.opts(Native)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, in+cutEvent)
	if s := c.summary(t); s.End != EndedEarly {
		t.Errorf("End = %q", s.End)
	}
}

func TestEarlyEOFMidEvent(t *testing.T) {
	var c rec
	head := chatEvents[0] + chatEvents[2]
	in := head + `data: {"choices":[{"index":0,"finish_re`
	got, err := readAll(New(whole(in), c.opts(Chat)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, head+cutEvent)
	if strings.Contains(string(got), "[DONE]") {
		t.Error("[DONE] after the terminal event")
	}
	if strings.Count(string(got), `{"error":`) != 1 {
		t.Error("want exactly one terminal event")
	}
	s := c.summary(t)
	if s.End != EndedEarly || s.Events != 2 {
		t.Errorf("End = %q Events = %d", s.End, s.Events)
	}
	if c.cutCount() != 1 {
		t.Errorf("Cut called %d times", c.cutCount())
	}
}

func TestEarlyEOFAtBoundary(t *testing.T) {
	var c rec
	head := chatEvents[0] + chatEvents[2] + chatEvents[5]
	got, err := readAll(New(whole(head), c.opts(Chat)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, head+cutEvent)
	if s := c.summary(t); s.End != EndedEarly {
		t.Errorf("End = %q", s.End)
	}
}

func TestEarlyEOFCompletionsDialect(t *testing.T) {
	var c rec
	head := `data: {"choices":[{"text":"a","index":0,"finish_reason":null}]}` + "\n\n"
	got, err := readAll(New(whole(head), c.opts(Completions)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, head+cutEvent)
}

func TestCompletionsDone(t *testing.T) {
	var c rec
	in := `data: {"choices":[{"text":"a","index":0,"finish_reason":null}]}` + "\n\n" + "data: [DONE]\n\n"
	got, err := readAll(New(whole(in), c.opts(Completions)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, in)
	if s := c.summary(t); s.End != Done {
		t.Errorf("End = %q", s.End)
	}
}

func TestReadErrorGivesTerminalEventThenEOF(t *testing.T) {
	var c rec
	head := chatEvents[0] + chatEvents[2]
	body := &pieceBody{pieces: [][]byte{[]byte(head + "data: {\"par")}, err: errors.New("connection reset")}
	r := New(body, c.opts(Chat))
	chunks, err := readChunks(r, 1<<20)
	if err != nil {
		t.Fatalf("the upstream's error leaked to the caller: %v", err)
	}
	mustSame(t, bytes.Join(chunks, nil), head+cutEvent)
	// And it stays at EOF.
	if n, err := r.Read(make([]byte, 16)); n != 0 || err != io.EOF {
		t.Errorf("after the end: %d, %v", n, err)
	}
	if s := c.summary(t); s.End != EndedEarly {
		t.Errorf("End = %q", s.End)
	}
}

func TestUpstreamErrorEventForwardedAsIs(t *testing.T) {
	var c rec
	in := chatEvents[0] + chatEvents[2] + `data: {"error":{"code":500,"message":"x"}}` + "\n\n"
	got, err := readAll(New(whole(in), c.opts(Chat)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, in)
	if s := c.summary(t); s.End != UpstreamError || s.Events != 3 {
		t.Errorf("End = %q Events = %d", s.End, s.Events)
	}
	if c.cutCount() != 0 {
		t.Error("Cut called after an upstream error event")
	}
}

func TestFinishReasonThenEOFWithoutDone(t *testing.T) {
	var c rec
	in := strings.Join(chatEvents[:8], "") // up to the finish_reason chunk, no usage, no [DONE]
	got, err := readAll(New(whole(in), c.opts(Chat)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, in)
	s := c.summary(t)
	if s.End != Done || s.FinishReason != "stop" {
		t.Errorf("End = %q FinishReason = %q", s.End, s.FinishReason)
	}
	// With no usage chunk the counts come from timings: prompt_n + cache_n, cache_n, predicted_n.
	if s.PromptTokens != 100 || s.CachedTokens != 64 || s.CompletionTokens != 19 {
		t.Errorf("tokens = %d/%d/%d, want 100/64/19", s.PromptTokens, s.CachedTokens, s.CompletionTokens)
	}
	if c.cutCount() != 0 {
		t.Error("Cut called")
	}
}

func TestUnknownDialectUntouched(t *testing.T) {
	var c rec
	in := chatEvents[0] + chatEvents[2] + "data: {\"half"
	got, err := readAll(New(whole(in), c.opts(Unknown)))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, in)
	if c.cutCount() != 0 {
		t.Error("Cut called for an Unknown stream")
	}
	if c.summary(t).Dialect != Unknown {
		t.Error("Summary dialect")
	}
}

func TestUnknownDialectKeepsReadError(t *testing.T) {
	var c rec
	boom := errors.New("boom")
	body := &pieceBody{pieces: [][]byte{[]byte(chatEvents[0])}, err: boom}
	got, err := readAll(New(body, c.opts(Unknown)))
	mustSame(t, got, chatEvents[0])
	if !errors.Is(err, boom) {
		t.Errorf("err = %v, want the upstream's", err)
	}
	if c.cutCount() != 0 {
		t.Error("Cut called")
	}
}

func TestOversizeEventFallsBackToRaw(t *testing.T) {
	small := "data: {}\n\n"
	big := "data: " + strings.Repeat("x", 300) // never gets its blank line
	for name, mk := range map[string]func(string) *pieceBody{
		"whole":   whole,
		"perbyte": perByte,
	} {
		t.Run(name, func(t *testing.T) {
			var c rec
			o := c.opts(Chat)
			o.MaxEvent = 64
			in := small + big
			got, err := readAll(New(mk(in), o))
			if err != nil {
				t.Fatal(err)
			}
			mustSame(t, got, in)
			if c.cutCount() != 0 {
				t.Error("Cut called after the fall back to raw")
			}
			if strings.Contains(string(got), `{"error":`) {
				t.Error("terminal event added to a raw stream")
			}
			c.summary(t)
		})
	}
}

func TestOversizeThenMoreBytesStaysRaw(t *testing.T) {
	var c rec
	o := c.opts(Chat)
	o.MaxEvent = 64
	in := "data: " + strings.Repeat("y", 100) + "\n\ndata: [DONE]\n\n"
	// Split so the long event is still open when the limit trips.
	b := &pieceBody{pieces: [][]byte{[]byte(in[:90]), []byte(in[90:])}}
	got, err := readAll(New(b, o))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, in)
	if c.cutCount() != 0 {
		t.Error("Cut called")
	}
}

func TestOnEndOnceOnEOF(t *testing.T) {
	var c rec
	r := New(whole(chatStream), c.opts(Chat))
	readAll(r)
	r.Read(make([]byte, 8)) // reading past the end must not fire again
	r.Close()
	r.Close()
	if c.endCount() != 1 {
		t.Fatalf("OnEnd fired %d times", c.endCount())
	}
}

func TestOnEndOnceOnReadError(t *testing.T) {
	var c rec
	body := &pieceBody{pieces: [][]byte{[]byte(chatEvents[0])}, err: errors.New("boom")}
	r := New(body, c.opts(Chat))
	readAll(r)
	r.Close()
	if c.endCount() != 1 {
		t.Fatalf("OnEnd fired %d times", c.endCount())
	}
	if c.summary(t).End != EndedEarly {
		t.Error("End after a read error should be ended_early")
	}
}

func TestOnEndOnCloseBeforeEnd(t *testing.T) {
	var c rec
	body := &pieceBody{pieces: [][]byte{[]byte(chatEvents[0] + chatEvents[2])}}
	r := New(body, c.opts(Chat))
	buf := make([]byte, 1<<20)
	if _, err := r.Read(buf); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	r.Close()
	s := c.summary(t)
	if s.End != ClientGone {
		t.Errorf("End = %q, want client_gone", s.End)
	}
	if s.Events != 2 {
		t.Errorf("Events = %d", s.Events)
	}
	if c.cutCount() != 0 {
		t.Error("Cut called on Close")
	}
}

func TestOnEndOnCloseAfterEnd(t *testing.T) {
	var c rec
	r := New(whole(chatStream), c.opts(Chat))
	readAll(r)
	if c.endCount() != 1 {
		t.Fatalf("not fired at the end: %d", c.endCount())
	}
	r.Close()
	if c.endCount() != 1 {
		t.Fatalf("Close fired again: %d", c.endCount())
	}
	if c.summary(t).End != Done {
		t.Error("End changed")
	}
}

func TestCloseAfterCompleteButBeforeEOFIsDone(t *testing.T) {
	var c rec
	// The caller got [DONE] and closed without reading to EOF.
	r := New(whole(chatStream), c.opts(Chat))
	buf := make([]byte, 1<<20)
	if _, err := r.Read(buf); err != nil {
		t.Fatal(err)
	}
	r.Close()
	if s := c.summary(t); s.End != Done {
		t.Errorf("End = %q, want done", s.End)
	}
}

func TestCloseReachesUpstream(t *testing.T) {
	body := whole(chatStream)
	r := New(body, Options{Dialect: Chat})
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if body.closes() != 1 {
		t.Fatalf("upstream closed %d times", body.closes())
	}
}

func TestGoneMeansNoTerminalEvent(t *testing.T) {
	head := chatEvents[0] + chatEvents[2]
	boom := errors.New("context canceled")
	cases := []struct {
		name    string
		body    *pieceBody
		wantErr error
	}{
		{"eof", &pieceBody{pieces: [][]byte{[]byte(head + "data: {\"pa")}}, nil},
		{"error", &pieceBody{pieces: [][]byte{[]byte(head + "data: {\"pa")}, err: boom}, boom},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var c rec
			o := c.opts(Chat)
			o.Gone = func() bool { return true }
			got, err := readAll(New(tc.body, o))
			if !errors.Is(err, tc.wantErr) && !(tc.wantErr == nil && err == nil) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
			if strings.Contains(string(got), `{"error":`) {
				t.Fatalf("terminal event sent to a client that left: %q", got)
			}
			mustSame(t, got, head)
			if c.cutCount() != 0 {
				t.Error("Cut called")
			}
			if s := c.summary(t); s.End != ClientGone {
				t.Errorf("End = %q, want client_gone", s.End)
			}
		})
	}
}

func TestGoneAfterCompleteStillDone(t *testing.T) {
	var c rec
	o := c.opts(Chat)
	o.Gone = func() bool { return true }
	got, err := readAll(New(whole(chatStream), o))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, chatStream)
	if s := c.summary(t); s.End != Done {
		t.Errorf("End = %q", s.End)
	}
}

func TestOnEventOncePerEvent(t *testing.T) {
	var c rec
	r := New(perByte(chatStream), c.opts(Chat))
	readAll(r)
	if c.eventCount() != len(chatEvents) {
		t.Fatalf("OnEvent called %d times, want %d", c.eventCount(), len(chatEvents))
	}
	if s := c.summary(t); s.Events != len(chatEvents) {
		t.Fatalf("Events = %d", s.Events)
	}
}

func TestSmallCallerBuffer(t *testing.T) {
	for _, size := range []int{1, 2, 7, 50} {
		var c rec
		cut := chatEvents[0] + chatEvents[1] // ends early: the terminal event is handed out in pieces too
		r := New(whole(cut), c.opts(Chat))
		chunks, err := readChunks(r, size)
		if err != nil {
			t.Fatal(err)
		}
		for _, ch := range chunks {
			if len(ch) > size {
				t.Fatalf("size %d: Read returned %d bytes", size, len(ch))
			}
		}
		mustSame(t, bytes.Join(chunks, nil), cut+cutEvent)
	}
	var c rec
	chunks, err := readChunks(New(whole(chatStream), c.opts(Chat)), 13)
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, bytes.Join(chunks, nil), chatStream)
	checkChatSummary(t, c.summary(t))
}

func TestCRLFSplitBetweenCRAndLF(t *testing.T) {
	in := "data: {\"a\":1}\r\n\r\ndata: [DONE]\r\n\r\n"
	for k := 0; k <= len(in); k++ {
		var c rec
		got, err := readAll(New(splitAt(in, k), c.opts(Chat)))
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != in {
			t.Fatalf("k=%d: got %q", k, got)
		}
		if s := c.summary(t); s.End != Done || s.Events != 2 {
			t.Fatalf("k=%d: End = %q Events = %d", k, s.End, s.Events)
		}
	}
}

func TestNilHooksAreFine(t *testing.T) {
	got, err := readAll(New(whole(chatEvents[0]), Options{Dialect: Chat}))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, chatEvents[0]) // Cut is nil: nothing is sent
}

func TestCutReturningNilSendsNothing(t *testing.T) {
	o := Options{Dialect: Chat, Cut: func() []byte { return nil }}
	got, err := readAll(New(whole(chatEvents[0]+"data: {"), o))
	if err != nil {
		t.Fatal(err)
	}
	mustSame(t, got, chatEvents[0])
}

// pipeBody closes the read side of a pipe, which unblocks a Read waiting on it.
type pipeBody struct {
	pr *io.PipeReader
	pw *io.PipeWriter
}

func (b *pipeBody) Read(p []byte) (int, error) { return b.pr.Read(p) }
func (b *pipeBody) Close() error               { b.pw.Close(); return b.pr.Close() }

func TestCloseFromAnotherGoroutineWhileReadBlocked(t *testing.T) {
	ctx := timeoutCtx(t)
	pr, pw := io.Pipe()
	var c rec
	r := New(&pipeBody{pr: pr, pw: pw}, c.opts(Chat))

	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		close(started)
		_, err := readAll(r)
		done <- err
	}()
	<-started
	r.Close()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Read ended with %v", err)
		}
	case <-ctx.Done():
		t.Fatal("Read stayed blocked after Close")
	}
	// Whichever side got there first, the summary fired once.
	if c.endCount() != 1 {
		t.Fatalf("OnEnd fired %d times", c.endCount())
	}
}

func TestConcurrentCloseDuringEvents(t *testing.T) {
	ctx := timeoutCtx(t)
	pr, pw := io.Pipe()
	var c rec
	r := New(&pipeBody{pr: pr, pw: pw}, c.opts(Chat))

	go func() {
		for _, e := range chatEvents {
			if _, err := pw.Write([]byte(e)); err != nil {
				return
			}
		}
		pw.Close()
	}()
	done := make(chan struct{})
	go func() {
		defer close(done)
		readAll(r)
	}()
	r.Close()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("Read stayed blocked after Close")
	}
	if c.endCount() != 1 {
		t.Fatalf("OnEnd fired %d times", c.endCount())
	}
}
