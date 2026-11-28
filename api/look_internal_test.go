package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The decode bound: maxLookDecoding looks decode at once, all callers; the next is refused at
// once, and a look whose images are built frees its share.
func TestLookGateDecodeBound(t *testing.T) {
	var g lookGate
	if !g.enter("loopback", true) || !g.enter("loopback", true) {
		t.Fatal("the first two looks are admitted")
	}
	if g.enter("loopback", true) {
		t.Fatal("a third look decoding at once is admitted")
	}
	if g.enter("client-1", false) {
		t.Fatal("a remote look over the decode bound is admitted")
	}
	g.decoded()
	if !g.enter("client-1", false) {
		t.Fatal("a share freed by decoded() is not given")
	}
	if g.enter("loopback", true) {
		t.Fatal("the freed share was taken twice")
	}
	g.decoded()
	g.decoded()
	g.release("client-1", false)
	if g.decoding != 0 || len(g.remote) != 0 {
		t.Fatalf("gate not empty: decoding %d remote %v", g.decoding, g.remote)
	}
}

// A caller from another machine has one look at a time (from its release on, not from decoded),
// each identity on its own; a local look is bounded by the decode bound only.
func TestLookGateOneRemoteLookPerIdentity(t *testing.T) {
	var g lookGate
	if !g.enter("client-1", false) {
		t.Fatal("first remote look refused")
	}
	g.decoded() // its images are built and it is now in flight
	if g.enter("client-1", false) {
		t.Fatal("a second look of the same remote identity is admitted while the first runs")
	}
	if !g.enter("client-2", false) {
		t.Fatal("a look of another identity is refused")
	}
	g.decoded()
	for i := 0; i < 2; i++ { // locals are not counted by identity
		if !g.enter("loopback", true) {
			t.Fatalf("local look %d refused", i)
		}
		g.decoded()
	}
	g.release("client-1", false)
	if !g.enter("client-1", false) {
		t.Fatal("client-1 is still refused after its look was released")
	}
}

// What parseLook accepts at the edges, and the defaults.
func TestParseLookEdges(t *testing.T) {
	q, ce := parseLook([]byte(`{"images":[{"data":"AAAA"}],"question":"q"}`))
	if ce != nil || q.detail != "high" || q.maxTok != defLookTokens || q.idSlot != -1 || q.stream {
		t.Fatalf("defaults: %+v %v", q, ce)
	}
	q, ce = parseLook([]byte(`{"images":[{"data":"AAAA","label":"` + strings.Repeat("l", maxLookLabel) + `"}],"question":"` +
		strings.Repeat("q", maxLookQuestion) + `","detail":"max","max_tokens":4096,"id_slot":1023,"stream":true}`))
	if ce != nil || q.detail != "max" || q.maxTok != 4096 || q.idSlot != 1023 || !q.stream || len(q.images[0].label) != maxLookLabel {
		t.Fatalf("the largest accepted values: %+v %v", q.maxTok, ce)
	}
	if _, ce = parseLook([]byte(`{"images":[{"data":"AAAA"}],"question":"q","id_slot":0,"max_tokens":1}`)); ce != nil {
		t.Fatalf("id_slot 0 and max_tokens 1: %v", ce)
	}
	// nesting: 8 deep passes the depth check (and then fails as an unknown shape), 9 does not
	_, ce = parseLook([]byte(`{"question":"q","images":[[[[[[[1]]]]]]]}`))
	if ce == nil || strings.Contains(ce.Msg, "nests") {
		t.Fatalf("8 deep: %v", ce)
	}
	_, ce = parseLook([]byte(`{"question":"q","images":[[[[[[[[1]]]]]]]]}`))
	if ce == nil || !strings.Contains(ce.Msg, "nests") {
		t.Fatalf("9 deep: %v", ce)
	}
}

// The look answer's figures: usage tokens as given, ms.prefill = round(prompt_ms),
// ms.total = round(prompt_ms + predicted_ms), tokens.image always -1; a figure the model server's
// answer does not carry is -1.
func TestLookAnswerFigures(t *testing.T) {
	figures := func(js string) lookAnswer {
		t.Helper()
		var u upstreamFigures
		if err := json.Unmarshal([]byte(js), &u); err != nil {
			t.Fatal(err)
		}
		la := newLookAnswer()
		la.take(u)
		return la
	}
	la := figures(`{"usage":{"prompt_tokens":150,"completion_tokens":7},"timings":{"prompt_ms":12.5,"predicted_ms":100.4}}`)
	if la.Tokens != (lookTotals{Image: -1, Prompt: 150, Completion: 7}) || la.MS != (lookMS{Prefill: 13, Total: 113}) {
		t.Fatalf("figures %+v", la)
	}
	la = figures(`{"usage":{"prompt_tokens":3.4,"completion_tokens":0},"timings":{"prompt_ms":0.4,"predicted_ms":0.4}}`)
	if la.Tokens.Prompt != 3 || la.Tokens.Completion != 0 || la.MS.Prefill != 0 || la.MS.Total != 1 {
		t.Fatalf("rounding %+v", la)
	}
	for name, js := range map[string]string{
		"nothing":     `{}`,
		"empty usage": `{"usage":{},"timings":{}}`,
		"negative":    `{"usage":{"prompt_tokens":-5,"completion_tokens":-1},"timings":{"prompt_ms":-1,"predicted_ms":-1}}`,
		"too big":     `{"usage":{"prompt_tokens":1e30},"timings":{"prompt_ms":1e30}}`,
	} {
		la = figures(js)
		if la.Tokens != (lookTotals{-1, -1, -1}) || la.MS != (lookMS{-1, -1}) {
			t.Errorf("%s: %+v", name, la)
		}
	}
	la = figures(`{"usage":{"prompt_tokens":9},"timings":{"prompt_ms":20}}`) // half the figures
	if la.Tokens.Prompt != 9 || la.Tokens.Completion != -1 || la.MS.Prefill != 20 || la.MS.Total != -1 {
		t.Fatalf("half: %+v", la)
	}
	// the last that carries a figure wins, a chunk without one leaves it
	la = newLookAnswer()
	for _, js := range []string{`{"usage":{"prompt_tokens":1,"completion_tokens":1}}`, `{"choices":[]}`, `{"usage":{"prompt_tokens":2,"completion_tokens":3}}`} {
		var u upstreamFigures
		json.Unmarshal([]byte(js), &u)
		la.take(u)
	}
	if la.Tokens.Prompt != 2 || la.Tokens.Completion != 3 {
		t.Fatalf("last wins: %+v", la)
	}
}

// A non-streamed answer through the recorder: a 200 chat completion becomes the look answer, any
// other status goes to the caller as it is, and a 200 that is not a chat completion is a 502.
func TestLookRecorderAnswers(t *testing.T) {
	run := func(status int, ct, body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		rec := &lookRecorder{w: w, h: http.Header{}}
		rec.Header().Set("Content-Type", ct)
		rec.WriteHeader(status)
		rec.Write([]byte(body))
		rec.answer()
		return w
	}
	w := run(200, "application/json", `{"choices":[{"message":{"content":"hi"}}]}`)
	var la lookAnswer
	json.Unmarshal(w.Body.Bytes(), &la)
	if w.Code != 200 || la.Text != "hi" || la.Tokens != (lookTotals{-1, -1, -1}) || la.MS != (lookMS{-1, -1}) {
		t.Fatalf("no usage, no timings: %d %s", w.Code, w.Body)
	}
	const upstreamErr = `{"error":{"code":500,"message":"image input is not supported"}}`
	w = run(500, "application/json", upstreamErr)
	if w.Code != 500 || w.Body.String() != upstreamErr || w.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("an error answer: %d %q", w.Code, w.Body)
	}
	w = run(200, "application/json", `not json`)
	if w.Code != 502 || !strings.Contains(w.Body.String(), "bad_gateway") {
		t.Fatalf("a bad answer: %d %s", w.Code, w.Body)
	}
	w = run(200, "application/json", `{"choices":[]}`)
	if w.Code != 502 {
		t.Fatalf("no choices: %d %s", w.Code, w.Body)
	}
}

// A streamed look: an error answer passes through unchanged; a stream without [DONE] gets no
// look.result; the look.result is one event before [DONE], with the deltas joined and the last
// figures; events split over writes are read whole.
func TestLookStream(t *testing.T) {
	stream := func(status int, ct string, writes ...string) (*httptest.ResponseRecorder, *lookStream) {
		w := httptest.NewRecorder()
		l := &lookStream{w: w}
		l.Header().Set("Content-Type", ct)
		l.WriteHeader(status)
		for _, p := range writes {
			l.Write([]byte(p))
		}
		l.end()
		return w, l
	}
	const errBody = `{"error":{"code":500,"message":"x"}}`
	w, _ := stream(500, "application/json", errBody)
	if w.Code != 500 || w.Body.String() != errBody {
		t.Fatalf("an error answer: %d %q", w.Code, w.Body)
	}
	w, _ = stream(503, "text/event-stream", "data: [DONE]\n\n")
	if strings.Contains(w.Body.String(), lookObject) {
		t.Fatalf("a non-200 event stream got a look.result: %q", w.Body)
	}
	const ct = "text/event-stream"
	w, _ = stream(200, ct, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n", "data: {\"error\":{\"message\":\"gone\"}}\n\n")
	if strings.Contains(w.Body.String(), lookObject) || !strings.Contains(w.Body.String(), "gone") {
		t.Fatalf("a stream without [DONE]: %q", w.Body)
	}
	c1 := "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n"
	c2 := "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n"
	c3 := "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2},\"timings\":{\"prompt_ms\":1.6,\"predicted_ms\":2.2}}\n\n"
	all := c1 + c2 + c3
	// the same stream cut into writes at every byte: split events are put together
	for cut := 1; cut < len(all); cut += 7 {
		w, _ = stream(200, ct, all[:cut], all[cut:], "data: [DONE]\n\n")
		out := w.Body.String()
		want := all + "data: {\"object\":\"look.result\",\"text\":\"hello\",\"tokens\":{\"image\":-1,\"prompt\":10,\"completion\":2},\"ms\":{\"prefill\":2,\"total\":4}}\n\ndata: [DONE]\n\n"
		if out != want {
			t.Fatalf("cut at %d:\n%q\nwant\n%q", cut, out, want)
		}
	}
	// no figures at all: -1
	w, _ = stream(200, ct, c1, "data: [DONE]\n\n")
	if !strings.Contains(w.Body.String(), `"tokens":{"image":-1,"prompt":-1,"completion":-1},"ms":{"prefill":-1,"total":-1}`) {
		t.Fatalf("no figures: %q", w.Body)
	}
	if n := strings.Count(w.Body.String(), lookObject); n != 1 {
		t.Fatalf("%d look.result events", n)
	}
}
