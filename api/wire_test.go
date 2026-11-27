package api_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/yield"
)

// The wire: facts with life and seq (the engine fact says api 1), the events stream from that
// seq, 409 for another life, and the book.
func TestTheWire(t *testing.T) {
	_, s, srv := readyEnv(t)
	// The room's first held reading comes a moment after ready (two reads agree): let it be
	// history, so that the first event after the facts is the mark.
	testkit.WaitFor(t, 10*time.Second, "the room's held reading", func() bool {
		return len(testkit.Said(t, s, 0, 0, "room.held")) > 0
	})
	c, body, _ := do(t, "GET", srv.URL+"/engine/v1/facts", "", nil)
	var f event.Standing
	if c != 200 || json.Unmarshal([]byte(body), &f) != nil || f.Life == "" || f.Seq == 0 {
		t.Fatalf("facts %d %s", c, body)
	}
	var api struct {
		API int `json:"api"`
	}
	for _, e := range f.Facts {
		if e.Name == "engine" {
			json.Unmarshal(e.Body, &api)
		}
	}
	if api.API != run.APIVersion || run.APIVersion != 1 {
		t.Fatalf("engine fact api %d", api.API)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", srv.URL+"/engine/v1/events?life="+f.Life+"&since="+seqStr(f.Seq), nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("events %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	testkit.Mark(t, s, "on the wire")
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	var e event.Event
	if err != nil || json.Unmarshal([]byte(line), &e) != nil || e.Name != "mark" || e.Seq != f.Seq+1 || e.Life != f.Life {
		t.Fatalf("first event %q %v", line, err)
	}
	if c, body, _ := do(t, "GET", srv.URL+"/engine/v1/events?life=other&since=0", "", nil); c != 409 || !strings.Contains(body, f.Life) {
		t.Fatalf("another life: %d %s", c, body)
	}
	if c, body, _ := do(t, "GET", srv.URL+"/engine/v1/book", "", nil); c != 200 || !strings.Contains(body, `"name":"door"`) {
		t.Fatalf("book %d %s", c, body)
	}
}

func seqStr(n uint64) string { b, _ := json.Marshal(n); return string(b) }

// A streamed reply's end is said with what the streamer read, paired with its begin by the
// request id; its decode rate is status tok_s.
func TestRequestEndCarriesTheStreamersSummary(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_TIMINGS"] = "1"
	e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms"
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	from := testkit.Mark(t, s, "a streamed request")
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	req.Header.Set("X-Request-Id", "rid-7")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	testkit.WaitFor(t, 10*time.Second, "the request's end", func() bool { return s.InFlight() == 0 && len(testkit.Said(t, s, from, 0, "request")) >= 2 })
	var begin, end map[string]any
	for _, ev := range testkit.Said(t, s, from, 0, "request") {
		var b map[string]any
		json.Unmarshal(ev.Body, &b)
		switch b["what"] {
		case "begin":
			begin = b
		case "end":
			end = b
		}
	}
	if begin["id"] != "rid-7" || end["id"] != "rid-7" {
		t.Fatalf("begin %v end %v", begin, end)
	}
	if end["end"] != "done" || end["finish_reason"] != "stop" || end["tok_s"] != float64(50) || end["slot"] != float64(0) || end["completion_tokens"] != float64(5) {
		t.Fatalf("end %v", end)
	}
	if st := s.Status(); st.TokS == nil || *st.TokS != 50 {
		t.Fatalf("status tok_s %v", st.TokS)
	}
}

// A request that arrives while the yield drains (a reply still in flight, the server not stopped
// yet) is answered yielded at once: not held, not proxied.
func TestRequestDuringTheYieldsDrainIsYielded(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_CHAT_DELAY"] = "3s"
	g := &gameOS{}
	cfg := e.Config()
	cfg.Yield = &run.YieldConfig{OS: func() yield.OS { return g }, Every: 50 * time.Millisecond}
	cfg.Yield.Policy.Grace = 100 * time.Millisecond
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	slow := send(t.Context(), srv.URL+"/v1/chat/completions", `{}`)
	testkit.WaitFor(t, 10*time.Second, "the slow reply in flight", func() bool { return s.InFlight() == 1 })
	g.on.Store(true)
	testkit.WaitFor(t, 10*time.Second, "the yield draining", func() bool { y := s.Status().Yield; return y != nil && y.Draining })
	c, body, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, nil)
	if c != 503 || h.Get("Retry-After") != "10" || !strings.Contains(body, `"code":"yielded"`) || s.Held() != 0 {
		t.Fatalf("during the drain: %d %s held %d", c, body, s.Held())
	}
	await(t, slow, "the slow reply")
	testkit.WaitFor(t, 10*time.Second, "yielded and unloaded", func() bool { st := s.Status(); return st.Phase == "yielded" && st.Shape == nil })
}

// No secret reaches the stream: the external API key and the token are in no fact, reading or
// event, and not in the book.
func TestTheStreamKeepsNoSecrets(t *testing.T) {
	testkit.SkipWindows(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, `{}`) }))
	defer up.Close()
	_, s, srv := readyEnv(t)
	const key = "sk-very-secret-4242"
	set := `{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":"m","api_key":"` + key + `"}}`
	if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", set, csrf); c != 200 {
		t.Fatalf("PUT %d %s", c, b)
	}
	testkit.WaitPhase(t, s, "external", 10*time.Second)
	for _, path := range []string{"facts", "book", "status", "settings"} {
		c, body, _ := do(t, "GET", srv.URL+"/engine/v1/"+path, "", nil)
		if c != 200 || strings.Contains(body, key) {
			t.Fatalf("%s %d carries the key: %s", path, c, body)
		}
	}
	evs, _ := s.Hub().Since(0, "")
	for _, ev := range evs {
		if strings.Contains(string(ev.Body), key) {
			t.Fatalf("event %s carries the key", ev.Name)
		}
	}
}
