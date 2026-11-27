package api_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// reply is one proxied request's answer, sent when it comes.
type reply struct {
	code int
	body string
	hdr  http.Header
	err  error
}

// send makes a request in the background; its answer comes on the channel.
func send(ctx context.Context, url, body string) <-chan reply {
	out := make(chan reply, 1)
	go func() {
		req, _ := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			out <- reply{err: err}
			return
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		out <- reply{code: resp.StatusCode, body: string(b), hdr: resp.Header}
	}()
	return out
}

// await is the answer, or a failure when none comes.
func await(t *testing.T, c <-chan reply, what string) reply {
	t.Helper()
	select {
	case r := <-c:
		return r
	case <-time.After(15 * time.Second): // watchdog for silence: the engine never answered
		t.Fatalf("no answer: %s", what)
	}
	return reply{}
}

// neverReady is an engine whose model server prints its ready lines and never answers /health:
// a request is held for as long as the test wants.
func neverReady(t *testing.T) (*testkit.Env, *run.Supervisor, string) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_READY_LINES"] = "early"
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "1h"
	cfg := e.Config()
	cfg.ReadyQuiet, cfg.LoadTimeout = time.Hour, time.Hour
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "loading", 15*time.Second)
	return e, s, srv.URL
}

// A request during a load is held and answered when the model is up; while held it is counted
// in the door fact and status, never in flight (no hang watchdog, no drain waits for it).
func TestHeldWhileLoadingIsAnswered(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "1500ms"
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "loading", 10*time.Second)
	got := send(t.Context(), srv.URL+"/v1/chat/completions", `{}`)
	testkit.WaitFor(t, 10*time.Second, "the request held", func() bool { return s.Held() == 1 })
	if n := s.InFlight(); n != 0 {
		t.Fatalf("a held request is in flight: %d", n)
	}
	if st := s.Status(); st.Held != 1 || st.API != run.APIVersion {
		t.Fatalf("status held %d api %d", st.Held, st.API)
	}
	if f, ok := s.Hub().Fact("door", ""); !ok || !strings.Contains(string(f.Body), `"held":1`) {
		t.Fatalf("door fact %s", f.Body)
	}
	r := await(t, got, "held request")
	if r.err != nil || r.code != 200 || !strings.Contains(r.body, "ready") {
		t.Fatalf("held request: %d %s %v", r.code, r.body, r.err)
	}
	if s.Held() != 0 {
		t.Fatalf("held after the answer: %d", s.Held())
	}
}

// A request held while loading is a word: the load asks /health once on it. The server here
// prints its ready lines first and turns healthy later with no line after it; ReadyQuiet and
// LoadTimeout are an hour, so only the request can find it ready.
func TestHeldRequestIsAWordForTheLoad(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_READY_LINES"] = "early"
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "300ms"
	cfg := e.Config()
	cfg.ReadyQuiet, cfg.LoadTimeout = time.Hour, time.Hour
	healthy := filepath.Join(e.Ctl, "healthy")
	e.WatchFile(healthy)
	s, srv := serve(t, e, cfg)
	testkit.WaitFor(t, 10*time.Second, "the fake answers /health", func() bool {
		_, err := os.Stat(healthy)
		return err == nil
	})
	if p := s.Phase(); p != "loading" {
		t.Fatalf("phase %s before the request: the load found ready by itself", p)
	}
	r := await(t, send(t.Context(), srv.URL+"/v1/chat/completions", `{}`), "the request that is a word")
	if r.err != nil || r.code != 200 {
		t.Fatalf("held request: %d %s %v", r.code, r.body, r.err)
	}
}

// A hold that ends in a person's Stop is answered at once with the JSON 503 of that state.
func TestHoldEndsInAStop(t *testing.T) {
	_, s, url := neverReady(t)
	got := send(t.Context(), url+"/v1/chat/completions", `{}`)
	testkit.WaitFor(t, 10*time.Second, "the request held", func() bool { return s.Held() == 1 })
	s.Stop()
	r := await(t, got, "the held request after a stop")
	var j struct {
		Error struct {
			Code    int
			Type    string
			Message string
		}
	}
	if r.code != 503 || r.hdr.Get("Retry-After") != "2" || json.Unmarshal([]byte(r.body), &j) != nil || j.Error.Code != 503 || j.Error.Type != "unavailable_error" {
		t.Fatalf("after stop: %d %q %s", r.code, r.hdr.Get("Retry-After"), r.body)
	}
	// stopped by a person: a new request is answered at once, not held, and wakes nothing
	from := testkit.Mark(t, s, "a request while stopped")
	r = await(t, send(t.Context(), url+"/v1/chat/completions", `{}`), "a request while stopped")
	if r.code != 503 || s.Held() != 0 || s.Phase() != "sleeping" {
		t.Fatalf("while stopped: %d held %d phase %s", r.code, s.Held(), s.Phase())
	}
	if ph := testkit.Phases(t, s, from, 0); len(ph) != 0 {
		t.Fatalf("a request woke a stopped model: %v", ph)
	}
}

// A client that goes away while held is no longer counted.
func TestHeldClientGoes(t *testing.T) {
	_, s, url := neverReady(t)
	ctx, cancel := context.WithCancel(t.Context())
	got := send(ctx, url+"/v1/chat/completions", `{}`)
	testkit.WaitFor(t, 10*time.Second, "the request held", func() bool { return s.Held() == 1 })
	cancel()
	await(t, got, "the client's own end")
	testkit.WaitFor(t, 10*time.Second, "the hold let go", func() bool { return s.Held() == 0 })
	if s.InFlight() != 0 {
		t.Fatalf("in flight %d", s.InFlight())
	}
}

// A sleeping model (not stopped by a person) is woken by a request, which is held until it is up.
func TestRequestWakesASleepingModel(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.StartSleeping = true
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "sleeping", 10*time.Second)
	r := await(t, send(t.Context(), srv.URL+"/v1/chat/completions", `{}`), "the waking request")
	if r.code != 200 || !strings.Contains(r.body, "ready") {
		t.Fatalf("waking request: %d %s", r.code, r.body)
	}
	// Sleep says its phase before the server stops: no request is proxied to a stopping server
	from := testkit.Mark(t, s, "sleep")
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	var phase, gone uint64
	for _, ev := range testkit.Said(t, s, from, 0, "") {
		if ev.Name == "phase" && phase == 0 && strings.Contains(string(ev.Body), `"sleeping"`) {
			phase = ev.Seq
		}
		if ev.Name == "mind.shape" && ev.Gone && gone == 0 {
			gone = ev.Seq
		}
	}
	if phase == 0 || gone == 0 || phase > gone {
		t.Fatalf("sleeping said at %d, the shape gone at %d", phase, gone)
	}
}

// A streamed request held through a load gets its stream opened with the waiting word, then the
// reply in the same stream.
func TestHeldStreamOpensAndFollows(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "3s"
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "loading", 10*time.Second)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("Content-Type") != "text/event-stream" || resp.Header.Get("X-Request-Id") == "" {
		t.Fatalf("held stream: %d %v", resp.StatusCode, resp.Header)
	}
	b, _ := io.ReadAll(resp.Body)
	body := string(b)
	if !strings.HasPrefix(body, ": easy waiting\n\n") || !strings.Contains(body, "data: ") || !strings.Contains(body, "[DONE]") {
		t.Fatalf("held stream body: %q", body)
	}
}

// A streamed request whose hold ends in a person's stop after the stream was opened ends with
// the terminal event "unavailable".
func TestHeldStreamEndsWithTheTerminalEvent(t *testing.T) {
	_, s, url := neverReady(t)
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	line, _ := br.ReadString('\n')
	if resp.StatusCode != 200 || line != ": easy waiting\n" {
		t.Fatalf("held stream opened: %d %q", resp.StatusCode, line)
	}
	s.Stop()
	rest, _ := io.ReadAll(br)
	want := `data: {"error":{"code":"unavailable","type":"unavailable_error","message":"Loading model","status":503,"request_id":"` + resp.Header.Get("X-Request-Id") + `"}}`
	if !strings.Contains(string(rest), want) || strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("after stop: %q, want %s", rest, want)
	}
}
