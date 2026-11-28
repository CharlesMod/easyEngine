//go:build !windows

package app_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

// The idle rule (docs/ENGINE.md "Idle sleep") in serve and in the app: what keeps the model
// loaded, what does not, the named moment and its settings. Every wait is on what the engine
// says (its event stream), never on a clock.

// startEngine runs the engine on an installed fake model, in serve (resident false) or as the
// app, and gives the test kit's env with it (starts.log, the room).
func startEngine(t *testing.T, resident bool, parents []int, tweak func(*app.Options)) (*residentRun, *testkit.Env) {
	t.Helper()
	e := testkit.New(t)
	e.InstallFake()
	o := app.Options{
		DataDir: e.Data, LingerSet: true, ParentPIDs: parents, Version: "t", Port: 47620, Profile: prof(),
		Configure: func(c *run.Config) {
			ec := e.Config()
			ec.Parents, ec.Yield, ec.Group, ec.StartSleeping = c.Parents, c.Yield, c.Group, c.StartSleeping
			*c = ec
		},
	}
	if tweak != nil {
		tweak(&o)
	}
	if resident {
		return startResident(t, o), e
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &residentRun{dir: o.DataDir, res: make(chan app.Result, 1), stop: cancel}
	ported := make(chan int, 1)
	o.Started = func(p int) { ported <- p }
	go func() {
		res, err := app.Serve(ctx, o)
		if err != nil {
			t.Errorf("serve: %v", err)
		}
		r.res <- res
	}()
	select {
	case r.port = <-ported:
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("engine did not start")
	}
	t.Cleanup(func() {
		cancel()
		select {
		case <-r.res:
		case <-time.After(20 * time.Second): // watchdog for silence
		}
	})
	return r, e
}

// put sends PUT settings (loopback: no token) and wants 200.
func (r *residentRun) put(t *testing.T, body string) {
	t.Helper()
	req, _ := http.NewRequest("PUT", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/engine/v1/settings", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("PUT settings %s: %d", body, resp.StatusCode)
	}
}

// postControl sends POST /engine/v1/<what> and answers the code.
func (r *residentRun) postControl(t *testing.T, what string) int {
	t.Helper()
	req, _ := http.NewRequest("POST", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/engine/v1/"+what, nil)
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

type idleFact struct {
	Armed      bool   `json:"armed"`
	Why        string `json:"why"`
	DelayMS    int64  `json:"delay_ms"`
	IdleSleep  bool   `json:"idle_sleep"`
	IdleSleepS int    `json:"idle_sleep_s"`
}

// armed says e is the idle fact armed for why, after delay.
func armed(why string, delay time.Duration) func(event.Event) bool {
	return func(e event.Event) bool {
		var f idleFact
		body(e, &f)
		return e.Name == "idle" && f.Armed && f.Why == why && f.DelayMS == delay.Milliseconds()
	}
}

func anyArmed(e event.Event) bool {
	var f idleFact
	body(e, &f)
	return e.Name == "idle" && f.Armed
}

// sleptFor says e is the phase fact sleeping with this reason.
// idleNow is the idle fact as it stands.
func (r *residentRun) idleNow(t *testing.T) idleFact {
	t.Helper()
	var st event.Standing
	r.get(t, "/engine/v1/facts", &st)
	for _, e := range st.Facts {
		if e.Name == "idle" {
			var f idleFact
			body(e, &f)
			return f
		}
	}
	return idleFact{}
}

// awaitArmed waits until the idle moment stands armed for why, after delay.
func (r *residentRun) awaitArmed(t *testing.T, why string, delay time.Duration) {
	t.Helper()
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("idle fact now %+v, phase %q", r.idleNow(t), r.status(t).Phase)
		}
	})
	r.await(t, "the idle moment armed: "+why, func() bool {
		f := r.idleNow(t)
		return f.Armed && f.Why == why && f.DelayMS == delay.Milliseconds()
	})
}

func sleptFor(reason string) func(event.Event) bool {
	return func(e event.Event) bool {
		var b struct {
			Phase       string
			SleepReason string `json:"sleep_reason"`
		}
		body(e, &b)
		return e.Name == "phase" && b.Phase == "sleeping" && b.SleepReason == reason
	}
}

func isReady(e event.Event) bool { return phaseIs(e, "ready") }

// fakeGone waits until the fake model server last started is no process at all.
func fakeGone(t *testing.T, r *residentRun, e *testkit.Env) {
	t.Helper()
	l := e.Starts()
	if len(l) == 0 {
		t.Fatal("no start")
	}
	pid, err := strconv.Atoi(testkit.Field(l[len(l)-1], "pid"))
	if err != nil {
		t.Fatal(err)
	}
	r.await(t, "the model server's process gone", func() bool {
		return errors.Is(syscall.Kill(pid, 0), syscall.ESRCH)
	})
}

type idleStatus struct {
	Phase       string `json:"phase"`
	Message     string `json:"message"`
	SleepReason string `json:"sleep_reason"`
	IdleSleep   bool   `json:"idle_sleep"`
	IdleSleepS  int    `json:"idle_sleep_s"`
	Parents     []struct {
		Pid   int  `json:"pid"`
		Alive bool `json:"alive"`
	} `json:"parents"`
}

// serve with a --parent-pid parent and nothing else: the model loads at the start, sleeps after
// idle_sleep (the parent is not a client) and its process is gone; the engine stays and answers,
// a POST wake loads it again, and the engine ends with its parent as before.
func TestServeParentKeepsTheEngineNotTheModel(t *testing.T) {
	par := newSleeper(t)
	const quiet = 300 * time.Millisecond
	r, e := startEngine(t, false, []int{par.pid}, idleAfter(quiet))
	s := r.follow(t)
	evs := s.until(t, "the model ready, then asleep for being idle", func(evs []event.Event) bool {
		return indexOf(evs, indexOf(evs, 0, isReady)+1, sleptFor("idle")) > 0
	})
	ready := indexOf(evs, 0, isReady)
	slept := evs[indexOf(evs, ready+1, sleptFor("idle"))]
	if ready >= 0 { // the stream may open after ready was said
		if d := slept.At.Sub(evs[ready].At); d < quiet {
			t.Fatalf("slept %v after ready, the quiet is %v", d, quiet)
		}
	}
	fakeGone(t, r, e)
	var st idleStatus
	r.get(t, "/engine/v1/status", &st)
	if st.Phase != "sleeping" || st.SleepReason != "idle" || st.Message != run.IdleMessage || !st.IdleSleep || st.IdleSleepS != settings.DefaultIdleSleepS {
		t.Fatalf("status %+v", st)
	}
	if len(st.Parents) != 1 || st.Parents[0].Pid != par.pid || !st.Parents[0].Alive {
		t.Fatalf("parents %+v", st.Parents)
	}
	// POST wake loads it again; idle again, it sleeps again
	from := len(s.events())
	if code := r.postControl(t, "wake"); code != 202 {
		t.Fatalf("wake: %d", code)
	}
	s.until(t, "woken, then asleep again", func(evs []event.Event) bool {
		return indexOf(evs, indexOf(evs, from, isReady)+1, sleptFor("idle")) > 0
	})
	select {
	case <-r.res:
		t.Fatal("the engine ended while its parent lives")
	default:
	}
	par.kill()
	select {
	case res := <-r.res: // the engine ends with its parent (linger 0), as before
		r.res <- res // for the cleanup, which waits for it too
	case <-time.After(20 * time.Second): // watchdog for silence
		t.Fatal("the engine did not end with its parent")
	}
}

// serve: a settings change while the moment is armed arms it again (idle_sleep_s 600 -> 0 sleeps
// at once); an attachment by pid wakes the model and keeps it loaded however long it is silent
// (idle_sleep_s 0); when it leaves, linger_s (0) applies, not idle_sleep_s (an hour by then).
func TestServeAttachmentKeepsTheModelThenLingerAfterItLeaves(t *testing.T) {
	par := newSleeper(t) // keeps the engine (an attach by pid is a parent of serve's life too)
	r, _ := startEngine(t, false, []int{par.pid}, nil)
	s := r.follow(t)
	r.awaitArmed(t, "idle_sleep/load", settings.DefaultIdleSleepS*time.Second)
	r.put(t, `{"idle_sleep_s": 0}`)
	s.until(t, "asleep at once with the new setting", func(evs []event.Event) bool {
		return indexOf(evs, 0, armed("idle_sleep/load", 0)) >= 0 && indexOf(evs, 0, sleptFor("idle")) >= 0
	})
	from := len(s.events())
	cl := newSleeper(t)
	if err := app.PostAttach(app.EngineTarget(r.dir, "127.0.0.1", r.port), cl.pid); err != nil {
		t.Fatal(err)
	}
	s.until(t, "the attach woke the model", func(evs []event.Event) bool { return indexOf(evs, from, isReady) >= 0 })
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	r.put(t, `{"idle_sleep_s": 3600}`) // now only linger_s (0) can explain a sleep at once
	left := len(s.events())
	cl.kill()
	evs := s.until(t, "asleep once the attachment left", func(evs []event.Event) bool {
		return indexOf(evs, left, sleptFor("idle")) >= 0
	})
	if i := indexOf(evs, from, isSleeping); i < left {
		t.Fatalf("the model slept (event %d) while the attachment stood (it left at %d)", i, left)
	}
	if i := indexOf(evs, from, anyArmed); i < left {
		t.Fatalf("the idle moment was armed (event %d) while the attachment stood", i)
	}
	if indexOf(evs, left, armed("linger/local_leave", 0)) < 0 {
		t.Fatal("the moment after the attachment left was not the linger")
	}
}

// The app with idle_sleep off: a request-only caller's quiet arms nothing and the model stays;
// turned on again (idle_sleep_s 0), the model sleeps at once.
func TestIdleSleepOffKeepsTheModel(t *testing.T) {
	r, _ := startEngine(t, true, nil, nil)
	s := r.follow(t)
	r.put(t, `{"idle_sleep": false, "idle_sleep_s": 0}`)
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	r.put(t, `{"linger_s": 0}`) // a word to the rule: it looks again, still off
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	var st idleStatus
	r.get(t, "/engine/v1/status", &st)
	if st.Phase != "ready" || st.IdleSleep {
		t.Fatalf("status %+v", st)
	}
	on := len(s.events())
	r.put(t, `{"idle_sleep": true}`)
	evs := s.until(t, "asleep once idle sleep is on", func(evs []event.Event) bool {
		return indexOf(evs, on, sleptFor("idle")) >= 0
	})
	if i := indexOf(evs, 0, anyArmed); i < on {
		t.Fatalf("the idle moment was armed (event %d) while idle sleep was off", i)
	}
	if i := indexOf(evs, indexOf(evs, 0, isReady), isSleeping); i < on {
		t.Fatalf("the model slept (event %d) while idle sleep was off", i)
	}
}

// serve --no-idle-sleep: the setting cannot turn it on; the model stays.
func TestNoIdleSleepFlag(t *testing.T) {
	r, _ := startEngine(t, false, nil, func(o *app.Options) {
		o.NoIdleSleep, o.IdleSleep, o.IdleSleepSet = true, 0, true
	})
	s := r.follow(t)
	s.until(t, "ready", func(evs []event.Event) bool { return indexOf(evs, 0, isReady) >= 0 || r.status(t).Phase == "ready" })
	r.put(t, `{"idle_sleep": true, "idle_sleep_s": 0}`)
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	r.put(t, `{"linger_s": 1}`)
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	evs := s.events()
	if i := indexOf(evs, 0, anyArmed); i >= 0 {
		t.Fatalf("armed (event %d) under --no-idle-sleep", i)
	}
	if i := indexOf(evs, 0, isSleeping); i >= 0 {
		t.Fatalf("slept (event %d) under --no-idle-sleep", i)
	}
}

// An edit of settings.json while the moment is armed applies to it: armed for 600 s, the file
// says 0, the model sleeps.
func TestIdleSleepFileEditRearms(t *testing.T) {
	r, _ := startEngine(t, false, nil, nil)
	s := r.follow(t)
	r.awaitArmed(t, "idle_sleep/load", settings.DefaultIdleSleepS*time.Second)
	path := filepath.Join(r.dir, "settings.json")
	f, err := settings.Read(path)
	if err != nil {
		t.Fatal(err)
	}
	f.IdleSleepS = 0
	if err := settings.Write(path, f); err != nil {
		t.Fatal(err)
	}
	s.until(t, "asleep after the edit", func(evs []event.Event) bool {
		return indexOf(evs, 0, armed("idle_sleep/load", 0)) >= 0 && indexOf(evs, 0, sleptFor("idle")) >= 0
	})
}

// A full card at wake: the held request is answered 503 with the reason (never hung), the
// engine says error and starts no process; the next request is the word that tries again, and
// once the room is free it is served.
func TestServeWakeOnABusyCard(t *testing.T) {
	r, e := startEngine(t, false, nil, nil)
	s := r.follow(t)
	r.put(t, `{"idle_sleep_s": 0}`)
	s.until(t, "asleep for being idle", func(evs []event.Event) bool { return indexOf(evs, 0, sleptFor("idle")) >= 0 })
	r.put(t, `{"idle_sleep_s": 3600}`) // what wakes now stays up for the count below
	starts := len(e.Starts())
	e.SetOthers(16384)
	for i := 0; i < 2; i++ {
		if code := r.post(t); code != 503 {
			t.Fatalf("request %d on a full card: %d", i, code)
		}
	}
	r.await(t, "error, saying the card is busy", func() bool {
		var st idleStatus
		r.get(t, "/engine/v1/status", &st)
		return st.Phase == "error" && bytes.Contains([]byte(st.Message), []byte("other programs hold 16384 MiB"))
	})
	if n := len(e.Starts()); n != starts {
		t.Fatalf("a model server was started on a full card: %v", e.Starts())
	}
	e.SetOthers(0)
	// the next request is the word that tries again: it is answered at once (503: the engine
	// stood in error when it came, or 200: the retry was quick), and the try loads the model
	if code := r.post(t); code != 200 && code != 503 {
		t.Fatalf("request after the room freed: %d", code)
	}
	waitPhase(t, r, "ready")
	if code := r.post(t); code != 200 {
		t.Fatalf("request once ready: %d", code)
	}
	// one start, at the full shape: the room was read afresh at the wake. (A later shrink can
	// follow here: the test kit's room watch never re-reads on its own, RoomEvery is an hour, so
	// the watchdog may still hold the busy reading; the run package's test moves the room.)
	if l := e.Starts(); len(l) <= starts || testkit.Field(l[starts], "c") != testkit.Field(l[0], "c") {
		t.Fatalf("starts %v, want one more at the first shape", l)
	}
}

// fwd makes a caller another machine's, as a forwarder says (api.peerLocal: never local).
var fwd = map[string]string{"X-Forwarded-For": "10.9.8.7"}

// postFrom sends a chat completion with extra headers and answers the status code.
func (r *residentRun) postFrom(t *testing.T, hdr map[string]string) int {
	t.Helper()
	req, _ := http.NewRequest("POST", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

// holdFrom makes a held attachment with extra headers; the returned func ends its connection
// (the attachment's leave) and waits until the engine said it gone.
func (r *residentRun) holdFrom(t *testing.T, s *stream, hdr map[string]string) func() {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	req, _ := http.NewRequestWithContext(ctx, "POST", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/engine/v1/attach",
		bytes.NewReader([]byte(`{"hold":true,"name":"far"}`)))
	req.Header.Set("X-Easy-Engine", "1")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		t.Fatalf("held attach: %d", resp.StatusCode)
	}
	var first struct {
		Attachment string `json:"attachment"`
		Local      bool   `json:"local"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&first); err != nil || first.Attachment == "" {
		t.Fatalf("held attach answer: %v %+v", err, first)
	}
	if first.Local == (hdr != nil) {
		t.Fatalf("held attach local=%v with headers %v", first.Local, hdr)
	}
	from := len(s.events())
	return func() {
		cancel()
		resp.Body.Close()
		s.until(t, "the attachment gone", func(evs []event.Event) bool {
			return indexOf(evs, from, func(e event.Event) bool { return e.Name == "attachment" && e.Key == first.Attachment && e.Gone }) >= 0
		})
	}
}

// A remote attachment that was the last to leave, with linger_s 0: the model does not sleep at
// once, the moment is idle_sleep_s (600) for a remote leave.
func TestRemoteLeaveWaitsTheLongDelay(t *testing.T) {
	r, _ := startEngine(t, false, nil, nil)
	s := r.follow(t)
	waitPhase(t, r, "ready")
	leave := r.holdFrom(t, s, fwd)
	from := len(s.events())
	leave()
	r.awaitArmed(t, "idle_sleep/remote_leave", settings.DefaultIdleSleepS*time.Second)
	if i := indexOf(s.events(), from, isSleeping); i >= 0 {
		t.Fatalf("the model slept (event %d) after a remote leave with linger_s 0", i)
	}
	if st := r.status(t); st.Phase != "ready" {
		t.Fatalf("phase %q", st.Phase)
	}
}

// A local held attachment that leaves: linger_s 0 sleeps the model at once, as before.
func TestLocalLeaveSleepsAtOnce(t *testing.T) {
	r, _ := startEngine(t, false, nil, nil)
	s := r.follow(t)
	waitPhase(t, r, "ready")
	leave := r.holdFrom(t, s, nil)
	from := len(s.events())
	leave()
	s.until(t, "asleep at once after the local leave", func(evs []event.Event) bool {
		return indexOf(evs, from, armed("linger/local_leave", 0)) >= 0 && indexOf(evs, from, sleptFor("idle")) >= 0
	})
}

// A remote identity cycling attach and detach with nothing else going on: no model start and no
// stop within the delay (each leave arms the long one; the model stays as it was).
func TestRemoteAttachDetachCycleNeitherStartsNorStops(t *testing.T) {
	r, e := startEngine(t, false, nil, nil)
	s := r.follow(t)
	waitPhase(t, r, "ready")
	starts := len(e.Starts())
	from := len(s.events())
	for i := 0; i < 5; i++ {
		r.holdFrom(t, s, fwd)()
	}
	r.awaitArmed(t, "idle_sleep/remote_leave", settings.DefaultIdleSleepS*time.Second)
	evs := s.events()
	if i := indexOf(evs, from, isSleeping); i >= 0 {
		t.Fatalf("the model slept (event %d) in the cycle", i)
	}
	if n := len(e.Starts()); n != starts {
		t.Fatalf("the cycle started the model server: %v", e.Starts())
	}
}

// idle_sleep_s 0: requests from another machine still wait the floor, a remote leave too, while
// a local caller that only sends requests sleeps the model at once.
func TestRemoteFloorWhenIdleSleepIsZero(t *testing.T) {
	r, _ := startEngine(t, false, nil, nil)
	s := r.follow(t)
	r.put(t, `{"idle_sleep_s": 0}`)
	s.until(t, "asleep at once after the load", func(evs []event.Event) bool { return indexOf(evs, 0, sleptFor("idle")) >= 0 })
	from := len(s.events())
	if code := r.postFrom(t, fwd); code != 200 {
		t.Fatalf("remote request: %d", code)
	}
	r.awaitArmed(t, "remote_floor/remote_requests", run.RemoteIdleFloor)
	leave := r.holdFrom(t, s, fwd)
	leave()
	r.awaitArmed(t, "remote_floor/remote_leave", run.RemoteIdleFloor)
	if i := indexOf(s.events(), indexOf(s.events(), from, isReady), isSleeping); i >= 0 {
		t.Fatalf("the model slept (event %d) before the floor", i)
	}
	local := len(s.events())
	if code := r.post(t); code != 200 {
		t.Fatalf("local request: %d", code)
	}
	s.until(t, "asleep at once after the local request", func(evs []event.Event) bool {
		return indexOf(evs, local, armed("idle_sleep/local_requests", 0)) >= 0 && indexOf(evs, local, sleptFor("idle")) >= 0
	})
}

// A request the rule never sees in flight (refused at the door: an unknown X-Easy-Client) still
// restarts the quiet: when the moment comes it finds traffic since it was armed, arms again from
// there, and the model sleeps a whole quiet after that request, not at the first moment.
func TestUnseenRequestRestartsTheQuiet(t *testing.T) {
	const quiet = 400 * time.Millisecond
	r, _ := startEngine(t, true, nil, idleAfter(quiet))
	s := r.follow(t)
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	evs := s.until(t, "armed after the request", func(evs []event.Event) bool {
		return indexOf(evs, indexOf(evs, 0, isEnd)+1, armed("idle_sleep/local_requests", quiet)) > 0
	})
	first := indexOf(evs, indexOf(evs, 0, isEnd)+1, armed("idle_sleep/local_requests", quiet))
	sent := time.Now()
	if code := r.postFrom(t, map[string]string{"X-Easy-Client": "no-such-attachment"}); code < 400 {
		t.Fatalf("the unknown attachment's request was answered %d", code)
	}
	evs = s.until(t, "asleep", func(evs []event.Event) bool { return indexOf(evs, first, sleptFor("idle")) > 0 })
	slept := indexOf(evs, first, sleptFor("idle"))
	if again := indexOf(evs, first+1, armed("idle_sleep/local_requests", quiet)); again < 0 || again > slept {
		t.Fatalf("the moment was not armed again before the sleep (armed %d, slept %d)", again, slept)
	}
	if d := evs[slept].At.Sub(sent); d < quiet {
		t.Fatalf("slept %v after the last request, the quiet is %v", d, quiet)
	}
}
