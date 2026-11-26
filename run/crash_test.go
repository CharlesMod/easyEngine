//go:build !windows

package run_test

import (
	"bufio"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/run"
)

func ctxOf(t *testing.T, line string) int {
	t.Helper()
	c, err := strconv.Atoi(testkit.Field(line, "c"))
	if err != nil {
		t.Fatalf("no c= in %q", line)
	}
	return c
}

func capped(e *testkit.Env, atMost int) bool {
	for _, c := range run.LoadState(e.Data).Book.Entries {
		if c.CapCtxTotal > 0 && c.CapCtxTotal <= atMost {
			return true
		}
	}
	return false
}

func writeCtl(t *testing.T, e *testkit.Env, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(e.Ctl, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// A crash with no memory error in its last words restarts the same shape after the first
// backoff, and the status says so.
func TestNonOOMCrashRestartsSameShape(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	e.Touch("crash")
	testkit.WaitFor(t, 10*time.Second, "restart after the crash", func() bool {
		return len(e.Starts()) >= 2 && s.Phase() == "ready"
	})
	lines := e.Starts()
	if ctxOf(t, lines[0]) != ctxOf(t, lines[1]) {
		t.Fatalf("a non-memory crash changed the shape: %v", lines)
	}
	st := s.Status()
	if st.Recovery == nil || st.Recovery.Crashes != 1 || !strings.Contains(st.Recovery.LastError, "Segmentation fault") {
		t.Fatalf("recovery = %+v", st.Recovery)
	}
	if capped(e, 1<<30) {
		t.Fatalf("a non-memory crash lowered the cap: %+v", run.LoadState(e.Data).Book)
	}
}

// A start that keeps failing retries with a growing wait, never in a tight loop, says how many
// times it failed, and recovers by itself once the cause goes; a stable run clears the record.
func TestFailedStartRetriesWithBackoff(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Touch("loadfail")
	cfg := e.Config()
	cfg.CrashBackoff = []time.Duration{50 * time.Millisecond, 150 * time.Millisecond, 400 * time.Millisecond}
	cfg.CrashQuick = 300 * time.Millisecond
	s := e.StartSup(cfg)
	testkit.WaitFor(t, 10*time.Second, "three failed starts", func() bool {
		st := s.Status()
		return st.Phase == "error" && st.Recovery != nil && st.Recovery.Failures >= 3
	})
	st := s.Status()
	if st.Recovery == nil || st.Recovery.Failures < 3 || !strings.Contains(st.Message, "in a row") {
		t.Fatalf("status = %s %q %+v", st.Phase, st.Message, st.Recovery)
	}
	// at the longest wait (400ms) the next second holds at most three more tries
	n := len(e.Starts())
	time.Sleep(time.Second)
	if more := len(e.Starts()) - n; more > 3 || more < 1 {
		t.Fatalf("%d tries in a second at a 400ms backoff", more)
	}
	e.Remove("loadfail")
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	testkit.WaitFor(t, 5*time.Second, "the record cleared after a stable run", func() bool { return s.Status().Recovery == nil })
}

// The model not fitting beside other apps is not given up on: the engine looks again slowly and
// starts by itself once they let go of the card.
func TestNoFitWaitsForTheCard(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.SetOthers(15000) // a game holds almost the whole card
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "error", 10*time.Second)
	st := s.Status()
	if !strings.Contains(st.Message, "does not fit") || st.Recovery == nil {
		t.Fatalf("status = %q %+v", st.Message, st.Recovery)
	}
	if len(e.Starts()) != 0 {
		t.Fatalf("started a server that cannot fit: %v", e.Starts())
	}
	e.SetOthers(0)
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
}

// The same shape crashing soon after it starts, again and again, is read as a memory problem:
// one step down after CrashSteps, and the engine stays below the shape that kept dying.
func TestQuickCrashesStepDownOnce(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	c0 := ctxOf(t, e.Starts()[0])
	writeCtl(t, e, "crashloop", strconv.Itoa(c0-1)) // every start at c0 dies soon after it is up
	testkit.WaitFor(t, 20*time.Second, "a smaller shape that stays up", func() bool {
		st := s.Status()
		return st.Phase == "ready" && st.Shape != nil && st.Shape.Slots*st.Shape.CtxPerSlot < c0
	})
	lines := e.Starts()
	atC0 := 0
	for _, l := range lines {
		if ctxOf(t, l) == c0 {
			atC0++
		}
	}
	if atC0 != 3 { // the first start plus two restarts, then the step down
		t.Fatalf("%d starts at %d before the step down: %v", atC0, c0, lines)
	}
	st := s.Status()
	if st.Recovery == nil || !st.Recovery.SuspectMemory || st.Recovery.CeilingCtx != c0 {
		t.Fatalf("recovery = %+v", st.Recovery)
	}
	time.Sleep(300 * time.Millisecond)
	if n := len(e.Starts()); n != len(lines) {
		t.Fatalf("restarted again after the step down: %v", e.Starts())
	}
}

// A shrink whose smaller shape runs out of memory while loading steps down from that shape,
// records it, and never goes back to the shape that no longer fit.
func TestFailedShrinkStepsDownFromThere(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	c0 := ctxOf(t, e.Starts()[0])
	e.SetUsed(16000)
	e.SetOthers(8000)
	// the first smaller shape will not load: anything above c0*3/4 runs out of memory
	writeCtl(t, e, "oom_above", strconv.Itoa(c0*3/4))
	testkit.WaitFor(t, 20*time.Second, "ready below the failed shrink", func() bool {
		st := s.Status()
		return len(e.Starts()) >= 3 && st.Phase == "ready"
	})
	lines := e.Starts()
	for _, l := range lines[1:] {
		if ctxOf(t, l) >= c0 {
			t.Fatalf("went back to the shape that no longer fit: %v", lines)
		}
	}
	last := ctxOf(t, lines[len(lines)-1])
	if last > c0*3/4 {
		t.Fatalf("serving above the shape known to run out of memory: %v", lines)
	}
	testkit.WaitFor(t, 2*time.Second, "the failed shrink recorded", func() bool { return capped(e, c0-1) })
}

// An out-of-memory the server survives (a failed request) is not a crash the watchdog sees:
// one burst is let go, a second within OOMHold shrinks the model and records the cap.
func TestSurvivedOOMShrinks(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	c0 := ctxOf(t, e.Starts()[0])
	e.Touch("oomlog")
	time.Sleep(400 * time.Millisecond)
	if len(e.Starts()) != 1 || s.Phase() != "ready" {
		t.Fatalf("one survived out-of-memory reshaped: %v %s", e.Starts(), s.Phase())
	}
	e.Touch("oomlog")
	testkit.WaitFor(t, 10*time.Second, "shrink after the second", func() bool {
		return len(e.Starts()) >= 2 && s.Phase() == "ready"
	})
	if c1 := ctxOf(t, e.Starts()[1]); c1 >= c0 {
		t.Fatalf("not smaller: %v", e.Starts())
	}
	testkit.WaitFor(t, 2*time.Second, "cap recorded", func() bool { return capped(e, c0-1) })
	logs, _ := s.Logs("llama-server", 200)
	if !strings.Contains(logs, "cudaMalloc failed") {
		t.Fatalf("the run-time output is not in llama-server.log:\n%s", logs)
	}
}

// Another app takes memory in the middle of a reply: the reply finishes whole, new requests
// wait meanwhile, and the model shrinks right after.
func TestMidReplyGrabFinishesTheReply(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.Llama["FAKELLAMA_STREAM_DELAY"] = "250ms" // five chunks: a reply of about 1.25 s
	s := ready(t, e)
	url := s.Status().ServerURL
	type result struct {
		chunks int
		done   bool
		err    error
	}
	got := make(chan result, 1)
	go func() {
		resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true,"messages":[]}`))
		if err != nil {
			got <- result{err: err}
			return
		}
		defer resp.Body.Close()
		var r result
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			l := sc.Text()
			if l == "data: [DONE]" {
				r.done = true
			} else if strings.HasPrefix(l, "data: ") {
				r.chunks++
			}
		}
		got <- r
	}()
	time.Sleep(100 * time.Millisecond)
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitFor(t, 5*time.Second, "draining", func() bool { return s.Phase() == "resizing" })
	if len(e.Starts()) != 1 {
		t.Fatalf("restarted under a reply: %v", e.Starts())
	}
	r := <-got
	if r.err != nil || r.chunks != 5 || !r.done {
		t.Fatalf("the reply was cut: %+v", r)
	}
	testkit.WaitFor(t, 10*time.Second, "shrink after the reply", func() bool { return len(e.Starts()) >= 2 })
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	if ctxOf(t, e.Starts()[1]) >= ctxOf(t, e.Starts()[0]) {
		t.Fatalf("not smaller: %v", e.Starts())
	}
}

// A small held shortfall during a reply never touches it: the shrink waits for a pause.
func TestSmallShortfallWaitsForIdle(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.DrainMinMiB = 100000 // every shortfall counts as small
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	e.Touch("busy")
	e.SetUsed(16000)
	e.SetOthers(8000)
	time.Sleep(500 * time.Millisecond)
	if len(e.Starts()) != 1 || s.Phase() != "ready" {
		t.Fatalf("acted on a small shortfall mid-reply: %v %s", e.Starts(), s.Phase())
	}
	e.Remove("busy")
	testkit.WaitFor(t, 10*time.Second, "shrink once idle", func() bool { return len(e.Starts()) >= 2 })
}

// A reply that outlasts DrainMax does not hold the shrink forever.
func TestDrainGivesUpAfterDrainMax(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.DrainMax = 300 * time.Millisecond
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	e.Touch("busy") // a slot that never finishes
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitFor(t, 10*time.Second, "shrink after DrainMax", func() bool { return len(e.Starts()) >= 2 })
	e.Remove("busy")
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	found := false
	for _, m := range s.Status().Errors {
		if strings.Contains(m, "replies still running") {
			found = true
		}
	}
	if !found {
		t.Fatalf("errors = %v", s.Status().Errors)
	}
}

// A shape shrunk away from because another app took room did not fail: once the app lets go the
// model grows back to it.
func TestGrowsBackAfterTheAppLetsGo(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.GrowIdle = 200 * time.Millisecond
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	c0 := ctxOf(t, e.Starts()[0])
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitFor(t, 10*time.Second, "shrink", func() bool { return len(e.Starts()) >= 2 })
	e.SetUsed(0)
	e.SetOthers(0)
	testkit.WaitFor(t, 15*time.Second, "grown back to the first shape", func() bool {
		st := s.Status()
		return st.Phase == "ready" && st.Shape != nil && st.Shape.Slots*st.Shape.CtxPerSlot == c0
	})
}
