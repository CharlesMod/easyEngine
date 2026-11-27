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

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
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

// A crash with no memory error in its last words restarts the same shape at once, and the
// status says so.
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

// A start that keeps failing is tried again once at once, then waits for a word that could
// change the outcome (no cadence, never a tight loop), says how many times it failed and what it
// waits for, and starts on the first word once the cause is gone; a stable run clears the record.
func TestFailedStartRetriesOnceThenWaitsForAWord(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Touch("loadfail")
	cfg := e.Config()
	cfg.CrashQuick = 300 * time.Millisecond
	s := e.StartSup(cfg)
	testkit.WaitFor(t, 10*time.Second, "two failed starts", func() bool {
		st := s.Status()
		return st.Phase == "error" && st.Recovery != nil && st.Recovery.Failures >= 2
	})
	st := s.Status()
	if st.Recovery.Failures != 2 || st.Recovery.WaitingFor == "" || st.Recovery.RetryInS != 0 || !strings.Contains(st.Message, "waiting for") {
		t.Fatalf("status = %s %q %+v", st.Phase, st.Message, st.Recovery)
	}
	if n := len(e.Starts()); n != 2 {
		t.Fatalf("%d starts, want the first and one retry at once: %v", n, e.Starts())
	}
	e.Remove("loadfail")
	s.Route() // a request at the door is the word
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	if n := len(e.Starts()); n != 3 {
		t.Fatalf("%d starts, want exactly one more on the word: %v", n, e.Starts())
	}
	testkit.WaitFor(t, 5*time.Second, "the record cleared after a stable run", func() bool { return s.Status().Recovery == nil })
}

// The model not fitting beside other apps is not given up on: the engine waits for a word that
// the room may have changed and starts once they let go of the card.
func TestNoFitWaitsForTheCard(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.SetOthers(15000) // a game holds almost the whole card
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "error", 10*time.Second)
	st := s.Status()
	if !strings.Contains(st.Message, "does not fit") || st.Recovery == nil || st.Recovery.WaitingFor == "" {
		t.Fatalf("status = %q %+v", st.Message, st.Recovery)
	}
	if len(e.Starts()) != 0 {
		t.Fatalf("started a server that cannot fit: %v", e.Starts())
	}
	e.SetOthers(0)
	s.Nudge("the game ended") // the word (in a real run: the game's end, by host.Exits)
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
}

// The same shape crashing soon after it starts, again and again, is read as a memory problem:
// one restart at once, one step down after CrashSteps (2), and the engine stays below the shape
// that kept dying.
func TestQuickCrashesStepDownOnce(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.CrashQuick = 400 * time.Millisecond // a crashloop server dies 150 ms after it is ready
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
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
	if atC0 != 2 { // the first start plus one restart at once, then the step down
		t.Fatalf("%d starts at %d before the step down: %v", atC0, c0, lines)
	}
	st := s.Status()
	if st.Recovery == nil || !st.Recovery.SuspectMemory || st.Recovery.CeilingCtx != c0 {
		t.Fatalf("recovery = %+v", st.Recovery)
	}
	// the record clears only after a run of CrashQuick with no crash: the smaller shape held
	testkit.WaitFor(t, 5*time.Second, "a stable run", func() bool { return s.Status().Recovery == nil })
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
	// the smaller shape the room asks for will not load: anything above 30000 runs out of memory
	const above = 30000
	writeCtl(t, e, "oom_above", strconv.Itoa(above))
	testkit.WaitFor(t, 20*time.Second, "ready below the failed shrink", func() bool {
		st := s.Status()
		return len(e.Starts()) >= 3 && st.Phase == "ready"
	})
	lines := e.Starts()
	if ctxOf(t, lines[1]) <= above {
		t.Fatalf("the shrink target already fits below %d; the test needs a target above it: %v", above, lines)
	}
	for _, l := range lines[1:] {
		if ctxOf(t, l) >= c0 {
			t.Fatalf("went back to the shape that no longer fit: %v", lines)
		}
	}
	if last := ctxOf(t, lines[len(lines)-1]); last > above {
		t.Fatalf("serving above the shape known to run out of memory: %v", lines)
	}
	testkit.WaitFor(t, 2*time.Second, "the failed shrink recorded", func() bool { return capped(e, c0-1) })
}

// oomSaid counts the out-of-memory lines the server survived, as the hub said them, and how many
// of them were held.
func oomSaid(t *testing.T, s *run.Supervisor) (n, held int) {
	for _, ev := range testkit.Said(t, s, 0, 0, "mind.oom") {
		n++
		if strings.Contains(string(ev.Body), `"held":true`) {
			held++
		}
	}
	return n, held
}

// An out-of-memory the server survives (a failed request) is not a crash: the lines of one
// request are let go, lines in a second request within OOMHold shrink the model (once the
// request in flight ends) and record the cap.
func TestSurvivedOOMShrinks(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	c0 := ctxOf(t, e.Starts()[0])
	done := s.Begin()
	e.Touch("oomlog") // two matching lines, in this one request
	testkit.WaitFor(t, 10*time.Second, "the lines heard", func() bool { n, _ := oomSaid(t, s); return n >= 2 })
	done()
	if n, held := oomSaid(t, s); n != 2 || held != 0 {
		t.Fatalf("%d lines, %d held, want 2 lines of one request and none held", n, held)
	}
	m := testkit.Mark(t, s, "after one request's lines")
	if ph := testkit.Phases(t, s, 0, m); len(e.Starts()) != 1 || ph[len(ph)-1] != "ready" {
		t.Fatalf("one survived out-of-memory reshaped: %v %v", e.Starts(), ph)
	}
	done = s.Begin()
	e.Touch("oomlog")
	testkit.WaitFor(t, 10*time.Second, "a held run", func() bool { _, held := oomSaid(t, s); return held == 1 })
	testkit.WaitPhase(t, s, "resizing", 10*time.Second) // the request in flight finishes first
	if len(e.Starts()) != 1 {
		t.Fatalf("restarted under a request: %v", e.Starts())
	}
	done()
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
	first := make(chan struct{})
	end := s.Begin() // the request goes through the door, as every request does
	go func() {
		defer end()
		resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(`{"stream":true,"messages":[]}`))
		if err != nil {
			close(first)
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
				if r.chunks++; r.chunks == 1 {
					close(first)
				}
			}
		}
		got <- r
	}()
	<-first // the reply has begun
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
	done := s.Begin() // a reply in flight
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitFor(t, 10*time.Second, "the new room held", func() bool { r, ok := s.HeldReserve(); return ok && r >= 8000 })
	s.React() // the engine has looked at the held room
	if len(e.Starts()) != 1 || s.Phase() != "ready" {
		t.Fatalf("acted on a small shortfall mid-reply: %v %s", e.Starts(), s.Phase())
	}
	done()
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
	done := s.Begin() // a request that never finishes
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitFor(t, 10*time.Second, "shrink after DrainMax", func() bool { return len(e.Starts()) >= 2 })
	done()
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

// Another app leaves less room than any shape needs: the model goes to the smallest shape in one
// restart, not one step per reading.
func TestNoRoomAtAllShrinksOnce(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	e.SetUsed(16300)
	e.SetOthers(12000) // budget well under the 5000 MiB the model needs at any ctx
	testkit.WaitFor(t, 10*time.Second, "shrink", func() bool { return len(e.Starts()) >= 2 && s.Phase() == "ready" })
	testkit.WaitFor(t, 10*time.Second, "the room held again", func() bool { r, ok := s.HeldReserve(); return ok && r >= 12000 })
	s.React() // the engine has looked at the room the smallest shape runs in
	if n := len(e.Starts()); n != 2 {
		t.Fatalf("%d starts, want one restart: %v", n, e.Starts())
	}
}
