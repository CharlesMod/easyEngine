//go:build !windows

package run_test

import (
	"errors"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// readyWith is ready with a tweak of the config (the idle seams).
func readyWith(t *testing.T, e *testkit.Env, tweak func(*run.Config)) *run.Supervisor {
	t.Helper()
	e.InstallFake()
	cfg := e.Config()
	tweak(&cfg)
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return s
}

// A request that comes after SleepIdle's decision is held (and wakes the model again), never
// forwarded to the server the sleep is about to stop.
func TestSleepIdleHoldsARequestAfterItsDecision(t *testing.T) {
	e := testkit.New(t)
	var s *run.Supervisor
	routes := make(chan run.Route, 1)
	s = readyWith(t, e, func(c *run.Config) {
		once := false
		c.IdleDecided = func() {
			if !once { // the first sleep's decision: a request arrives now
				once = true
				routes <- s.Route()
			}
		}
	})
	if err := s.SleepIdle(); err != nil {
		t.Fatal(err)
	}
	r := <-routes
	if r.Kind == "local" || !r.Hold {
		t.Fatalf("a request after the decision was routed %+v: it would go to a server being stopped", r)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second) // the held request woke the model again
	// the held request woke the model: one more start (a port another program took meanwhile
	// can make it two: the engine's own retry of a bind clash, not this test's matter)
	if n := len(e.Starts()); n < 2 {
		t.Fatalf("starts %v, want the first and the wake", e.Starts())
	}
}

// A request that begins while SleepIdle waits for the operation lock (a reshape or a swap holds
// it) makes it decline: the model stays, the request runs on the server it began on.
func TestSleepIdleDeclinesARequestBegunWhileItWaited(t *testing.T) {
	e := testkit.New(t)
	waiting := make(chan struct{}, 1)
	s := readyWith(t, e, func(c *run.Config) {
		c.IdleBeforeOp = func() { waiting <- struct{}{} }
	})
	unlock := s.HoldOp()
	got := make(chan error, 1)
	go func() { got <- s.SleepIdle() }()
	<-waiting
	rt, end := s.Enter("late")
	if rt.Kind != "local" || end == nil {
		t.Fatalf("route %+v", rt)
	}
	unlock()
	if err := <-got; !errors.Is(err, run.ErrIdleBusy) {
		t.Fatalf("SleepIdle with a request in flight: %v", err)
	}
	if p := s.Phase(); p != "ready" {
		t.Fatalf("phase %q", p)
	}
	end(nil)
	if err := s.SleepIdle(); err != nil { // the door empty again: now it sleeps
		t.Fatal(err)
	}
}

// A quiet that began before the phase left ready (a reshape) and came back is not the quiet
// that counts: SleepIdleSince declines; a new quiet begins.
func TestSleepIdleDeclinesAfterThePhaseMoved(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	seq := s.RestSeq()
	s.SayPhase("resizing", "Resizing")
	if s.IdleSleepable() {
		t.Fatal("resizing is not at rest")
	}
	s.SayPhase("ready", "Ready")
	if err := s.SleepIdleSince(seq); !errors.Is(err, run.ErrIdleBusy) {
		t.Fatalf("a quiet from before the reshape: %v", err)
	}
	if err := s.SleepIdleSince(s.RestSeq()); err != nil {
		t.Fatal(err)
	}
}

// During a swap the engine is not at rest; a swap that ends while the model sleeps for being
// idle keeps the sleep's reason and words (and a client's attach can still wake it).
func TestIdleSleepAndTheSwap(t *testing.T) {
	st := newStage(t)
	sup := st.start(t)
	sup.Install("target", "")
	waitStatus(t, sup, "starter ready", func(x run.Status) bool { return x.Phase == "ready" && x.Model == "starter" })
	done := sup.Begin() // keeps the swap draining
	st.release()
	waitStatus(t, sup, "swapping", func(x run.Status) bool { return x.ModelStage == "swapping" })
	if sup.IdleSleepable() {
		t.Fatal("at rest during a swap")
	}
	done()
	waitStatus(t, sup, "swap done", func(x run.Status) bool { return x.Phase == "ready" && x.Model == "target" })

	// a second stage: the idle sleep comes first, the swap ends while asleep
	st2 := newStage(t)
	sup2 := st2.start(t)
	sup2.Install("target", "")
	waitStatus(t, sup2, "starter ready", func(x run.Status) bool { return x.Phase == "ready" && x.Model == "starter" })
	if err := sup2.SleepIdle(); err != nil {
		t.Fatal(err)
	}
	st2.release()
	x := waitStatus(t, sup2, "the upgrade done while asleep", func(x run.Status) bool {
		return x.Upgrade != nil && x.Upgrade.State == "done"
	})
	if x.Phase != "sleeping" || x.SleepReason != "idle" || x.Message != run.IdleMessage || sup2.StoppedByPerson() {
		t.Fatalf("after the swap's end: phase %q reason %q message %q stopped %v", x.Phase, x.SleepReason, x.Message, sup2.StoppedByPerson())
	}
}
