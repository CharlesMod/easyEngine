//go:build !windows

package run_test

import (
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// lastPid is the pid of the fake server the last line of starts.log started.
func lastPid(t *testing.T, e *testkit.Env) int {
	t.Helper()
	l := e.Starts()
	if len(l) == 0 {
		t.Fatal("no starts")
	}
	pid, err := strconv.Atoi(testkit.Field(l[len(l)-1], "pid"))
	if err != nil {
		t.Fatalf("pid of %q: %v", l[len(l)-1], err)
	}
	return pid
}

// The idle sleep ends the model server; when others took part of the card meanwhile, the next
// request wakes the model at a smaller shape.
func TestIdleSleepWakesSmallerWhenOthersHoldPart(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	lines := e.Starts()
	if len(lines) != 1 {
		t.Fatalf("starts %v", lines)
	}
	c0, _ := strconv.Atoi(testkit.Field(lines[0], "c"))
	pid := lastPid(t, e)

	if err := s.SleepIdle(); err != nil {
		t.Fatal(err)
	}
	st := s.Status()
	if st.Phase != "sleeping" || st.SleepReason != "idle" || st.Message != run.IdleMessage {
		t.Fatalf("after SleepIdle: %q %q %q", st.Phase, st.SleepReason, st.Message)
	}
	testkit.WaitFor(t, 10*time.Second, "the fake server gone", func() bool {
		return syscall.Kill(pid, 0) == syscall.ESRCH
	})

	e.SetOthers(8000) // room for a smaller shape, not none
	if r := s.Route(); r.Kind != "unavailable" || !r.Hold {
		t.Fatalf("route asleep = %+v", r)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	lines = e.Starts()
	if len(lines) != 2 {
		t.Fatalf("starts %v, want one more", lines)
	}
	c1, _ := strconv.Atoi(testkit.Field(lines[1], "c"))
	if c1 >= c0 {
		t.Fatalf("not smaller: %v", lines)
	}
	st = s.Status()
	if st.SleepReason != "" {
		t.Fatalf("sleep_reason %q after the wake", st.SleepReason)
	}
	if st.Shape == nil || st.Shape.Slots*st.Shape.CtxPerSlot != c1 {
		t.Fatalf("shape %+v does not reflect c=%d", st.Shape, c1)
	}
}

// A wake from the idle sleep on a full card fails slowly: it says why, starts nothing, and each
// word (a request) is one attempt, never a loop; the room freeing starts it.
func TestIdleSleepWakeOnAFullCardErrorsThenStartsWhenTheRoomFrees(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	if err := s.SleepIdle(); err != nil {
		t.Fatal(err)
	}
	e.SetOthers(16384)
	s.Route() // the request that wakes it
	testkit.WaitPhase(t, s, "error", 15*time.Second)
	st := s.Status()
	if !strings.Contains(st.Message, "does not fit") || !strings.Contains(st.Message, "other programs hold 16384 MiB") {
		t.Fatalf("message %q", st.Message)
	}
	if st.Recovery == nil || st.Recovery.Failures < 1 {
		t.Fatalf("recovery %+v", st.Recovery)
	}
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("starts %v", e.Starts())
	}
	t.Logf("error message %q; recovery %+v", st.Message, *st.Recovery)
	if r := s.Route(); r.Kind != "unavailable" || r.Hold {
		t.Fatalf("route in error = %+v", r)
	}
	// that Route was a word too: wait until its attempt has counted, then count from there
	testkit.WaitFor(t, 10*time.Second, "the retry the request caused", func() bool {
		x := s.Status()
		return x.Phase == "error" && x.Recovery != nil && x.Recovery.Failures > st.Recovery.Failures
	})
	base := s.Status().Recovery.Failures
	for i := 1; i <= 2; i++ {
		s.Route()
		want := base + i
		testkit.WaitFor(t, 10*time.Second, "one more failed attempt", func() bool {
			x := s.Status()
			return x.Phase == "error" && x.Recovery != nil && x.Recovery.Failures >= want
		})
	}
	if f := s.Status().Recovery.Failures; f != base+2 {
		t.Fatalf("failures %d, want %d: a loop", f, base+2)
	}
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("a process started on a full card: %v", e.Starts())
	}
	t.Logf("recovery after two words %+v", *s.Status().Recovery)

	e.SetOthers(0)
	s.RoomMoved()
	s.React()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if n := len(e.Starts()); n != 2 {
		t.Fatalf("starts %v, want exactly one new", e.Starts())
	}
}

// The idle sleep keeps the conversation: the slots are saved at the sleep and laid back at the wake.
func TestIdleSleepKeepsTheConversation(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	want := savedTokens(4000)
	from := testkit.Mark(t, s, "before the idle sleep")
	if err := s.SleepIdle(); err != nil {
		t.Fatal(err)
	}
	if sv := whatsOf(contextsSince(t, s, from), "saved", 0); len(sv) != 1 || sv[0].Tokens != want {
		t.Fatalf("not saved at the idle sleep: %+v", contextsSince(t, s, from))
	}
	if c := s.Status().Contexts; c == nil || c.Slots != 1 {
		t.Fatalf("status contexts asleep = %+v", c)
	}
	if err := s.Wake(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if lb := whatsOf(contextsSince(t, s, from), "laid_back", 0); len(lb) != 1 || lb[0].Tokens != want {
		t.Fatalf("not laid back at the wake: %+v", contextsSince(t, s, from))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 4100); c != want {
		t.Fatalf("cache_n after the wake = %d, want %d", c, want)
	}
}
