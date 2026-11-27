//go:build !windows

package run_test

import (
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// After the hang watchdog fires once and /health answers, a server that then hangs while the
// same request is still in flight is restarted: the watchdog is made again after a fire.
// (Adopted from the review's probe; it waits on the engine's own words, not on a clock.)
func TestHangWatchdogArmsAgainAfterAFire(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.HTTP = nil
	cfg.HangSilence = 300 * time.Millisecond
	s := e.StartSup(withShortHTTP(cfg))
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	pid, _ := strconv.Atoi(testkit.Field(e.Starts()[0], "pid"))
	from := testkit.Mark(t, s, "before")
	done := s.Begin()
	defer done()
	// the watchdog fires while the server is healthy: it answers /health and is left alone
	answered := func() bool {
		for _, ev := range testkit.Said(t, s, from, 0, "mind.hang") {
			if strings.Contains(string(ev.Body), `"answered":true`) {
				return true
			}
		}
		return false
	}
	testkit.WaitFor(t, 15*time.Second, "a healthy server asked /health once", answered)
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("healthy server restarted: %d", n)
	}
	if err := syscall.Kill(pid, syscall.SIGSTOP); err != nil {
		t.Fatal(err)
	}
	defer syscall.Kill(pid, syscall.SIGCONT)
	testkit.WaitFor(t, 15*time.Second, "the hung server restarted after the second fire", func() bool { return len(e.Starts()) >= 2 })
}

// A start that failed for want of room (a game holds the card, no yield) starts by itself when the
// game lets go: the room watch stays armed while that wait stands. The room's level is the
// mechanism under test here, so this test sets its period.
func TestStartWaitingForRoomStartsWhenTheRoomFrees(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	e.SetOthers(16000) // another app holds nearly all of the card
	cfg := e.Config()
	cfg.RoomEvery = 50 * time.Millisecond // watchdog for silence under test: memory gives no event
	s := e.StartSup(cfg)
	testkit.WaitFor(t, 15*time.Second, "the start failing for want of room", func() bool {
		st := s.Status()
		return st.Phase == "error" && st.Recovery != nil && st.Recovery.WaitingFor != ""
	})
	if n := len(e.Starts()); n != 0 {
		t.Fatalf("a model server started with no room: %v", e.Starts())
	}
	e.SetOthers(0) // the game exits
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

// A build that never prints a listening line is still found ready: a quiet stretch after its last
// line asks /health once (ReadyQuiet), long before LoadTimeout.
func TestReadyWithNoListeningLine(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_READY_LINES"] = "none"
	cfg := e.Config()
	cfg.LoadTimeout = time.Minute
	cfg.ReadyQuiet = 200 * time.Millisecond
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

// A build whose last line comes before /health turns 200 is still found ready: nothing more is
// said, and the quiet stretch asks once more.
func TestReadyWhenHealthTurnsAfterTheLastLine(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_READY_LINES"] = "early"
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "400ms"
	cfg := e.Config()
	cfg.LoadTimeout = time.Minute
	cfg.ReadyQuiet = 600 * time.Millisecond // past the health delay: the one ask finds it ready
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}
