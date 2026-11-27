//go:build !windows

package run_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

func TestMain(m *testing.M) {
	code := m.Run()
	testkit.Cleanup()
	os.Exit(code)
}

func ready(t *testing.T, e *testkit.Env) *run.Supervisor {
	t.Helper()
	e.InstallFake()
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return s
}

func TestStartToReady(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	st := s.Status()
	if st.Shape == nil || st.Shape.Slots != 3 || st.Shape.CtxPerSlot != 65536 || st.Shape.KVType != "q4_0" {
		t.Fatalf("shape = %+v", st.Shape)
	}
	if st.ServerURL == "" {
		t.Fatal("no server url")
	}
	lines := e.Starts()
	if len(lines) != 1 || testkit.Field(lines[0], "np") != "3" || testkit.Field(lines[0], "c") != "196608" {
		t.Fatalf("starts = %v", lines)
	}
	// calibration used the measured VRAM from the fake nvidia-smi (5500 + 0.05*196608 = 15330)
	testkit.WaitFor(t, 3*time.Second, "calibrated book", func() bool {
		for _, c := range run.LoadState(e.Data).Book.Entries {
			if c.FixedMiB > 4900 && c.FixedMiB < 5100 {
				return true
			}
		}
		return false
	})
	if st.Shape.VRAMMiB < 15000 || st.Shape.VRAMMiB > 15600 {
		t.Fatalf("measured vram = %d", st.Shape.VRAMMiB)
	}
	t.Logf("command line: %s", cmdLine(lines[0]))
}

func cmdLine(line string) string {
	i := strings.Index(line, "args=")
	return "llama-server " + line[i+5:]
}

func TestFlagDroppedWhenUnknown(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.Llama["FAKELLAMA_HIDE"] = "--cache-reuse"
	s := ready(t, e)
	if s.Phase() != "ready" {
		t.Fatal("not ready")
	}
	if strings.Contains(e.Starts()[0], "--cache-reuse") {
		t.Fatalf("unknown flag was passed: %v", e.Starts())
	}
	logs, _ := s.Logs("engine", 100)
	if !strings.Contains(logs, "does not know [--cache-reuse]") {
		t.Fatalf("no log line about the dropped flag:\n%s", logs)
	}
}

func TestLoadOOMMeasuredRetry(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.Llama["FAKELLAMA_OOM_ABOVE"] = "150000"
	s := ready(t, e)
	lines := e.Starts()
	if len(lines) != 2 {
		t.Fatalf("want 2 starts (fail, measured retry), got %v", lines)
	}
	if testkit.Field(lines[0], "c") != "196608" {
		t.Fatalf("first start %v", lines[0])
	}
	c, _ := strconv.Atoi(testkit.Field(lines[1], "c"))
	if c <= 0 || c > 150000 {
		t.Fatalf("retry did not shrink below the failing size: %v", lines[1])
	}
	logs, _ := s.Logs("engine", 100)
	if !strings.Contains(logs, "measured retry") {
		t.Fatalf("log lacks the measured retry:\n%s", logs)
	}
}

func TestLoadOOMTwiceStepsDownAndRecords(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	// Fails at the first shape and again at the measured retry (which only drops a little).
	e.Llama["FAKELLAMA_OOM_ABOVE"] = "170000"
	e.Llama["FAKELLAMA_BYTES_PER_TOKEN"] = "1000"
	s := ready(t, e)
	lines := e.Starts()
	if len(lines) < 3 {
		t.Fatalf("want at least 3 starts, got %v", lines)
	}
	st := run.LoadState(e.Data)
	capped := false
	for _, c := range st.Book.Entries {
		if c.CapCtxTotal > 0 && c.CapCtxTotal <= 170000 {
			capped = true
		}
	}
	if !capped {
		t.Fatalf("step-down was not recorded in the book: %+v", st.Book)
	}
	_ = s
}

func TestRuntimeOOMLowersCap(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	e.Touch("oom")
	testkit.WaitFor(t, 10*time.Second, "restart after the runtime OOM", func() bool {
		return len(e.Starts()) >= 2 && s.Phase() == "ready"
	})
	lines := e.Starts()
	c0, _ := strconv.Atoi(testkit.Field(lines[0], "c"))
	c1, _ := strconv.Atoi(testkit.Field(lines[len(lines)-1], "c"))
	if c1 >= c0 {
		t.Fatalf("restart was not smaller: %v", lines)
	}
	testkit.WaitFor(t, 2*time.Second, "cap saved", func() bool {
		for _, c := range run.LoadState(e.Data).Book.Entries {
			if c.CapCtxTotal == c1 {
				return true
			}
		}
		return false
	})
}

func TestShrinkWhenIdleNeverMidRequest(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	done := s.Begin() // a request is processing a line
	e.SetUsed(16000)  // the player loaded a heavy save: free 384 MiB < margin 655
	e.SetOthers(8000)
	// a held shortfall this large does not wait for a pause: new requests wait ("Loading
	// model") while the line in flight finishes, then the model shrinks
	testkit.WaitPhase(t, s, "resizing", 10*time.Second)
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("restarted while busy: %v", e.Starts())
	}
	if r := s.Route(); r.Kind != "unavailable" || r.Message != "Loading model" {
		t.Fatalf("route while draining = %+v", r)
	}
	done()
	testkit.WaitFor(t, 10*time.Second, "shrink once idle", func() bool { return len(e.Starts()) >= 2 })
	e.SetUsed(0)
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	lines := e.Starts()
	c0, _ := strconv.Atoi(testkit.Field(lines[0], "c"))
	c1, _ := strconv.Atoi(testkit.Field(lines[1], "c"))
	if c1 >= c0 {
		t.Fatalf("not smaller: %v", lines)
	}
}

func TestGrowWhenIdleNeverMidRequest(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.SetOthers(6000) // the player's apps hold 6 GB: starts smaller
	cfg := e.Config()
	cfg.GrowIdle = 300 * time.Millisecond
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	c0, _ := strconv.Atoi(testkit.Field(e.Starts()[0], "c"))
	if c0 >= 196608 {
		t.Fatalf("expected a reduced first shape, got %v", e.Starts())
	}
	done := s.Begin()
	e.SetOthers(0) // room is back, but a line is being served
	s.RoomMoved()  // the stranger's word
	testkit.WaitFor(t, 10*time.Second, "the room held", func() bool { r, ok := s.HeldReserve(); return ok && r == 0 })
	s.React() // the engine has looked at the room that came back
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("grew while busy: %v", e.Starts())
	}
	done()
	testkit.WaitFor(t, 10*time.Second, "grow once idle for GrowIdle", func() bool { return len(e.Starts()) >= 2 })
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	c1, _ := strconv.Atoi(testkit.Field(e.Starts()[1], "c"))
	if c1 <= c0 {
		t.Fatalf("not bigger: %v", e.Starts())
	}
}

func TestFailedGrowGoesBack(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.SetOthers(6000)
	first := 0
	_ = first
	cfg := e.Config()
	cfg.GrowIdle = 200 * time.Millisecond
	cfg.GrowBackoff = time.Hour
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	c0, _ := strconv.Atoi(testkit.Field(e.Starts()[0], "c"))
	// the grown shape will not load, the old one will
	e.Llama["FAKELLAMA_OOM_ABOVE"] = strconv.Itoa(c0 + 1)
	s.Shutdown()
	cfg = e.Config()
	cfg.GrowIdle = 200 * time.Millisecond
	cfg.GrowBackoff = time.Hour
	s2 := e.StartSup(cfg)
	testkit.WaitPhase(t, s2, "ready", 15*time.Second)
	n := len(e.Starts())
	e.SetOthers(0)
	testkit.WaitFor(t, 10*time.Second, "grow attempt", func() bool { return len(e.Starts()) >= n+2 })
	testkit.WaitPhase(t, s2, "ready", 10*time.Second)
	st := s2.Status()
	if st.Shape == nil || st.Shape.CtxPerSlot*st.Shape.Slots > c0 {
		t.Fatalf("did not go back to the old shape: %+v", st.Shape)
	}
}

// A server that says nothing while a request is in flight is asked /health once when the
// watchdog for silence fires; not answering, it is restarted.
func TestHungServerRestarts(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.HTTP = nil
	cfg.HangSilence = 300 * time.Millisecond
	s := e.StartSup(withShortHTTP(cfg))
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	pid, _ := strconv.Atoi(testkit.Field(e.Starts()[0], "pid"))
	done := s.Begin() // a request in flight
	defer done()
	if err := syscall.Kill(pid, syscall.SIGSTOP); err != nil { // alive but not answering
		t.Fatal(err)
	}
	testkit.WaitFor(t, 15*time.Second, "restart after the silence", func() bool { return len(e.Starts()) >= 2 })
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if s.Status().Restarts < 1 {
		t.Fatal("restart not counted")
	}
	found := false
	for _, m := range s.Status().Errors {
		found = found || strings.Contains(m, "said nothing")
	}
	if !found {
		t.Fatalf("errors = %v", s.Status().Errors)
	}
}

// A server that is quiet with nothing in flight is never asked anything: the watchdog for
// silence is armed only while requests are in flight.
func TestQuietIdleServerIsLeftAlone(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.HTTP = nil
	cfg.HangSilence = 50 * time.Millisecond
	s := e.StartSup(withShortHTTP(cfg))
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	pid, _ := strconv.Atoi(testkit.Field(e.Starts()[0], "pid"))
	if err := syscall.Kill(pid, syscall.SIGSTOP); err != nil {
		t.Fatal(err)
	}
	defer syscall.Kill(pid, syscall.SIGCONT)
	from := testkit.Mark(t, s, "before")
	done := s.Begin() // a request begins and ends at once: the silence is disarmed again
	done()
	// named moment: well past HangSilence; a watchdog left armed would have fired by now
	past := time.NewTimer(10 * cfg.HangSilence)
	<-past.C
	s.React()
	if hs := testkit.Said(t, s, from, 0, "mind.hang"); len(hs) != 0 {
		t.Fatalf("a quiet idle server was asked /health: %v", hs)
	}
	if n := len(e.Starts()); n != 1 || s.Phase() != "ready" {
		t.Fatalf("a quiet idle server was restarted: %v %s", e.Starts(), s.Phase())
	}
}

func TestStopSleepWake(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	if s.Phase() != "sleeping" {
		t.Fatalf("phase %s", s.Phase())
	}
	if r := s.Route(); r.Kind != "unavailable" {
		t.Fatalf("route %+v", r)
	}
	// a proxied request (Route) wakes it
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if err := s.Stop(); err != nil {
		t.Fatal(err)
	}
	// a stopped server is not woken by a request: Route starts nothing
	if r := s.Route(); r.Kind != "unavailable" || s.Phase() != "sleeping" {
		t.Fatalf("stopped server must stay stopped: %+v %s", r, s.Phase())
	}
}

func TestStateSurvivesRestart(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	if err := s.SetSettings(run.Settings{Mode: "local", LingerS: 7, KVType: "q8_0"}); err != nil {
		t.Fatal(err)
	}
	st := run.LoadState(e.Data)
	if st.LingerS != 7 || st.KVType != "q8_0" || st.Model != "m1" || st.Backend != "fake" {
		t.Fatalf("state = %+v", st)
	}
	if _, err := os.Stat(filepath.Join(e.Data, "state.json")); err != nil {
		t.Fatal(err)
	}
}

func TestStatusReportsFallbackModel(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := ready(t, e)
	if st := s.Status(); st.ModelFallback || st.ModelNote != "" {
		t.Fatalf("a normal model must not be flagged: %+v", st)
	}
	e.Cat.Models[0].Fallback, e.Cat.Models[0].FallbackNote = true, "smallest model"
	st := s.Status()
	if !st.ModelFallback || st.ModelNote != "smallest model" {
		t.Fatalf("fallback not reported: %+v", st)
	}
	b, _ := json.Marshal(st)
	if !strings.Contains(string(b), `"model_fallback":true`) || !strings.Contains(string(b), `"model_note":"smallest model"`) {
		t.Fatalf("json %s", b)
	}
}
