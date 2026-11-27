package run_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/probe/gpusense"
	"hermesengine/run"
	"hermesengine/yield"
)

// fakeYieldOS is the other programs on a pretend machine.
type fakeYieldOS struct {
	mu       sync.Mutex
	launches []yield.Launch
	openers  map[int]string
	procs    []gpusense.Process // memory on adapter 0, not ours
	wake     chan struct{}
}

func newFakeYieldOS() *fakeYieldOS {
	return &fakeYieldOS{openers: map[int]string{}, wake: make(chan struct{}, 1)}
}

func (f *fakeYieldOS) setLaunches(ls ...yield.Launch) { f.mu.Lock(); f.launches = ls; f.mu.Unlock() }
func (f *fakeYieldOS) setProcs(ps ...gpusense.Process) {
	f.mu.Lock()
	f.procs = ps
	f.mu.Unlock()
}
func (f *fakeYieldOS) open(pid int, name string) {
	f.mu.Lock()
	f.openers[pid] = name
	f.mu.Unlock()
	select {
	case f.wake <- struct{}{}:
	default:
	}
}
func (f *fakeYieldOS) close(pid int) { f.mu.Lock(); delete(f.openers, pid); f.mu.Unlock() }

func (f *fakeYieldOS) Launches() []yield.Launch {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]yield.Launch(nil), f.launches...)
}
func (f *fakeYieldOS) Openers(skip func(int) bool) map[int]string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := map[int]string{}
	for p, n := range f.openers {
		if !skip(p) {
			out[p] = n
		}
	}
	return out
}
func (f *fakeYieldOS) Under(pid, anc int) bool { return false }
func (f *fakeYieldOS) Wait(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	case <-f.wake:
		return true
	}
	return false
}
func (f *fakeYieldOS) Drain()         {}
func (f *fakeYieldOS) Adapter(string) {}
func (f *fakeYieldOS) Close()         {}

func (f *fakeYieldOS) memProcs(ctx context.Context, as []gpusense.Adapter) []gpusense.Process {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]gpusense.Process(nil), f.procs...)
}

func (f *fakeYieldOS) sense(ctx context.Context, ours []int, host int) gpusense.Picture {
	ps := f.memProcs(ctx, nil)
	used := 0
	for _, p := range ps {
		used += p.DedicatedMiB
	}
	return gpusense.Picture{Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UsedMiB: used, UUID: "GPU-aaaa"},
	}, Processes: ps, Sources: map[string]string{"fake": "ok"}}
}

// settled waits for ready and for the yield loop's first look (its baseline).
func settled(t *testing.T, s *run.Supervisor) {
	t.Helper()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	time.Sleep(400 * time.Millisecond)
}

func yieldSup(t *testing.T, e *testkit.Env, f *fakeYieldOS, tweak func(*run.YieldConfig)) *run.Supervisor {
	t.Helper()
	e.InstallFake()
	cfg := e.Config()
	cfg.Sense = f.sense
	y := &run.YieldConfig{Except: []string{"294"}, Every: 50 * time.Millisecond, Drain: 2 * time.Second,
		OS: func() yield.OS { return f }, Procs: f.memProcs}
	y.Policy = yield.Config{Grace: 400 * time.Millisecond, Steady: 600 * time.Millisecond, Floor: 128}
	if tweak != nil {
		tweak(y)
	}
	cfg.Yield = y
	return e.StartSup(cfg)
}

var game = yield.Launch{PID: 777, Kind: "steam", AppID: "999", Exe: "x"}

func TestYieldToGameAndBack(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	t0 := time.Now()
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if d := time.Since(t0); d > 2*time.Second {
		t.Fatalf("yield took %s", d)
	}
	st := s.Status()
	if st.Yield == nil || st.Yield.Kind != "game" || st.Yield.Reason != "AppId 999 (x)" || st.Yield.Draining || st.Shape != nil {
		t.Fatalf("status %+v yield %+v", st, st.Yield)
	}
	if st.Recovery != nil || st.Restarts != 0 || len(st.Errors) != 0 {
		t.Fatalf("a yield is not a failure: recovery %+v restarts %d errors %v", st.Recovery, st.Restarts, st.Errors)
	}
	if r := s.Route(); r.Kind != "unavailable" || r.Code != "yielded" || !strings.Contains(r.Message, "Paused while AppId 999 (x) runs") {
		t.Fatalf("route %+v", r)
	}
	if err := s.Start(); !errors.Is(err, run.ErrYielded) {
		t.Fatalf("Start while yielded: %v", err)
	}
	if err := s.Sleep(); err != nil || s.Phase() != "yielded" {
		t.Fatalf("Sleep while yielded: %v %s", err, s.Phase())
	}
	gone := time.Now()
	f.setLaunches()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if d := time.Since(gone); d < 400*time.Millisecond {
		t.Fatalf("resumed %s after the game left, before the grace", d)
	}
	if s.Status().Yield != nil || s.Route().Code != "" {
		t.Fatal("yield left behind after resume")
	}
	if n := len(e.Starts()); n != 2 {
		t.Fatalf("starts %d, want 2", n)
	}
}

func TestYieldSupportedAppIDStays(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	f.setLaunches(yield.Launch{PID: 778, Kind: "steam", AppID: "294"})
	time.Sleep(time.Second)
	if s.Phase() != "ready" {
		t.Fatalf("a supported app id yielded: %s %+v", s.Phase(), s.Status().Yield)
	}
}

func TestYieldGameReturnsDuringGrace(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Policy.Grace = 800 * time.Millisecond })
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	f.setLaunches()
	time.Sleep(400 * time.Millisecond)
	f.setLaunches(game) // a launcher restarts the game
	time.Sleep(600 * time.Millisecond)
	if s.Phase() != "yielded" {
		t.Fatalf("resumed while the game was back: %s", s.Phase())
	}
	last := time.Now()
	f.setLaunches()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if d := time.Since(last); d < 800*time.Millisecond {
		t.Fatalf("grace not restarted: resumed %s after the last sighting", d)
	}
}

func TestYieldToNewMemoryThenBesideAndAgainOnGrowth(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s)
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 1000})
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if y := s.Status().Yield; y == nil || y.Kind != "gpu" || !strings.Contains(y.Reason, "editor (1000 MiB)") {
		t.Fatalf("yield %+v", y)
	}
	// steady for Steady: back beside it
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	// it grows past the floor: yield again
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 1500})
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if y := s.Status().Yield; y == nil || !strings.Contains(y.Reason, "grew to 1500 MiB") {
		t.Fatalf("yield %+v", y)
	}
	f.setProcs()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

func TestYieldStaysWhenNothingFitsBeside(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s)
	f.setProcs(gpusense.Process{PID: 4243, Name: "hog", Adapter: 0, DedicatedMiB: 15000})
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	time.Sleep(1500 * time.Millisecond) // well past Steady
	if s.Phase() != "yielded" {
		t.Fatalf("loaded into memory the other program holds: %s %s", s.Phase(), s.Status().Message)
	}
	f.setProcs()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

func TestYieldStrictWaitsForExit(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Policy.Strict = true })
	settled(t, s)
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 500})
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	time.Sleep(1500 * time.Millisecond)
	if s.Phase() != "yielded" {
		t.Fatalf("strict resumed beside: %s", s.Phase())
	}
	f.setProcs()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

func TestYieldOnDeviceOpen(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Every = 5 * time.Second }) // memory reads too slow to matter
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	time.Sleep(300 * time.Millisecond) // baseline taken
	t0 := time.Now()
	f.open(5151, "viewer")
	testkit.WaitFor(t, 3*time.Second, "yield on open", func() bool { return s.Status().Yield != nil })
	if d := time.Since(t0); d > time.Second {
		t.Fatalf("open to yield took %s", d)
	}
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if y := s.Status().Yield; !strings.Contains(y.Reason, "viewer (opened the GPU)") {
		t.Fatalf("yield %+v", y)
	}
	f.close(5151)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

func TestYieldShortOpenIgnored(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Ignore = []string{"Monitor.EXE"} })
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	time.Sleep(300 * time.Millisecond)
	f.open(6000, "probe")
	time.Sleep(50 * time.Millisecond)
	f.close(6000) // gone before the second look
	f.open(6001, "monitor")
	time.Sleep(time.Second)
	if s.Phase() != "ready" {
		t.Fatalf("a short-lived or ignored opener yielded: %+v", s.Status().Yield)
	}
}

func TestYieldDrainsReplyInFlight(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	done := s.Begin() // a reply streaming
	f.setLaunches(game)
	testkit.WaitFor(t, 3*time.Second, "draining", func() bool { y := s.Status().Yield; return y != nil && y.Draining })
	if r := s.Route(); r.Code != "yielded" {
		t.Fatalf("new requests must be refused at once: %+v", r)
	}
	time.Sleep(500 * time.Millisecond)
	if s.Phase() != "ready" {
		t.Fatalf("unloaded under a running reply: %s", s.Phase())
	}
	t0 := time.Now()
	done()
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if d := time.Since(t0); d > time.Second {
		t.Fatalf("unload %s after the reply ended", d)
	}
}

func TestYieldDrainCapped(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Drain = 300 * time.Millisecond })
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	done := s.Begin()
	defer done()
	t0 := time.Now()
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if d := time.Since(t0); d < 300*time.Millisecond || d > 3*time.Second {
		t.Fatalf("capped drain took %s", d)
	}
}

func TestYieldWhileLoading(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "4s"
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	testkit.WaitFor(t, 10*time.Second, "loading", func() bool { return s.Phase() == "loading" })
	t0 := time.Now()
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if d := time.Since(t0); d > 2*time.Second {
		t.Fatalf("yield while loading took %s", d)
	}
	if st := s.Status(); st.Recovery != nil || len(st.Errors) != 0 {
		t.Fatalf("a cancelled load for a yield is not a failure: %+v %v", st.Recovery, st.Errors)
	}
}

func TestStopWhileYielded(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	if err := s.Stop(); err != nil {
		t.Fatal(err)
	}
	if s.Phase() != "sleeping" || s.Status().Yield != nil || s.Route().Code == "yielded" {
		t.Fatalf("after Stop: %s %+v %+v", s.Phase(), s.Status().Yield, s.Route())
	}
	time.Sleep(500 * time.Millisecond) // the game still runs, but a person stopped it: no yield
	if s.Phase() != "sleeping" || s.Status().Yield != nil {
		t.Fatalf("stopped engine yielded: %s", s.Phase())
	}
	f.setLaunches()
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	f.setLaunches(game) // the watch is back
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
}

// A program already on the card when the engine starts is baseline, even when the first looks
// come before the engine knows its adapter (a slow first sense): it must not read as growth.
func TestYieldBaselineWaitsForMemory(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	f.setProcs(gpusense.Process{PID: 9, Name: "resident", Adapter: 0, DedicatedMiB: 2000})
	f.open(9, "resident")
	e.InstallFake()
	cfg := e.Config()
	var once sync.Once
	cfg.Sense = func(ctx context.Context, ours []int, host int) gpusense.Picture {
		once.Do(func() { time.Sleep(800 * time.Millisecond) })
		return f.sense(ctx, ours, host)
	}
	y := &run.YieldConfig{Every: 50 * time.Millisecond, OS: func() yield.OS { return f }, Procs: f.memProcs}
	y.Policy = yield.Config{Grace: 400 * time.Millisecond, Steady: time.Minute, Floor: 128}
	cfg.Yield = y
	s := e.StartSup(cfg)
	testkit.WaitFor(t, 15*time.Second, "ready or yielded", func() bool { p := s.Phase(); return p == "ready" || p == "yielded" })
	time.Sleep(1500 * time.Millisecond)
	if st := s.Status(); st.Phase != "ready" || st.Yield != nil || len(e.Starts()) != 1 {
		t.Fatalf("a program there from the start yielded the engine: %s %+v starts %d", st.Phase, st.Yield, len(e.Starts()))
	}
}

func TestYieldIgnoresNvidiaSMI(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s)
	f.open(7000, "nvidia-smi") // a monitor that outlives the second look
	time.Sleep(time.Second)
	if s.Phase() != "ready" {
		t.Fatalf("nvidia-smi yielded the engine: %+v", s.Status().Yield)
	}
}
