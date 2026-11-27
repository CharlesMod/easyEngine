package run_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/yield"
)

// fakeYieldOS is the other programs on a pretend machine. It gives the words a Linux or Windows
// machine gives: a program opening the device (opens), a launcher's game coming or going
// (launchWord). The yield reactor waits on those and on the host's level; no test here waits on
// a clock.
type fakeYieldOS struct {
	mu       sync.Mutex
	launches []yield.Launch
	openers  map[int]string
	procs    []gpusense.Process // memory on adapter 0, not ours
	flash    *gpusense.Process  // seen by exactly one memory read

	flashSeen  chan struct{} // closed when the flash was read
	opens      chan struct{}
	launchWord chan struct{}

	watchers  atomic.Int32 // Watch calls not yet released
	watchTick chan struct{}
	senses    atomic.Int32
	senseTick chan struct{}

	hasLook    bool
	lookEngine int
	lookUses   []yield.Use
	lookGames  []yield.Launch
	lookTick   chan struct{}
}

func newFakeYieldOS() *fakeYieldOS {
	return &fakeYieldOS{openers: map[int]string{}, opens: make(chan struct{}, 1), launchWord: make(chan struct{}, 1),
		watchTick: make(chan struct{}, 1), senseTick: make(chan struct{}, 1), lookTick: make(chan struct{}, 1)}
}

func ping(c chan struct{}) {
	select {
	case c <- struct{}{}:
	default:
	}
}

func (f *fakeYieldOS) setLaunches(ls ...yield.Launch) {
	f.mu.Lock()
	f.launches = ls
	f.mu.Unlock()
	ping(f.launchWord)
}
func (f *fakeYieldOS) setProcs(ps ...gpusense.Process) {
	f.mu.Lock()
	f.procs = ps
	f.mu.Unlock()
}
func (f *fakeYieldOS) open(pid int, name string) {
	f.mu.Lock()
	f.openers[pid] = name
	f.mu.Unlock()
	ping(f.opens)
}

// flashProc is a program the next memory read sees and no other: a probe that came and went.
func (f *fakeYieldOS) flashProc(p gpusense.Process) {
	f.mu.Lock()
	f.flash, f.flashSeen = &p, make(chan struct{})
	f.mu.Unlock()
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
func (f *fakeYieldOS) Adapter(string)          {}
func (f *fakeYieldOS) Close()                  {}

func (f *fakeYieldOS) Watch() yield.Words {
	f.watchers.Add(1)
	var once sync.Once
	return yield.Words{Launches: f.launchWord, Opens: f.opens, Stop: func() {
		once.Do(func() { f.watchers.Add(-1); ping(f.watchTick) })
	}}
}

// noWordsOS is the fake without the OS's words: a launcher's game is read on the host's level.
type noWordsOS struct{ f *fakeYieldOS }

func (n noWordsOS) Launches() []yield.Launch                   { return n.f.Launches() }
func (n noWordsOS) Openers(skip func(int) bool) map[int]string { return n.f.Openers(skip) }
func (n noWordsOS) Under(pid, anc int) bool                    { return false }
func (n noWordsOS) Adapter(string)                             {}
func (n noWordsOS) Close()                                     {}

func (f *fakeYieldOS) memProcs(ctx context.Context, as []gpusense.Adapter) []gpusense.Process {
	f.mu.Lock()
	defer f.mu.Unlock()
	ps := append([]gpusense.Process(nil), f.procs...)
	if f.flash != nil {
		ps = append(ps, *f.flash)
		f.flash = nil
		close(f.flashSeen)
	}
	return ps
}

func (f *fakeYieldOS) sense(ctx context.Context, ours []int, host int) gpusense.Picture {
	f.senses.Add(1)
	ping(f.senseTick)
	f.mu.Lock()
	ps := append([]gpusense.Process(nil), f.procs...)
	f.mu.Unlock()
	used := 0
	for _, p := range ps {
		used += p.DedicatedMiB
	}
	return gpusense.Picture{Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UsedMiB: used, UUID: "GPU-aaaa"},
	}, Processes: ps, Sources: map[string]string{"fake": "ok"}}
}

// looked is YieldConfig.Looked: the engine has decided on this reading.
func (f *fakeYieldOS) looked(engine int, uses []yield.Use, games []yield.Launch) {
	f.mu.Lock()
	f.hasLook, f.lookEngine, f.lookUses, f.lookGames = true, engine, uses, games
	f.mu.Unlock()
	ping(f.lookTick)
}

// waitTick waits for a word on c; limit only makes a broken engine fail the test.
func waitTick(t *testing.T, c chan struct{}, what string) {
	t.Helper()
	select {
	case <-c:
	// watchdog for silence: a test whose engine never says what it waits for fails after a bound
	case <-time.After(20 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

// waitLook waits until the engine has decided on a reading that pred likes, one it took after
// the call (the earlier ones are forgotten).
func (f *fakeYieldOS) waitLook(t *testing.T, what string, pred func(engine int, uses []yield.Use, games []yield.Launch) bool) {
	t.Helper()
	for {
		f.mu.Lock()
		ok := f.hasLook && pred(f.lookEngine, f.lookUses, f.lookGames)
		f.mu.Unlock()
		if ok {
			return
		}
		waitTick(t, f.lookTick, what)
	}
}

// forgetLooks drops what the engine decided on so far: waitLook then waits for a later look.
func (f *fakeYieldOS) forgetLooks() {
	f.mu.Lock()
	f.hasLook = false
	f.mu.Unlock()
}

func hasUse(pid, mib int) func(int, []yield.Use, []yield.Launch) bool {
	return func(_ int, us []yield.Use, _ []yield.Launch) bool {
		for _, u := range us {
			if u.PID == pid && (mib == 0 || u.MiB == mib) {
				return true
			}
		}
		return false
	}
}

func hasGames(n int) func(int, []yield.Use, []yield.Launch) bool {
	return func(_ int, _ []yield.Use, gs []yield.Launch) bool { return len(gs) == n }
}

// onCard is a look taken with the engine's adapter known (its baseline, or a change after it).
func onCard(int, []yield.Use, []yield.Launch) bool             { return true }
func onCard0(engine int, _ []yield.Use, _ []yield.Launch) bool { return engine == 0 }

// settled waits for ready and for the yield reactor's first look (its baseline).
func settled(t *testing.T, s *run.Supervisor, f *fakeYieldOS) {
	t.Helper()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	f.waitLook(t, "the baseline", onCard0)
}

func yieldSup(t *testing.T, e *testkit.Env, f *fakeYieldOS, tweak func(*run.YieldConfig)) *run.Supervisor {
	t.Helper()
	e.InstallFake()
	cfg := e.Config()
	cfg.Sense = f.sense
	y := &run.YieldConfig{Except: []string{"294"}, Every: 50 * time.Millisecond, Drain: 2 * time.Second,
		OS: func() yield.OS { return f }, Procs: f.memProcs, Looked: f.looked}
	y.Policy = yield.Config{Grace: 400 * time.Millisecond, Steady: 600 * time.Millisecond, Floor: 128}
	if tweak != nil {
		tweak(y)
	}
	cfg.Yield = y
	return e.StartSup(cfg)
}

// yieldReason is what the card is yielded to, once yielded.
func yieldReason(t *testing.T, s *run.Supervisor) string {
	t.Helper()
	y := s.Status().Yield
	if y == nil {
		t.Fatalf("not yielded: %s", s.Phase())
	}
	return y.Reason
}

var game = yield.Launch{PID: 777, Kind: "steam", AppID: "999", Exe: "x"}

// A device open and a launcher's word are the whole signal: the level's period is an hour, so
// only words (and the level's two reads at arming) can move the engine.
func TestYieldToGameAndBack(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Every = time.Hour })
	settled(t, s, f)
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
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
	// The grace is a named moment that runs from the game's going: not resumed before it.
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

// Where the OS gives no word for its launchers (argv in /proc on Linux), the host's level reads
// them: the same yield and resume.
func TestYieldLauncherReadOnTheLevel(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.OS = func() yield.OS { return noWordsOS{f} } })
	settled(t, s, f)
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if r := yieldReason(t, s); r != "AppId 999 (x)" {
		t.Fatalf("reason %q", r)
	}
	f.setLaunches()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

func TestYieldSupportedAppIDStays(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s, f)
	mark := testkit.Mark(t, s, "supported app id")
	// The supported launch comes first, so a yield to it would name it; the game after it is
	// the control that the engine looked.
	f.setLaunches(yield.Launch{PID: 778, Kind: "steam", AppID: "294"}, game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if r := yieldReason(t, s); r != "AppId 999 (x)" {
		t.Fatalf("a supported app id yielded: %q", r)
	}
	f.waitLook(t, "the game", hasGames(1))
	if ph := testkit.Phases(t, s, mark, 0); len(ph) == 0 || ph[len(ph)-1] != "yielded" {
		t.Fatalf("phases %v", ph)
	}
}

func TestYieldGameReturnsDuringGrace(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Policy.Grace = 800 * time.Millisecond })
	settled(t, s, f)
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	f.forgetLooks()
	f.setLaunches()
	f.waitLook(t, "the game gone", hasGames(0))
	mark := testkit.Mark(t, s, "game left")
	f.forgetLooks()
	f.setLaunches(game) // a launcher restarts the game
	f.waitLook(t, "the game back", hasGames(1))
	if ph := testkit.Phases(t, s, mark, 0); len(ph) != 0 {
		t.Fatalf("phases while the game was back: %v", ph)
	}
	if s.Phase() != "yielded" {
		t.Fatalf("resumed while the game was back: %s", s.Phase())
	}
	last := time.Now()
	f.setLaunches()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	// The grace runs from the game's last going (the pure machine's test holds that it restarts):
	// not resumed before it.
	if d := time.Since(last); d < 800*time.Millisecond {
		t.Fatalf("grace not restarted: resumed %s after the game left", d)
	}
}

func TestYieldToNewMemoryThenBesideAndAgainOnGrowth(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s, f)
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 1000})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if y := s.Status().Yield; y == nil || y.Kind != "gpu" || !strings.Contains(y.Reason, "editor (1000 MiB)") {
		t.Fatalf("yield %+v", y)
	}
	// steady for Steady: back beside it
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	// it grows past the floor: yield again
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 1500})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
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
	settled(t, s, f)
	f.setProcs(gpusense.Process{PID: 4243, Name: "hog", Adapter: 0, DedicatedMiB: 15000})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	mark := testkit.Mark(t, s, "yielded to the hog")
	// The steady moment fires and the engine looks for a shape beside the hog (it senses the card):
	// once, and again after the next steady wait. The second look began after the first one ended.
	n := f.senses.Load()
	for f.senses.Load() < n+2 {
		waitTick(t, f.senseTick, "the engine to look for room beside the hog")
	}
	if s.Phase() != "yielded" {
		t.Fatalf("loaded into memory the other program holds: %s %s", s.Phase(), s.Status().Message)
	}
	if ph := testkit.Phases(t, s, mark, 0); len(ph) != 0 {
		t.Fatalf("phases while nothing fit: %v", ph)
	}
	f.setProcs()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

func TestYieldStrictWaitsForExit(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Policy.Strict = true })
	settled(t, s, f)
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 500})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	mark := testkit.Mark(t, s, "yielded to the editor")
	// It changes (past the floor): the engine looks again at a changed card and stays away.
	f.forgetLooks()
	f.setProcs(gpusense.Process{PID: 4242, Name: "editor", Adapter: 0, DedicatedMiB: 800})
	f.waitLook(t, "the editor's change", hasUse(4242, 800))
	if s.Phase() != "yielded" {
		t.Fatalf("strict resumed beside: %s", s.Phase())
	}
	if ph := testkit.Phases(t, s, mark, 0); len(ph) != 0 {
		t.Fatalf("phases while the editor stayed: %v", ph)
	}
	f.setProcs()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

// A program opening the device is heard at once (the level's period is an hour), and counts when
// it is there in two reads in a row.
func TestYieldOnDeviceOpen(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Every = time.Hour })
	settled(t, s, f)
	f.open(5151, "viewer")
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if y := s.Status().Yield; !strings.Contains(y.Reason, "viewer (opened the GPU)") {
		t.Fatalf("yield %+v", y)
	}
	f.close(5151)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

// A probe that is gone by the next read never held, and a program the keeper ignores never
// counts. A real newcomer after them is the control: it yields, and only it is named.
func TestYieldShortOpenIgnored(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Ignore = []string{"Monitor.EXE"} })
	settled(t, s, f)
	mark := testkit.Mark(t, s, "probes")
	f.open(6001, "monitor")
	f.flashProc(gpusense.Process{PID: 6000, Name: "probe", Adapter: 0, DedicatedMiB: 300})
	flashed := func() chan struct{} { f.mu.Lock(); defer f.mu.Unlock(); return f.flashSeen }()
	waitTick(t, flashed, "the probe to be read")
	f.setProcs(gpusense.Process{PID: 6002, Name: "real", Adapter: 0, DedicatedMiB: 400})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if r := yieldReason(t, s); r != "real (400 MiB)" {
		t.Fatalf("a short-lived or ignored program counted: %q", r)
	}
	if ph := testkit.Phases(t, s, mark, 0); len(ph) == 0 || ph[len(ph)-1] != "yielded" {
		t.Fatalf("phases %v", ph)
	}
}

func TestYieldDrainsReplyInFlight(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s, f)
	done := s.Begin() // a reply streaming
	mark := testkit.Mark(t, s, "reply began")
	f.setLaunches(game)
	testkit.WaitFor(t, 15*time.Second, "draining", func() bool { y := s.Status().Yield; return y != nil && y.Draining })
	if r := s.Route(); r.Code != "yielded" {
		t.Fatalf("new requests must be refused at once: %+v", r)
	}
	done()
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	// The unload waited for the reply: its end was said before the first phase after the yield began.
	var ended uint64
	for _, ev := range testkit.Said(t, s, mark, 0, "request") {
		var b struct {
			What string `json:"what"`
		}
		json.Unmarshal(ev.Body, &b)
		if b.What == "end" {
			ended = ev.Seq
		}
	}
	if ended == 0 {
		t.Fatal("the reply's end was not said")
	}
	for _, ev := range testkit.Said(t, s, mark, 0, "phase") {
		if ev.Seq < ended {
			t.Fatalf("phase said at seq %d, before the reply ended at seq %d", ev.Seq, ended)
		}
	}
}

func TestYieldDrainCapped(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, func(y *run.YieldConfig) { y.Drain = 300 * time.Millisecond })
	settled(t, s, f)
	done := s.Begin()
	defer done()
	t0 := time.Now()
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	// Drain is a named moment: the reply never ended, and the unload did not come before it.
	if d := time.Since(t0); d < 300*time.Millisecond {
		t.Fatalf("unloaded after %s, before the drain moment", d)
	}
}

func TestYieldWhileLoading(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "4s"
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	testkit.WaitFor(t, 15*time.Second, "loading", func() bool { return s.Phase() == "loading" })
	f.waitLook(t, "the baseline", onCard)
	mark := testkit.Mark(t, s, "loading")
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	for _, p := range testkit.Phases(t, s, mark, 0) {
		if p == "ready" {
			t.Fatalf("the load finished before the yield: %v", testkit.Phases(t, s, mark, 0))
		}
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
	settled(t, s, f)
	f.setLaunches(game)
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if err := s.Stop(); err != nil {
		t.Fatal(err)
	}
	if s.Phase() != "sleeping" || s.Status().Yield != nil || s.Route().Code == "yielded" {
		t.Fatalf("after Stop: %s %+v %+v", s.Phase(), s.Status().Yield, s.Route())
	}
	// The game still runs, but a person stopped the engine: the reactor lets its watches go.
	for f.watchers.Load() != 0 {
		waitTick(t, f.watchTick, "the yield watch to be let go")
	}
	f.setLaunches(game) // a word that changes nothing: nothing is watched
	if s.Phase() != "sleeping" || s.Status().Yield != nil {
		t.Fatalf("stopped engine yielded: %s", s.Phase())
	}
	f.setLaunches()
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	f.setLaunches(game) // the watch is back
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
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
	slow := make(chan struct{}) // the first sense waits for this: the adapter is not known yet
	var once sync.Once
	cfg.Sense = func(ctx context.Context, ours []int, host int) gpusense.Picture {
		once.Do(func() {
			select {
			case <-slow:
			case <-ctx.Done():
			}
		})
		return f.sense(ctx, ours, host)
	}
	y := &run.YieldConfig{Every: 50 * time.Millisecond, OS: func() yield.OS { return f }, Procs: f.memProcs, Looked: f.looked}
	y.Policy = yield.Config{Grace: 400 * time.Millisecond, Steady: time.Minute, Floor: 128}
	cfg.Yield = y
	s := e.StartSup(cfg)
	f.waitLook(t, "a look before the adapter is known", func(engine int, us []yield.Use, _ []yield.Launch) bool {
		return engine == -1 && hasUse(9, 0)(engine, us, nil)
	}) // baseline: the opener, no memory yet
	close(slow)
	testkit.WaitFor(t, 15*time.Second, "ready or yielded", func() bool { p := s.Phase(); return p == "ready" || p == "yielded" })
	f.waitLook(t, "a look with the memory known", func(engine int, us []yield.Use, _ []yield.Launch) bool {
		return engine == 0 && hasUse(9, 2000)(engine, us, nil)
	})
	mark := testkit.Mark(t, s, "resident is baseline")
	// The control: a real newcomer yields, and the resident is not named.
	f.setProcs(gpusense.Process{PID: 9, Name: "resident", Adapter: 0, DedicatedMiB: 2000},
		gpusense.Process{PID: 10000, Name: "newcomer", Adapter: 0, DedicatedMiB: 1000})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if r := yieldReason(t, s); r != "newcomer (1000 MiB)" {
		t.Fatalf("a program there from the start counted: %q", r)
	}
	// The engine yielded once, to the newcomer: no phase "yielded" before it.
	for _, p := range testkit.Phases(t, s, 0, mark) {
		if p == "yielded" {
			t.Fatalf("yielded before the newcomer: %v", testkit.Phases(t, s, 0, mark))
		}
	}
	if ph := testkit.Phases(t, s, mark, 0); len(ph) == 0 || ph[len(ph)-1] != "yielded" {
		t.Fatalf("phases %v", ph)
	}
}

func TestYieldIgnoresNvidiaSMI(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := newFakeYieldOS()
	s := yieldSup(t, e, f, nil)
	settled(t, s, f)
	f.open(7000, "nvidia-smi") // a monitor that outlives the reads
	f.setProcs(gpusense.Process{PID: 7002, Name: "real", Adapter: 0, DedicatedMiB: 400})
	testkit.WaitPhase(t, s, "yielded", 15*time.Second)
	if r := yieldReason(t, s); r != "real (400 MiB)" {
		t.Fatalf("nvidia-smi counted: %q", r)
	}
}
