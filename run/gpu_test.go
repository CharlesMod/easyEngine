//go:build !windows

package run_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/plan"
	"hermesengine/probe"
	"hermesengine/probe/gpusense"
	"hermesengine/run"
)

// senseFake is an injected Sense: the fake card (nvidia, 16384 MiB) with other apps holding
// `others` MiB, and an integrated adapter the host may render on. It records the pids it got.
type senseFake struct {
	others atomic.Int32
	spike  atomic.Int32 // the next read answers this many others, once (0: none)
	calls  atomic.Int32
	mu     sync.Mutex
	host   int
	ours   []int
}

func (f *senseFake) sense(ctx context.Context, ours []int, host int) gpusense.Picture {
	f.calls.Add(1)
	f.mu.Lock()
	f.host, f.ours = host, ours
	f.mu.Unlock()
	o := int(f.others.Load())
	if sp := f.spike.Swap(0); sp > 0 {
		o = int(sp)
	}
	procs := []gpusense.Process{{PID: 4242, Name: "browser", Adapter: 0, DedicatedMiB: o}}
	for _, p := range ours {
		procs = append(procs, gpusense.Process{PID: p, Name: "llama-server", Adapter: 0, DedicatedMiB: 15000, Ours: true})
	}
	return gpusense.Picture{
		Adapters: []gpusense.Adapter{
			{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UsedMiB: o + 15000*len(ours)},
			{Index: 1, Vendor: "intel", VendorID: 0x8086, Name: "Intel(R) Iris(R) Xe Graphics", DedicatedMiB: 128, SharedMiB: 8000, UsedMiB: 100, Integrated: true},
		},
		Processes: procs, Sources: map[string]string{"fake": "ok"},
	}
}

func (f *senseFake) hostPID() int { f.mu.Lock(); defer f.mu.Unlock(); return f.host }

func sensedSup(t *testing.T, e *testkit.Env, f *senseFake, mod func(*run.Config)) *run.Supervisor {
	t.Helper()
	e.InstallFake()
	cfg := e.Config()
	cfg.Sense = f.sense
	cfg.Others = func(context.Context, probe.GPU, []int) (int, error) { return 0, nil }
	if mod != nil {
		mod(&cfg)
	}
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return s
}

// The watchdog's shrink used the card's used memory, which holds our own server: with nothing
// else on the card the model must not be shrunk however full the card reads.
func TestWatchdogDoesNotShrinkOnItsOwnUse(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := &senseFake{}
	s := sensedSup(t, e, f, nil)
	e.SetUsed(16300) // the card reads full: our own llama-server holds almost all of it
	time.Sleep(700 * time.Millisecond)
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("shrank with no other app on the card: %v", e.Starts())
	}
	if st := s.Status(); st.GPU == nil || st.GPU.Reserve.OthersMiB != 0 || st.GPU.EngineAdapter != 0 {
		t.Fatalf("status gpu = %+v", st.GPU)
	}
}

// A change in the room reshapes the model once two reads agree, never on one read.
func TestHeldChangeReshapes(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := &senseFake{}
	s := sensedSup(t, e, f, nil)
	f.spike.Store(9000) // one read of a game loading, then it is gone
	time.Sleep(600 * time.Millisecond)
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("reshaped on a single read: %v", e.Starts())
	}
	f.others.Store(9000) // held
	testkit.WaitFor(t, 10*time.Second, "shrink on a held reserve", func() bool { return len(e.Starts()) >= 2 })
	fits := func() bool { // each held tick takes one step down until the model fits
		st := s.Status()
		return st.Phase == "ready" && st.Shape != nil && st.GPU != nil && st.Shape.VRAMMiB <= st.GPU.FreeForModelMiB+st.GPU.Reserve.MarginMiB
	}
	testkit.WaitFor(t, 30*time.Second, "the shrunk model fits the room", fits)
	st := s.Status()
	if st.GPU.Reserve.OthersMiB != 9000 || st.GPU.Reserve.OthersBasis != "processes" || st.GPU.OthersNote == "" {
		t.Fatalf("gpu = %+v", st.GPU)
	}
	// the new shape fits what is left, and the processes of other apps are never touched
	if st.Shape == nil || st.Shape.VRAMMiB > st.GPU.FreeForModelMiB+st.GPU.Reserve.MarginMiB {
		t.Fatalf("shape %+v does not fit %+v", st.Shape, st.GPU)
	}
}

func TestHostPIDOrder(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := &senseFake{}
	e.Prof.HostPID = 0
	s := sensedSup(t, e, f, func(c *run.Config) { c.Parents = func() []int { return []int{777, 888} } })
	ctx := context.Background()
	if _, err := s.GPUReport(ctx, true); err != nil {
		t.Fatal(err)
	}
	if got := f.hostPID(); got != 777 {
		t.Fatalf("host pid %d, want the first parent 777", got)
	}
	me := os.Getpid()
	if err := s.SetHost(run.HostSet{PID: me}); err != nil {
		t.Fatal(err)
	}
	s.GPUReport(ctx, true)
	if got := f.hostPID(); got != me {
		t.Fatalf("host pid %d, want the posted %d", got, me)
	}
	for _, bad := range []run.HostSet{{PID: -1}, {PID: 2147483000}, {ReserveMiB: -5}, {Adapter: &plan.HostAdapter{}}, {Adapter: &plan.HostAdapter{Name: "x", MemoryMiB: -1}}} {
		if err := s.SetHost(bad); err == nil {
			t.Errorf("accepted %+v", bad)
		}
	}
}

func TestProfileHostFields(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "profile.json")
	os.WriteFile(p, []byte(`{"game":"g","game_reserve_mib":1500,"host_pid":4321,"host_adapter":{"name":"Intel(R) Iris(R) Xe Graphics","vendor_id":32902,"memory_mib":128}}`), 0o644)
	pr, err := run.LoadProfile(p)
	if err != nil {
		t.Fatal(err)
	}
	if pr.HostPID != 4321 || pr.HostAdapter == nil || pr.HostAdapter.VendorID != 0x8086 || pr.HostAdapter.MemoryMiB != 128 || pr.GameReserveMiB != 1500 {
		t.Fatalf("profile %+v %+v", pr, pr.HostAdapter)
	}
	b, _ := json.Marshal(pr)
	var m map[string]any
	json.Unmarshal(b, &m)
	if m["host_pid"] != float64(4321) || m["host_adapter"] == nil {
		t.Fatalf("json %s", b)
	}
	// the profile's adapter reaches the reserve: the host renders on the integrated adapter
	e := testkit.New(t)
	f := &senseFake{}
	e.Prof = pr
	e.Prof.Slots, e.Prof.MinSlots = 3, 1
	e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 65536, 8192
	s := sensedSup(t, e, f, nil)
	rep, err := s.GPUReport(context.Background(), true)
	if err != nil {
		t.Fatal(err)
	}
	if rep.HostAdapter != 1 || rep.HostIntegrated != "yes" || rep.Reserve.HostMiB != 0 || rep.Reserve.HostBasis != "other-adapter" {
		t.Fatalf("report %+v", rep.GPUSummary)
	}
	if f.hostPID() != 4321 {
		t.Fatalf("host pid %d", f.hostPID())
	}
}

func TestGPUReportCached(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	f := &senseFake{}
	s := sensedSup(t, e, f, func(c *run.Config) { c.WatchEvery = time.Hour })
	ctx := context.Background()
	if _, err := s.GPUReport(ctx, true); err != nil {
		t.Fatal(err)
	}
	n := f.calls.Load()
	for i := 0; i < 5; i++ {
		if _, err := s.GPUReport(ctx, false); err != nil {
			t.Fatal(err)
		}
	}
	if f.calls.Load() != n {
		t.Fatalf("sensed again within 5 s: %d -> %d", n, f.calls.Load())
	}
	s.GPUReport(ctx, true)
	if f.calls.Load() == n {
		t.Fatal("refresh did not sense")
	}
}
