package run_test

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
	"github.com/CharlesMod/easyEngine/run"
)

// roomEnv is a supervisor that is not started (no model, no loop) with a room watch of its own
// kind: the card's device nodes are files in a temp dir, and the probe and the sense open them
// as nvidia-smi opens the card, n times each.
func roomEnv(t *testing.T, opens int) (s *run.Supervisor, node string, probes *atomic.Int32, readDone chan struct{}) {
	if runtime.GOOS != "linux" {
		t.Skip("the opens of a device are heard on linux only")
	}
	dev := t.TempDir()
	node = filepath.Join(dev, "nvidia0")
	if err := os.WriteFile(node, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	touch := func() {
		for i := 0; i < opens; i++ {
			if f, err := os.Open(node); err == nil {
				f.Close()
			}
		}
	}
	readDone = make(chan struct{}, 1024)
	// registered before the supervisor, so that they are undone after its Shutdown
	t.Cleanup(run.SetRoomDevices([]string{filepath.Join(dev, "nvidia[0-9]*")}))
	t.Cleanup(run.SetRoomReadDone(func() {
		select {
		case readDone <- struct{}{}:
		default:
		}
	}))
	e := testkit.New(t)
	cfg := e.Config()
	cfg.RoomEvery = time.Hour // the watchdog never comes round in a test
	probes = new(atomic.Int32)
	inner := cfg.Probe
	cfg.Probe = func(ctx context.Context) (probe.Hardware, error) {
		probes.Add(1)
		touch()
		return inner(ctx)
	}
	cfg.Sense = func(ctx context.Context, ours []int, host int) gpusense.Picture {
		touch()
		return gpusense.Picture{Adapters: []gpusense.Adapter{}, Processes: []gpusense.Process{}, Sources: map[string]string{"test": "nothing"}}
	}
	s, err := run.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Shutdown)
	return s, node, probes, readDone
}

func readEnds(t *testing.T, c chan struct{}, what string) {
	t.Helper()
	select {
	case <-c:
	case <-time.After(10 * time.Second): // watchdog for silence: a read that never ends fails the test
		t.Fatalf("%s: no read ended", what)
	}
}

// One read of the room opens the card's device several times (its probe and its sense). Those
// opens are the engine's own: they make no further read, however late the kernel's events are
// read; and a stranger's open, at any time, makes exactly one.
func TestRoomReadMakesNoReadOfItsOwn(t *testing.T) {
	s, node, probes, readDone := roomEnv(t, 4)
	s.RoomArm()
	readEnds(t, readDone, "the arming's first read") // arming reads twice, so the baseline holds
	readEnds(t, readDone, "the arming's second read")
	base := probes.Load()
	const rounds = 12
	for i := 0; i < rounds; i++ {
		f, err := os.Open(node) // a stranger's open
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
		readEnds(t, readDone, "after a stranger's open")
	}
	// a read that owed another would be reading nonstop by now
	if n := probes.Load() - base; n != rounds {
		t.Fatalf("%d reads for %d strangers' opens", n, rounds)
	}
	if len(readDone) > 0 {
		t.Fatal("a read the engine's own opens made")
	}
}

// arm and disarm are safe in any order and twice over; each arming's reading goroutine ends with
// its disarm, so the goroutines do not grow with the model loads.
func TestRoomDisarmEndsTheOpensGoroutine(t *testing.T) {
	ends := make(chan struct{}, 1024)
	t.Cleanup(run.SetRoomOpensEnd(func() { ends <- struct{}{} }))
	s, _, _, _ := roomEnv(t, 1)
	s.RoomDisarm() // before any arm
	s.RoomArm()
	s.RoomArm() // twice
	s.RoomDisarm()
	s.RoomDisarm()
	readEnds(t, ends, "first disarm") // the same wait, for the goroutine's end
	before := runtime.NumGoroutine()
	const cycles = 60
	for i := 0; i < cycles; i++ {
		s.RoomArm()
		s.RoomDisarm()
		readEnds(t, ends, "disarm")
	}
	if len(ends) != 0 {
		t.Fatal("more goroutines ended than were armed")
	}
	// the reads the armings started end by themselves; a leak would be one goroutine per cycle
	if after := runtime.NumGoroutine(); after > before+cycles/4 {
		t.Fatalf("goroutines grew from %d to %d over %d arm/disarm cycles", before, after, cycles)
	}
}
