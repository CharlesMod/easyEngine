//go:build !windows

package run_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/probe"
	"hermesengine/probe/gpusense"
	"hermesengine/run"
)

// twoCards senses two equal NVIDIA cards, the first busy: the plan picks the second (the emptier).
func twoCards(ctx context.Context, ours []int, host int) gpusense.Picture {
	return gpusense.Picture{Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UsedMiB: 9000, UUID: "GPU-aaaa"},
		{Index: 1, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UsedMiB: 100, UUID: "GPU-bbbb"},
	}, Processes: []gpusense.Process{{PID: 4242, Name: "other", Adapter: 0, DedicatedMiB: 9000}}, Sources: map[string]string{"fake": "ok"}}
}

// cudaSup installs the fake server as a CUDA build on a two-card machine and starts it.
func cudaSup(t *testing.T, e *testkit.Env) *run.Supervisor {
	t.Helper()
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	e.InstallFake()
	src := filepath.Join(e.Data, "backends", "fake")
	dst := filepath.Join(e.Data, "backends", "cuda-linux-x64")
	if err := os.Rename(src, dst); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(map[string]any{"mode": "local", "model": "m1", "backend": "cuda-linux-x64", "kv_type": "q4_0"})
	if err := os.WriteFile(filepath.Join(e.Data, "state.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	e.Llama["FAKELLAMA_DEVICES"] = "CUDA0: Fake GPU (16384 MiB, 7000 MiB free);CUDA1: Fake GPU (16384 MiB, 16000 MiB free)"
	cfg := e.Config()
	cfg.Sense = twoCards
	cfg.Probe = func(ctx context.Context) (probe.Hardware, error) {
		return probe.Hardware{OS: runtime.GOOS, Arch: runtime.GOARCH, Cores: 8, RAMTotalMiB: 32768, RAMAvailMiB: 24000, DiskFreeMiB: 100000,
			GPUs: []probe.GPU{{Index: 0, Vendor: "nvidia", Name: "Fake GPU", VRAMTotalMiB: 16384, VRAMUsedMiB: 9000}, {Index: 1, Vendor: "nvidia", Name: "Fake GPU", VRAMTotalMiB: 16384, VRAMUsedMiB: 100}}}, nil
	}
	cfg.Others = func(context.Context, probe.GPU, []int) (int, error) { return 0, nil }
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return s
}

// On two CUDA cards llama-server is held to the planned one, and the status and plan say so.
func TestStartPinsPlannedCard(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := cudaSup(t, e)
	start := e.Starts()[0]
	if testkit.Field(start, "cvd") != "GPU-bbbb" || !strings.Contains(start, "--split-mode none") {
		t.Fatalf("start %s", start)
	}
	st := s.Status()
	if st.GPU == nil || st.GPU.EngineAdapter != 1 || st.Device == nil || !st.Device.Pinned || st.Device.Device != "CUDA0" || st.Device.Adapter != 1 {
		t.Fatalf("status device %+v gpu %+v", st.Device, st.GPU)
	}
	p, err := s.Plan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if p.Device == nil || !p.Device.Pinned || p.Device.Env[0] != "CUDA_VISIBLE_DEVICES=GPU-bbbb" {
		t.Fatalf("plan device %+v", p.Device)
	}
	logs, _ := s.Logs("engine", 200)
	if !strings.Contains(logs, "device: adapter 1 (Fake GPU) only: CUDA_VISIBLE_DEVICES=GPU-bbbb") {
		t.Fatalf("no device line:\n%s", logs)
	}
}

// The planned card cannot be confirmed (the UUID leaves nothing, two cards share a name): the
// server starts as it always has, and the status says why.
func TestStartUnmatchedCardStartsAnyway(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.Llama["FAKELLAMA_DEVICES_PINNED"] = "none"
	s := cudaSup(t, e)
	start := e.Starts()[0]
	if testkit.Field(start, "cvd") != "" || strings.Contains(start, "--split-mode") || strings.Contains(start, "--device") {
		t.Fatalf("start %s", start)
	}
	st := s.Status()
	if st.Device == nil || st.Device.Pinned || !strings.Contains(st.Device.Note, "match equally") || !strings.Contains(st.Device.Note, "0 CUDA devices") {
		t.Fatalf("status device %+v", st.Device)
	}
}
