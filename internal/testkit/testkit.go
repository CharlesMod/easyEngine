// Package testkit builds the fake llama-server and a ready-made data dir for tests.
package testkit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"hermesengine/catalog"
	"hermesengine/fit"
	"hermesengine/probe"
	"hermesengine/run"
)

var (
	buildOnce sync.Once
	buildDir  string
	buildPath string
	buildErr  error
)

// Cleanup removes the temp dir that holds the fake llama-server (call from TestMain).
func Cleanup() {
	if buildDir != "" {
		os.RemoveAll(buildDir) // a dir this package created with MkdirTemp
	}
}

// FakeLlama builds internal/fakellama once per test binary and returns its path.
func FakeLlama(t testing.TB) string {
	t.Helper()
	buildOnce.Do(func() {
		buildDir, buildErr = os.MkdirTemp("", "fakellama-build-")
		if buildErr != nil {
			return
		}
		_, file, _, _ := runtime.Caller(0)
		root := filepath.Join(filepath.Dir(file), "..", "..")
		name := "llama-server"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		buildPath = filepath.Join(buildDir, name)
		cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), "build", "-o", buildPath, "./internal/fakellama")
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOFLAGS=-mod=mod", "GOPROXY=off")
		if out, err := cmd.CombinedOutput(); err != nil {
			buildErr = fmt.Errorf("build fakellama: %v\n%s", err, out)
		}
	})
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	return buildPath
}

// Env is a data dir with the fake backend and a model "m1" installed.
type Env struct {
	T      *testing.T
	Data   string
	Ctl    string
	Llama  map[string]string // extra FAKELLAMA_* environment for the server
	Agent  []string          // extra environment for the agent
	Cat    *catalog.Catalog
	Prof   run.Profile
	used   int32
	others int32
}

const fakeNvidiaSMI = `#!/bin/sh
# fake nvidia-smi: compute apps come from the vram.<pid> files the fake llama-server writes
CTL="$(dirname "$0")/.."
case "$*" in
  *query-compute-apps*)
    for f in "$CTL"/vram.*; do
      [ -f "$f" ] || continue
      pid="${f##*.}"
      kill -0 "$pid" 2>/dev/null || continue
      echo "$pid, $(cat "$f")"
    done ;;
  *query-gpu*)
    sum=0
    for f in "$CTL"/vram.*; do
      [ -f "$f" ] || continue
      pid="${f##*.}"
      kill -0 "$pid" 2>/dev/null || continue
      sum=$((sum + $(cat "$f")))
    done
    others=0; [ -f "$CTL/others" ] && others=$(cat "$CTL/others")
    echo "0, Fake GPU, 16384, $((sum + others))" ;;
esac
`

// New makes the env: temp dirs, fake nvidia-smi first on PATH, the fake backend and model installed.
func New(t *testing.T) *Env {
	t.Helper()
	root := t.TempDir()
	e := &Env{T: t, Data: filepath.Join(root, "data"), Ctl: filepath.Join(root, "ctl"), Llama: map[string]string{}}
	must(t, os.MkdirAll(filepath.Join(e.Ctl, "bin"), 0o755))
	must(t, os.MkdirAll(e.Data, 0o755))
	if runtime.GOOS != "windows" {
		must(t, os.WriteFile(filepath.Join(e.Ctl, "bin", "nvidia-smi"), []byte(fakeNvidiaSMI), 0o755))
		t.Setenv("PATH", filepath.Join(e.Ctl, "bin")+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	e.Llama["FAKELLAMA_VRAM_BASE_MIB"] = "5500"
	e.Llama["FAKELLAMA_VRAM_PER_TOKEN"] = "0.05"
	e.Prof = run.Profile{Game: "TestGame", Slots: 3, MinSlots: 1, GameReserveMiB: 0}
	e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 65536, 8192
	return e
}

// InstallFake puts the fake backend (backends/fake) and model m1 in place and selects them.
func (e *Env) InstallFake() {
	t := e.T
	bin := FakeLlama(t)
	dst := filepath.Join(e.Data, "backends", "fake", filepath.Base(bin))
	CopyFile(t, bin, dst, 0o755)
	model := []byte("not a real gguf, the fake server ignores it")
	must(t, os.MkdirAll(filepath.Join(e.Data, "models"), 0o755))
	must(t, os.WriteFile(filepath.Join(e.Data, "models", "m1.gguf"), model, 0o644))
	sum := sha256.Sum256(model)
	e.Cat = &catalog.Catalog{Version: 1, Models: []catalog.Model{{ID: "m1", Name: "Model one", File: "m1.gguf", URL: "http://invalid.example/m1", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(model)), MoE: false, Quality: 50}}}
	st := map[string]any{"mode": "local", "model": "m1", "backend": "fake", "kv_type": "q4_0"}
	b, _ := json.Marshal(st)
	must(t, os.WriteFile(filepath.Join(e.Data, "state.json"), b, 0o600))
}

// SetUsed sets the total VRAM in use as the probe reports it; SetOthers what other processes use.
func (e *Env) SetUsed(mib int)   { atomic.StoreInt32(&e.used, int32(mib)) }
func (e *Env) SetOthers(mib int) { atomic.StoreInt32(&e.others, int32(mib)) }

// Config is the run config for this env, with fast timings.
func (e *Env) Config() run.Config {
	env := []string{"FAKELLAMA_CTL=" + e.Ctl}
	for k, v := range e.Llama {
		env = append(env, k+"="+v)
	}
	cfg := run.Config{
		DataDir: e.Data, Profile: e.Prof, Catalog: e.Cat, Version: "test",
		ExtraEnv: env, AgentEnv: e.Agent,
		Probe: func(ctx context.Context) (probe.Hardware, error) {
			return probe.Hardware{OS: runtime.GOOS, Arch: runtime.GOARCH, Cores: 8, RAMTotalMiB: 32768, RAMAvailMiB: 24000,
				GPUs: []probe.GPU{{Vendor: "nvidia", Name: "Fake GPU", VRAMTotalMiB: 16384, VRAMUsedMiB: int(atomic.LoadInt32(&e.used))}}}, nil
		},
		Others: func(ctx context.Context, g probe.GPU, pids []int) (int, error) {
			return int(atomic.LoadInt32(&e.others)), nil
		},
		Seed: func(m catalog.Model, path, kv string) (fit.Footprint, error) {
			return fit.Footprint{FixedMiB: 5000, PerTokenMiB: 0.05, ScratchMiB: 500}, nil
		},
		WatchEvery: 40 * time.Millisecond, GrowIdle: 10 * time.Minute, HealthPoll: 20 * time.Millisecond,
		LoadTimeout: 20 * time.Second, StopGrace: 500 * time.Millisecond, HealthMisses: 3, IdleChecks: 2,
		AgentBackoffBase: 40 * time.Millisecond, AgentBackoffMax: 200 * time.Millisecond, AgentStable: time.Hour,
		CmdTimeout: 30 * time.Second,
	}
	return cfg
}

// Starts returns the fake server's start log (one line per start attempt).
func (e *Env) Starts() []string {
	b, err := os.ReadFile(filepath.Join(e.Ctl, "starts.log"))
	if err != nil {
		return nil
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

// Touch creates, Remove deletes a control file.
func (e *Env) Touch(name string)  { must(e.T, os.WriteFile(filepath.Join(e.Ctl, name), nil, 0o644)) }
func (e *Env) Remove(name string) { os.Remove(filepath.Join(e.Ctl, name)) }

// StartSup builds a supervisor from the config, runs it and shuts it down at cleanup.
func (e *Env) StartSup(cfg run.Config) *run.Supervisor {
	s, err := run.New(cfg)
	if err != nil {
		e.T.Fatal(err)
	}
	e.T.Cleanup(s.Shutdown)
	s.Run()
	return s
}

// Field extracts key=value from a starts.log line.
func Field(line, key string) string {
	for _, f := range strings.Fields(line) {
		if strings.HasPrefix(f, key+"=") {
			return strings.TrimPrefix(f, key+"=")
		}
	}
	return ""
}

// WaitFor polls until cond is true or the timeout passes.
func WaitFor(t testing.TB, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	end := time.Now().Add(timeout)
	for time.Now().Before(end) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for: %s", timeout, what)
}

// WaitPhase waits for a phase.
func WaitPhase(t testing.TB, s *run.Supervisor, phase string, timeout time.Duration) {
	t.Helper()
	end := time.Now().Add(timeout)
	for time.Now().Before(end) {
		if s.Phase() == phase {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for phase %q (now %q: %s)", phase, s.Status().Phase, s.Status().Message)
}

// CopyFile copies a file.
func CopyFile(t testing.TB, src, dst string, mode os.FileMode) {
	t.Helper()
	must(t, os.MkdirAll(filepath.Dir(dst), 0o755))
	in, err := os.Open(src)
	must(t, err)
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	must(t, err)
	_, err = io.Copy(out, in)
	must(t, err)
	must(t, out.Close())
}

func must(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// SkipWindows skips tests that need a POSIX shell.
func SkipWindows(t testing.TB) {
	if runtime.GOOS == "windows" {
		t.Skip("needs a POSIX shell")
	}
}
