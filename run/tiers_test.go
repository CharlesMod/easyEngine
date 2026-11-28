package run_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/run"
)

// tierEnv is the fake install with a ladder of four models on disk: m1 (quality 50), a Campfire
// (100), an opt-in model that fits (110) and an opt-in dense Bonfire (120), with Bonfire selected.
// bonfireMiB is Bonfire's fixed footprint; every other model takes 5000 MiB of the 16384 MiB card.
func tierEnv(t *testing.T) (*testkit.Env, func() []string) {
	e := testkit.New(t)
	e.InstallFake()
	add := func(m catalog.Model) {
		body := []byte("fake gguf for " + m.ID)
		if err := os.WriteFile(filepath.Join(e.Data, "models", m.File), body, 0o644); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(body)
		m.SHA256, m.Size, m.URL = hex.EncodeToString(sum[:]), int64(len(body)), "http://invalid.example/"+m.File
		e.Cat.Models = append(e.Cat.Models, m)
	}
	add(catalog.Model{ID: "camp", Name: "Gemma 4 26B RP", Codename: "Campfire", File: "camp.gguf", MoE: true, Quality: 100})
	add(catalog.Model{ID: "pick", Name: "A picked model", Codename: "Torch", File: "pick.gguf", OptIn: true, Quality: 110})
	add(catalog.Model{ID: "bon", Name: "Qwen 27B", Codename: "Bonfire", File: "bon.gguf", OptIn: true, Quality: 120})
	b, _ := json.Marshal(map[string]any{"mode": "local", "model": "bon", "backend": "fake", "kv_type": "q4_0"})
	if err := os.WriteFile(filepath.Join(e.Data, "state.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	starts := func() []string {
		var files []string
		for _, l := range e.Starts() {
			for _, f := range []string{"m1.gguf", "camp.gguf", "pick.gguf", "bon.gguf"} {
				if strings.Contains(l, string(filepath.Separator)+f) {
					files = append(files, f)
				}
			}
		}
		return files
	}
	return e, starts
}

func tierSeed(bonfireMiB float64) func(m catalog.Model, path, kv string) (fit.Footprint, error) {
	return func(m catalog.Model, path, kv string) (fit.Footprint, error) {
		fp := fit.Footprint{FixedMiB: 5000, PerTokenMiB: 0.05, ScratchMiB: 500}
		if m.ID == "bon" {
			fp.FixedMiB = bonfireMiB // dense: no expert layer to move off the card
		}
		return fp, nil
	}
}

// A selected dense model the card cannot hold whole is never split between card and RAM: the best
// installed tier below it that is not opt-in serves instead (Campfire, not the opt-in model that
// would fit), the status says so, and the selection stays Bonfire.
func TestDenseTooBigStepsDownToTheNextTier(t *testing.T) {
	testkit.SkipWindows(t)
	e, starts := tierEnv(t)
	cfg := e.Config()
	cfg.Seed = tierSeed(20000)
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	st := s.Status()
	if st.Model != "camp" || st.ModelCodename != "Campfire" || st.ModelName != "Gemma 4 26B RP" {
		t.Fatalf("serving %q %q %q, want camp Campfire", st.Model, st.ModelCodename, st.ModelName)
	}
	if !st.ModelFallback || !strings.Contains(st.ModelNote, "Bonfire (Qwen 27B)") || !strings.Contains(st.ModelNote, "never split") || !strings.Contains(st.ModelNote, "Campfire (Gemma 4 26B RP) serves instead") {
		t.Fatalf("fallback %v note %q", st.ModelFallback, st.ModelNote)
	}
	if got := starts(); len(got) != 1 || got[0] != "camp.gguf" {
		t.Fatalf("started %v, want only camp.gguf (Bonfire never split, never the opt-in pick)", got)
	}
	if st.Sampler.Basis != "local" {
		t.Fatalf("sampler basis %q", st.Sampler.Basis)
	}
	b, err := os.ReadFile(filepath.Join(e.Data, "state.json"))
	if err != nil || !strings.Contains(string(b), `"model": "bon"`) {
		t.Fatalf("the selection must stay Bonfire: %s %v", b, err)
	}
}

// A selected model that would fit the empty card but not beside what others hold is not stepped
// down: the engine waits for the card, as before.
func TestBusyCardWaitsRatherThanStepping(t *testing.T) {
	testkit.SkipWindows(t)
	e, starts := tierEnv(t)
	e.SetOthers(9000)
	cfg := e.Config()
	cfg.Seed = tierSeed(9000)
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "error", 10*time.Second)
	if st := s.Status(); !strings.Contains(st.Message, "does not fit") || st.Model != "bon" || st.ModelCodename != "Bonfire" {
		t.Fatalf("status %q %q %q", st.Message, st.Model, st.ModelCodename)
	}
	if got := starts(); len(got) != 0 {
		t.Fatalf("started %v while the card was busy", got)
	}
	e.SetOthers(0)
	s.Nudge("the game ended")
	testkit.WaitPhase(t, s, "ready", 10*time.Second)
	if st := s.Status(); st.Model != "bon" || st.ModelFallback {
		t.Fatalf("serving %q fallback %v, want bon", st.Model, st.ModelFallback)
	}
	if got := starts(); len(got) != 1 || got[0] != "bon.gguf" {
		t.Fatalf("started %v", got)
	}
}

// With no tier below that is installed and not opt-in, a too-big model waits as before.
func TestNoTierBelowWaits(t *testing.T) {
	testkit.SkipWindows(t)
	e, starts := tierEnv(t)
	for i := range e.Cat.Models {
		if e.Cat.Models[i].ID != "bon" {
			e.Cat.Models[i].OptIn = true
		}
	}
	cfg := e.Config()
	cfg.Seed = tierSeed(20000)
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "error", 10*time.Second)
	if st := s.Status(); !strings.Contains(st.Message, "does not fit") || st.Model != "bon" || st.ModelFallback {
		t.Fatalf("status %q %q %v", st.Message, st.Model, st.ModelFallback)
	}
	if got := starts(); len(got) != 0 {
		t.Fatalf("started %v", got)
	}
}

// stepped is the tier env with Bonfire too big for the card, ready on Campfire by a step.
func stepped(t *testing.T) (*testkit.Env, *run.Supervisor, func() []string) {
	e, starts := tierEnv(t)
	cfg := e.Config()
	cfg.Seed = tierSeed(20000)
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if st := s.Status(); st.Model != "camp" {
		t.Fatalf("serving %q, want camp by a step", st.Model)
	}
	return e, s, starts
}

// lastModelFact is the body of the last model fact s said.
func lastModelFact(t *testing.T, s *run.Supervisor) map[string]any {
	var last map[string]any
	for _, ev := range testkit.Said(t, s, 0, 0, "model") {
		if ev.Name == "model" {
			last = nil
			if err := json.Unmarshal(ev.Body, &last); err != nil {
				t.Fatal(err)
			}
		}
	}
	return last
}

// The step says what is selected beside what serves: status model_selected, and the model fact's
// serving.
func TestStepSaysSelectedAndServing(t *testing.T) {
	testkit.SkipWindows(t)
	_, s, _ := stepped(t)
	st := s.Status()
	if st.ModelSelected != "bon" || st.Model != "camp" {
		t.Fatalf("model_selected %q model %q", st.ModelSelected, st.Model)
	}
	b, _ := json.Marshal(st)
	if !strings.Contains(string(b), `"model_selected":"bon"`) {
		t.Fatalf("json %s", b)
	}
	if f := lastModelFact(t, s); f["model"] != "bon" || f["serving"] != "camp" {
		t.Fatalf("model fact %v", f)
	}
}

// The file of the tier that serves under a step is in use: it cannot be deleted (nor the
// selection's).
func TestStepServedFileIsInUse(t *testing.T) {
	testkit.SkipWindows(t)
	_, s, _ := stepped(t)
	if _, err := s.DeleteModel("camp", ""); !errors.Is(err, run.ErrInUse) {
		t.Fatalf("delete of the serving tier: %v, want ErrInUse", err)
	}
	if _, err := s.DeleteModel("bon", ""); !errors.Is(err, run.ErrInUse) {
		t.Fatalf("delete of the selection: %v, want ErrInUse", err)
	}
	if _, err := s.DeleteModel("m1", ""); err != nil {
		t.Fatalf("a model neither selected nor serving can go: %v", err)
	}
}

// Selecting the tier that already serves for a step moves the selection and ends the step with no
// restart.
func TestSelectingTheServedTierDoesNotRestart(t *testing.T) {
	testkit.SkipWindows(t)
	e, s, starts := stepped(t)
	if err := s.Install("camp", ""); err != nil {
		t.Fatal(err)
	}
	st := s.Status()
	if st.Model != "camp" || st.ModelSelected != "camp" || st.ModelFallback || st.Phase != "ready" {
		t.Fatalf("after selecting camp: model %q selected %q fallback %v phase %q", st.Model, st.ModelSelected, st.ModelFallback, st.Phase)
	}
	if got := starts(); len(got) != 1 {
		t.Fatalf("starts %v: selecting the serving tier must not restart it", got)
	}
	if ss := run.LoadState(e.Data); ss.Model != "camp" {
		t.Fatalf("state.json model %q", ss.Model)
	}
	if f := lastModelFact(t, s); f["model"] != "camp" || f["serving"] != "camp" {
		t.Fatalf("model fact %v", f)
	}
}

// A model that fits this card, with its expert layers in the machine's memory, but not in the
// memory free today, is not stepped down: the engine waits as before.
func TestShortFreeMemoryWaitsRatherThanStepping(t *testing.T) {
	testkit.SkipWindows(t)
	e, starts := tierEnv(t)
	for i := range e.Cat.Models {
		if e.Cat.Models[i].ID == "bon" {
			e.Cat.Models[i].MoE = true
		}
	}
	cfg := e.Config()
	seed := tierSeed(20000)
	cfg.Seed = func(m catalog.Model, path, kv string) (fit.Footprint, error) {
		fp, err := seed(m, path, kv)
		if m.ID == "bon" { // the card holds it with most expert layers in RAM: up to 16000 MiB of them
			fp.MoE, fp.ExpertLayers, fp.ExpertMiBPerLayer = true, 40, 400
		}
		return fp, err
	}
	sensed := cfg.Probe
	cfg.Probe = func(ctx context.Context) (probe.Hardware, error) {
		hw, err := sensed(ctx)
		hw.RAMAvailMiB = 3000 // the machine has 32 GiB, little of it free today
		return hw, err
	}
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "error", 10*time.Second)
	if st := s.Status(); !strings.Contains(st.Message, "does not fit") || st.Model != "bon" || st.ModelFallback {
		t.Fatalf("status %q %q %v", st.Message, st.Model, st.ModelFallback)
	}
	if got := starts(); len(got) != 0 {
		t.Fatalf("started %v: a passing memory shortage is not a card too small", got)
	}
}
