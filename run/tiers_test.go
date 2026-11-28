package run_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/internal/testkit"
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
