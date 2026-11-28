package run_test

import (
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/run"
)

// A first-run swap to a target the card cannot hold never steps down a tier: the swap fails at
// the fit, the target is never started, no lower tier is started in between, and the starter
// comes back with one load.
func TestSwapToATooBigTargetFailsFastBackOnTheStarter(t *testing.T) {
	s := newStage(t)
	mid := []byte(strings.Repeat("midtier", 7000)) // installed, between the starter and the target
	if err := os.MkdirAll(filepath.Join(s.e.Data, "models"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.e.Data, "models", "mid.gguf"), mid, 0o644); err != nil {
		t.Fatal(err)
	}
	s.cat.Models = append(s.cat.Models, catalog.Model{ID: "mid", Name: "Mid", File: "mid.gguf", URL: s.hs.URL + "/mid.gguf",
		SHA256: sum(mid), Size: int64(len(mid)), Quality: 70})
	s.e.Cat = s.cat
	cfg := s.e.Config()
	cfg.SwapDrain = 5 * time.Second
	cfg.DiskFree = func(dir string) (int, error) { return int(atomic.LoadInt32(&s.free)), nil }
	cfg.Seed = func(m catalog.Model, path, kv string) (fit.Footprint, error) {
		fp := fit.Footprint{FixedMiB: 5000, PerTokenMiB: 0.05, ScratchMiB: 500}
		if m.ID == "target" {
			fp.FixedMiB = 20000 // dense, bigger than the 16384 MiB card
		}
		return fp, nil
	}
	sup := s.e.StartSup(cfg)
	s.sup.Store(sup)
	sup.SetPort(4242)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "starter serving", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "starter" })
	s.release()
	st := waitStatus(t, sup, "the swap failed, back on the starter", func(st run.Status) bool {
		return st.Phase == "ready" && st.Upgrade != nil && st.Upgrade.State == "failed"
	})
	if st.Model != "starter" || st.ModelSelected != "starter" {
		t.Fatalf("serving %q selected %q, want the starter", st.Model, st.ModelSelected)
	}
	var files []string
	for _, l := range s.e.Starts() {
		for _, f := range []string{"starter.gguf", "mid.gguf", "target.gguf"} {
			if strings.Contains(l, string(filepath.Separator)+f) {
				files = append(files, f)
			}
		}
	}
	if strings.Join(files, ",") != "starter.gguf,starter.gguf" {
		t.Fatalf("starts %v, want the starter, then the starter again (no target, no lower tier)", files)
	}
}
