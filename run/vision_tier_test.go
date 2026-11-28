//go:build !windows

package run_test

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// Vision under a tier step (run/tiers.go): the projector is the serving tier's, never the
// selected model's.

// giveProjector writes a projector file for model id into models/ and sets it in the catalog.
func giveProjector(t *testing.T, e *testkit.Env, id string, loadMiB int, low int) (*catalog.Projector, string) {
	t.Helper()
	body := []byte("fake projector for " + id)
	sum := sha256.Sum256(body)
	p := &catalog.Projector{File: id + "-mmproj.gguf", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(body)),
		LoadMiB: loadMiB, ImageMiB: 200, MaxImageTokens: 512, Detail: catalog.Detail{Low: low, High: 512, Max: 1024}}
	path := filepath.Join(e.Data, "models", p.File)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	for i := range e.Cat.Models {
		if e.Cat.Models[i].ID == id {
			e.Cat.Models[i].Projector = p
			return p, path
		}
	}
	t.Fatalf("no model %s", id)
	return nil, ""
}

// steppedVision is the tier env with Bonfire too big, vision wanted, Campfire's projector given.
func steppedVision(t *testing.T, campLoad int, bonHas bool) (*testkit.Env, *run.Supervisor, string, string) {
	testkit.SkipWindows(t)
	e, _ := tierEnv(t)
	_, campPath := giveProjector(t, e, "camp", campLoad, 111)
	bonPath := ""
	if bonHas {
		_, bonPath = giveProjector(t, e, "bon", 800, 999)
	}
	cfg := e.Config()
	cfg.Seed = tierSeed(20000)
	cfg.Vision = true
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, campPath, bonPath
}

func hasMMPath(line, path string) bool { return strings.Contains(line, "--mmproj "+path) }

// 4a. The selection has no projector, the serving tier has one: the start loads the tier's, the
// storage protects it, and the look's detail table is the tier's.
func TestVisionStepLoadsTheTiersProjector(t *testing.T) {
	e, s, campPath, _ := steppedVision(t, 800, false)
	st := s.Status()
	if st.Model != "camp" || st.ModelSelected != "bon" || !st.Vision {
		t.Fatalf("model %q selected %q vision %v note %q", st.Model, st.ModelSelected, st.Vision, st.VisionNote)
	}
	if starts := e.Starts(); len(starts) != 1 || !hasMMPath(starts[0], campPath) {
		t.Fatalf("starts: %v", starts)
	}
	if _, err := s.DeleteModel("", filepath.Base(campPath)); !errors.Is(err, run.ErrInUse) {
		t.Fatalf("delete of the serving tier's projector: %v, want ErrInUse", err)
	}
	if p, ok := s.LookSpec(); !ok || p.Detail.Low != 111 || p.File != "camp-mmproj.gguf" {
		t.Fatalf("look spec %+v ok %v: want the tier's projector", p, ok)
	}
}

// 4b. Both have a projector: the tier's is loaded, the selection's is not, and the selection's
// file is not protected as a projector in use.
func TestVisionStepPrefersTheTiersOverTheSelections(t *testing.T) {
	e, s, campPath, bonPath := steppedVision(t, 800, true)
	st := s.Status()
	if st.Model != "camp" || !st.Vision {
		t.Fatalf("model %q vision %v note %q", st.Model, st.Vision, st.VisionNote)
	}
	starts := e.Starts()
	if len(starts) != 1 || !hasMMPath(starts[0], campPath) || strings.Contains(starts[0], bonPath) {
		t.Fatalf("starts: %v", starts)
	}
	if p, ok := s.LookSpec(); !ok || p.Detail.Low != 111 {
		t.Fatalf("look spec %+v ok %v: want the tier's low edge", p, ok)
	}
	if _, err := s.DeleteModel("", filepath.Base(campPath)); !errors.Is(err, run.ErrInUse) {
		t.Fatalf("delete of the tier's projector: %v, want ErrInUse", err)
	}
	if _, err := s.DeleteModel("", filepath.Base(bonPath)); err != nil {
		t.Fatalf("the selection's projector is not the served one's, delete: %v", err)
	}
}

// 4c. The tier fits only without its projector: text on the tier, and the note says so.
func TestVisionStepTierFitsOnlyWithoutProjector(t *testing.T) {
	e, s, _, _ := steppedVision(t, 20000, false)
	st := s.Status()
	if st.Model != "camp" || st.ModelSelected != "bon" || st.Vision || !strings.Contains(st.VisionNote, "does not fit") {
		t.Fatalf("model %q selected %q vision %v note %q", st.Model, st.ModelSelected, st.Vision, st.VisionNote)
	}
	if starts := e.Starts(); len(starts) != 1 || strings.Contains(starts[0], "--mmproj") {
		t.Fatalf("starts: %v", starts)
	}
	if _, ok := s.LookSpec(); ok {
		t.Fatal("a look spec with no projector loaded")
	}
}
