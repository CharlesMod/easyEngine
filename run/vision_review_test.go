//go:build !windows

package run_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// The model server's environment loses every LLAMA_ARG_* of the parent (the engine gives the
// flags), and the engine log names the dropped variables once.
func TestParentLlamaArgEnvNeverReachesTheChild(t *testing.T) {
	names := []string{"LLAMA_ARG_HF_REPO", "LLAMA_ARG_MODEL_URL", "LLAMA_ARG_DOCKER_REPO", "LLAMA_ARG_MMPROJ"}
	for _, k := range names {
		t.Setenv(k, "x")
	}
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	log, err := os.ReadFile(filepath.Join(e.Data, "logs", "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(log), "left out of the parent's environment"); n != 1 {
		t.Errorf("the log names the dropped variables %d times", n)
	}
	for _, k := range names {
		if !strings.Contains(string(log), k) {
			t.Errorf("the log does not name %s", k)
		}
	}
	for _, l := range e.Starts() {
		if strings.Contains(l, "envarg=") && !strings.Contains(l, "envarg=none") {
			t.Errorf("a LLAMA_ARG_* reached the child: %s", l)
		}
	}
}
