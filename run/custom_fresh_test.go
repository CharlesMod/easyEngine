//go:build !windows

package run_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/run"
)

// A fresh machine: custom/use installs llama.cpp and the agent (no catalog model), then serves the file.
func TestCustomUseOnFreshMachine(t *testing.T) {
	s := newSite(t)
	sup := s.sup(t)
	if sup.Phase() != "not_installed" {
		t.Fatalf("phase %s", sup.Phase())
	}
	p := testkit.WriteGGUF(t, t.TempDir(), "mine.gguf", true)
	if err := sup.UseCustom(p); err != nil {
		t.Fatal(err)
	}
	if err := sup.UseCustom(p); err != run.ErrBusy {
		t.Fatalf("second use: %v (want ErrBusy)", err)
	}
	testkit.WaitFor(t, 30*time.Second, "custom file served", func() bool {
		return sup.Phase() == "ready" && strings.HasSuffix(strings.Join(s.e.Starts(), "\n"), "-a custom:mine")
	})
	// The install job was reported as one: the phase went through downloading. A small archive
	// from a local server is down in less than one look at the status, so the phase is read from
	// the engine's log, which has a line for every phase it set.
	if lg, err := sup.Logs("engine", 500); err != nil || !strings.Contains(lg, "phase downloading: Installing the model server") {
		t.Errorf("no downloading phase in the engine log (%v):\n%s", err, lg)
	}
	st := run.LoadState(s.e.Data)
	if st.Model != "custom" || st.Backend != "cpu-test" || st.Custom == nil || st.Custom.Path != p {
		t.Fatalf("state = %+v", st)
	}
	if !strings.Contains(strings.Join(s.e.Starts(), "\n"), "-m "+p) {
		t.Fatalf("starts: %v", s.e.Starts())
	}
	if ents, _ := os.ReadDir(filepath.Join(s.e.Data, "models")); len(ents) != 0 {
		t.Fatalf("models/ has files: %v", ents)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "runtime", "uv")); err != nil {
		t.Fatal("agent runtime not installed")
	}
	testkit.WaitFor(t, 5*time.Second, "agent to start", func() bool { return len(lines(s.agent)) >= 1 })
}

// An engine restart with state.model="custom" and a backend resumes serving the file.
func TestCustomResumeAfterRestart(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	p := testkit.WriteGGUF(t, t.TempDir(), "mine.gguf", true)
	fi, _ := os.Stat(p)
	b, _ := os.ReadFile(filepath.Join(e.Data, "state.json"))
	var m map[string]any
	json.Unmarshal(b, &m)
	m["model"] = "custom"
	m["custom"] = map[string]any{"path": p, "name": "mine", "size": fi.Size(), "mtime": fi.ModTime().Unix()}
	b, _ = json.Marshal(m)
	os.WriteFile(filepath.Join(e.Data, "state.json"), b, 0o600)
	sup := e.StartSup(e.Config())
	testkit.WaitPhase(t, sup, "ready", 15*time.Second)
	if st := sup.Status(); st.Model != "custom:mine" || !strings.HasSuffix(strings.Join(e.Starts(), "\n"), "-a custom:mine") {
		t.Fatalf("status %s starts %v", st.Model, e.Starts())
	}
}
