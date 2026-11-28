package run_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

// external puts the engine in external mode before it starts (the keeper's settings.json).
func (s *site) external(t *testing.T) {
	t.Helper()
	f := settings.Defaults()
	f.Mode = "external"
	f.External.BaseURL = s.srv.URL + "/v1"
	if err := settings.Write(settings.Path(s.e.Data), f); err != nil {
		t.Fatal(err)
	}
}

// noJob says the hub holds no job fact: no install job (the agent's own included) runs.
func noJob(s *run.Supervisor) bool {
	_, ok := s.Hub().Fact("job", "")
	return !ok
}

func hasError(st run.Status, part string) bool {
	for _, e := range st.Errors {
		if strings.Contains(e, part) {
			return true
		}
	}
	return false
}

// In external mode no model is ever installed, so no install job ever installed the agent: it
// stayed not_installed and nothing answered the game. The engine now installs it by itself at
// start: uv and the wheel, no model, no model server.
func TestExternalModeInstallsTheAgentAtStart(t *testing.T) {
	s := newSite(t)
	s.external(t)
	sup := s.sup(t)
	testkit.WaitFor(t, 30*time.Second, "the agent to be installed and running in external mode", func() bool {
		return sup.Status().Agent.Phase == "running" && len(lines(s.agent)) >= 1
	})
	st := sup.Status()
	if st.Phase != "external" || st.Progress != nil {
		t.Fatalf("phase %q progress %+v after the agent install, want external and none", st.Phase, st.Progress)
	}
	if hasError(st, "Agent install") {
		t.Fatalf("errors: %v", st.Errors)
	}
	if l := lines(s.agent)[0]; !strings.Contains(l, "url=http://127.0.0.1:4242/v1") || !strings.Contains(l, "mode=openai") {
		t.Fatalf("agent line: %s", l)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "runtime", "uv")); err != nil {
		t.Fatal("uv not installed")
	}
	// the agent alone: no model and no model server were fetched
	if m, _ := filepath.Glob(filepath.Join(s.e.Data, "models", "*")); len(m) != 0 {
		t.Fatalf("a model was installed in external mode: %v", m)
	}
	if got := run.LoadState(s.e.Data); got.Model != "" || got.Backend != "" {
		t.Fatalf("state names a model or a build in external mode: %+v", got)
	}
	// a second start finds it installed and runs uv no more
	n := len(lines(s.uvLog))
	sup.Shutdown()
	sup2 := s.sup(t)
	testkit.WaitFor(t, 20*time.Second, "the agent to run again after a restart", func() bool {
		return sup2.Status().Agent.Phase == "running"
	})
	if got := len(lines(s.uvLog)); got != n {
		t.Fatalf("uv ran again for an installed agent: %v", lines(s.uvLog)[n:])
	}
}

// A switch to external mode while the engine runs installs the agent too.
func TestSwitchToExternalInstallsTheAgent(t *testing.T) {
	s := newSite(t)
	sup := s.sup(t)
	if p := sup.Phase(); p != "not_installed" {
		t.Fatalf("phase %s", p)
	}
	if a := sup.Status().Agent.Phase; a != "not_installed" {
		t.Fatalf("agent %s before any install", a)
	}
	if err := sup.SetSettings(run.Settings{Mode: "external", External: run.External{BaseURL: s.srv.URL + "/v1"}}); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 30*time.Second, "the agent to be installed and running after the switch", func() bool {
		st := sup.Status()
		return st.Agent.Phase == "running" && st.Phase == "external"
	})
}

// A failed agent install is said and blocks nothing: the engine stays in external mode, and the
// next settings change tries again.
func TestAgentInstallFailureDoesNotBlockExternal(t *testing.T) {
	s := newSite(t)
	s.external(t)
	wheel, err := os.ReadFile(s.wheelP)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(s.wheelP); err != nil { // made by newSite under the test's temp dir
		t.Fatal(err)
	}
	sup := s.sup(t)
	testkit.WaitFor(t, 20*time.Second, "the failed agent install to be said", func() bool {
		return hasError(sup.Status(), "Agent install failed")
	})
	st := sup.Status()
	if st.Phase != "external" {
		t.Fatalf("phase %q after a failed agent install, want external", st.Phase)
	}
	if st.Agent.Phase != "error" || !strings.Contains(st.Agent.Message, "install failed") {
		t.Fatalf("agent status %+v", st.Agent)
	}
	if err := os.WriteFile(s.wheelP, wheel, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := sup.SetSettings(run.Settings{Mode: "external", External: run.External{BaseURL: s.srv.URL + "/v1"}}); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 30*time.Second, "the agent to be installed on the next settings change", func() bool {
		st := sup.Status()
		return st.Agent.Phase == "running" && st.Phase == "external"
	})
}

// A failed agent install waits for a word like the engine's other waits, not only for the next
// settings change: a request at the door, a client's nudge or the network coming back tries again.
func TestAgentInstallFailureTriesAgainOnAWord(t *testing.T) {
	for _, tc := range []struct {
		name string
		word func(*run.Supervisor)
	}{
		{"request", func(s *run.Supervisor) { s.Begin()() }},
		{"nudge", func(s *run.Supervisor) { s.Nudge("attach") }},
		{"network", func(s *run.Supervisor) { s.Network() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newSite(t)
			s.external(t)
			wheel, err := os.ReadFile(s.wheelP)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Remove(s.wheelP); err != nil { // made by newSite under the test's temp dir
				t.Fatal(err)
			}
			sup := s.sup(t)
			testkit.WaitFor(t, 20*time.Second, "the failed agent install to wait for a word", func() bool {
				a := sup.Status().Agent
				if a.Phase == "error" && strings.Contains(a.Message, "waiting for a request") {
					// the status says it waits only once a request counts as a word for it
					if sup.Listeners() == 0 {
						t.Errorf("status %q seen with no listener counted: a request now would be lost", a.Message)
					}
					return true
				}
				return false
			})
			if err := os.WriteFile(s.wheelP, wheel, 0o644); err != nil {
				t.Fatal(err)
			}
			tc.word(sup)
			testkit.WaitFor(t, 30*time.Second, "the agent installed and running after the word", func() bool {
				st := sup.Status()
				return st.Agent.Phase == "running" && st.Phase == "external"
			})
		})
	}
}

// An engine that starts asleep (the resident app) still installs a missing agent at start, in
// external mode and beside an installed model; the model stays asleep, and the agent runs from
// the first start on.
func TestStartSleepingStillInstallsTheAgent(t *testing.T) {
	asleep := func(c *run.Config) { c.StartSleeping = true }
	t.Run("external", func(t *testing.T) {
		s := newSite(t)
		s.external(t)
		sup := s.supWith(t, asleep)
		testkit.WaitFor(t, 30*time.Second, "the agent installed while asleep", func() bool {
			return len(lines(s.uvLog)) > 0 && noJob(sup)
		})
		if hasError(sup.Status(), "Agent install") {
			t.Fatalf("errors: %v", sup.Status().Errors)
		}
		sup.Wake()
		testkit.WaitFor(t, 30*time.Second, "the agent running after the first start", func() bool {
			return sup.Status().Agent.Phase == "running"
		})
	})
	t.Run("local", func(t *testing.T) {
		s := newSite(t)
		sup := s.sup(t)
		if err := sup.Install("dl", ""); err != nil {
			t.Fatal(err)
		}
		testkit.WaitFor(t, 30*time.Second, "the install to finish", func() bool {
			st := sup.Status()
			return st.Phase == "ready" && st.Agent.Phase == "running"
		})
		sup.Shutdown()
		if err := os.RemoveAll(filepath.Join(s.e.Data, "runtime", "venvs")); err != nil { // the engine's own, under the test's temp dir
			t.Fatal(err)
		}
		n := len(lines(s.uvLog))
		starts := len(s.e.Starts())
		sup2 := s.supWith(t, asleep)
		testkit.WaitFor(t, 30*time.Second, "the agent reinstalled while the model sleeps", func() bool {
			return len(lines(s.uvLog)) > n && noJob(sup2) && sup2.Phase() == "sleeping"
		})
		if st := sup2.Status(); st.Phase != "sleeping" || len(s.e.Starts()) != starts {
			t.Fatalf("the model did not stay asleep: phase %q, starts %v", st.Phase, s.e.Starts())
		}
	})
}

// A data dir whose model is installed but whose agent was removed gets its agent back at start,
// and the model still loads.
func TestLocalStartReinstallsARemovedAgent(t *testing.T) {
	s := newSite(t)
	sup := s.sup(t)
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 30*time.Second, "the install to finish", func() bool {
		st := sup.Status()
		return st.Phase == "ready" && st.Agent.Phase == "running"
	})
	sup.Shutdown()
	venvs := filepath.Join(s.e.Data, "runtime", "venvs") // the engine's own, under the test's temp dir
	if err := os.RemoveAll(venvs); err != nil {
		t.Fatal(err)
	}
	sup2 := s.sup(t)
	testkit.WaitFor(t, 30*time.Second, "the agent to be reinstalled and the model ready", func() bool {
		st := sup2.Status()
		return st.Phase == "ready" && st.Agent.Phase == "running"
	})
	if st := sup2.Status(); hasError(st, "Agent install") {
		t.Fatalf("errors: %v", st.Errors)
	}
}
