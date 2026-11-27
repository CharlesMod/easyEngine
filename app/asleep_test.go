//go:build !windows

package app_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// The resident app starts with no model loaded and says so; a client's attach is the event that
// loads it (the fake server's start is heard in its start log, by the kernel's word).
func TestResidentStartsAsleepUntilAnAttach(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	r := startResident(t, app.Options{
		DataDir: e.Data, Linger: time.Hour, LingerSet: true,
		Configure: func(c *run.Config) {
			ec := e.Config()
			ec.Parents, ec.Yield, ec.Group, ec.StartSleeping = c.Parents, c.Yield, c.Group, c.StartSleeping
			*c = ec
		},
	})
	var st struct {
		Phase   string `json:"phase"`
		Message string `json:"message"`
	}
	r.get(t, "/engine/v1/status", &st) // Run has said its phase before the engine reported its port
	if st.Phase != "sleeping" || st.Message != "Model not loaded" {
		t.Fatalf("phase %q %q, want sleeping with no model loaded", st.Phase, st.Message)
	}
	log := filepath.Join(e.Ctl, "starts.log")
	if _, err := os.Stat(log); err == nil {
		t.Fatalf("a model server started before any client: %v", e.Starts())
	}
	started, stop, err := host.Changes(log)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	par := newSleeper(t)
	defer par.kill()
	if err := app.PostAttach(app.EngineTarget(r.dir, "127.0.0.1", r.port), par.pid); err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(15 * time.Second): // watchdog for silence: the attach never loaded the model
		t.Fatal("the attach did not load the model")
	}
	if n := len(e.Starts()); n != 1 {
		t.Fatalf("%d model starts, want one: %v", n, e.Starts())
	}
}
