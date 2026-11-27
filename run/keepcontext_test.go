//go:build !windows

package run_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

// A settings.json edit of keep_context takes effect live through ReloadSettings; SetSettings
// refuses a bad value and persists a good one.
func TestKeepContextReloadsLive(t *testing.T) {
	e := testkit.New(t)
	p := filepath.Join(e.Data, "settings.json")
	s := newSup(t, e)
	if s.KeepContext() != "on" || s.Settings().KeepContext != "on" {
		t.Fatalf("default %q", s.KeepContext())
	}
	f := settings.Defaults()
	f.KeepContext = "always"
	os.WriteFile(p, settings.Encode(f), 0o600)
	s.ReloadSettings()
	if s.KeepContext() != "always" || s.Settings().KeepContext != "always" {
		t.Fatalf("after reload %q", s.KeepContext())
	}
	if err := s.SetSettings(run.Settings{Mode: "local", KeepContext: "bogus"}); err == nil {
		t.Fatal("bogus keep_context accepted")
	}
	if err := s.SetSettings(run.Settings{Mode: "local", KeepContext: "off"}); err != nil || s.KeepContext() != "off" {
		t.Fatalf("set off: %v %q", err, s.KeepContext())
	}
	if g, err := settings.Read(p); err != nil || g.KeepContext != "off" {
		t.Fatalf("persisted %q %v", g.KeepContext, err)
	}
}
