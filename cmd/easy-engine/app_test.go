//go:build linux

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/settings"
)

// noTrayEnv gives the child no session bus, so it can show no tray icon and touch nothing of
// the real desktop; its data dir is a temp dir.
func noTrayEnv(t *testing.T, home string) []string {
	return []string{
		"EASY_ENGINE_HOME=" + home,
		"DBUS_SESSION_BUS_ADDRESS=unix:path=" + filepath.Join(t.TempDir(), "no-bus"),
		"XDG_CONFIG_HOME=" + t.TempDir(),
	}
}

func TestAppCommandRunsWithoutATrayAndFlagsAreNotWritten(t *testing.T) {
	bin := buildBin(t)
	home := t.TempDir()
	env := noTrayEnv(t, home)
	c := exec.Command(bin, "app", "--port", "47633")
	c.Env = append(os.Environ(), env...)
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- c.Wait() }()
	t.Cleanup(func() { c.Process.Kill() })
	deadline := time.Now().Add(20 * time.Second)
	var port int
	for {
		if p, ok := app.Running(home); ok {
			port = p
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the app did not start")
		}
		time.Sleep(50 * time.Millisecond) // watchdog for silence
	}
	if port != 47633 {
		t.Fatalf("port %d: the flag did not override", port)
	}
	// settings.json is whole, with the defaults: the flag was not written back.
	f, err := settings.Read(filepath.Join(home, "settings.json"))
	if err != nil || !f.Equal(settings.Defaults()) {
		t.Fatalf("%+v %v", f, err)
	}
	// A second app says so and ends with 0.
	out, code := runBin(t, bin, env, "app", "--port", "47633")
	if code != 0 || !strings.Contains(out, "already running") {
		t.Fatalf("second app: %q exit %d", out, code)
	}
	// stop ends the first, with 0, and the lock goes.
	if out, code := runBin(t, bin, env, "stop"); code != 0 {
		t.Fatalf("stop: %q exit %d", out, code)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("app exit: %v", err)
		}
	case <-time.After(20 * time.Second): // watchdog for silence
		t.Fatal("the app did not end")
	}
	if _, err := os.Stat(filepath.Join(home, "engine.lock")); err == nil {
		t.Fatal("lock left behind")
	}
}

// The listener comes from settings.json when the app starts without flags.
func TestAppReadsTheListenerFromSettings(t *testing.T) {
	bin := buildBin(t)
	home := t.TempDir()
	env := noTrayEnv(t, home)
	f := settings.Defaults()
	f.Listen.Port = 47643
	if err := settings.Write(filepath.Join(home, "settings.json"), f); err != nil {
		t.Fatal(err)
	}
	c := exec.Command(bin, "app")
	c.Env = append(os.Environ(), env...)
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { c.Wait(); close(done) }()
	t.Cleanup(func() { c.Process.Kill(); <-done })
	deadline := time.Now().Add(20 * time.Second)
	for {
		if p, ok := app.Running(home); ok {
			if p != 47643 {
				t.Fatalf("port %d, want 47643", p)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the app did not start")
		}
		time.Sleep(50 * time.Millisecond) // watchdog for silence
	}
	runBin(t, bin, env, "stop")
}

// serve is unchanged: it still needs a parent, and no settings file is read for its listener.
func TestServeStillNeedsAParent(t *testing.T) {
	bin := buildBin(t)
	out, code := runBin(t, bin, noTrayEnv(t, t.TempDir()), "serve")
	if code != 2 || !strings.Contains(out, "--parent-pid") {
		t.Fatalf("%q exit %d", out, code)
	}
}
