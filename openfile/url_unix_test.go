//go:build !windows

package openfile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestOpenURL(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "log")
	script := filepath.Join(dir, "opener")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho \"$1\" >> '"+log+"'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	saved := opener
	opener = script
	defer func() { opener = saved }()

	if err := OpenURL("http://127.0.0.1:47480/chat/#/chat/abc"); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(log)
	if strings.TrimSpace(string(b)) != "http://127.0.0.1:47480/chat/#/chat/abc" {
		t.Fatalf("opener got %q", b)
	}
	for _, bad := range []string{"file:///etc/passwd", "/etc/passwd", "javascript:alert(1)", "http://"} {
		if OpenURL(bad) == nil {
			t.Errorf("%q must be refused", bad)
		}
	}
}

// Window falls back to the ordinary opener when the default browser is not one known to take
// --new-window, and starts the browser itself when it is.
func TestWindowFallsBackToOpener(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "log")
	script := filepath.Join(dir, "opener")
	os.WriteFile(script, []byte("#!/bin/sh\necho \"$1\" >> '"+log+"'\n"), 0o755)
	saved := opener
	opener = script
	defer func() { opener = saved }()
	// A PATH with an xdg-settings that names a browser, and an XDG_DATA_HOME holding its entry.
	bin := filepath.Join(dir, "bin")
	os.MkdirAll(bin, 0o755)
	os.WriteFile(filepath.Join(bin, "xdg-settings"), []byte("#!/bin/sh\necho web.desktop\n"), 0o755)
	data := filepath.Join(dir, "data")
	os.MkdirAll(filepath.Join(data, "applications"), 0o755)
	entry := filepath.Join(data, "applications", "web.desktop")
	t.Setenv("PATH", bin+":/usr/bin:/bin")
	t.Setenv("XDG_DATA_HOME", data)
	t.Setenv("XDG_DATA_DIRS", dir+"/none")
	addr := "http://127.0.0.1:47480/chat/"

	// An unknown browser (Epiphany): the ordinary opener gets the address.
	os.WriteFile(entry, []byte("[Desktop Entry]\nExec=epiphany %u\n"), 0o644)
	if err := Window(addr); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(log); strings.TrimSpace(string(b)) != addr {
		t.Fatalf("fallback opener got %q", b)
	}
	// A browser that is named but cannot be started: the same fallback.
	os.Remove(log)
	os.WriteFile(entry, []byte("[Desktop Entry]\nExec="+dir+"/missing/firefox %u\n"), 0o644)
	if err := Window(addr); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(log); strings.TrimSpace(string(b)) != addr {
		t.Fatalf("launch failure: opener got %q", b)
	}
	// A browser that starts: it gets --new-window <addr>, and the opener is not used.
	os.Remove(log)
	fx := filepath.Join(dir, "firefox")
	started := filepath.Join(dir, "started")
	os.WriteFile(fx, []byte("#!/bin/sh\necho \"$@\" > '"+started+"'\n"), 0o755)
	os.WriteFile(entry, []byte("[Desktop Entry]\nExec="+fx+" %u\n"), 0o644)
	if err := Window(addr); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second) // network timeout: the child runs on its own
	for time.Now().Before(deadline) {
		if b, _ := os.ReadFile(started); strings.TrimSpace(string(b)) == "--new-window "+addr {
			break
		}
		time.Sleep(10 * time.Millisecond) // named moment: poll for the child's file; no event to wait on
	}
	if b, _ := os.ReadFile(started); strings.TrimSpace(string(b)) != "--new-window "+addr {
		t.Fatalf("browser got %q", b)
	}
	if _, err := os.Stat(log); err == nil {
		t.Fatal("the opener ran as well")
	}
	if Window("file:///etc/passwd") == nil {
		t.Fatal("a non-http address must be refused")
	}
}
