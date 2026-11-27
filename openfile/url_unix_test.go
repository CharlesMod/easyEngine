//go:build !windows

package openfile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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

// Popup falls back to the ordinary opener when the default browser is not Chromium family.
func TestPopupFallsBackToOpener(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "log")
	script := filepath.Join(dir, "opener")
	os.WriteFile(script, []byte("#!/bin/sh\necho \"$1\" >> '"+log+"'\n"), 0o755)
	saved := opener
	opener = script
	defer func() { opener = saved }()
	// A PATH with an xdg-settings that names Firefox, and an XDG_DATA_HOME holding its entry.
	bin := filepath.Join(dir, "bin")
	os.MkdirAll(bin, 0o755)
	os.WriteFile(filepath.Join(bin, "xdg-settings"), []byte("#!/bin/sh\necho firefox.desktop\n"), 0o755)
	data := filepath.Join(dir, "data")
	os.MkdirAll(filepath.Join(data, "applications"), 0o755)
	os.WriteFile(filepath.Join(data, "applications", "firefox.desktop"), []byte("[Desktop Entry]\nExec=firefox %u\n"), 0o644)
	t.Setenv("PATH", bin+":/usr/bin:/bin")
	t.Setenv("XDG_DATA_HOME", data)
	t.Setenv("XDG_DATA_DIRS", dir+"/none")

	addr := "http://127.0.0.1:47480/chat/?mini=1"
	if err := Popup(addr); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(log); strings.TrimSpace(string(b)) != addr {
		t.Fatalf("fallback opener got %q", b)
	}
}
