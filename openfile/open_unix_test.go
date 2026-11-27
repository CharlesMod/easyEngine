//go:build !windows

package openfile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The opener is a script that logs what it is given and refuses paths ending in ".nohandler",
// as xdg-open does for a type nothing handles.
func TestOpenFallsBackToFolder(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "log")
	script := filepath.Join(dir, "opener")
	body := "#!/bin/sh\necho \"$1\" >> '" + log + "'\ncase \"$1\" in *.nohandler) exit 3;; esac\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	saved := opener
	opener = script
	defer func() { opener = saved }()

	settings := filepath.Join(dir, "settings.json")
	odd := filepath.Join(dir, "x.nohandler")
	for _, p := range []string{settings, odd} {
		if err := os.WriteFile(p, []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct{ path, want string }{
		{settings, settings}, // opened as it is
		{odd, dir},           // no handler: its folder
		{filepath.Join(dir, "missing.json"), dir},  // missing: its folder
		{filepath.Join(dir, "gone", "a.json"), ""}, // no folder either: an error
	} {
		got, err := Open(c.path)
		if got != c.want || (c.want == "") != (err != nil) {
			t.Errorf("Open(%s) = %q, %v; want %q", c.path, got, err, c.want)
		}
	}
	b, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Join([]string{settings, odd, dir, dir}, "\n") + "\n"
	if string(b) != want {
		t.Errorf("opener saw:\n%s\nwant:\n%s", b, want)
	}
}
