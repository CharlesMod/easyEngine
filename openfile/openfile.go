// Package openfile opens a file with the desktop's default handler: xdg-open on Linux, open on
// macOS, ShellExecute "open" on Windows. When the file is missing or the system cannot open it
// (no program for its type), it opens the file's folder instead.
package openfile

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Open opens path with the default handler, falling back to its folder. It waits for the
// opener (xdg-open, open), which returns once the handler has started, so call it from a
// goroutine when the caller must not wait. It reports which one it opened.
func Open(path string) (opened string, err error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if st, statErr := os.Stat(abs); statErr == nil && !st.IsDir() {
		if err = open(abs); err == nil {
			return abs, nil
		}
	}
	dir := abs
	if st, statErr := os.Stat(abs); statErr != nil || !st.IsDir() {
		dir = filepath.Dir(abs)
	}
	if _, statErr := os.Stat(dir); statErr != nil {
		return "", errors.Join(err, fmt.Errorf("openfile: %w", statErr))
	}
	if derr := open(dir); derr != nil {
		return "", errors.Join(err, derr)
	}
	return dir, nil
}

// Folder opens a folder in the file manager.
func Folder(dir string) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	return open(abs)
}
