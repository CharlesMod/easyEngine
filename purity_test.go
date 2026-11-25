package engine_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The engine is game-agnostic: no source file may name the game or its vocabulary.
// The banned words are built from pieces so this file does not trip itself (and is skipped).
func TestPurity(t *testing.T) {
	banned := []string{"rim" + "world", "pa" + "wn", "col" + "on"}
	n := 0
	err := filepath.WalkDir(".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" || p == "dist" || p == "docs" { // dist holds compiled binaries; docs may name games as examples
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Base(p) == "purity_test.go" || p == "README.md" { // the README lists the games that use the engine
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		n++
		low := strings.ToLower(string(b))
		lowName := strings.ToLower(p)
		for _, w := range banned {
			if strings.Contains(low, w) {
				t.Errorf("%s contains %q", p, w)
			}
			if strings.Contains(lowName, w) {
				t.Errorf("file name %s contains %q", p, w)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("walked no files")
	}
}
