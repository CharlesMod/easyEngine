package run

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestCorruptStateSetAside(t *testing.T) {
	dir := t.TempDir()
	orig := []byte("{not json")
	if err := os.WriteFile(filepath.Join(dir, "state.json"), orig, 0o600); err != nil {
		t.Fatal(err)
	}
	st, note := loadState(dir)
	if !strings.Contains(note, "set aside as") || st.Mode != "local" {
		t.Fatalf("note=%q mode=%q", note, st.Mode)
	}
	if _, err := os.Stat(filepath.Join(dir, "state.json")); err == nil {
		t.Fatal("corrupt file still in place")
	}
	m, _ := filepath.Glob(filepath.Join(dir, "state.json.corrupt-*"))
	if len(m) != 1 {
		t.Fatalf("aside files: %v", m)
	}
	if b, _ := os.ReadFile(m[0]); string(b) != string(orig) {
		t.Fatalf("bytes changed: %q", b)
	}
	// a second corrupt file is kept beside the first
	os.WriteFile(filepath.Join(dir, "state.json"), []byte("x"), 0o600)
	loadState(dir)
	m, _ = filepath.Glob(filepath.Join(dir, "state.json.corrupt-*"))
	if len(m) != 2 {
		t.Fatalf("aside files: %v", m)
	}
}

func TestCorruptStateReachesErrors(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "state.json"), []byte("{oops"), 0o600)
	s, err := New(Config{DataDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Shutdown()
	var found bool
	for _, e := range s.Status().Errors {
		if strings.Contains(e, "was set aside as") {
			found = true
		}
	}
	if !found {
		t.Fatalf("errors: %v", s.Status().Errors)
	}
}

func TestMissingStateIsSilent(t *testing.T) {
	st, note := loadState(t.TempDir())
	if note != "" || st.Mode != "local" {
		t.Fatalf("note=%q", note)
	}
}

func TestConcurrentWriteState(t *testing.T) {
	dir := t.TempDir()
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				st := State{Mode: "local", LingerS: i*100 + j}
				st.defaults()
				if err := writeState(dir, st); err != nil {
					t.Error(err)
				}
			}
		}(i)
	}
	wg.Wait()
	_, note := loadState(dir)
	if note != "" {
		t.Fatal(note)
	}
	if m, _ := filepath.Glob(filepath.Join(dir, "state.json.tmp*")); len(m) != 0 {
		t.Fatalf("temp files left: %v", m)
	}
	if fi, _ := os.Stat(filepath.Join(dir, "state.json")); fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", fi.Mode())
	}
}
