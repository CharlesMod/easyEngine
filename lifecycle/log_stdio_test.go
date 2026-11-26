package lifecycle

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRotateStdioLog(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, StdioLogName)

	// below the cap: left alone
	small := make([]byte, StdioLogCap-1)
	os.WriteFile(p, small, 0o644)
	RotateStdioLog(p)
	if st, err := os.Stat(p); err != nil || st.Size() != int64(len(small)) {
		t.Fatalf("a log under the cap moved: %v %v", st, err)
	}
	if _, err := os.Stat(p + ".1"); !os.IsNotExist(err) {
		t.Error(".1 made for a small log")
	}

	// at the cap: becomes .1, replacing an older copy
	os.WriteFile(p+".1", []byte("old"), 0o644)
	big := make([]byte, StdioLogCap)
	os.WriteFile(p, big, 0o644)
	RotateStdioLog(p)
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Error("the full log stayed in place")
	}
	if st, err := os.Stat(p + ".1"); err != nil || st.Size() != StdioLogCap {
		t.Errorf(".1 = %v %v, want the full log", st, err)
	}

	// no file: no panic, nothing made
	os.Remove(p + ".1")
	RotateStdioLog(p)
	if _, err := os.Stat(p + ".1"); !os.IsNotExist(err) {
		t.Error("rotating a missing log made a file")
	}
}

func TestStdioLogCeiling(t *testing.T) {
	if c := StdioLogCeiling(); c < 2*StdioLogCap || c%4096 != 0 {
		t.Errorf("ceiling %d must cover two files at the cap and be block-aligned", c)
	}
}
