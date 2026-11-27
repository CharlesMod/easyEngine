package lifecycle

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLogSinkReopensAfterFailure(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("directory permissions do not bind root")
	}
	old := MaxLogSize
	MaxLogSize = 10
	defer func() { MaxLogSize = old }()
	dir := t.TempDir()
	path := filepath.Join(dir, "x.log")
	s, err := OpenLog(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Write([]byte("0123456789ab\n")) // over the cap: the next write rotates
	// An unwritable directory makes the reopen fail.
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	s.Write([]byte("lost while the dir is read-only\n"))
	os.Chmod(dir, 0o755)
	s.Write([]byte("after\n"))
	b, _ := os.ReadFile(path)
	if !strings.Contains(string(b), "after") {
		t.Fatalf("log after a failed reopen: %q", b)
	}
}
