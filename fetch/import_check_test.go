package fetch

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// A hard link needs no space: beforeCopy must not be asked.
func TestImportCheckSkipsBeforeCopyWhenLinked(t *testing.T) {
	data, sum := payload(50000)
	dir := t.TempDir()
	src, dest := filepath.Join(dir, "src.bin"), filepath.Join(dir, "m", "dest.bin")
	os.WriteFile(src, data, 0o644)
	called := false
	err := ImportCheck(src, dest, sum, func(int64) error { called = true; return errors.New("no") })
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Error("beforeCopy was asked although the file was hard-linked")
	}
	a, _ := os.Stat(src)
	b, _ := os.Stat(dest)
	if !os.SameFile(a, b) {
		t.Skip("filesystem did not link; link path not exercised")
	}
}

// A link that fails (here: the .import temp name is taken by a non-empty directory, which the
// import cannot remove, on any OS and without root) falls to the copy, and beforeCopy is asked
// the file's size first. Its refusal is returned unchanged and nothing is written at dest.
func TestImportCheckRefusesCopy(t *testing.T) {
	data, sum := payload(50000)
	dir := t.TempDir()
	src, dest := filepath.Join(dir, "src.bin"), filepath.Join(dir, "m", "dest.bin")
	os.WriteFile(src, data, 0o644)
	blocker := dest + ".import"
	if err := os.MkdirAll(filepath.Join(blocker, "x"), 0o755); err != nil {
		t.Fatal(err)
	}
	refuse := errors.New("not enough disk space for the copy")
	var asked []int64
	err := ImportCheck(src, dest, sum, func(n int64) error { asked = append(asked, n); return refuse })
	if !errors.Is(err, refuse) {
		t.Fatalf("err = %v, want the beforeCopy error", err)
	}
	if len(asked) != 1 || asked[0] != int64(len(data)) {
		t.Errorf("beforeCopy asked %v, want one call with %d", asked, len(data))
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Errorf("dest exists after a refused copy: %v", err)
	}
	// without a check the same setup is not refused by the check (it fails later, on the taken name)
	asked = nil
	if err := ImportCheck(src, dest, sum, func(n int64) error { asked = append(asked, n); return nil }); err == nil || len(asked) != 1 {
		t.Errorf("an accepting check must let the copy proceed to its own error: err=%v asked=%v", err, asked)
	}
}

// A real cross-device import (Linux only, when /dev/shm is a different filesystem): the check
// is asked, accepts, and the file is copied byte for byte.
func TestImportCheckCopiesAcrossDevices(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("needs /dev/shm")
	}
	shm, err := os.MkdirTemp("/dev/shm", "importcheck-")
	if err != nil {
		t.Skip("no /dev/shm")
	}
	t.Cleanup(func() { os.RemoveAll(shm) }) // a dir this test made
	data, sum := payload(30000)
	src := filepath.Join(shm, "src.bin")
	os.WriteFile(src, data, 0o644)
	destDir := t.TempDir()
	if err := os.Link(src, filepath.Join(destDir, "probe")); err == nil {
		t.Skip("temp dir is on the same filesystem as /dev/shm")
	}
	dest := filepath.Join(destDir, "dest.bin")
	var asked int64
	if err := ImportCheck(src, dest, sum, func(n int64) error { asked = n; return nil }); err != nil {
		t.Fatal(err)
	}
	if asked != int64(len(data)) {
		t.Errorf("asked %d, want %d", asked, len(data))
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, data) {
		t.Error("copy differs")
	}
	if _, err := os.Stat(dest + ".import"); !os.IsNotExist(err) {
		t.Error(".import left behind")
	}
}
