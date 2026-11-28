package run

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// pruneBuildDirs removes only the other builds of one backend id, by exact names, and nothing
// past a symlink. Each case says what must still be on disk.
func TestPruneBuildDirsBounds(t *testing.T) {
	d := t.TempDir()
	mk := func(rel string, marker bool) {
		p := filepath.Join(d, rel)
		os.MkdirAll(p, 0o755)
		os.WriteFile(filepath.Join(p, serverBinName()), []byte("x"), 0o755)
		if marker {
			os.WriteFile(filepath.Join(p, ".installed"), []byte("sha"), 0o644)
		}
	}
	file := func(rel string) {
		os.MkdirAll(filepath.Dir(filepath.Join(d, rel)), 0o755)
		os.WriteFile(filepath.Join(d, rel), []byte("x"), 0o644)
	}
	outside := t.TempDir()
	os.WriteFile(filepath.Join(outside, "keep.txt"), []byte("x"), 0o644)

	mk("backends/cuda-linux-x64", true)                                    // main's layout: goes
	mk("backends/cuda-linux-x64@b1", true)                                 // an older tag: goes
	mk("backends/cuda-linux-x64@b2-old", false)                            // a failed build of an old tag, no marker: goes
	mk("backends/cuda-linux-x64@b12651-easy1", true)                       // the running build: stays
	mk("backends/cuda-linux-x64@b12652-easy1", false)                      // the staged build: stays
	mk("backends/vulkan-linux-x64@b1", true)                               // another id with @ and a marker: stays
	mk("backends/vulkan-linux-x64", true)                                  // another id: stays
	mk("backends/cuda-linux-x64-extra", true)                              // a name that only starts with the id: stays
	file("backends/notes.txt")                                             // a user's file: stays
	file("backends/cuda-linux-x64@b3")                                     // a file where a build dir would be: stays
	os.Symlink(outside, filepath.Join(d, "backends", "cuda-linux-x64@b4")) // a symlinked entry: stays, and what it points to
	file("downloads/cuda-linux-x64.tar.gz.part")                           // main's old partial: goes
	file("downloads/cudart-linux-x64.tar.gz")                              // main's old extra archive: goes
	file("downloads/cuda-linux-x64@b12652-easy1.tar.gz.part")              // the move's partial, new naming: stays
	file("downloads/vulkan-linux-x64.tar.gz.part")                         // another id's: stays
	file("downloads/model.gguf.part")                                      // a model's: stays

	keep := map[string]bool{"cuda-linux-x64@b12651-easy1": true, "cuda-linux-x64@b12652-easy1": true}
	got := pruneBuildDirs(d, "cuda-linux-x64", keep, []string{"cudart-linux-x64"})
	want := []string{"backends/cuda-linux-x64", "backends/cuda-linux-x64@b1", "backends/cuda-linux-x64@b2-old",
		"downloads/cuda-linux-x64.tar.gz.part", "downloads/cudart-linux-x64.tar.gz"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("removed %v\nwant    %v", got, want)
	}
	for _, rel := range want {
		if _, err := os.Lstat(filepath.Join(d, rel)); err == nil {
			t.Errorf("%s still there", rel)
		}
	}
	for _, rel := range []string{"backends/cuda-linux-x64@b12651-easy1/.installed", "backends/cuda-linux-x64@b12652-easy1/" + serverBinName(),
		"backends/vulkan-linux-x64@b1", "backends/vulkan-linux-x64", "backends/cuda-linux-x64-extra", "backends/notes.txt",
		"backends/cuda-linux-x64@b3", "backends/cuda-linux-x64@b4", "downloads/cuda-linux-x64@b12652-easy1.tar.gz.part",
		"downloads/vulkan-linux-x64.tar.gz.part", "downloads/model.gguf.part"} {
		if _, err := os.Lstat(filepath.Join(d, rel)); err != nil {
			t.Errorf("%s was removed", rel)
		}
	}
	if _, err := os.Stat(filepath.Join(outside, "keep.txt")); err != nil {
		t.Error("a file behind a symlinked entry was removed")
	}

	// a symlinked backends/ and downloads/: nothing is entered
	d2 := t.TempDir()
	target := t.TempDir()
	os.MkdirAll(filepath.Join(target, "cuda-linux-x64"), 0o755)
	os.WriteFile(filepath.Join(target, "cuda-linux-x64.tar.gz"), []byte("x"), 0o644)
	os.Symlink(target, filepath.Join(d2, "backends"))
	os.Symlink(target, filepath.Join(d2, "downloads"))
	if got := pruneBuildDirs(d2, "cuda-linux-x64", nil, nil); len(got) != 0 {
		t.Fatalf("entered a symlinked dir: %v", got)
	}
	if _, err := os.Stat(filepath.Join(target, "cuda-linux-x64")); err != nil {
		t.Fatal("removed behind a symlinked backends/")
	}
	if _, err := os.Stat(filepath.Join(target, "cuda-linux-x64.tar.gz")); err != nil {
		t.Fatal("removed behind a symlinked downloads/")
	}
}

// A failure of a server is recorded under its own build, not one a restart moved to.
func TestRecordLCUsesTheBuildThatRanOut(t *testing.T) {
	newLC := loadCtx{key: "m|card|cuda@b2-easy1", in: install{build: "cuda@b2-easy1@x"}}
	old := &server{key: "m|card|cuda", build: "cuda@y"}
	if r := recordLC(old, newLC); r.key != "m|card|cuda" || r.in.build != "cuda@y" {
		t.Fatalf("recorded under %q %q", r.key, r.in.build)
	}
	same := &server{key: newLC.key, build: "cuda@b2-easy1@x"}
	if r := recordLC(same, newLC); r.key != newLC.key || r.in.build != newLC.in.build {
		t.Fatal("the same build is recorded as it is")
	}
	if r := recordLC(nil, newLC); r.key != newLC.key {
		t.Fatal("no server: the load's own")
	}
}

// The smoke test tells its caller each process's pid while it runs and 0 after; smokeOwned
// holds it in s.smoking (the room watch's own pids) while it runs, and lets it go after.
func TestSmokeTellsItsPid(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script")
	}
	d := t.TempDir()
	bin := filepath.Join(d, serverBinName())
	pidf := filepath.Join(d, "pids")
	os.WriteFile(bin, []byte("#!/bin/sh\necho $$ >> \""+pidf+"\"\necho 'Vulkan0: AMD Radeon RX 7600 (8176 MiB, 8000 MiB free)'\n"), 0o755)
	var mu sync.Mutex
	var got []int
	if err := smokeTestWith(context.Background(), bin, "vulkan", func(pid int) { mu.Lock(); got = append(got, pid); mu.Unlock() }); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(pidf)
	ran := strings.Fields(string(b))
	if len(ran) != 2 || len(got) != 4 || fmt.Sprint(got[0]) != ran[0] || got[1] != 0 || fmt.Sprint(got[2]) != ran[1] || got[3] != 0 {
		t.Fatalf("told %v, ran %v", got, ran)
	}
	var s Supervisor
	if err := s.smokeOwned(context.Background(), bin, "vulkan"); err != nil || s.smoking.pids() != nil {
		t.Fatalf("smokeOwned: %v, smoking left %v", err, s.smoking.pids())
	}
}
