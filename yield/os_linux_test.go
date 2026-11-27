package yield

import (
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"
)

type fakePID struct {
	cmdline []string
	comm    string
	stat    string
	fds     []string
}

func buildProc(t *testing.T, procs map[int]fakePID) string {
	t.Helper()
	root := t.TempDir()
	// Non-numeric entries must be ignored.
	if err := os.MkdirAll(filepath.Join(root, "self-not-a-pid"), 0o755); err != nil {
		t.Fatal(err)
	}
	for pid, p := range procs {
		d := filepath.Join(root, strconv.Itoa(pid))
		if err := os.MkdirAll(filepath.Join(d, "fd"), 0o755); err != nil {
			t.Fatal(err)
		}
		cmd := ""
		for _, a := range p.cmdline {
			cmd += a + "\x00"
		}
		write := func(name, s string) {
			if err := os.WriteFile(filepath.Join(d, name), []byte(s), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write("cmdline", cmd)
		write("comm", p.comm+"\n")
		write("stat", p.stat)
		for i, target := range p.fds {
			if err := os.Symlink(target, filepath.Join(d, "fd", strconv.Itoa(i))); err != nil {
				t.Fatal(err)
			}
		}
	}
	return root
}

func TestGPUNode(t *testing.T) {
	tests := map[string]bool{
		"/dev/nvidia0":           true,
		"/dev/nvidia12":          true,
		"/dev/nvidiactl":         false,
		"/dev/nvidia-uvm":        false,
		"/dev/nvidia-modeset":    false,
		"/dev/nvidia":            false,
		"/dev/dri/renderD128":    true,
		"/dev/dri/card0":         true,
		"/dev/dri/by-path":       false,
		"/dev/null":              false,
		"socket:[1234]":          false,
		"":                       false,
		"/home/u/dev/nvidia0":    false,
		"/dev/nvidia0 (deleted)": false,
	}
	for in, want := range tests {
		if got := gpuNode(in); got != want {
			t.Errorf("gpuNode(%q) = %v, want %v", in, got, want)
		}
	}
}

func fakeProcTree(t *testing.T) string {
	return buildProc(t, map[int]fakePID{
		1:   {cmdline: []string{"/sbin/init"}, comm: "init", stat: "1 (init) S 0 1 1 0"},
		10:  {cmdline: []string{"/usr/bin/bash"}, comm: "bash", stat: "10 (bash) S 1 10 10 0"},
		20:  {cmdline: []string{"/s/ubuntu12_32/reaper", "SteamLaunch", "AppId=77", "--", "/p/SomeGame.exe"}, comm: "reaper", stat: "20 (reaper) S 10 20 20 0"},
		30:  {cmdline: []string{"/p/SomeGame.exe"}, comm: "SomeGame.exe", stat: "30 (some name) with ) paren) S 20 20 20 0", fds: []string{"/dev/nvidia0", "/dev/null"}},
		40:  {cmdline: []string{"nvidia-smi"}, comm: "nvidia-smi", stat: "40 (nvidia-smi) S 10 40 40 0", fds: []string{"/dev/nvidiactl"}},
		50:  {cmdline: []string{"vk"}, comm: "vk", stat: "50 (vk) S 10 50 50 0", fds: []string{"/dev/null", "/dev/dri/renderD128"}},
		60:  {cmdline: []string{"other"}, comm: "other", stat: "60 (other) S 1 60 60 0"},
		70:  {cmdline: nil, comm: "oom_reaper", stat: "70 (oom_reaper) S 2 0 0 0"},
		777: {cmdline: []string{"lutris-wrapper: Some Title"}, comm: "lutris-wrapper", stat: "777 (lutris-wrapper) S 1 1 1 0"},
	})
}

func TestLinuxLaunches(t *testing.T) {
	o := newLinuxOS(fakeProcTree(t), t.TempDir())
	defer o.Close()
	got := map[int]Launch{}
	for _, l := range o.Launches() {
		got[l.PID] = l
	}
	want := map[int]Launch{
		20:  {PID: 20, Kind: "steam", AppID: "77", Exe: "SomeGame.exe"},
		777: {PID: 777, Kind: "lutris", Exe: "Some Title"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestLinuxOpeners(t *testing.T) {
	o := newLinuxOS(fakeProcTree(t), t.TempDir())
	defer o.Close()
	got := o.Openers(nil)
	want := map[int]string{30: "SomeGame.exe", 50: "vk"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	got = o.Openers(func(pid int) bool { return pid == 30 })
	want = map[int]string{50: "vk"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("skip: got %v want %v", got, want)
	}
	// A proc root that cannot be read yields nothing.
	if got := newLinuxOS(filepath.Join(t.TempDir(), "missing"), t.TempDir()).Openers(nil); len(got) != 0 {
		t.Fatalf("missing root: %v", got)
	}
}

func TestLinuxUnder(t *testing.T) {
	o := newLinuxOS(fakeProcTree(t), t.TempDir())
	defer o.Close()
	tests := []struct {
		pid, anc int
		want     bool
	}{
		{30, 20, true},  // parent; name has spaces and ')'
		{30, 10, true},  // grandparent
		{30, 30, true},  // self
		{30, 1, true},   // reaches init
		{20, 30, false}, // descendant is not an ancestor
		{60, 10, false}, // unrelated
		{50, 20, false}, // sibling subtree
		{999, 10, false},
	}
	for _, tc := range tests {
		if got := o.Under(tc.pid, tc.anc); got != tc.want {
			t.Errorf("Under(%d,%d) = %v, want %v", tc.pid, tc.anc, got, tc.want)
		}
	}
}

func TestLinuxParent(t *testing.T) {
	o := newLinuxOS(fakeProcTree(t), t.TempDir())
	defer o.Close()
	if got := o.parent(30); got != 20 {
		t.Fatalf("parent(30) = %d, want 20", got)
	}
	if got := o.parent(12345); got != 0 {
		t.Fatalf("missing pid parent = %d", got)
	}
}

func fakeDev(t *testing.T) (dev, nvidia, render string) {
	t.Helper()
	dev = t.TempDir()
	nvidia = filepath.Join(dev, "nvidia0")
	render = filepath.Join(dev, "dri", "renderD128")
	if err := os.MkdirAll(filepath.Join(dev, "dri"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{nvidia, render} {
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return
}

func TestLinuxNodes(t *testing.T) {
	dev, nvidia, render := fakeDev(t)
	o := newLinuxOS(t.TempDir(), dev)
	defer o.Close()
	want := []string{render, nvidia} // sorted: dri/renderD128 before nvidia0
	if got := o.Nodes(); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	if got := newLinuxOS(t.TempDir(), t.TempDir()).Nodes(); len(got) != 0 {
		t.Fatalf("no devices: %v", got)
	}
}

// Watch is the kernel's word that a program opened a node: no clock, so the test waits on the
// word itself, bounded only so a broken watch fails the test.
func TestLinuxWatchOpens(t *testing.T) {
	dev, nvidia, render := fakeDev(t)
	o := newLinuxOS(t.TempDir(), dev)
	defer o.Close()
	w := o.Watch()
	if w.Opens == nil || w.Launches != nil {
		t.Fatalf("words %+v", w)
	}
	defer w.Release()
	for _, path := range []string{nvidia, render} {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
		select {
		case <-w.Opens:
		// watchdog for silence: a test whose watch never speaks fails after a bound
		case <-time.After(5 * time.Second):
			t.Fatalf("%s: no word that it was opened", path)
		}
	}
}

func TestLinuxWatchNoDevice(t *testing.T) {
	o := newLinuxOS(t.TempDir(), t.TempDir())
	defer o.Close()
	if w := o.Watch(); w.Opens != nil || w.Launches != nil {
		t.Fatalf("words with no devices: %+v", w)
	}
}

func TestLinuxNodesAfterAdapter(t *testing.T) {
	dev, _, _ := fakeDev(t)
	o := newLinuxOS(t.TempDir(), dev)
	defer o.Close()
	o.nodes, o.pci = map[string]bool{"/dev/nvidia1": true, "/dev/dri/renderD129": true}, "x"
	want := []string{"/dev/dri/renderD129", "/dev/nvidia1"}
	if got := o.Nodes(); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestLinuxAdapterNodes(t *testing.T) {
	root := t.TempDir()
	proc, sys, drv := filepath.Join(root, "proc"), filepath.Join(root, "sys"), filepath.Join(root, "drv")
	devs := filepath.Join(root, "devices")
	for _, d := range []string{filepath.Join(drv, "0000:0b:00.0"), filepath.Join(drv, "0000:84:00.0"), filepath.Join(devs, "0000:0b:00.0"), filepath.Join(devs, "0000:84:00.0"), filepath.Join(sys, "renderD128"), filepath.Join(sys, "renderD129"), filepath.Join(sys, "card1-DP-1")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(filepath.Join(drv, "0000:0b:00.0", "information"), []byte("Model: x\nDevice Minor: \t 0\n"), 0o644)
	os.WriteFile(filepath.Join(drv, "0000:84:00.0", "information"), []byte("Model: x\nDevice Minor: \t 1\n"), 0o644)
	os.Symlink(filepath.Join(devs, "0000:0b:00.0"), filepath.Join(sys, "renderD128", "device"))
	os.Symlink(filepath.Join(devs, "0000:84:00.0"), filepath.Join(sys, "renderD129", "device"))
	// pid 10 holds GPU 0's card node, pid 11 GPU 1's, pid 12 GPU 1's render node
	for pid, link := range map[int]string{10: "/dev/nvidia0", 11: "/dev/nvidia1", 12: "/dev/dri/renderD129", 13: "/dev/dri/renderD128"} {
		fd := filepath.Join(proc, strconv.Itoa(pid), "fd")
		os.MkdirAll(fd, 0o755)
		os.Symlink(link, filepath.Join(fd, "3"))
		os.WriteFile(filepath.Join(proc, strconv.Itoa(pid), "comm"), []byte("p"+strconv.Itoa(pid)+"\n"), 0o644)
	}
	o := newLinuxOS(proc, filepath.Join(root, "nodev"))
	defer o.Close()
	o.sys, o.drv = sys, drv
	if got := o.Openers(nil); len(got) != 4 {
		t.Fatalf("no adapter: want all 4 openers, got %v", got)
	}
	o.Adapter("0000:0B:00.0")
	got := o.Openers(nil)
	if len(got) != 2 || got[10] == "" || got[13] == "" {
		t.Fatalf("adapter 0: got %v", got)
	}
	o.Adapter("0000:99:00.0") // unknown: every node again
	if got := o.Openers(nil); len(got) != 4 {
		t.Fatalf("unknown adapter: got %v", got)
	}
}
