package yield

import (
	"os"
	"path/filepath"
	"testing"
)

// nodeRig is a fake /dev, /sys/class/drm and /proc/driver/nvidia/gpus for three adapters: two
// NVIDIA cards (minors 0 and 1, their dri nodes driven by "nvidia") and an AMD card whose dri
// nodes are driven by "amdgpu".
type nodeRig struct {
	dev, sys, drv string
}

const (
	pciA = "0000:01:00.0" // NVIDIA, minor 0, card0 + renderD128
	pciB = "0000:65:00.0" // NVIDIA, minor 1, card1 + renderD129
	pciC = "0000:0b:00.0" // AMD, card2 + renderD130
)

func newNodeRig(t *testing.T) nodeRig {
	t.Helper()
	root := t.TempDir()
	r := nodeRig{dev: filepath.Join(root, "dev"), sys: filepath.Join(root, "sys"), drv: filepath.Join(root, "drv")}
	devs, drivers := filepath.Join(root, "devices"), filepath.Join(root, "drivers")
	mk := func(p string) {
		if err := os.MkdirAll(p, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	touch := func(p string) {
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mk(filepath.Join(r.dev, "dri"))
	for _, n := range []string{"nvidia0", "nvidia1", "nvidiactl", "dri/card0", "dri/card1", "dri/card2", "dri/renderD128", "dri/renderD129", "dri/renderD130"} {
		touch(filepath.Join(r.dev, n))
	}
	for _, d := range []string{"nvidia", "amdgpu"} {
		mk(filepath.Join(drivers, d))
	}
	for _, c := range []struct {
		pci, driver, minor string
		nodes              []string
	}{
		{pciA, "nvidia", "0", []string{"card0", "renderD128"}},
		{pciB, "nvidia", "1", []string{"card1", "renderD129"}},
		{pciC, "amdgpu", "", []string{"card2", "renderD130"}},
	} {
		mk(filepath.Join(devs, c.pci))
		if err := os.Symlink(filepath.Join(drivers, c.driver), filepath.Join(devs, c.pci, "driver")); err != nil {
			t.Fatal(err)
		}
		if c.minor != "" {
			mk(filepath.Join(r.drv, c.pci))
			os.WriteFile(filepath.Join(r.drv, c.pci, "information"), []byte("Model: x\nDevice Minor: \t "+c.minor+"\n"), 0o644)
		}
		for _, n := range c.nodes {
			mk(filepath.Join(r.sys, n))
			if err := os.Symlink(filepath.Join(devs, c.pci), filepath.Join(r.sys, n, "device")); err != nil {
				t.Fatal(err)
			}
		}
	}
	return r
}

func (r nodeRig) os(t *testing.T, procs map[int]fakePID) *linuxOS {
	o := newLinuxOS(buildProc(t, procs), r.dev)
	o.sys, o.drv = r.sys, r.drv
	t.Cleanup(o.Close)
	return o
}

func (r nodeRig) at(names ...string) []string {
	var out []string
	for _, n := range names {
		out = append(out, filepath.Join(r.dev, n))
	}
	return out
}

func sameList(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// An NVIDIA adapter is listened to at its /dev/nvidia<minor> alone: not its dri nodes, not the
// other NVIDIA card's node. An open of its dri node makes no word.
func TestNodesNvidiaAdapterListensToItsMinorOnly(t *testing.T) {
	r := newNodeRig(t)
	o := r.os(t, nil)
	o.Adapter(pciA)
	if got, want := o.Nodes(), r.at("nvidia0"); !sameList(got, want) {
		t.Fatalf("nodes %v want %v", got, want)
	}
	w := o.Watch()
	defer w.Release()
	if w.Opens == nil {
		t.Fatal("no watch")
	}
	for _, n := range []string{"dri/renderD128", "dri/card0", "nvidia1", "dri/renderD129"} {
		f, err := os.Open(filepath.Join(r.dev, n)) // not listened to: no word
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
	}
	f, _ := os.Open(filepath.Join(r.dev, "nvidia0"))
	f.Close()
	<-w.Opens // the only word is the nvidia0 open: events come in order
	select {
	case <-w.Opens:
		t.Fatal("a second word: a node that is not the adapter's was heard")
	default:
	}
}

// The scan agrees: a holder of the adapter's dri node that does not hold /dev/nvidia<minor> is not
// a holder (the NVIDIA driver's clients open the device node); one of the minor node is.
func TestNodesNvidiaScanWantsTheMinorOnly(t *testing.T) {
	r := newNodeRig(t)
	o := r.os(t, map[int]fakePID{
		10: {comm: "dri-only", fds: []string{"/dev/dri/renderD128"}},
		11: {comm: "minor", fds: []string{"/dev/nvidia0", "/dev/dri/renderD128"}},
		12: {comm: "other-card", fds: []string{"/dev/nvidia1"}},
	})
	o.Adapter(pciA)
	got := o.Openers(nil)
	if len(got) != 1 || got[11] == "" {
		t.Fatalf("holders %v", got)
	}
}

// Where the minor cannot be read, every NVIDIA node is listened to and still no dri node.
func TestNodesNvidiaWithoutMinorTakesEveryNvidiaNode(t *testing.T) {
	r := newNodeRig(t)
	os.Remove(filepath.Join(r.drv, pciA, "information"))
	o := r.os(t, nil)
	o.Adapter(pciA)
	if got, want := o.Nodes(), r.at("nvidia0", "nvidia1"); !sameList(got, want) {
		t.Fatalf("nodes %v want %v", got, want)
	}
}

// An adapter of another driver has its own dri nodes only.
func TestNodesAmdAdapterListensToItsDriNodes(t *testing.T) {
	r := newNodeRig(t)
	o := r.os(t, nil)
	o.Adapter(pciC)
	if got, want := o.Nodes(), r.at("dri/card2", "dri/renderD130"); !sameList(got, want) {
		t.Fatalf("nodes %v want %v", got, want)
	}
}

// With the adapter not known every node is listened to, and the set is made again when it arrives
// (NodeSet moves: the caller that sees it listens again).
func TestNodesUnknownThenKnown(t *testing.T) {
	r := newNodeRig(t)
	o := r.os(t, nil)
	o.Adapter("")
	all := o.Nodes()
	if len(all) != 8 { // nvidia0, nvidia1, three cards, three render nodes (nvidiactl is no node)
		t.Fatalf("unknown adapter: %v", all)
	}
	before := o.NodeSet()
	o.Adapter(pciA)
	if o.NodeSet() == before || o.NodeSet() != r.dev+"/nvidia0" {
		t.Fatalf("set did not narrow: %q", o.NodeSet())
	}
	o.Adapter(pciC)
	if o.NodeSet() == r.dev+"/nvidia0" {
		t.Fatalf("set did not follow the change of adapter: %q", o.NodeSet())
	}
}
