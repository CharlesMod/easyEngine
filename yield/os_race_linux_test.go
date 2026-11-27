package yield

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Adapter (the level's read goroutine) and Watch / NodeSet / Openers (the reactor) meet on the
// adapter's nodes: under -race this fails if they are read and written without a lock.
func TestLinuxAdapterAndWatchTogether(t *testing.T) {
	root := t.TempDir()
	drv, sys := filepath.Join(root, "drv"), filepath.Join(root, "sys")
	for _, pci := range []string{"0000:0b:00.0", "0000:84:00.0"} {
		if err := os.MkdirAll(filepath.Join(drv, pci), 0o755); err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(drv, pci, "information"), []byte("Device Minor: 0\n"), 0o644)
	}
	if err := os.MkdirAll(sys, 0o755); err != nil {
		t.Fatal(err)
	}
	dev, _, _ := fakeDev(t)
	o := newLinuxOS(buildProc(t, nil), dev)
	o.sys, o.drv = sys, drv
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			o.Adapter([]string{"0000:0b:00.0", "0000:84:00.0", "0000:99:00.0"}[i%3])
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			w := o.Watch()
			_ = o.NodeSet()
			o.Openers(nil)
			w.Release()
		}
	}()
	wg.Wait()
}

// The nodes may be found after a first Watch: NodeSet then differs from the Set that Watch
// reported, which is how the caller knows to watch again; and Watch then reports the new set.
func TestLinuxNodeSetMovesWhenNodesAppear(t *testing.T) {
	root := t.TempDir()
	drv, sys := filepath.Join(root, "drv"), filepath.Join(root, "sys")
	pci := "0000:0b:00.0"
	if err := os.MkdirAll(filepath.Join(drv, pci), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(sys, 0o755); err != nil {
		t.Fatal(err)
	}
	dev, _, _ := fakeDev(t)
	o := newLinuxOS(buildProc(t, nil), dev)
	o.sys, o.drv = sys, drv
	o.Adapter(pci) // the driver's file is not there yet: every node
	w := o.Watch()
	defer w.Release()
	if w.Opens == nil || w.Set != o.NodeSet() {
		t.Fatalf("first watch %+v, set now %q", w, o.NodeSet())
	}
	// the adapter's node appears: a real device file the minor names
	node := filepath.Join(dev, "nvidia7")
	os.WriteFile(node, nil, 0o644)
	os.WriteFile(filepath.Join(drv, pci, "information"), []byte("Device Minor: 7\n"), 0o644)
	o.Adapter(pci) // the same address, looked at again: its nodes are found now
	if o.NodeSet() == w.Set {
		t.Fatalf("the node set did not move: %q", w.Set)
	}
	if got := o.NodeSet(); got != "/dev/nvidia7" {
		t.Fatalf("node set %q", got)
	}
}
