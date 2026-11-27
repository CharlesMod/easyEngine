package host

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// opensHook hears each event the watch reads (true: dropped by the gate, false: the word), so a
// test counts them by waiting on the events themselves, never on a clock.
func opensHook(t *testing.T) chan bool {
	t.Helper()
	seen := make(chan bool, 256)
	f := func(dropped bool) { seen <- dropped }
	testOpens.Store(&f)
	t.Cleanup(func() { testOpens.Store(nil) })
	return seen
}

func nextEvent(t *testing.T, seen chan bool) bool {
	t.Helper()
	tm := time.NewTimer(bound) // watchdog for silence: test-only, a broken watch fails instead of hanging
	defer tm.Stop()
	select {
	case d := <-seen:
		return d
	case <-tm.C:
		t.Fatal("no event heard")
		return false
	}
}

func openOnce(t *testing.T, path string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
}

func nodeFile(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "node")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// Own swallows exactly the opens made inside f (the kernel queues them before f returns, the
// reader may come to them long after) and an open made after f is heard.
func TestOwnSwallowsOwnOpensOnly(t *testing.T) {
	a, b := nodeFile(t), nodeFile(t)
	seen := opensHook(t)
	ch, stop, err := Opens([]string{a, b})
	if err != nil {
		t.Fatal(err)
	}
	Own(func() {
		for i := 0; i < 5; i++ { // alternating nodes: the kernel joins identical events in a row
			openOnce(t, []string{a, b}[i%2])
		}
	})
	openOnce(t, a) // a stranger's, after the read
	dropped := 0
	for {
		if !nextEvent(t, seen) {
			break // the word
		}
		dropped++
	}
	if dropped != 5 {
		t.Fatalf("%d of the 5 own opens were dropped", dropped)
	}
	within(t, ch, "the open after the own read")
	stop()
	within(t, ch, "the channel closed after stop") // a closed channel answers at once
}

// An open that is not ours, made while no own read runs, is the word; two own reads that overlap
// compose, and what follows the later one's end is heard.
func TestOwnComposes(t *testing.T) {
	path := nodeFile(t)
	seen := opensHook(t)
	ch, stop, err := Opens([]string{path})
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	inA, inB, endA := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		Own(func() { openOnce(t, path); close(inA); <-endA })
	}()
	go func() {
		defer wg.Done()
		<-inA
		Own(func() { openOnce(t, path); close(inB); <-endA; openOnce(t, path) })
	}()
	<-inB
	close(endA)
	wg.Wait()
	openOnce(t, path)
	for nextEvent(t, seen) {
	}
	within(t, ch, "the open after both own reads")
}

func TestOpensChannelClosesOnStop(t *testing.T) {
	ch, stop, err := Opens([]string{nodeFile(t)})
	if err != nil {
		t.Fatal(err)
	}
	stop()
	for {
		tm := time.NewTimer(bound) // watchdog for silence: test-only
		select {
		case _, ok := <-ch:
			tm.Stop()
			if !ok {
				return
			}
		case <-tm.C:
			t.Fatal("the channel stayed open after stop")
		}
	}
}
