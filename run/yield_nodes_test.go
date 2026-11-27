package run_test

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/yield"
)

// nodedOS is the fake with the OS's node set: what Watch holds, and what Adapter has found since.
type nodedOS struct {
	*fakeYieldOS
	set     atomic.Value // string
	watched chan string  // the set each Watch call was made on
}

func (n *nodedOS) NodeSet() string { return n.set.Load().(string) }
func (n *nodedOS) Watch() yield.Words {
	w := n.fakeYieldOS.Watch()
	w.Set = n.NodeSet()
	n.watched <- w.Set
	return w
}

// A watch made before the adapter's nodes were known is made again once they are, on the nodes
// found: the reactor compares the set the watch holds with the OS's now, not only the address.
func TestYieldWatchesAgainWhenTheNodesAppear(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	n := &nodedOS{fakeYieldOS: newFakeYieldOS(), watched: make(chan string, 16)}
	n.set.Store("")
	s := yieldSup(t, e, n.fakeYieldOS, func(y *run.YieldConfig) {
		y.Every = time.Hour
		y.Ignore = []string{"probe"}
		y.OS = func() yield.OS { return n }
	})
	settled(t, s, n.fakeYieldOS)
	select {
	case first := <-n.watched:
		if first != "" {
			t.Fatalf("first watch on %q", first)
		}
	default:
		t.Fatal("no watch was made when the reactor armed")
	}
	n.set.Store("/dev/nvidia0") // Adapter found the nodes on a read
	n.open(4242, "probe")       // a look of the reactor's: it sees the nodes moved
	select {
	case again := <-n.watched:
		if again != "/dev/nvidia0" {
			t.Fatalf("watched again on %q", again)
		}
	// watchdog for silence: a reactor that never watches again fails the test after a bound
	case <-time.After(20 * time.Second):
		t.Fatal("the watch was not made again on the new nodes")
	}
}
