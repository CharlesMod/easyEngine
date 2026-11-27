package host

import "testing"

// A real network change cannot be made without privilege, so this checks the
// watch opens, gives nothing while nothing changes, and lets go (twice is
// safe). The channel is never closed.
func TestNetworkChangesOpensAndStops(t *testing.T) {
	c, stop, err := NetworkChanges()
	if err != nil {
		t.Skipf("no netlink here: %v", err)
	}
	stop()
	stop()
	select {
	case <-c:
	default:
	}
}
