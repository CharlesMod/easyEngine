package host

import (
	"io"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

// ENOBUFS means changes were dropped: it is a change and the reading goes on.
// Only a closed fd (or an error that will not mend) ends the reading.
func TestReadVerdict(t *testing.T) {
	for _, tc := range []struct {
		err         error
		change, end bool
	}{
		{nil, true, false},
		{unix.ENOBUFS, true, false},
		{&os.PathError{Op: "read", Err: unix.ENOBUFS}, true, false},
		{os.ErrClosed, false, true},
		{io.EOF, false, true},
	} {
		c, e := readVerdict(tc.err)
		if c != tc.change || e != tc.end {
			t.Errorf("%v: got change=%v end=%v", tc.err, c, e)
		}
	}
}

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
