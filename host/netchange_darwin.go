package host

// netchange_darwin.go — the network coming or going as the kernel says it: a
// routing socket, which is told when a link, an address or a route changes.

import (
	"os"

	"golang.org/x/sys/unix"
)

// NetworkChanges is a channel of one coalesced signal for each time the
// system says a network link or address changed (the way "the network came
// back" is heard), and a func that lets the watch go. The channel is not
// closed. On darwin it is a PF_ROUTE socket.
func NetworkChanges() (<-chan struct{}, func(), error) {
	fd, err := unix.Socket(unix.AF_ROUTE, unix.SOCK_RAW, unix.AF_UNSPEC)
	if err != nil {
		return nil, nil, err
	}
	unix.CloseOnExec(fd)
	if err := unix.SetNonblock(fd, true); err != nil {
		unix.Close(fd)
		return nil, nil, err
	}
	f := os.NewFile(uintptr(fd), "route")
	c := make(chan struct{}, 1)
	go func() {
		buf := make([]byte, 64<<10)
		for {
			if _, err := f.Read(buf); err != nil {
				return // closed by the stop func
			}
			signal(c)
		}
	}()
	return c, func() { f.Close() }, nil
}
