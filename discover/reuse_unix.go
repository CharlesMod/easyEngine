//go:build !windows

package discover

import (
	"syscall"

	"golang.org/x/sys/unix"
)

// reuse lets this socket share the mDNS port with other responders on the
// machine (avahi, Bonjour).
func reuse(network, address string, c syscall.RawConn) error {
	var serr error
	err := c.Control(func(fd uintptr) {
		if e := unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEADDR, 1); e != nil {
			serr = e
			return
		}
		if e := unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEPORT, 1); e != nil && e != unix.ENOPROTOOPT {
			serr = e
		}
	})
	if err != nil {
		return err
	}
	return serr
}
