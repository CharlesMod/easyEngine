//go:build windows

package discover

import (
	"syscall"

	"golang.org/x/sys/windows"
)

// reuse lets this socket share the mDNS port with other responders on the
// machine (Bonjour, the DNS Client).
func reuse(network, address string, c syscall.RawConn) error {
	var serr error
	err := c.Control(func(fd uintptr) {
		serr = windows.SetsockoptInt(windows.Handle(fd), windows.SOL_SOCKET, windows.SO_REUSEADDR, 1)
	})
	if err != nil {
		return err
	}
	return serr
}
