package host

// netchange_linux.go — the network coming or going as the kernel says it: a
// netlink route socket, read through the runtime's poller, that is told when
// a link or an address changes. It needs no privilege.

import (
	"os"

	"golang.org/x/sys/unix"
)

// NetworkChanges is a channel of one coalesced signal for each time the
// system says a network link or address changed (the way "the network came
// back" is heard), and a func that lets the watch go. The channel is not
// closed. On linux it is a netlink route socket.
func NetworkChanges() (<-chan struct{}, func(), error) {
	fd, err := unix.Socket(unix.AF_NETLINK, unix.SOCK_RAW|unix.SOCK_CLOEXEC|unix.SOCK_NONBLOCK, unix.NETLINK_ROUTE)
	if err != nil {
		return nil, nil, err
	}
	err = unix.Bind(fd, &unix.SockaddrNetlink{
		Family: unix.AF_NETLINK,
		Groups: unix.RTMGRP_LINK | unix.RTMGRP_IPV4_IFADDR | unix.RTMGRP_IPV6_IFADDR,
	})
	if err != nil {
		unix.Close(fd)
		return nil, nil, err
	}
	f := os.NewFile(uintptr(fd), "netlink")
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
