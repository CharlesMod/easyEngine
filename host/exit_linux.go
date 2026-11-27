package host

// exit_linux.go — a process's end as the kernel says it: a pidfd, which the
// runtime's poller hears become readable when the process exits; and its
// start, from /proc, to tell one process from a later one with its number.

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/sys/unix"
)

// Exits is closed when the process ends. The func lets the watch go before
// then; the channel is never closed for a stop. A pid with no process has
// ended: a closed channel and a nil error.
func Exits(pid int) (<-chan struct{}, func(), error) {
	if pid <= 0 {
		return nil, nil, errors.New("host: no process has pid " + strconv.Itoa(pid))
	}
	fd, err := unix.PidfdOpen(pid, unix.PIDFD_NONBLOCK)
	if err == unix.ESRCH {
		return gone(), func() {}, nil // no such process: it has ended, and that is its end
	}
	if err != nil {
		return nil, nil, errors.New("host: pidfd_open " + strconv.Itoa(pid) + ": " + err.Error())
	}
	f := os.NewFile(uintptr(fd), "pidfd")
	raw, err := f.SyscallConn()
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	ch := make(chan struct{})
	go func() {
		// the poller calls this first, and again each time the pidfd is
		// readable; the pidfd is asked each time whether its process has
		// ended, so an end that came before the wait began is not lost. A
		// close (stop) ends the wait with an error
		err := raw.Read(func(fd uintptr) bool { return pidfdEnded(int(fd)) })
		if err == nil {
			close(ch)
		}
		f.Close()
	}()
	var once sync.Once
	return ch, func() { once.Do(func() { f.Close() }) }, nil
}

// pidfdEnded is the kernel's word on a pidfd, asked once and never waited on:
// whether its process has ended (the pidfd is readable).
func pidfdEnded(fd int) bool {
	p := []unix.PollFd{{Fd: int32(fd), Events: unix.POLLIN}}
	n, err := unix.Poll(p, 0) // asked, never waited on: a zero wait
	return err == nil && n == 1 && p[0].Revents&unix.POLLIN != 0
}

// StartOf is the process's start in the kernel's ticks since boot (the 22nd
// field of /proc/<pid>/stat): the same for one process every time asked, and
// another for a later process given the same number.
func StartOf(pid int) (string, error) {
	raw, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return "", errors.New("host: no process has pid " + strconv.Itoa(pid))
	}
	// the name (field 2) is in parentheses and may hold spaces or a ')':
	// the fields after it begin after the last ')'
	s := string(raw)
	i := strings.LastIndexByte(s, ')')
	if i < 0 {
		return "", errors.New("host: /proc/" + strconv.Itoa(pid) + "/stat does not parse")
	}
	f := strings.Fields(s[i+1:]) // f[0] is field 3
	if len(f) < 20 {
		return "", errors.New("host: /proc/" + strconv.Itoa(pid) + "/stat is short")
	}
	return f[19], nil
}
