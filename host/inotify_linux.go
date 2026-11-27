package host

// inotify_linux.go — a file's change and a device's open as the kernel says
// them: inotify, read through the runtime's poller, so a watch costs a
// blocked read and nothing while nothing happens.

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
)

// inotify is one inotify instance and the goroutine that reads it.
type inotify struct {
	fd   int // kept: f.Fd() would set the file blocking, out of the poller's hands
	f    *os.File
	once sync.Once
}

func newInotify() (*inotify, error) {
	fd, err := unix.InotifyInit1(unix.IN_NONBLOCK | unix.IN_CLOEXEC)
	if err != nil {
		return nil, errors.New("host: inotify: " + err.Error())
	}
	return &inotify{fd: fd, f: os.NewFile(uintptr(fd), "inotify")}, nil
}

func (n *inotify) add(path string, mask uint32) error {
	_, err := unix.InotifyAddWatch(n.fd, path, mask)
	if err != nil {
		return errors.New("host: inotify on " + path + ": " + err.Error())
	}
	return nil
}

func (n *inotify) stop() { n.once.Do(func() { n.f.Close() }) }

// read calls each with the name each event carries (empty for an event on a
// watched file itself) until the instance is closed.
func (n *inotify) read(each func(name string)) {
	buf := make([]byte, 64<<10)
	for {
		k, err := n.f.Read(buf) // blocks in the poller until the kernel says something
		if err != nil {
			n.stop()
			return
		}
		for off := 0; off+unix.SizeofInotifyEvent <= k; {
			ev := (*unix.InotifyEvent)(unsafe.Pointer(&buf[off]))
			name := ""
			if ev.Len > 0 {
				b := buf[off+unix.SizeofInotifyEvent : off+unix.SizeofInotifyEvent+int(ev.Len)]
				for i, c := range b {
					if c == 0 {
						b = b[:i]
						break
					}
				}
				name = string(b)
			}
			each(name)
			off += unix.SizeofInotifyEvent + int(ev.Len)
		}
	}
}

// Changes is signalled once (coalesced, never blocking the kernel's word)
// for each change of the file at path: its content written, a file renamed
// onto its name, created, removed. The file need not exist; its directory
// must. Changes to other files of the directory do not signal. The func
// lets the watch go.
func Changes(path string) (<-chan struct{}, func(), error) {
	dir, base := filepath.Split(filepath.Clean(path))
	if dir == "" {
		dir = "."
	}
	n, err := newInotify()
	if err != nil {
		return nil, nil, err
	}
	// the directory, not the file: a file replaced by a rename is a new
	// inode, and a watch on the old one would hear nothing more
	if err := n.add(dir, unix.IN_CLOSE_WRITE|unix.IN_MODIFY|unix.IN_ATTRIB|unix.IN_CREATE|unix.IN_DELETE|
		unix.IN_MOVED_FROM|unix.IN_MOVED_TO|unix.IN_ONLYDIR); err != nil {
		n.stop()
		return nil, nil, err
	}
	ch, seen := make(chan struct{}, 1), testSeen
	go n.read(func(name string) {
		if name == base {
			signal(ch)
		}
		if seen != nil {
			seen(name)
		}
	})
	return ch, n.stop, nil
}

// testSeen, set by a test before a watch is made, hears every name the
// watch's directory said, after the watch has acted on it: the kernel says
// a directory's events in order, so a test learns that everything before a
// marker file has been heard without waiting on a clock.
var testSeen func(name string)

// Opens is signalled once (coalesced) each time one of paths is opened by
// any process: on device nodes, how another program taking the card is
// heard. It has no time cap. The func lets the watch go.
func Opens(paths []string) (<-chan struct{}, func(), error) {
	if len(paths) == 0 {
		return nil, nil, errors.New("host: Opens of no path")
	}
	n, err := newInotify()
	if err != nil {
		return nil, nil, err
	}
	for _, p := range paths {
		if err := n.add(p, unix.IN_OPEN); err != nil {
			n.stop()
			return nil, nil, err
		}
	}
	ch := make(chan struct{}, 1)
	go n.read(func(string) { signal(ch) })
	return ch, n.stop, nil
}
