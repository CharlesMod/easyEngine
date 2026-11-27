package host

import (
	"path/filepath"
	"sync"

	"golang.org/x/sys/unix"
)

// Changes is a channel of one coalesced signal for each change of the file
// at path (written, replaced by a rename, created, removed), and a func that
// lets the watch go. The directory must exist; the file need not. On darwin
// it is a kqueue EVFILT_VNODE watch of the directory and of the file.
func Changes(path string) (<-chan struct{}, func(), error) {
	dir := filepath.Dir(path)
	w, err := newWaker()
	if err != nil {
		return nil, nil, err
	}
	dirfd, err := unix.Open(dir, unix.O_EVTONLY, 0)
	if err != nil {
		w.close()
		return nil, nil, err
	}
	unix.CloseOnExec(dirfd)
	err = w.register([]unix.Kevent_t{{
		Ident:  uint64(dirfd),
		Filter: unix.EVFILT_VNODE,
		Flags:  unix.EV_ADD | unix.EV_CLEAR,
		Fflags: unix.NOTE_WRITE,
	}})
	if err != nil {
		unix.Close(dirfd)
		w.close()
		return nil, nil, err
	}
	c := &fileWatch{w: w, path: path, dirfd: dirfd, filefd: -1, out: make(chan struct{}, 1)}
	c.reconcile() // the directory is watched already, so nothing slips by
	go c.run()
	var once sync.Once
	return c.out, func() { once.Do(w.wake) }, nil
}

// fileWatch is the state of one Changes watch; only its goroutine touches it
// after the start.
type fileWatch struct {
	w      *waker
	path   string
	dirfd  int
	filefd int // -1 when the file is not watched
	seen   bool
	ino    uint64
	out    chan struct{}
}

// reconcile looks at the file by name, watches the file now there, and says
// whether it appeared, vanished or is another file than last seen.
func (c *fileWatch) reconcile() bool {
	var st unix.Stat_t
	exists := unix.Stat(c.path, &st) == nil
	changed := exists != c.seen || (exists && st.Ino != c.ino)
	c.seen, c.ino = exists, 0
	if exists {
		c.ino = st.Ino
	}
	if changed && c.filefd >= 0 {
		unix.Close(c.filefd) // closing drops its registration
		c.filefd = -1
	}
	if exists && c.filefd < 0 {
		fd, err := unix.Open(c.path, unix.O_EVTONLY, 0)
		if err == nil {
			unix.CloseOnExec(fd)
			err = c.w.register([]unix.Kevent_t{{
				Ident:  uint64(fd),
				Filter: unix.EVFILT_VNODE,
				Flags:  unix.EV_ADD | unix.EV_CLEAR,
				Fflags: unix.NOTE_WRITE | unix.NOTE_EXTEND | unix.NOTE_ATTRIB | unix.NOTE_DELETE | unix.NOTE_RENAME | unix.NOTE_REVOKE,
			}})
			if err != nil {
				unix.Close(fd)
			} else {
				c.filefd = fd
			}
		}
	}
	return changed
}

// run is the goroutine: it blocks in kevent and says what it hears.
func (c *fileWatch) run() {
	defer func() {
		unix.Close(c.dirfd)
		if c.filefd >= 0 {
			unix.Close(c.filefd)
		}
		c.w.close()
	}()
	const lost = unix.NOTE_DELETE | unix.NOTE_RENAME | unix.NOTE_REVOKE
	events := make([]unix.Kevent_t, 8)
	for {
		n, err := c.w.wait(events)
		if err != nil {
			return
		}
		for _, e := range events[:n] {
			switch {
			case e.Filter == unix.EVFILT_USER:
				return // let go
			case e.Filter != unix.EVFILT_VNODE:
			case e.Ident == uint64(c.dirfd):
				if c.reconcile() {
					signal(c.out)
				}
			case c.filefd >= 0 && e.Ident == uint64(c.filefd):
				signal(c.out)
				if e.Fflags&lost != 0 {
					// the file this fd held is no longer the name's
					unix.Close(c.filefd)
					c.filefd = -1
					c.seen = false
					c.reconcile()
				}
			}
		}
	}
}
