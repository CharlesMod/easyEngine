package host

import (
	"sync"

	"golang.org/x/sys/windows"
)

// latch is a manual-reset event a watch's goroutine waits on beside the
// thing it watches, so the watch can be let go.
type latch struct {
	mu   sync.Mutex
	ev   windows.Handle
	shut bool // the goroutine has left and ev is closed
}

// newLatch makes the event, not yet set.
func newLatch() (*latch, error) {
	ev, err := windows.CreateEvent(nil, 1, 0, nil)
	if err != nil {
		return nil, err
	}
	return &latch{ev: ev}, nil
}

// wake sets the event, if it is still open.
func (l *latch) wake() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.shut {
		windows.SetEvent(l.ev)
	}
}

// close closes the event; only the goroutine that waited calls it, on leaving.
func (l *latch) close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.shut {
		l.shut = true
		windows.CloseHandle(l.ev)
	}
}

// Exits is a channel that is closed when the process pid ends, and a func
// that lets the watch go (the channel is then never closed). A pid that has
// no process is ended already. On windows it waits on the process handle.
func Exits(pid int) (<-chan struct{}, func(), error) {
	h, err := windows.OpenProcess(windows.SYNCHRONIZE|windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		if err == windows.ERROR_INVALID_PARAMETER {
			return gone(), func() {}, nil
		}
		return nil, nil, err
	}
	l, err := newLatch()
	if err != nil {
		windows.CloseHandle(h)
		return nil, nil, err
	}
	ended := make(chan struct{})
	go func() {
		defer windows.CloseHandle(h)
		defer l.close()
		ev, err := windows.WaitForMultipleObjects([]windows.Handle{h, l.ev}, false, windows.INFINITE)
		if err == nil && ev == windows.WAIT_OBJECT_0 {
			close(ended)
		}
	}()
	var once sync.Once
	return ended, func() { once.Do(l.wake) }, nil
}
