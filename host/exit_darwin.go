package host

import (
	"sync"

	"golang.org/x/sys/unix"
)

// waker is a kqueue that also holds a user event, so a watch can be let go:
// the goroutine blocked in kevent is woken by triggering that event.
type waker struct {
	mu   sync.Mutex
	kq   int
	shut bool // the goroutine has left and kq is closed
}

// newWaker makes the kqueue and registers the user event on it.
func newWaker() (*waker, error) {
	kq, err := unix.Kqueue()
	if err != nil {
		return nil, err
	}
	unix.CloseOnExec(kq)
	user := []unix.Kevent_t{{Ident: 1, Filter: unix.EVFILT_USER, Flags: unix.EV_ADD | unix.EV_CLEAR}}
	if _, err := unix.Kevent(kq, user, nil, nil); err != nil {
		unix.Close(kq)
		return nil, err
	}
	return &waker{kq: kq}, nil
}

// register adds events to the kqueue.
func (w *waker) register(changes []unix.Kevent_t) error {
	for {
		_, err := unix.Kevent(w.kq, changes, nil, nil)
		if err != unix.EINTR {
			return err
		}
	}
}

// wait blocks in kevent with no timeout until there are events.
func (w *waker) wait(events []unix.Kevent_t) (int, error) {
	for {
		n, err := unix.Kevent(w.kq, nil, events, nil)
		if err == unix.EINTR {
			continue
		}
		return n, err
	}
}

// wake triggers the user event, if the kqueue is still open.
func (w *waker) wake() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.shut {
		return
	}
	user := []unix.Kevent_t{{Ident: 1, Filter: unix.EVFILT_USER, Fflags: unix.NOTE_TRIGGER}}
	unix.Kevent(w.kq, user, nil, nil)
}

// close shuts the kqueue; only the goroutine that waited calls it, on leaving.
func (w *waker) close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.shut {
		w.shut = true
		unix.Close(w.kq)
	}
}

// Exits is a channel that is closed when the process pid ends, and a func
// that lets the watch go (the channel is then never closed). A pid that has
// no process is ended already. On darwin it is a kqueue EVFILT_PROC watch.
func Exits(pid int) (<-chan struct{}, func(), error) {
	w, err := newWaker()
	if err != nil {
		return nil, nil, err
	}
	err = w.register([]unix.Kevent_t{{
		Ident:  uint64(pid),
		Filter: unix.EVFILT_PROC,
		Flags:  unix.EV_ADD | unix.EV_ONESHOT,
		Fflags: unix.NOTE_EXIT,
	}})
	if err != nil {
		w.close()
		if err == unix.ESRCH {
			return gone(), func() {}, nil
		}
		return nil, nil, err
	}
	ended := make(chan struct{})
	go func() {
		defer w.close()
		events := make([]unix.Kevent_t, 4)
		for {
			n, err := w.wait(events)
			if err != nil {
				return // the kqueue is unusable: no word will come
			}
			for _, e := range events[:n] {
				switch e.Filter {
				case unix.EVFILT_USER:
					return // let go
				case unix.EVFILT_PROC:
					if e.Fflags&unix.NOTE_EXIT != 0 {
						close(ended)
						return
					}
				}
			}
		}
	}()
	var once sync.Once
	return ended, func() { once.Do(w.wake) }, nil
}
