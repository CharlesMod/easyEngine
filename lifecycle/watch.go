package lifecycle

import (
	"context"
	"errors"
	"sort"
	"sync"

	"github.com/CharlesMod/easyEngine/host"
)

type parent struct {
	pid   int
	start string
	stop  func() // lets the watch go; nil when the system gave no word for this parent
}

// Watcher follows parent processes by the operating system's word of their
// end (host.ExitsOf: the pid plus its start token, so a reused pid is never
// followed) and reports when the last one is gone. It reads no clock.
type Watcher struct {
	mu      sync.Mutex
	parents []parent
	seen    bool
	change  chan struct{} // coalesced: something changed in the list
	closed  chan struct{} // closed by Close
	once    sync.Once
}

func NewWatcher() *Watcher {
	return &Watcher{change: make(chan struct{}, 1), closed: make(chan struct{})}
}

func (w *Watcher) signal() {
	select {
	case w.change <- struct{}{}:
	default: // one is owed already
	}
}

// Add registers a parent. It fails if the pid is not alive. A pid that turns
// out to be another process by the time it is watched (reused) counts as a
// parent that has already gone: Add succeeds and the parent is never listed.
func (w *Watcher) Add(pid int) error {
	if pid <= 0 || !PidAlive(pid) {
		return errors.New("no such process")
	}
	st, err := host.StartOf(pid)
	if err != nil && !errors.Is(err, host.ErrNoEvent) {
		return errors.New("no such process")
	}
	w.mu.Lock()
	for _, p := range w.parents {
		if p.pid == pid && p.start == st {
			w.mu.Unlock()
			return nil
		}
	}
	w.mu.Unlock()
	ended, stop, err := host.ExitsOf(pid, st)
	switch {
	case errors.Is(err, host.ErrNotThatProcess):
		w.mu.Lock()
		w.seen = true
		w.mu.Unlock()
		w.signal()
		return nil
	case errors.Is(err, host.ErrNoEvent):
		// This system gives no word of a process's end. The parent stays listed
		// for as long as the engine runs; any fallback belongs inside package
		// host, not here.
		w.mu.Lock()
		w.parents = append(w.parents, parent{pid: pid, start: st})
		w.seen = true
		w.mu.Unlock()
		return nil
	case err != nil:
		return err
	}
	w.mu.Lock()
	w.parents = append(w.parents, parent{pid: pid, start: st, stop: stop})
	w.seen = true
	w.mu.Unlock()
	go func() {
		select {
		case <-ended:
			w.mu.Lock()
			keep := w.parents[:0]
			for _, p := range w.parents {
				if !(p.pid == pid && p.start == st) {
					keep = append(keep, p)
				}
			}
			w.parents = keep
			w.mu.Unlock()
			w.signal()
		case <-w.closed:
			stop()
		}
	}()
	return nil
}

// PIDs lists the parents still watched.
func (w *Watcher) PIDs() []int {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := []int{}
	for _, p := range w.parents {
		out = append(out, p.pid)
	}
	sort.Ints(out)
	return out
}

// Close lets every watch go (host's stop funcs). The parents stay listed but
// are no longer followed, so Run after Close only ends with its ctx. Close is
// safe to call more than once.
func (w *Watcher) Close() {
	w.once.Do(func() { close(w.closed) })
}

// Run returns true when every parent that was ever added has gone (and at
// least one was added), false when ctx ends. It waits on the OS's word and on
// ctx, never on a clock. With no parent ever added it waits for ctx.
func (w *Watcher) Run(ctx context.Context) bool {
	for {
		w.mu.Lock()
		empty := len(w.parents) == 0 && w.seen
		w.mu.Unlock()
		if empty {
			return true
		}
		select {
		case <-ctx.Done():
			return false
		case <-w.change:
		}
	}
}
