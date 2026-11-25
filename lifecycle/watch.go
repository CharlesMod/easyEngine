package lifecycle

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"
)

type parent struct {
	pid   int
	start string
}

// Watcher polls parent processes (pid plus start time, so pid reuse cannot fool
// it) and reports when the last one is gone.
type Watcher struct {
	Poll    time.Duration
	StartOf func(pid int) (string, error) // default PidStart
	Alive   func(pid int) bool            // default PidAlive

	mu      sync.Mutex
	parents []parent
	seen    bool
}

func NewWatcher() *Watcher {
	return &Watcher{Poll: time.Second, StartOf: PidStart, Alive: PidAlive}
}

// Add registers a parent. It fails if the pid is not alive.
func (w *Watcher) Add(pid int) error {
	if pid <= 0 || !w.Alive(pid) {
		return errors.New("no such process")
	}
	st, _ := w.StartOf(pid)
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, p := range w.parents {
		if p.pid == pid && p.start == st {
			return nil
		}
	}
	w.parents = append(w.parents, parent{pid, st})
	w.seen = true
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

func (w *Watcher) gone(p parent) bool {
	if !w.Alive(p.pid) {
		return true
	}
	if p.start == "" {
		return false
	}
	st, err := w.StartOf(p.pid)
	return err != nil || st != p.start
}

// Run returns when every parent that was ever added has gone, or ctx ends.
// With no parent ever added it waits for ctx.
func (w *Watcher) Run(ctx context.Context) bool {
	t := time.NewTicker(w.Poll)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return false
		case <-t.C:
		}
		w.mu.Lock()
		keep := w.parents[:0]
		for _, p := range w.parents {
			if !w.gone(p) {
				keep = append(keep, p)
			}
		}
		w.parents = keep
		empty := len(w.parents) == 0 && w.seen
		w.mu.Unlock()
		if empty {
			return true
		}
	}
}
