package host

// opens_linux.go — a device opened by any process (inotify IN_OPEN), and the gate that keeps
// the engine's own opens (the nvidia-smi it runs to read the card) from being heard as a
// stranger's.

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

// The gate. Own(f) runs a read of ours that opens the devices; every Opens watch drops what it
// hears of those opens, exactly: the events made while f runs are dropped as they are read, and
// when f returns each live watch gets a marker of its own (a file made in a private directory
// the watch also watches: one more event on the same queue, in the kernel's order, so it comes
// after every event f made). Everything before the marker is dropped, whatever time it takes
// the reader to come to it, and what follows it is heard. No settle delay and no one-shot flag.
//
// The trade-off: a stranger's open that falls in the same stretch (while any own read runs, or
// between its events and its marker) is dropped with ours. Our own read sees the card as it is
// at that moment anyway, and the level watchdogs (host.Level) read it again if it moved.
var gate struct {
	mu   sync.Mutex
	own  int                  // own reads running now
	live map[*opensWatch]bool // the Opens watches alive
}

// markSeq makes every marker file's name its own: inotify joins an event to the one just before
// it when they are alike and the first is unread, and a joined marker would never be counted.
var markSeq atomic.Uint64

// opensWatch is one Opens watch.
type opensWatch struct {
	dir    string // the private directory the markers are made in
	markWD int32  // its watch
	pend   int    // markers made and not yet read (guarded by gate.mu)
}

// Own runs f, a read of ours that opens the card's device nodes, and keeps the Opens watches
// from signalling for what f opened (see the gate above). Calls compose: while any runs, every
// watch drops what it reads. Off Linux it only runs f.
func Own(f func()) {
	gate.mu.Lock()
	gate.own++
	gate.mu.Unlock()
	defer func() {
		gate.mu.Lock()
		for w := range gate.live {
			w.mark()
		}
		gate.own--
		gate.mu.Unlock()
	}()
	f()
}

// mark puts a marker after everything queued now. The caller holds gate.mu, and the marker is
// counted before own falls, so a reader that sees the own count fall sees the marker owed.
func (w *opensWatch) mark() {
	name := filepath.Join(w.dir, "m"+strconv.FormatUint(markSeq.Add(1), 10))
	f, err := os.Create(name) // its open is the one event, queued after f's
	if err != nil {
		return
	}
	w.pend++
	f.Close()
	os.Remove(name)
}

// heard is what the watch does with one event it has read: a marker is counted off, an event
// while own reads run or markers are owed is dropped, any other is the word.
func (w *opensWatch) heard(wd int32, mask uint32, ch chan struct{}) {
	gate.mu.Lock()
	if mask&unix.IN_Q_OVERFLOW != 0 {
		w.pend = 0 // the queue lost events, markers among them: no marker is owed any more
		gate.mu.Unlock()
		signal(ch) // and what was lost may have been anything
		return
	}
	if wd == w.markWD {
		if w.pend > 0 {
			w.pend--
		}
		gate.mu.Unlock()
		return
	}
	drop := gate.own > 0 || w.pend > 0
	gate.mu.Unlock()
	if !drop {
		signal(ch)
	}
	if f := testOpens.Load(); f != nil {
		(*f)(drop)
	}
}

// testOpens, set by a test before a watch is made, hears each event the watch decided on after
// the decision (true: dropped by the gate).
var testOpens atomic.Pointer[func(dropped bool)]

// Opens is signalled once (coalesced) each time one of paths is opened by
// any process other than the engine's own reads (Own): on device nodes, how
// another program taking the card is heard. It has no time cap. The func
// lets the watch go; the channel is closed once the watch has ended.
func Opens(paths []string) (<-chan struct{}, func(), error) {
	if len(paths) == 0 {
		return nil, nil, errors.New("host: Opens of no path")
	}
	n, err := newInotify()
	if err != nil {
		return nil, nil, err
	}
	w := &opensWatch{}
	for _, p := range paths {
		if _, err := n.add(p, unix.IN_OPEN); err != nil {
			n.stop()
			return nil, nil, err
		}
	}
	if w.dir, err = os.MkdirTemp("", "easyengine-gate-"); err != nil {
		n.stop()
		return nil, nil, errors.New("host: Opens: " + err.Error())
	}
	if w.markWD, err = n.add(w.dir, unix.IN_OPEN); err != nil {
		n.stop()
		os.RemoveAll(w.dir)
		return nil, nil, err
	}
	ch := make(chan struct{}, 1)
	gate.mu.Lock()
	if gate.live == nil {
		gate.live = map[*opensWatch]bool{}
	}
	gate.live[w] = true
	gate.mu.Unlock()
	go func() {
		n.read(func(wd int32, mask uint32, _ string) { w.heard(wd, mask, ch) })
		gate.mu.Lock()
		delete(gate.live, w)
		gate.mu.Unlock()
		os.RemoveAll(w.dir)
		close(ch)
	}()
	return ch, n.stop, nil
}
