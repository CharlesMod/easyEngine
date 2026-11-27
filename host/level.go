package host

// level.go — the watchdog for a level the system gives no word about: another
// program's memory use on a graphics card, a launcher's argv in /proc. It is
// the one place such a level is read on a period; everything that can say
// when it moves is read once, by that word (Now).

import (
	"context"
	"errors"
	"reflect"
	"runtime"
	"sync"
	"syscall"
	"time"
)

// Level reports the changes of a value read from a system that gives no
// event when it moves. A change is reported only once it has held: two
// reads in a row agree with each other and differ from the value last
// reported. One odd reading (a card's memory mid-allocation) is never a
// change.
type Level[T any] struct {
	period  time.Duration
	read    func() (T, bool)
	same    func(a, b T) bool
	changed func(T)

	mu      sync.Mutex
	armed   bool
	stopped bool
	reading bool        // a read goroutine is running
	again   bool        // a read was asked for while one ran
	gen     uint64      // bumped by Disarm: a read of an earlier arming is dropped
	timer   *time.Timer // the next period read, while armed
	last    T
	hasLast bool
	pend    T // a reading that differs from last and has not yet been confirmed
	hasPend bool
}

// NewLevel makes a Level, not yet armed. read answers the level, or false
// for no reading (ignored). same says two readings are one value (nil:
// reflect.DeepEqual). changed is called with each held change, never
// concurrently with itself and never under a lock of the Level.
func NewLevel[T any](period time.Duration, read func() (T, bool), same func(a, b T) bool, changed func(T)) *Level[T] {
	if same == nil {
		same = func(a, b T) bool { return reflect.DeepEqual(a, b) }
	}
	return &Level[T]{period: period, read: read, same: same, changed: changed}
}

// Arm starts the reads every period; the first is at once, in a goroutine.
// The first value that holds is reported, as the baseline. Arm of an armed
// Level does nothing.
func (l *Level[T]) Arm() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.armed || l.stopped {
		return
	}
	l.armed = true
	l.request()
}

// Disarm stops the period reads and forgets everything: the last reported
// value and any reading not yet confirmed. The next Arm reports a fresh
// baseline. A read in flight is dropped.
func (l *Level[T]) Disarm() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.armed = false
	l.quiet()
}

// Now says an event elsewhere may have moved the level: it is read at once,
// in a goroutine, and counts toward the hold like any read. A Now while a
// read is running is one more read after it. It works armed or not; while
// not armed a reading that differs stays pending, and the next Now or Arm
// read confirms it.
func (l *Level[T]) Now() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.stopped {
		return
	}
	l.request()
}

// Last is the value last reported, and false if none has been since Arm.
func (l *Level[T]) Last() (T, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.last, l.hasLast
}

// Stop ends the Level for good: no more reads are started and none is
// reported. A read already running finishes in its goroutine and is dropped.
func (l *Level[T]) Stop() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.stopped = true
	l.armed = false
	l.quiet()
}

// quiet drops the reads of the arming that ends, the timer, the last value
// and the pending one. The lock is held.
func (l *Level[T]) quiet() {
	l.gen++
	var zero T
	l.last, l.pend = zero, zero
	l.hasLast, l.hasPend = false, false
	if l.timer != nil {
		l.timer.Stop()
		l.timer = nil
	}
}

// request starts a read, or owes one after the one running. The lock is held.
func (l *Level[T]) request() {
	if l.reading {
		l.again = true
		return
	}
	l.reading = true
	go l.run()
}

// run reads until no read is owed, one at a time.
func (l *Level[T]) run() {
	for {
		l.mu.Lock()
		gen := l.gen
		l.mu.Unlock()

		v, ok := l.read()

		l.mu.Lock()
		report := ok && l.take(gen, v)
		l.mu.Unlock()
		if report {
			l.changed(v)
		}

		l.mu.Lock()
		if l.again && !l.stopped {
			l.again = false
			l.mu.Unlock()
			continue
		}
		l.again = false
		l.reading = false
		if l.armed && !l.stopped {
			if l.timer != nil {
				l.timer.Stop()
			}
			// watchdog for silence: the system says nothing when this level moves, so it is read every period while armed
			l.timer = time.AfterFunc(l.period, l.tick)
		}
		l.mu.Unlock()
		return
	}
}

// tick is a period come round.
func (l *Level[T]) tick() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.armed && !l.stopped {
		l.request()
	}
}

// take counts reading v of arming gen toward the hold and says whether it
// is a change to report (and has become the last value). The lock is held.
func (l *Level[T]) take(gen uint64, v T) bool {
	if gen != l.gen || l.stopped {
		return false
	}
	if l.hasLast && l.same(v, l.last) {
		l.hasPend = false
		return false
	}
	if l.hasPend && l.same(v, l.pend) {
		l.last, l.hasLast = v, true
		l.hasPend = false
		return true
	}
	l.pend, l.hasPend = v, true
	return false
}

// unheldPeriod is how often a held file is looked at again.
const unheldPeriod = 150 * time.Millisecond

// IsHeld says whether err means "another process has the file open" (on
// Windows a sharing or lock violation, or access denied while a scan has it),
// the one answer that is worth looking at again. Elsewhere nothing is held
// that way, and it is false.
func IsHeld(err error) bool {
	var en syscall.Errno
	if runtime.GOOS != "windows" || !errors.As(err, &en) {
		return false
	}
	return en == 5 || en == 32 || en == 33 // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
}

// Unheld runs try (an operation on a file) until it works. When it fails
// because another process holds the file (IsHeld: an antivirus scan, an
// indexer, a just-stopped process whose handle lingers), the system gives no
// event for the letting go, so the file is looked at again every
// unheldPeriod; any other failure is returned at once. It returns nil when
// try worked, and ctx's error (joined with the last held error) when ctx ends
// first. A caller whose holder is a process it knows (its own child) should
// wait for that process's exit (Exits) before calling it, and give ctx a
// bound.
func Unheld(ctx context.Context, try func() error) error {
	return unheld(ctx, try, IsHeld)
}

// unheld is Unheld with the meaning of "held" given.
func unheld(ctx context.Context, try func() error, held func(error) bool) error {
	for {
		err := try()
		if err == nil || !held(err) {
			return err
		}
		// watchdog for silence: a watchdog for a thing with no event (nothing says a held file was let go), armed only while the answer matters (ctx not ended)
		t := time.NewTimer(unheldPeriod)
		select {
		case <-t.C:
		case <-ctx.Done():
			t.Stop()
			return errors.Join(ctx.Err(), err)
		}
	}
}
