package host

// level.go — the watchdog for a level the system gives no word about: another
// program's memory use on a graphics card, a launcher's argv in /proc. It is
// the one place such a level is read on a period; everything that can say
// when it moves is read once, by that word (Now).

import (
	"reflect"
	"sync"
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

// Unheld tries for a file another process holds (an antivirus scan, an
// indexer): there is no event for its letting go. try is called up to 8
// times, 150 ms apart, and stops at the first that answers true; Unheld
// says whether one did.
func Unheld(try func() bool) bool {
	for i := 0; i < 8; i++ {
		if try() {
			return true
		}
		if i < 7 {
			// watchdog for silence: a held file says nothing when it is let go: it is tried again after a short silence, 8 times at most
			t := time.NewTimer(150 * time.Millisecond)
			<-t.C
		}
	}
	return false
}
