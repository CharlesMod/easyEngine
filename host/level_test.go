package host

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// A rig is a Level whose reads wait for the test: each read says it has
// started, then answers what the test feeds it. Reads are one at a time, so
// when the next read has started the one before has been counted and
// reported: the tests are exact, not timed.
type rig struct {
	t       *testing.T
	feed    chan rd
	started chan struct{}
	got     chan int
	reads   atomic.Int64
	lv      *Level[int]
}

type rd struct {
	v  int
	ok bool
}

func newRig(t *testing.T, period time.Duration) *rig {
	r := &rig{t: t, feed: make(chan rd), started: make(chan struct{}, 1024), got: make(chan int, 1024)}
	r.lv = NewLevel(period, func() (int, bool) {
		r.reads.Add(1)
		r.started <- struct{}{}
		x := <-r.feed // closed at cleanup: not ok
		return x.v, x.ok
	}, nil, func(v int) { r.got <- v })
	t.Cleanup(func() {
		r.lv.Stop()
		close(r.feed)
	})
	return r
}

// begun waits for a read to start.
func (r *rig) begun() {
	r.t.Helper()
	tm := time.NewTimer(levelBound) // watchdog for silence: test-only, a broken Level fails instead of hanging
	defer tm.Stop()
	select {
	case <-r.started:
	case <-tm.C:
		r.t.Fatal("no read started")
	}
}

// next asks for a read with Now and answers it v.
func (r *rig) next(v int) {
	r.t.Helper()
	r.lv.Now()
	r.begun()
	r.feed <- rd{v, true}
}

// settle makes the read before it counted and reported, then answers its own
// read with no reading.
func (r *rig) settle() {
	r.t.Helper()
	r.lv.Now()
	r.begun()
	r.feed <- rd{}
}

// reported is what has been reported since the last call.
func (r *rig) reported() []int {
	var out []int
	for {
		select {
		case v := <-r.got:
			out = append(out, v)
		default:
			return out
		}
	}
}

func eq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestLevelBaselineOnce(t *testing.T) {
	r := newRig(t, time.Hour)
	r.next(5)
	r.next(5)
	r.settle()
	if got := r.reported(); !eq(got, []int{5}) {
		t.Fatalf("baseline: reported %v, want [5]", got)
	}
	if v, ok := r.lv.Last(); !ok || v != 5 {
		t.Fatalf("Last = %v %v", v, ok)
	}
	r.next(5)
	r.next(5)
	r.settle()
	if got := r.reported(); len(got) != 0 {
		t.Fatalf("same value reported again: %v", got)
	}
}

func TestLevelOneDifferingReadIsNotAChange(t *testing.T) {
	r := newRig(t, time.Hour)
	r.next(5)
	r.next(5)
	r.next(9) // odd one
	r.next(5)
	r.settle()
	if got := r.reported(); !eq(got, []int{5}) {
		t.Fatalf("reported %v, want [5]", got)
	}
	// an odd one then another odd one that differs from it: still none
	r.next(9)
	r.next(8)
	r.next(5)
	r.settle()
	if got := r.reported(); len(got) != 0 {
		t.Fatalf("reported %v, want none", got)
	}
}

func TestLevelTwoAgreeingReadsAreOneChange(t *testing.T) {
	r := newRig(t, time.Hour)
	r.next(5)
	r.next(5)
	r.next(9)
	r.next(9)
	r.next(9)
	r.settle()
	if got := r.reported(); !eq(got, []int{5, 9}) {
		t.Fatalf("reported %v, want [5 9]", got)
	}
	if v, _ := r.lv.Last(); v != 9 {
		t.Fatalf("Last = %d", v)
	}
}

func TestLevelNoReadingIsIgnored(t *testing.T) {
	r := newRig(t, time.Hour)
	r.next(5)
	r.lv.Now()
	r.begun()
	r.feed <- rd{} // no reading: neither confirms nor breaks
	r.next(5)
	r.settle()
	if got := r.reported(); !eq(got, []int{5}) {
		t.Fatalf("reported %v, want [5]", got)
	}
}

func TestLevelNowWhileReadingIsOneMoreRead(t *testing.T) {
	r := newRig(t, time.Hour)
	r.lv.Now()
	r.begun()
	r.lv.Now() // while the read waits: coalesces to one more
	r.lv.Now()
	r.feed <- rd{5, true}
	r.begun() // the one more
	r.feed <- rd{5, true}
	r.settle()
	if n := r.reads.Load(); n != 3 {
		t.Fatalf("%d reads, want 3 (one, one more, settle)", n)
	}
	if got := r.reported(); !eq(got, []int{5}) {
		t.Fatalf("reported %v, want [5]", got)
	}
}

func TestLevelReadsAreOneAtATime(t *testing.T) {
	var in, worst atomic.Int64
	got := make(chan int, 16)
	lv := NewLevel(time.Hour, func() (int, bool) {
		n := in.Add(1)
		for {
			w := worst.Load()
			if n <= w || worst.CompareAndSwap(w, n) {
				break
			}
		}
		in.Add(-1)
		return 1, true
	}, nil, func(v int) { got <- v })
	defer lv.Stop()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); lv.Now() }()
	}
	wg.Wait()
	tm := time.NewTimer(levelBound) // watchdog for silence: test-only
	defer tm.Stop()
	select {
	case <-got:
	case <-tm.C:
		t.Fatal("nothing reported")
	}
	if w := worst.Load(); w != 1 {
		t.Fatalf("%d reads at once", w)
	}
}

func TestLevelArmReadsEveryPeriodAndDisarmStops(t *testing.T) {
	r := newRig(t, time.Millisecond)
	r.lv.Arm()
	r.lv.Arm() // idempotent
	for i := 0; i < 4; i++ {
		r.begun() // the first at once, the rest by the period
		r.feed <- rd{5, true}
	}
	r.begun() // a fifth is in flight
	r.lv.Disarm()
	r.feed <- rd{5, true} // dropped: it was of the arming that ended
	if _, ok := r.lv.Last(); ok {
		t.Fatal("Disarm kept the last value")
	}
	n := r.reads.Load()
	tm := time.NewTimer(100 * time.Millisecond) // watchdog for silence: test-only, no read must come after Disarm
	defer tm.Stop()
	select {
	case <-r.started:
		t.Fatal("read after Disarm")
	case <-tm.C:
	}
	if r.reads.Load() != n {
		t.Fatal("read after Disarm")
	}
	if got := r.reported(); !eq(got, []int{5}) {
		t.Fatalf("reported %v, want one baseline", got)
	}
	// armed again: a fresh baseline
	r.lv.Arm()
	r.begun()
	r.feed <- rd{5, true}
	r.begun()
	r.feed <- rd{5, true}
	r.begun()
	if got := r.reported(); !eq(got, []int{5}) {
		t.Fatalf("after re-Arm reported %v, want [5]", got)
	}
}

func TestLevelNowWhileDisarmedKeepsPending(t *testing.T) {
	r := newRig(t, time.Hour)
	r.next(7) // pending, not reported: no second read was made
	r.lv.Now()
	r.begun()
	if got := r.reported(); len(got) != 0 {
		t.Fatalf("reported %v after one read", got)
	}
	r.feed <- rd{7, true} // the next Now confirms
	r.settle()
	if got := r.reported(); !eq(got, []int{7}) {
		t.Fatalf("reported %v, want [7]", got)
	}
}

func TestLevelStopEndsIt(t *testing.T) {
	r := newRig(t, time.Millisecond)
	r.lv.Stop()
	r.lv.Arm()
	r.lv.Now()
	tm := time.NewTimer(100 * time.Millisecond) // watchdog for silence: test-only, no read must come after Stop
	defer tm.Stop()
	select {
	case <-r.started:
		t.Fatal("read after Stop")
	case <-tm.C:
	}
}

func TestLevelCustomSame(t *testing.T) {
	got := make(chan int, 8)
	var n atomic.Int64
	lv := NewLevel(time.Hour, func() (int, bool) {
		return int(n.Add(1)), true // always a new number
	}, func(a, b int) bool { return a/10 == b/10 }, func(v int) { got <- v })
	defer lv.Stop()
	// readings 1 and 2 are the same value under same
	lv.Now()
	lv.Now()
	tm := time.NewTimer(levelBound) // watchdog for silence: test-only
	defer tm.Stop()
	select {
	case <-got:
	case <-tm.C:
		t.Fatal("nothing reported")
	}
}

func TestUnheld(t *testing.T) {
	errHeld := errors.New("held")
	errOther := errors.New("other")
	held := func(err error) bool { return err == errHeld }
	ctx := context.Background()

	calls := 0
	if err := unheld(ctx, func() error { calls++; return nil }, held); err != nil || calls != 1 {
		t.Fatalf("first try: %v, %d calls", err, calls)
	}
	// another error returns at once, never looked at again
	calls = 0
	if err := unheld(ctx, func() error { calls++; return errOther }, held); err != errOther || calls != 1 {
		t.Fatalf("other error: %v, %d calls", err, calls)
	}
	// held keeps looking (no fixed count) until it is let go
	calls = 0
	if err := unheld(ctx, func() error {
		calls++
		if calls < 10 {
			return errHeld
		}
		return nil
	}, held); err != nil || calls != 10 {
		t.Fatalf("let go: %v, %d calls", err, calls)
	}
	// held for ever ends with ctx
	cctx, cancel := context.WithCancel(ctx)
	calls = 0
	err := unheld(cctx, func() error {
		calls++
		if calls == 2 {
			cancel()
		}
		return errHeld
	}, held)
	if !errors.Is(err, context.Canceled) || !errors.Is(err, errHeld) {
		t.Fatalf("ctx end: %v", err)
	}
	// IsHeld is false off Windows
	if runtime.GOOS != "windows" && IsHeld(syscall.Errno(32)) {
		t.Fatal("IsHeld true off windows")
	}
}

// levelBound is the test-only watchdog for silence on the waits above.
const levelBound = 10 * time.Second
