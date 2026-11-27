// Package host is every word the operating system gives the engine: a
// process's end, a file's change, a device opened, a registry key changed;
// and the only watchdogs for what it does not give (docs/EVENTS.md sections
// 1 and 4). Nothing outside host shells out or reads the system on a clock.
//
// Each word is a channel that is signalled (or closed, for an end) when the
// system says the thing, and a func that lets the watch go. Where a system
// gives no such word, the function answers ErrNoEvent, and the caller knows
// there is none rather than guessing at a cadence.
//
// The package imports only the standard library and golang.org/x/sys, and
// builds with CGO_ENABLED=0 for linux, windows and darwin.
package host

import (
	"errors"
	"sync"
	"time"
)

// ErrNoEvent is the answer where this system gives no event for the thing
// asked.
var ErrNoEvent = errors.New("host: this system gives no event for that")

// ErrNotThatProcess is ExitsOf's answer when the pid is now another
// process's: its start differs from the token given.
var ErrNotThatProcess = errors.New("host: that pid is another process now")

// gone is an end that has come already.
func gone() <-chan struct{} {
	c := make(chan struct{})
	close(c)
	return c
}

// signal puts one coalesced signal on c, never blocking.
func signal(c chan struct{}) {
	select {
	case c <- struct{}{}:
	default: // it is already owed one
	}
}

// ExitsOf is Exits for the process that has pid and was started at token
// (from StartOf): a pid that is another process's now is refused with
// ErrNotThatProcess, so a number the system gave again is never followed. A
// pid that has no process is ended: a closed channel, nil error.
func ExitsOf(pid int, token string) (<-chan struct{}, func(), error) {
	ended, stop, err := Exits(pid)
	if err != nil {
		return nil, nil, err
	}
	// asked after the watch is made, so the watch holds the process the
	// token is checked against (a pidfd or a handle pins it on linux and
	// windows; on darwin the window between is the kernel's own)
	now, err := StartOf(pid)
	if err != nil {
		select {
		case <-ended:
			return ended, stop, nil // it ended meanwhile
		default:
		}
		stop()
		return nil, nil, err
	}
	if now != token {
		stop()
		return nil, nil, ErrNotThatProcess
	}
	return ended, stop, nil
}

// Silence is a watchdog for silence: fired is called once when Heard has not
// been called for d. It is the honest timer of a child or a stream that may
// hang without a word, armed only while that silence could mean something.
type Silence struct {
	mu    sync.Mutex
	d     time.Duration
	fired func()
	timer *time.Timer
	armed uint64 // which arming is live: a timer of an earlier one fires nothing
}

// NewSilence arms a watchdog for silence of d.
func NewSilence(d time.Duration, fired func()) *Silence {
	s := &Silence{d: d, fired: fired}
	s.mu.Lock()
	s.arm()
	s.mu.Unlock()
	return s
}

// arm starts a new arming. The lock is held.
func (s *Silence) arm() {
	s.armed++
	n := s.armed
	s.timer = time.AfterFunc(s.d, func() { s.come(n) }) // watchdog for silence
}

// come is arming n come round with nothing heard.
func (s *Silence) come(n uint64) {
	s.mu.Lock()
	live := s.timer != nil && s.armed == n
	if live {
		s.timer = nil
	}
	s.mu.Unlock()
	if live {
		s.fired()
	}
}

// Heard puts the watchdog back: something was heard. After it has fired or
// been stopped, Heard does nothing.
func (s *Silence) Heard() {
	s.mu.Lock()
	if s.timer != nil {
		s.timer.Stop()
		s.arm()
	}
	s.mu.Unlock()
}

// Stop disarms the watchdog. It says whether it was still armed (fired has
// not been called and will not be).
func (s *Silence) Stop() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.timer == nil {
		return false
	}
	s.timer.Stop()
	s.timer = nil
	return true
}

// Moment calls fired once after d: a named moment, a time somebody chose
// (the keeper's --linger, --yield-grace, --yield-steady). The func it
// answers cancels it and says whether it was still to come.
func Moment(d time.Duration, fired func()) (cancel func() bool) {
	t := time.AfterFunc(d, fired) // named moment
	return t.Stop
}
