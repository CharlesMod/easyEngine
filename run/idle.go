package run

import (
	"errors"
	"sync/atomic"
	"time"
)

// The idle rule's side in the supervisor (docs/ENGINE.md "Idle sleep"): why the phase is
// sleeping, and the idle fact the rule's named moment is said by. The rule itself, which knows
// the app's parents and attachments, is app/idle.go; it puts the model to sleep with SleepIdle.

// RemoteIdleFloor is the least quiet before an idle sleep that follows remote traffic (a remote
// client's leave, a request from another machine, anywhere in the busy stretch) or a wake
// somebody asked for (POST wake, an attach that woke the model): max(delay, RemoteIdleFloor).
// With linger_s or idle_sleep_s 0, a remote identity could otherwise load and unload the model
// as often as it liked (a start from down is free, an idle sleep is the engine's own stop), and
// a local caller would pay a reload each time; a wake asked for would be undone at once. Local
// leaves and local requests keep the exact settings. Config.IdleFloor overrides it (tests).
const RemoteIdleFloor = 60 * time.Second

// ErrIdleBusy is SleepIdle finding, at its decision, a request in flight or held, or the engine
// not at rest (not ready, a swap, a job, a reshape, a yield): nothing was changed.
var ErrIdleBusy = errors.New("idle sleep: not idle at the decision")

// SleepIdle is the idle rule's sleep (docs/ENGINE.md "Idle sleep"). The decision is one step
// with the door: under the operation lock and the supervisor's lock (the lock a request takes
// to be routed and begun), it looks that nothing is in flight or held and the engine is at rest,
// and says phase sleeping. A request begun before that makes it answer ErrIdleBusy (no sleep);
// one after it is held and wakes the model as at any sleeping engine (its Start waits for the
// operation lock). So no request is forwarded to a server this is about to stop. The sleep is
// then a sleep like any other: the slots saved, the process group stopped, budgets whole.
func (s *Supervisor) SleepIdle() error { return s.SleepIdleSince(s.RestSeq()) }

// RestSeq counts the phase's changes: the idle rule takes it when its quiet begins and gives it
// to SleepIdleSince, which declines when the phase has left ready (a swap, a reshape) since.
func (s *Supervisor) RestSeq() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.phaseSeq
}

// SleepIdleSince is SleepIdle for a quiet that began at rest seq: the phase must not have
// changed since (coming back to ready begins a new quiet).
func (s *Supervisor) SleepIdleSince(seq uint64) error {
	if f := s.cfg.IdleBeforeOp; f != nil {
		f()
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	s.mu.Lock()
	if s.phaseSeq != seq || !s.idleSleepableLocked() || s.inflightNow() != 0 || s.door.held.Load() != 0 {
		s.mu.Unlock()
		return ErrIdleBusy
	}
	s.sleepReason = "idle"
	s.setPhaseLocked("sleeping", IdleMessage)
	s.mu.Unlock()
	s.logf("phase sleeping: %s", IdleMessage)
	if f := s.cfg.IdleDecided; f != nil {
		f()
	}
	s.keepContexts("a sleep", false)
	s.stopServerLocked()
	s.refillBudgets("the model was unloaded")
	return nil
}

// IdleSleepable says the engine is at rest with a model loaded: phase ready, local, no swap,
// job, reshape, drain or yield under way. The idle rule's quiet counts only while it holds.
func (s *Supervisor) IdleSleepable() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.idleSleepableLocked()
}

func (s *Supervisor) idleSleepableLocked() bool {
	if s.phase != "ready" || s.st.Mode != "local" || s.swapping || s.jobRunning || s.removing ||
		s.reshape || s.yieldInfo != nil || s.srv == nil || s.srv.drain != nil {
		return false
	}
	if u := s.upg; u != nil && u.state == upSwapping {
		return false
	}
	return true
}

// IdleFloor is RemoteIdleFloor, or Config.IdleFloor when set.
func (s *Supervisor) IdleFloor() time.Duration {
	if s.cfg.IdleFloor > 0 {
		return s.cfg.IdleFloor
	}
	return RemoteIdleFloor
}

// idleSeen counts what the idle rule asks of the clients' side: requests that came to the door
// (held or not) and attachments that left, and of each how many were not from this machine (the
// attachment's own local flag, the request's DoorAsk.Local: decided by the api, never a header).
type idleSeen struct {
	arrivals, remoteArrivals atomic.Uint64
	leaves, remoteLeaves     atomic.Uint64
	wakes                    atomic.Uint64 // Wake calls: POST wake, an attach's wake, a request's
	last                     atomic.Int64  // unix ns of the last arrival or request end
}

func (c *idleSeen) arrive(local bool) {
	if !local {
		c.remoteArrivals.Add(1)
	}
	c.touch()
	c.arrivals.Add(1)
}

// touch records when the door last saw traffic (an arrival, a request's end).
func (c *idleSeen) touch() { c.last.Store(time.Now().UnixNano()) }

func (c *idleSeen) leave(local bool) {
	if !local {
		c.remoteLeaves.Add(1)
	}
	c.leaves.Add(1)
}

// IdleTraffic is what the idle rule compares between looks (Supervisor.IdleSeen).
type IdleTraffic struct {
	Arrivals, RemoteArrivals, Leaves, RemoteLeaves, Wakes uint64
	Last                                                  time.Time // the last arrival or request end
}

// IdleSeen is the door's arrivals and the attachments' leaves so far, each with how many were
// remote, the wakes, and when the door last saw traffic.
func (s *Supervisor) IdleSeen() IdleTraffic {
	c := &s.idleSeen
	t := IdleTraffic{Arrivals: c.arrivals.Load(), RemoteArrivals: c.remoteArrivals.Load(),
		Leaves: c.leaves.Load(), RemoteLeaves: c.remoteLeaves.Load(), Wakes: c.wakes.Load()}
	if n := c.last.Load(); n != 0 {
		t.Last = time.Unix(0, n)
	}
	return t
}

// IdleForce is what the command line forces for this run: --no-idle-sleep (Off), --idle-sleep D
// (Quiet, with QuietSet). Status and the idle fact say the effective values and that they are
// forced; PUT settings refuses a value a flag overrides.
type IdleForce struct {
	Off      bool
	Quiet    time.Duration
	QuietSet bool
}

// SetIdleForce records the flags (the app does, before it answers anyone).
func (s *Supervisor) SetIdleForce(f IdleForce) {
	s.mu.Lock()
	s.idleForce = f
	s.sayIdleLocked()
	s.mu.Unlock()
}

// IdleForced is the flags SetIdleForce recorded.
func (s *Supervisor) IdleForced() IdleForce {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.idleForce
}

// idleEffectiveLocked is the idle rule's settings as they act: the file's, under the flags.
func (s *Supervisor) idleEffectiveLocked() (on bool, quiet time.Duration, forced bool) {
	on, quiet = s.st.IdleSleep && !s.idleForce.Off, time.Duration(s.st.IdleSleepS)*time.Second
	if s.idleForce.QuietSet {
		quiet = s.idleForce.Quiet
	}
	return on, quiet, s.idleForce.Off || s.idleForce.QuietSet
}

// idleMoment is the idle rule's named moment as the rule last said it.
type idleMoment struct {
	armed       bool
	why         string // <delay>/<cause>: linger/local_leave, idle_sleep/remote_requests, ...
	delay, left time.Duration
}

// sleepPhase says phase sleeping and why (status sleep_reason). An empty reason keeps the one
// said (a sleep that a swap's end says again), else it is asked.
func (s *Supervisor) sleepPhase(reason, msg string) {
	s.mu.Lock()
	switch {
	case reason != "":
		s.sleepReason = reason
	case s.phase != "sleeping" || s.sleepReason == "":
		s.sleepReason = "asked"
	}
	s.setPhaseLocked("sleeping", msg)
	s.mu.Unlock()
	s.logf("phase sleeping: %s", msg)
}

// StoppedByPerson says a person's stop stands (POST stop, the tray): a client's attach does not
// wake the model then, as a request does not (routeLocked).
func (s *Supervisor) StoppedByPerson() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.userStopped
}

// SleepReason is status sleep_reason.
func (s *Supervisor) SleepReason() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sleepReason
}

// IdleSettings are the idle rule's settings as they act (the flags over the file): on, the
// quiet for a caller that only sends requests (idle_sleep_s), the linger after the last local
// attachment left (linger_s).
func (s *Supervisor) IdleSettings() (on bool, quiet, linger time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	on, quiet, _ = s.idleEffectiveLocked()
	return on, quiet, time.Duration(s.st.LingerS) * time.Second
}

// SetIdleMoment is the idle rule saying its named moment: armed (why, for delay from the start
// of the quiet, left of it when armed) or not. The idle fact is said only when this changes.
func (s *Supervisor) SetIdleMoment(armed bool, why string, delay, left time.Duration) {
	m := idleMoment{armed: armed, why: why, delay: delay, left: left}
	if !armed {
		m = idleMoment{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.idleMoment == m {
		return
	}
	s.idleMoment = m
	s.sayIdleLocked()
}

// sayIdleLocked says the idle fact: the settings and the moment (book: idle).
func (s *Supervisor) sayIdleLocked() {
	on, quiet, forced := s.idleEffectiveLocked()
	b := map[string]any{"idle_sleep": on, "idle_sleep_s": int(quiet / time.Second), "idle_sleep_forced": forced,
		"linger_s": s.st.LingerS, "armed": s.idleMoment.armed}
	if s.idleMoment.armed {
		b["why"] = s.idleMoment.why
		b["delay_ms"] = s.idleMoment.delay.Milliseconds()
		b["left_ms"] = s.idleMoment.left.Milliseconds()
	}
	s.say("idle", "", b)
}
