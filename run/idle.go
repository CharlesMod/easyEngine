package run

import (
	"time"
)

// The idle rule's side in the supervisor (docs/ENGINE.md "Idle sleep"): why the phase is
// sleeping, and the idle fact the rule's named moment is said by. The rule itself, which knows
// the app's parents and attachments, is app/idle.go; it puts the model to sleep with SleepIdle.

// idleMoment is the idle rule's named moment as the rule last said it.
type idleMoment struct {
	armed bool
	why   string // linger | idle_sleep
	delay time.Duration
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

// SleepReason is status sleep_reason.
func (s *Supervisor) SleepReason() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sleepReason
}

// IdleSettings are the idle rule's settings: on, the quiet for a caller that only sends
// requests (idle_sleep_s), the linger after the last attachment left (linger_s).
func (s *Supervisor) IdleSettings() (on bool, quiet, linger time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.IdleSleep, time.Duration(s.st.IdleSleepS) * time.Second, time.Duration(s.st.LingerS) * time.Second
}

// SetIdleMoment is the idle rule saying its named moment: armed (why: linger | idle_sleep, for
// delay from the start of the quiet) or not. The idle fact is said only when this changes.
func (s *Supervisor) SetIdleMoment(armed bool, why string, delay time.Duration) {
	m := idleMoment{armed: armed, why: why, delay: delay}
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
	b := map[string]any{"idle_sleep": s.st.IdleSleep, "idle_sleep_s": s.st.IdleSleepS, "linger_s": s.st.LingerS,
		"armed": s.idleMoment.armed}
	if s.idleMoment.armed {
		b["why"] = s.idleMoment.why
		b["delay_ms"] = s.idleMoment.delay.Milliseconds()
	}
	s.say("idle", "", b)
}
