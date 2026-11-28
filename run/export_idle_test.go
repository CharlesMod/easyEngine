package run

// SetIdleBeforeOp runs f when SleepIdle is about to wait for the operation lock.
func SetIdleBeforeOp(f func()) func() {
	idleBeforeOp.Store(&f)
	return func() { idleBeforeOp.Store(nil) }
}

// SetIdleDecided runs f when SleepIdle has decided (phase sleeping said) and before the stop.
func SetIdleDecided(f func()) func() {
	idleDecided.Store(&f)
	return func() { idleDecided.Store(nil) }
}

// HoldOp takes the operation lock (as a reshape or a swap holds it); the func lets it go.
func (s *Supervisor) HoldOp() func() {
	s.opMu.Lock()
	return s.opMu.Unlock
}
