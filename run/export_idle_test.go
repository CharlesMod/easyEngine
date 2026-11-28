package run

// HoldOp takes the operation lock (as a reshape or a swap holds it); the func lets it go.
func (s *Supervisor) HoldOp() func() {
	s.opMu.Lock()
	return s.opMu.Unlock
}

// SayPhase says a phase as the engine's own work would (a test standing in for a reshape).
func (s *Supervisor) SayPhase(phase, msg string) { s.setPhase(phase, msg) }
