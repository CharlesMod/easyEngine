package run

// CtxKeyString is the prompt-cache key of the running model server ("" when none runs).
func (s *Supervisor) CtxKeyString() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.srv == nil {
		return ""
	}
	return s.srv.ctxKey.String()
}
