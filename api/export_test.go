package api

// RemoteReadsNow is how many remote bodies are being read now.
func (a *API) RemoteReadsNow() int32 {
	a.heldMu.Lock()
	defer a.heldMu.Unlock()
	return int32(a.reads.all)
}

// SayReads tells f each new count of remote bodies being read (set before the handler serves).
func (a *API) SayReads(f func(int32)) { a.readsSaid = f }
