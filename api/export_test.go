package api

// RemoteReadsNow is how many remote bodies are being read now.
func (a *API) RemoteReadsNow() int32 { return a.remoteReads.Load() }

// SayReads tells f each new count of remote bodies being read (set before the handler serves).
func (a *API) SayReads(f func(int32)) { a.readsSaid = f }

// SayKept tells f each new count of the remote bytes kept (remoteKept), in order.
func (a *API) SayKept(f func(int64)) { a.keptSaid.Store(&f) }
