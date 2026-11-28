package api

// RemoteReadsNow is how many remote bodies are being read now.
func (a *API) RemoteReadsNow() int32 { return a.remoteReads.Load() }
