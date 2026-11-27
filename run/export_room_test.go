package run

// Seams of the room watch for the external tests (room_test.go).

// RoomArm, RoomDisarm and RoomEvent are the room watch's arm, disarm and event.
func (s *Supervisor) RoomArm()    { s.card.arm() }
func (s *Supervisor) RoomDisarm() { s.card.disarm() }
func (s *Supervisor) RoomEvent()  { s.card.event() }

// SetRoomDevices points the room's device-open watch at these globs; it returns the undo.
func SetRoomDevices(globs []string) func() {
	old := roomDevices
	roomDevices = globs
	return func() { roomDevices = old }
}

// SetRoomReadDone sets what is called when a read of the room has ended (nil: nothing); SetRoomOpensEnd
// what is called when the goroutine reading an arming's opens has ended. Each returns the undo.
func SetRoomReadDone(f func()) func() {
	roomReadDone.Store(&f)
	return func() { roomReadDone.Store(nil) }
}

func SetRoomOpensEnd(f func()) func() {
	roomOpensEnd.Store(&f)
	return func() { roomOpensEnd.Store(nil) }
}

// SetRoomWordDone sets what is called when a word of the device nodes has been looked at, whether
// a read followed or not; SetHolderScan replaces the scan of who holds the nodes (pid -> name).
// Each returns the undo.
func SetRoomWordDone(f func()) func() {
	roomWordDone.Store(&f)
	return func() { roomWordDone.Store(nil) }
}

func SetHolderScan(f func() map[int]string) func() {
	holderScanHook.Store(&f)
	return func() { holderScanHook.Store(nil) }
}

// HolderScans is how many scans of the holders have been made.
func (s *Supervisor) HolderScans() int {
	s.holders.mu.Lock()
	defer s.holders.mu.Unlock()
	return s.holders.scans
}
