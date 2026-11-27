package run

import "sync"

// The door holds (docs/ENGINE.md, "The door holds"; docs/EVENTS.md section 3): a request that
// arrives while the model is on its way up is held and answered when the model is ready, instead
// of a 503. The holding itself is api's (it waits on the phase fact and the client's context);
// run says what holds (Route.Hold), counts the held (the door fact) and takes a held request as
// a word: a load asks /health once on it.

// APIVersion is the version of the control API (/engine/v1), said in the engine fact and in
// status. A client that needs the stream or the hold reads it.
const APIVersion = 1

// Enter is a request at the door. It answers the route and, unless the route holds, counts the
// request in flight (call end when it ends; end is nil for a held route). The route and the
// count are taken under one lock, so a reshape that says its phase before it drains cannot slip
// in between. A held request is not in flight: the hang watchdog does not arm for a request the
// model server has not seen, and a drain does not wait for a request that waits for it.
func (s *Supervisor) Enter() (Route, func()) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rt := s.routeLocked()
	if rt.Hold {
		return rt, nil
	}
	return rt, s.Begin()
}

// Hold counts one request held at the door until the model is up; call the result when the
// hold ends (it entered, or the client went away). The count is the door fact {held}. A request
// held while the server starts or loads is a word: the load asks /health once on it (a build
// whose /health turns 200 long after its last line is found ready by the next request, not by
// LoadTimeout). One ask per held request, never a loop. The resident app counts held requests
// (Held) so it never puts the model to sleep under one.
func (s *Supervisor) Hold() func() {
	s.heldMu.Lock() // the count and its fact in one step: two holds never say it out of order
	n := s.door.held.Add(1)
	s.say("door", "", map[string]int32{"held": n})
	s.heldMu.Unlock()
	if p := s.Phase(); p == "starting" || p == "loading" {
		select {
		case s.loadAsk <- struct{}{}:
		default:
		}
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			s.heldMu.Lock()
			n := s.door.held.Add(-1)
			s.say("door", "", map[string]int32{"held": n})
			s.heldMu.Unlock()
		})
	}
}

// Held is the count of requests held at the door until the model is up (not in InFlight).
func (s *Supervisor) Held() int { return int(s.door.held.Load()) }
