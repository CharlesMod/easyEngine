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
//
// id is the request's id (X-Request-Id), said in the request begin and end so a client pairs
// them; end takes what the streamer read of the reply (nil for one it did not read).
func (s *Supervisor) Enter(id string) (Route, func(*RequestEnd)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rt := s.routeLocked()
	if rt.Hold {
		return rt, nil
	}
	return rt, s.begin(id)
}

// RequestEnd is what the streamer read of a streamed reply (api fills it from stream.Summary):
// how it ended, the finish reason, the token counts, the decode rate and timings, the slot. It
// is said in the request end happening; a decode rate it carries is status tok_s.
type RequestEnd struct {
	End              string  `json:"end,omitempty"` // done | upstream_error | ended_early | client_gone
	Events           int     `json:"events,omitempty"`
	FinishReason     string  `json:"finish_reason,omitempty"`
	PromptTokens     int     `json:"prompt_tokens,omitempty"`
	CachedTokens     int     `json:"cached_tokens,omitempty"`
	CompletionTokens int     `json:"completion_tokens,omitempty"`
	TokS             float64 `json:"tok_s,omitempty"`
	PromptMs         float64 `json:"prompt_ms,omitempty"`
	PredictedMs      float64 `json:"predicted_ms,omitempty"`
	Slot             *int    `json:"slot,omitempty"`
	// Attachment is the client's attachment the request was for (X-Easy-Client), ClientSlot the
	// slot in its own numbering (run/clients.go); Slot stays the engine's.
	Attachment string `json:"attachment,omitempty"`
	ClientSlot *int   `json:"client_slot,omitempty"`
}

// requestSaid is the body of the request happening.
type requestSaid struct {
	What     string `json:"what"` // begin | end
	Inflight int    `json:"inflight"`
	ID       string `json:"id,omitempty"`
	// Attachment is the attachment a request is for (X-Easy-Client), said in its begin.
	Attachment string `json:"attachment,omitempty"`
	*RequestEnd
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

// Begun is the count of requests that have passed the door in this life; it only grows. A
// caller that must know whether a request came since some moment compares two readings of it
// (what stands, not the history of happenings).
func (s *Supervisor) Begun() uint64 { return s.reqSeq() }
