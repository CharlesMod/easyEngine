package run

import (
	"github.com/CharlesMod/easyEngine/event"
)

// The book: every name the supervisor says on its hub (docs/EVENTS.md section 2, "the first
// names"). GET /engine/v1/book serves it. A fact stands until replaced or said gone, a reading is
// a level said when it changes, a happening is history no choice may depend on.
var book = []event.Decl{
	{Name: "engine", Kind: event.Fact, What: "this engine: {version, api, port}; api is the control API's version (run.APIVersion)"},
	{Name: "door", Kind: event.Fact, What: "the door: {held}, the requests held until the model is up (not in flight; request counts those)"},
	{Name: "phase", Kind: event.Fact, What: "where the engine stands: {phase, message}, as status says them"},
	{Name: "model", Kind: event.Fact, What: "the selected model and the settings it runs with, as state.json holds them (no keys): {mode, model, backend, kv_type, target, previous, starter_owned, custom, sampler_policy, linger_s}"},
	{Name: "mind.shape", Kind: event.Fact, What: "the running shape (status shape); gone when no model server is up"},
	{Name: "vision", Kind: event.Fact, What: "the vision projector (status vision, vision_note): {vision, note}; vision is true while the running model server has it loaded, note says why not while vision is wanted"},
	{Name: "vision.progress", Kind: event.Reading, What: "the vision projector's download (status vision_progress), said per step of bytes; gone when none runs"},
	{Name: "mind.device", Kind: event.Fact, What: "what the last model server start was given to hold it to its card (status device)"},
	{Name: "mind.exit", Kind: event.Happening, What: "the model server ended on its own: {pid, why, quick, oom}"},
	{Name: "mind.hang", Kind: event.Happening, What: "the model server said nothing for HangSilence with requests in flight and was asked /health once: {answered}"},
	{Name: "mind.oom", Kind: event.Happening, What: "the model server said it ran out of memory and kept running: {line, held}"},
	{Name: "agent", Kind: event.Fact, Keyed: true, What: "the agent (key: the profile's game name, else \"agent\"; or the name of the attachment whose agent runs): status agent"},
	{Name: "install.progress", Kind: event.Reading, What: "the download or install step in progress (status progress), said per step of bytes; gone when none"},
	{Name: "job", Kind: event.Fact, What: "an install, import or remove job: {kind, running}; gone when none runs"},
	{Name: "build", Kind: event.Fact, What: "the llama.cpp build (status build): {backend, tag, patched, rung, why, pin, vmm, next: {backend, tag, state: downloading|installing|waiting|staged|failed, error, attempt, progress}}; tag is the release (b12651-easy1), rung its place in the catalog's ladder (0 the pin, -1 not named), why says why a build is not patched, vmm the patched server's CUDA buffer line, next the move to the pin (staged: fetched and verified, the next load smoke-tests and runs it; failed: its smoke test failed at a load); gone with no build installed"},
	{Name: "build.pruned", Kind: event.Happening, What: "the other builds of the backend that served ready were removed: {backend, removed: [paths under the data dir]}"},
	{Name: "upgrade", Kind: event.Fact, What: "the starter-to-target upgrade (status upgrade); gone when none"},
	{Name: "room", Kind: event.Reading, What: "the adapters and the measured reserve, as last sensed (status gpu)"},
	{Name: "room.held", Kind: event.Reading, What: "the reserve the room watch holds to (two reads agreed): {reserve_mib}; gone when the watch is let go"},
	{Name: "yield", Kind: event.Fact, What: "the card is yielded to another program, or replies drain before that (status yield); gone when not"},
	{Name: "recovery", Kind: event.Fact, What: "the crash and failed-start record (status recovery), with what a retry waits for; gone when none"},
	{Name: "error", Kind: event.Happening, What: "something went wrong: {message}, as status errors lists it"},
	{Name: "request", Kind: event.Happening, What: "a request at the door began or ended, or the door went idle: {what: begin|end|idle, inflight, id (X-Request-Id)}; a streamed reply's end adds what the streamer read: {end: done|upstream_error|ended_early|client_gone, events, finish_reason, prompt_tokens, cached_tokens, completion_tokens, tok_s, prompt_ms, predicted_ms, slot}; idle says tok_s read from /metrics; a request of an attachment (X-Easy-Client) adds attachment and client_slot (its own number; slot stays the engine's)"},
	{Name: "context", Kind: event.Happening, Keyed: true, What: "the prompt cache across a reshape (key: the slot, or \"all\" for a save that did not happen for any slot, slot -1): {what: saved|laid_back|let_go|not_saved, slot, tokens, bytes, why, client, client_slot}; client is the key of the attachment the slot served and client_slot its own number; status contexts lists what is kept"},
	{Name: "nudge", Kind: event.Happening, What: "a word that could change an outcome: a request at the door, a client's start or install: {why}"},
	{Name: "network", Kind: event.Happening, What: "the system said a network link or address changed"},
	{Name: "pairing", Kind: event.Fact, Keyed: true, What: "a pairing request waiting for a person's Allow or Deny (key: the request id): {name, code, from}; never a token; gone when answered, withdrawn or pushed out"},
	{Name: "client", Kind: event.Fact, Keyed: true, What: "a paired client seen in this life (key: its client id): {name, attachments}; attachments are its standing attachments' ids, said again when they change; never a token"},
	{Name: "attachment", Kind: event.Fact, Keyed: true, What: "one use of the engine by a client (key: the attachment id): {attachment, name, key, client_id, local, presence, pid, slots: {want, min, given, first, pending, unserved}, ctx_per_slot: {asked, given}, server: {taken, not_taken, checked}, model, agent}; gone when it leaves"},
	{Name: "wants", Kind: event.Fact, What: "the engine's wants, the attachments' merged: {slots, min_slots, ctx_max, ctx_floor, from, unserved: [{attachment, why}], vacant: [{first, count, key}], server}; vacant ranges are the running mind's slots whose attachment left"},
	{Name: "mark", Kind: event.Happening, What: "a marker said from outside (a test, a client) to order what came before it: {what}"},
}

// say puts one event on the hub. A refusal is a programming error (an undeclared name, a body
// too large); it is logged, never fatal.
func (s *Supervisor) say(name, key string, body any) {
	if s.hub == nil {
		return
	}
	if _, err := s.hub.Say(event.Event{Name: name, Key: key}, body); err != nil && err != event.ErrClosed {
		s.logf("hub: %v", err)
	}
}

// sayGone says a fact or reading stands no longer.
func (s *Supervisor) sayGone(name, key string) {
	if s.hub == nil {
		return
	}
	if _, err := s.hub.Say(event.Event{Name: name, Key: key, Gone: true}, nil); err != nil && err != event.ErrClosed {
		s.logf("hub: %v", err)
	}
}

// Hub is the supervisor's hub: what it says, for GET /engine/v1/facts, events and book.
func (s *Supervisor) Hub() *event.Hub { return s.hub }

// sayPtr says a fact from a pointer: the value when set, gone when nil.
func sayPtr[T any](s *Supervisor, name, key string, p *T) {
	if p == nil {
		s.sayGone(name, key)
		return
	}
	s.say(name, key, p)
}
