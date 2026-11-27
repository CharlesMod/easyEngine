package run

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/host"
)

// The mind's reactor (docs/EVENTS.md section 3, "Watchdog tick"): there is no tick. What the
// tick did is done when the event that means it is heard:
//   - the model server ended: its Proc.Done;
//   - idle: the door's count of requests in flight reaching zero (every request reaches the
//     model server through the engine's door; the server listens on a private loopback port);
//   - a hung server: a watchdog for silence, armed only while requests are in flight and put
//     back by a line from the server, a request beginning or ending, and Heard (response bytes);
//     when it fires /health is asked once;
//   - tok/s: read once when the count returns to zero;
//   - out-of-memory lines the server survives: counted by request (run/recover.go);
//   - the room: a held change of the reading (run/room.go);
//   - grow: the named moment GrowIdle, armed when the count reaches zero, cancelled when a
//     request begins;
//   - a retry after a failure: at once for the first, else a word that could change the outcome
//     (the room moved, a request at the door, a client's word: Nudge).

// causes is what the reactor has been woken for since it last looked.
type causes struct {
	exit  bool // the model server ended
	hang  bool // the watchdog for silence on requests in flight fired
	idle  bool // the count of requests in flight reached zero
	oom   bool // out-of-memory lines in two requests (oomLine says the last)
	room  bool // the room moved (a held change, or a word that it may have)
	nudge bool // a word that could change an outcome
	retry bool // the first failure in a row: try again at once
	drain bool // a drain's DrainMax came
	grow  bool // the named moment GrowIdle came

	oomLine string
}

// door is the count of requests in flight and what it arms.
type door struct {
	mu       sync.Mutex
	inflight int
	seq      uint64        // requests begun in this life; out-of-memory lines are counted by it
	hang     *host.Silence // watchdog for silence on the requests in flight
	growGen  uint64
	growStop func() bool
	idleLong bool // GrowIdle has passed since the count last reached zero
	why      causes
}

// wakeMind records why and wakes the reactor (a writer never blocks: wakes coalesce).
func (s *Supervisor) wakeMind(f func(c *causes)) {
	s.door.mu.Lock()
	f(&s.door.why)
	s.door.mu.Unlock()
	select {
	case s.poke <- struct{}{}:
	default:
	}
}

// nudge is a word that could change an outcome: a retry that waits for one looks again.
func (s *Supervisor) nudge(why string) {
	s.say("nudge", "", map[string]string{"why": why})
	s.wakeMind(func(c *causes) { c.nudge = true })
}

// Nudge is a client's or the keeper's word that could change an outcome (an attach, a settings
// change, a client's own retry): a failed start that waits for an event tries again, and so do an
// agent and a download that wait for one. The API calls it.
func (s *Supervisor) Nudge(why string) { s.nudge(why) }

// Begin marks a request at the door in flight; call the result when it ends.
func (s *Supervisor) Begin() func() {
	d := &s.door
	d.mu.Lock()
	d.inflight++
	d.seq++
	n := d.inflight
	d.idleLong = false
	d.growGen++
	if d.growStop != nil {
		d.growStop()
		d.growStop = nil
	}
	if d.hang == nil {
		// watchdog for silence: a server that says nothing while requests are in flight may be hung
		d.hang = host.NewSilence(s.cfg.HangSilence, func() { s.wakeMind(func(c *causes) { c.hang = true }) })
	} else {
		d.hang.Heard()
	}
	d.mu.Unlock()
	s.say("request", "", map[string]any{"what": "begin", "inflight": n})
	var once sync.Once
	return func() { once.Do(s.end) }
}

// end is one request ending.
func (s *Supervisor) end() {
	d := &s.door
	d.mu.Lock()
	d.inflight--
	n := d.inflight
	if n == 0 {
		if d.hang != nil {
			d.hang.Stop()
			d.hang = nil
		}
		s.armGrowLocked()
		d.why.idle = true
	} else if d.hang != nil {
		d.hang.Heard()
	}
	d.mu.Unlock()
	s.say("request", "", map[string]any{"what": "end", "inflight": n})
	if n == 0 {
		select {
		case s.poke <- struct{}{}:
		default:
		}
	}
}

// Heard says the model server was heard from (a line in its log, response bytes the proxy
// forwards): the watchdog for silence on the requests in flight is put back. The proxy calls it
// as a response's bytes flow.
func (s *Supervisor) Heard() {
	s.door.mu.Lock()
	if s.door.hang != nil {
		s.door.hang.Heard()
	}
	s.door.mu.Unlock()
}

func (s *Supervisor) inflightNow() int32 {
	s.door.mu.Lock()
	defer s.door.mu.Unlock()
	return int32(s.door.inflight)
}

func (s *Supervisor) reqSeq() uint64 {
	s.door.mu.Lock()
	defer s.door.mu.Unlock()
	return s.door.seq
}

// armGrowLocked arms the named moment GrowIdle: idle that long, a grow is looked at. Caller holds
// door.mu.
func (s *Supervisor) armGrowLocked() {
	d := &s.door
	d.growGen++
	gen := d.growGen
	if d.growStop != nil {
		d.growStop()
	}
	d.idleLong = false
	// named moment: GrowIdle, how long the engine is idle before it grows back
	d.growStop = host.Moment(s.cfg.GrowIdle, func() {
		d.mu.Lock()
		if d.growGen == gen {
			d.idleLong = true
			d.why.grow = true
		}
		d.mu.Unlock()
		select {
		case s.poke <- struct{}{}:
		default:
		}
	})
}

// waitIdle waits until no request is in flight, on the request happenings (never a clock). It
// answers false when ctx ends first.
func (s *Supervisor) waitIdle(ctx context.Context) bool {
	w, err := s.hub.Wake("request")
	if err != nil {
		return false
	}
	defer w.Close()
	for s.inflightNow() != 0 {
		select {
		case <-ctx.Done():
			return false
		case <-s.hub.Done():
			return false
		case <-w.C():
		}
	}
	return true
}

// waitFact waits until ok answers true, woken by the hub on names beginning with prefixes. It
// answers false when ctx ends first.
func (s *Supervisor) waitFact(ctx context.Context, ok func() bool, prefixes ...string) bool {
	w, err := s.hub.Wake(prefixes...)
	if err != nil {
		return false
	}
	defer w.Close()
	for !ok() {
		select {
		case <-ctx.Done():
			return false
		case <-s.hub.Done():
			return false
		case <-w.C():
		}
	}
	return true
}

// listenWord starts listening for a word that could change an outcome: a nudge (a request at the
// door, a client's word) or the network coming back. Listen before saying that you wait, so a
// word said in between is not lost; then awaitWord.
func (s *Supervisor) listenWord() (*event.Waker, error) { return s.hub.Wake("nudge", "network") }

// awaitWord waits for the word w listens for, and lets w go. It answers ctx's error when ctx
// ends first.
func (s *Supervisor) awaitWord(ctx context.Context, w *event.Waker) error {
	defer w.Close()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.hub.Done():
		return event.ErrClosed
	case <-w.C():
		return nil
	}
}

// watchNetwork says the network happening when the system says a link or address changed.
func (s *Supervisor) watchNetwork() {
	ch, stop, err := host.NetworkChanges()
	if err != nil {
		return // host.ErrNoEvent: no word for it here; a client's word still nudges
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer stop()
		for {
			select {
			case <-s.root.Done():
				return
			case <-ch:
				s.say("network", "", nil)
			}
		}
	}()
}

// mindUp is a model server that has become ready: follow its end, keep the room while it runs,
// arm GrowIdle when nothing is in flight and the named moment that clears the crash record.
func (s *Supervisor) mindUp(srv *server) {
	s.door.mu.Lock()
	if s.door.inflight == 0 {
		s.armGrowLocked()
	}
	s.door.mu.Unlock()
	// named moment: CrashQuick, a run that long is stable and clears the crash record
	srv.stableCancel = host.Moment(s.cfg.CrashQuick, func() { s.stable(srv) })
	s.card.arm()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		select {
		case <-srv.proc.Done():
			if !srv.planned.Load() {
				s.wakeMind(func(c *causes) { c.exit = true })
			}
		case <-s.root.Done():
		}
	}()
}

// mindDown is a planned stop of srv (a reshape, a swap, a yield, a person's stop).
func (s *Supervisor) mindDown(srv *server) {
	srv.planned.Store(true)
	if srv.stableCancel != nil {
		srv.stableCancel()
	}
	if srv.drain != nil && srv.drain.cancel != nil {
		srv.drain.cancel()
	}
	s.card.disarm()
	s.door.mu.Lock()
	s.door.growGen++
	if s.door.growStop != nil {
		s.door.growStop()
		s.door.growStop = nil
	}
	s.door.idleLong = false
	s.door.mu.Unlock()
}

// mind is the reactor's goroutine: it looks when woken, never on a clock.
func (s *Supervisor) mind() {
	for {
		select {
		case <-s.root.Done():
			return
		case <-s.poke:
		}
		s.react()
	}
}

// react looks at what woke the reactor. The reads it needs (tok/s, /health, the room before a
// grow) are made before opMu is taken.
func (s *Supervisor) react() {
	d := &s.door
	d.mu.Lock()
	c := d.why
	d.why = causes{}
	idleLong, n := d.idleLong, d.inflight
	d.mu.Unlock()
	s.mu.Lock()
	local := s.st.Mode == "local"
	url := s.llamaURL
	s.mu.Unlock()
	if !local {
		return
	}
	if c.idle && n == 0 && url != "" {
		s.readTokS(url)
	}
	restart := false
	if c.hang && url != "" {
		// the watchdog for silence fired: ask once (a network timeout on the one call)
		if s.healthy(url) {
			d.mu.Lock()
			if d.inflight > 0 && d.hang == nil {
				// watchdog for silence: put back, the server answered
				d.hang = host.NewSilence(s.cfg.HangSilence, func() { s.wakeMind(func(c *causes) { c.hang = true }) })
			}
			d.mu.Unlock()
		} else {
			restart = true
		}
	}
	var fresh *roomRead
	if idleLong && n == 0 && url != "" { // before a grow, the room is read again
		if r, ok := s.card.read(); ok {
			fresh = &r
		}
	}

	s.opMu.Lock()
	defer s.opMu.Unlock()
	s.mu.Lock()
	phase := s.phase
	stopped := s.userStopped
	retryable := s.retryable
	reshape := s.reshape
	s.mu.Unlock()
	ctx := s.root
	srv := s.srv
	if srv == nil {
		if phase == "error" && retryable && !stopped && !s.cfg.NoAutoStart && (c.retry || c.room || c.nudge) {
			s.startLocked()
		}
		return
	}
	// 1. the process died on its own
	if srv.proc.Exited() {
		s.handleExit(ctx, srv)
		return
	}
	// 2. hung: it said nothing while requests were in flight, and /health did not answer
	if restart {
		s.addError("llama-server said nothing for %s with requests in flight and /health did not answer: restarting", s.cfg.HangSilence)
		s.restartSame(ctx, srv)
		return
	}
	idle := s.inflightNow() == 0
	if srv.drain != nil && phase == "resizing" {
		late := srv.drain.late.Load()
		if !idle && !late {
			return
		}
		dr := srv.drain
		srv.drain = nil
		if !idle {
			s.addError("replies still running after %s: making room now (%s)", s.cfg.DrainMax, dr.why)
		}
		s.shrink(ctx, srv, dr.to)
		return
	}
	srv.drain = nil
	if phase != "ready" {
		return
	}
	if reshape && idle {
		s.mu.Lock()
		s.reshape = false
		s.mu.Unlock()
		s.reload(ctx, srv, "KV type changed")
		return
	}
	// 3. make room for the game: the held reading of the room (the reserve is measured, never
	// the card's used memory, which holds our own server too); shrink when the model's memory no
	// longer fits the budget the reserve leaves
	rr, ok := s.card.held()
	if !ok || rr.room.NoGPU || srv.shape.CPUOnly {
		return
	}
	room := rr.room
	// 3a. the server keeps failing requests for want of memory: it is out of room whatever the
	// reading says. One step down, recorded; at idle now, else once the replies in flight end.
	if c.oom {
		w := s.wantsFor(srv.key)
		next, err := fit.StepDown(srv.fp, srv.shape, room, w)
		if err != nil {
			s.addError("llama-server is out of memory at the smallest shape (%s)", c.oomLine)
			return
		}
		s.addError("llama-server keeps running out of memory at %d x %d (%s): shrinking", srv.shape.Slots, srv.shape.CtxPerSlot, c.oomLine)
		if lc, room2, _, err := s.loadContext(ctx); err == nil {
			s.lowerCap(lc, srv.shape, next.Slots*next.CtxPerSlot, room2)
		}
		s.setCeiling(srv.key, srv.shape, "it kept running out of memory while serving")
		if idle {
			s.resize(ctx, srv, next, false)
		} else {
			s.startDrain(srv, next, "out of memory while serving")
		}
		return
	}
	if short, need, vram, ok := needShort(srv, room); ok && short > 0 {
		if !idle && short < s.cfg.DrainMinMiB {
			return // a small shortfall waits for a pause between lines (the count reaching zero)
		}
		w := s.wantsFor(srv.key)
		next, err := fit.StepDown(srv.fp, srv.shape, room, w)
		if err != nil {
			return
		}
		// straight to the shape that fits the room when one step is not enough: one restart
		if c, err := fit.Choose(srv.fp, room, w); err == nil {
			if cn, _, err1 := srv.fp.Need(c); err1 == nil {
				if nn, _, err2 := srv.fp.Need(next); err2 == nil && cn < nn {
					next = c
				}
			}
		} else {
			// nothing fits the room: the smallest shape the steps reach, in one restart
			for i := 0; i < 64; i++ {
				if nn, _, err := srv.fp.Need(next); err != nil || int(math.Ceil(nn)) <= vram {
					break
				}
				n2, err := fit.StepDown(srv.fp, next, room, w)
				if err != nil {
					break
				}
				next = n2
			}
		}
		r := rr.sum.Reserve
		s.logf("model needs %d MiB, the room leaves %d (others %d, host %d, margin %d): shrinking to %d x %d (moe %d)", need, vram, r.OthersMiB, r.HostMiB, r.MarginMiB, next.Slots, next.CtxPerSlot, next.NCpuMoe)
		if idle {
			s.resize(ctx, srv, next, false)
		} else {
			// mid-reply: llama-server holds its memory from load, so the reply in flight is safe
			// to finish; new requests wait while it does, then the model shrinks
			s.startDrain(srv, next, fmt.Sprintf("another app took memory (short %d MiB)", short))
		}
		return
	}
	// 4. grow back when idle for GrowIdle, on a reading read again just now that agrees with the
	// held one
	if !idleLong || !idle || fresh == nil {
		return
	}
	if !sameRoom(*fresh, rr) {
		s.card.level.Now() // the room is moving: the level reports it once it holds
		return
	}
	if wait := srv.nextGrow.Sub(s.cfg.Now()); wait > 0 {
		// named moment: GrowBackoff after a failed grow; look again when it has passed
		host.Moment(wait, func() { s.wakeMind(func(c *causes) { c.grow = true }) })
		return
	}
	w := s.wantsFor(srv.key)
	if next, ok, err := fit.Grow(srv.fp, srv.shape, fresh.room, w); err == nil && ok && !s.atCeiling(srv.key, srv.fp, next) {
		s.logf("idle with room: growing to %d x %d (moe %d)", next.Slots, next.CtxPerSlot, next.NCpuMoe)
		s.resize(ctx, srv, next, true)
	}
}
