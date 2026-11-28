package run

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/fetch"
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
	// clients: an attachment came or left (run/clients.go); clientReg.reclient says what it asks
	clients bool

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
	held     atomic.Int32 // requests held until the model is up (Hold); never in flight
	tokSaid  bool         // a reply's own decode rate was said since the last idle: no /metrics read
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
func (s *Supervisor) Nudge(why string) {
	s.words.Add(1) // a person's or a client's word: a move that waits for one tries again (build.go)
	s.nudge(why)
}

// Begin marks a request in flight with no route taken (tests, and the engine's own work that
// routes nothing); call the result when it ends. A request that is forwarded to the model server
// takes Enter or EnterAs, which route and count it in one step under the door's lock.
func (s *Supervisor) Begin() func() {
	end := s.begin("")
	return func() { end(nil) }
}

// begin is Begin with the request's id (said in the begin and the end, so a client pairs them);
// the end may carry what the streamer read of the reply.
func (s *Supervisor) begin(id string) func(*RequestEnd) { return s.beginAs(id, "") }

// beginAs is begin for a request of an attachment (said in the begin; its end carries it too).
func (s *Supervisor) beginAs(id, att string) func(*RequestEnd) {
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
		d.hang = s.newHang()
	} else {
		d.hang.Heard()
	}
	d.mu.Unlock()
	s.say("request", "", requestSaid{What: "begin", Inflight: n, ID: id, Attachment: att})
	if s.listeners.Load() > 0 { // a request is a word for whatever waits for one (an agent, a download)
		s.nudge("request")
	}
	var once sync.Once
	return func(re *RequestEnd) { once.Do(func() { s.end(id, re) }) }
}

// newHang makes the watchdog on the requests in flight.
func (s *Supervisor) newHang() *host.Silence {
	// watchdog for silence: a server that says nothing while requests are in flight may be hung
	return host.NewSilence(s.cfg.HangSilence, func() { s.wakeMind(func(c *causes) { c.hang = true }) })
}

// end is one request ending.
func (s *Supervisor) end(id string, re *RequestEnd) {
	d := &s.door
	streamed := re != nil && re.TokS > 0
	if streamed { // the reply said its own decode rate: status says it, no /metrics read for it
		s.mu.Lock()
		s.tokS = re.TokS
		s.mu.Unlock()
	}
	s.idleSeen.touch() // the idle rule counts its quiet from the last end (idle.go)
	d.mu.Lock()
	d.inflight--
	n := d.inflight
	if streamed {
		d.tokSaid = true
	}
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
	said := requestSaid{What: "end", Inflight: n, ID: id, RequestEnd: re}
	if re != nil {
		said.Attachment = re.Attachment
	}
	s.say("request", "", said)
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

// word is a listener for a word that could change an outcome. While any listens, a request at the
// door is said as a nudge (Begin); with none listening a request says no nudge.
type word struct {
	s       *Supervisor
	w       *event.Waker
	mu      sync.Mutex
	counted bool // among the listeners Begin counts (it waits, or is about to)
	closed  bool
}

// C is signalled when a word has been said since the listener was made.
func (w *word) C() <-chan struct{} { return w.w.C() }

// count puts the listener among those a request is said to (it waits now, or is about to).
func (w *word) count() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.counted && !w.closed {
		w.counted = true
		w.s.listeners.Add(1)
	}
}

// Close lets the listener go.
func (w *word) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return
	}
	w.closed = true
	w.w.Close()
	if w.counted {
		w.s.listeners.Add(-1)
	}
}

// listenWord starts listening for a word that could change an outcome: a nudge (a request at the
// door, a client's word) or the network coming back, and counts as waiting for one. Listen before
// saying that you wait, so a word said in between is not lost; then awaitWord.
func (s *Supervisor) listenWord() (*word, error) {
	w, err := s.listenQuiet()
	if err == nil {
		w.count()
	}
	return w, err
}

// listenQuiet listens without counting as waiting: a download listens before each attempt, so a
// word said while the attempt still runs is heard, but requests are said as nudges only once it
// waits (wordAgain counts it then).
func (s *Supervisor) listenQuiet() (*word, error) {
	w, err := s.hub.Wake("nudge", "network")
	if err != nil {
		return nil, err
	}
	return &word{s: s, w: w}, nil
}

// awaitWord waits for the word w listens for, and lets w go. It answers ctx's error when ctx
// ends first.
func (s *Supervisor) awaitWord(ctx context.Context, w *word) error {
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

// wordAgain is a download's Again that listens before each attempt: the listener for the wait
// after an attempt is made before the attempt starts, so a network word said while the attempt
// was still timing out is heard. close lets the last listener go when the download returns.
type wordAgain struct {
	s *Supervisor
	w *word
}

func (s *Supervisor) newWordAgain() *wordAgain {
	a := &wordAgain{s: s}
	a.w, _ = s.listenQuiet()
	return a
}

// again is the fetch.Again: waiting says so (onWait), a word ends the wait, and the next
// attempt's listener is made before it starts (then onResume).
func (a *wordAgain) again(onWait func(err error), onResume func()) fetch.Again {
	return func(ctx context.Context, err error) error {
		w := a.w
		a.w = nil
		if w == nil {
			var lerr error
			if w, lerr = a.s.listenQuiet(); lerr != nil {
				return lerr
			}
		}
		w.count() // it waits now: a request is a word for it
		onWait(err)
		if werr := a.s.awaitWord(ctx, w); werr != nil {
			return werr
		}
		a.w, _ = a.s.listenQuiet()
		onResume()
		return nil
	}
}

func (a *wordAgain) close() {
	if a.w != nil {
		a.w.Close()
		a.w = nil
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
				s.words.Add(1) // the network came back: a word for a move that waits (build.go)
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
	if srv.growWait != nil {
		srv.growWait()
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
	s.stopMap()
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
	tokSaid := d.tokSaid
	if c.idle && n == 0 {
		d.tokSaid = false
	}
	d.mu.Unlock()
	s.mu.Lock()
	local := s.st.Mode == "local"
	url := s.llamaURL
	s.mu.Unlock()
	if !local {
		url = "" // requests go elsewhere: no server of ours to read or ask
	}
	if c.idle && n == 0 && url != "" && !tokSaid {
		s.readTokS(url)
	}
	restart := false
	if c.hang {
		// the watchdog for silence fired: ask once (a network timeout on the one call); with no
		// server there is nobody to ask, and nothing to restart
		ok := url == "" || s.healthy(url)
		if url != "" {
			s.say("mind.hang", "", map[string]bool{"answered": ok})
		}
		// The watchdog that fired is dead (Heard does nothing after a fire): it is let go and,
		// while requests are still in flight, a new one watches them. Whatever stands now (the
		// fired one, or a live one a Begin made after the count reached zero) is replaced. After a
		// failed ask the restart ends the requests in flight; the next Begin arms a new one.
		d.mu.Lock()
		if d.hang != nil {
			d.hang.Stop()
			d.hang = nil
		}
		if ok && d.inflight > 0 {
			d.hang = s.newHang()
		}
		d.mu.Unlock()
		restart = !ok
	}
	if !local {
		return
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
	if c.oom { // kept on the server it was said of until it can be acted on (not mid-drain, mid-swap)
		srv.oomPend, srv.oomLine = true, c.oomLine
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
		if dr.rechoose {
			s.rechoose(ctx, srv, dr.why, !idle)
			return
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
	// 2b. the clients changed (run/clients.go): an attach the running mind cannot serve grows it;
	// a local caller's grow drains (new requests wait at the door, the replies in flight end or
	// DrainMax cuts them), a remote one's waits for the door's next empty moment; a leave that
	// gives someone something reshapes when idle
	s.clients.mu.Lock()
	rc, local := s.clients.reclient, s.clients.reLocal
	s.clients.mu.Unlock()
	switch {
	case rc == 2 && idle:
		s.rechoose(ctx, srv, "a client needs room", false)
		return
	case rc == 2 && local:
		s.startClientDrain(srv, "a client needs room")
		return
	case rc == 1 && idle:
		s.rechoose(ctx, srv, "a client left", false)
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
	if srv.oomPend {
		srv.oomPend = false
		w := s.wantsFor(srv.key)
		next, err := fit.StepDown(srv.fp, srv.shape, room, w)
		if err != nil {
			s.addError("llama-server is out of memory at the smallest shape (%s)", srv.oomLine)
			return
		}
		s.addError("llama-server keeps running out of memory at %d x %d (%s): shrinking", srv.shape.Slots, srv.shape.CtxPerSlot, srv.oomLine)
		if lc, room2, _, err := s.loadContext(ctx); err == nil {
			s.lowerCap(lc, srv.shape, next.Slots*next.CtxPerSlot, room2)
		}
		s.setCeiling(srv.key, srv.shape, "it kept running out of memory while serving")
		srv.oomShrink = true // an out-of-memory restart keeps nothing (run/context.go)
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
		// the room is moving: the level reports it once it holds at a new level; if it settles
		// back at the held one the level says nothing, so idle that long again and look again
		s.card.level.Now()
		d.mu.Lock()
		if d.inflight == 0 {
			s.armGrowLocked()
		}
		d.mu.Unlock()
		return
	}
	if wait := srv.nextGrow.Sub(s.cfg.Now()); wait > 0 {
		if srv.growWait != nil {
			srv.growWait()
		}
		// named moment: GrowBackoff after a failed grow; look again when it has passed
		srv.growWait = host.Moment(wait, func() { s.wakeMind(func(c *causes) { c.grow = true }) })
		return
	}
	w := s.wantsFor(srv.key)
	if next, ok, err := fit.Grow(srv.fp, srv.shape, fresh.room, w); err == nil && ok && !s.atCeiling(srv.key, srv.fp, next) {
		s.logf("idle with room: growing to %d x %d (moe %d)", next.Slots, next.CtxPerSlot, next.NCpuMoe)
		s.resize(ctx, srv, next, true)
	}
}
