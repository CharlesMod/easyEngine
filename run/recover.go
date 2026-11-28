package run

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/host"
)

// Recovery when llama-server crashes, runs out of memory, or another app takes the card's
// memory while it runs (docs/ENGINE.md section 5, "At run time"). There is no retry cadence
// (docs/EVENTS.md section 3):
//   - a crash restarts at once; the first failed start in a row is tried again at once;
//   - CrashSteps quick crashes in a row at one shape are read as a memory problem: one step down,
//     at once;
//   - with nothing left to try, a retry waits for an event that could change the outcome: the
//     room moved, a request at the door, a client's word (Nudge); the recovery fact says so;
//   - a shape known to fail is a ceiling the engine does not return to until it expires;
//   - an out-of-memory the server survives (failed requests) is watched in its log, by request;
//   - a held shortfall during a reply stops new requests and shrinks once the reply ends.

// waitingWord is what a retry with nothing left to try waits for (RecoveryInfo.WaitingFor).
const waitingWord = "the room on the card to change, a request, or a word from a client"

// recovery is the crash and failure record. Guarded by s.mu.
type recovery struct {
	fails      int    // failed starts and crashes in a row since the last stable run
	crashes    int    // quick crashes in a row at crashShape
	crashShape string // shapeKey of the shape that crashed
	lastErr    string
	waitingFor string // what a retry waits for ("" while it tries at once or runs)
	suspect    bool   // the last step down came from crashes read as a memory problem
	ceil       map[string]ceiling
}

// ceiling is a shape that failed under the model's book key: the engine runs only shapes that
// need less until it expires.
type ceiling struct {
	shape fit.Shape
	until time.Time
	why   string
}

// RecoveryInfo is Status.Recovery: present while the engine is recovering from crashes or
// failed starts, null once a run has been stable for CrashQuick.
type RecoveryInfo struct {
	Failures      int    `json:"failures"`       // failed starts and crashes in a row
	Crashes       int    `json:"crashes"`        // quick crashes in a row at one shape
	LastError     string `json:"last_error"`     // what the last one said
	RetryInS      int    `json:"retry_in_s"`     // always 0: there is no retry cadence (kept for old clients)
	WaitingFor    string `json:"waiting_for"`    // what the next try waits for ("": trying now or running)
	SuspectMemory bool   `json:"suspect_memory"` // crashes were read as another app taking memory
	CeilingCtx    int    `json:"ceiling_ctx"`    // total ctx of the shape it will not return to (0: none)
}

// recoveryLocked is the status block; caller holds s.mu.
func (s *Supervisor) recoveryLocked() *RecoveryInfo {
	r := s.rec
	if r.fails == 0 && r.crashes == 0 {
		return nil
	}
	ri := &RecoveryInfo{Failures: r.fails, Crashes: r.crashes, LastError: r.lastErr, SuspectMemory: r.suspect}
	if s.phase == "error" && s.retryable {
		ri.WaitingFor = r.waitingFor
	}
	now := s.cfg.Now()
	for _, c := range r.ceil {
		if now.Before(c.until) {
			ri.CeilingCtx = c.shape.Slots * c.shape.CtxPerSlot
		}
	}
	return ri
}

// noteFailure records a failed start and says whether to try again at once: the first failure in
// a row is; a later one, or one with nothing left to try (wait), waits for an event.
func (s *Supervisor) noteFailure(err error, wait bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rec.fails++
	s.rec.lastErr = err.Error()
	now := !wait && s.rec.fails == 1
	s.rec.waitingFor = waitingWord
	if now {
		s.rec.waitingFor = ""
	}
	s.refreshLocked()
	return now
}

// noteRestart records a crash or failure that is restarted at once (a step down, a crash's one
// restart).
func (s *Supervisor) noteRestart(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rec.fails++
	s.rec.lastErr = err.Error()
	s.rec.waitingFor = ""
	s.refreshLocked()
}

// noteCrash counts a crash at shape: quick ones in a row at the same shape add up, anything
// else starts the count again. It returns the count.
func (s *Supervisor) noteCrash(shape fit.Shape, quick bool, why string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := shapeKey(shape)
	if quick && s.rec.crashShape == k {
		s.rec.crashes++
	} else {
		s.rec.crashes = 1
	}
	s.rec.crashShape = k
	s.rec.lastErr = why
	s.refreshLocked()
	return s.rec.crashes
}

// shapeKey names a layout by what llama-server is given (not the estimate it was priced at,
// which moves as the book calibrates).
func shapeKey(s fit.Shape) string {
	return fmt.Sprintf("%dx%d moe %d kv %s cpu %v", s.Slots, s.CtxPerSlot, s.NCpuMoe, s.KVType, s.CPUOnly)
}

// stable clears the record once a server has run CrashQuick without trouble: called by the named
// moment mindUp arms, for the server it was armed for.
func (s *Supervisor) stable(srv *server) {
	s.mu.Lock()
	cleared := false
	if s.rec.fails != 0 || s.rec.crashes != 0 {
		s.rec.fails, s.rec.crashes, s.rec.crashShape, s.rec.suspect, s.rec.waitingFor = 0, 0, "", false, ""
		cleared = true
		s.refreshLocked()
	}
	s.mu.Unlock()
	if cleared {
		s.logf("stable for %s: the crash record is cleared", s.cfg.CrashQuick)
	}
}

// setCeiling marks shape as failed for key: no shape that needs as much is run again until
// GrowBackoff passes (a grow then tries once more; a failure sets it again).
func (s *Supervisor) setCeiling(key string, shape fit.Shape, why string) {
	s.mu.Lock()
	if s.rec.ceil == nil {
		s.rec.ceil = map[string]ceiling{}
	}
	c, had := s.rec.ceil[key]
	now := s.cfg.Now()
	// keep the lower of two live ceilings
	if !had || !now.Before(c.until) || shape.Slots*shape.CtxPerSlot < c.shape.Slots*c.shape.CtxPerSlot || shape.NCpuMoe > c.shape.NCpuMoe {
		s.rec.ceil[key] = ceiling{shape: shape, until: now.Add(s.cfg.GrowBackoff), why: why}
	} else {
		c.until = now.Add(s.cfg.GrowBackoff)
		s.rec.ceil[key] = c
	}
	s.mu.Unlock()
	s.logf("will not return to %d x %d (moe %d) for %s: %s", shape.Slots, shape.CtxPerSlot, shape.NCpuMoe, s.cfg.GrowBackoff, why)
}

// atCeiling reports whether shape needs as much as a live ceiling for key (priced with fp).
func (s *Supervisor) atCeiling(key string, fp fit.Footprint, shape fit.Shape) bool {
	s.mu.Lock()
	c, ok := s.rec.ceil[key]
	now := s.cfg.Now()
	s.mu.Unlock()
	if !ok || !now.Before(c.until) || shape.CPUOnly {
		return false
	}
	need, _, err1 := fp.Need(shape)
	cn, _, err2 := fp.Need(c.shape)
	if err1 != nil || err2 != nil {
		return shape.Slots*shape.CtxPerSlot >= c.shape.Slots*c.shape.CtxPerSlot && shape.NCpuMoe <= c.shape.NCpuMoe
	}
	return need >= cn-1
}

// belowCeiling steps shape down until it needs less than the live ceiling for lc.key.
func (s *Supervisor) belowCeiling(lc loadCtx, shape fit.Shape, room fit.Room, w fit.Wants) (fit.Shape, error) {
	for i := 0; i < 64 && s.atCeiling(lc.key, lc.fp, shape); i++ {
		next, err := fit.StepDown(lc.fp, shape, room, w)
		if err != nil {
			return shape, fmt.Errorf("no shape below the one that failed: %w", err)
		}
		if i == 0 {
			s.logf("chosen %d x %d is at a shape that failed: stepping below it", shape.Slots, shape.CtxPerSlot)
		}
		shape = next
	}
	return shape, nil
}

// ---- out-of-memory the server survives ---------------------------------------------------

// oomWatch counts out-of-memory lines in llama-server's output while it runs, by request: lines
// said during two different requests (Begin's sequence) within the named moment OOMHold are a
// held run, and wake the reactor to step down. One failed request is not a shrink signal; a
// second one is. Lines during the load are let go when it is ready (take).
type oomWatch struct {
	s     *Supervisor
	live  atomic.Bool // the load is over: lines count
	mu    sync.Mutex
	first uint64 // the request the first line of a run was said in (0: no run)
	in    bool   // a run stands (OOMHold has not passed since its first line)
	gen   uint64
}

func (w *oomWatch) line(l string) {
	if !w.live.Load() || strings.Contains(l, "pinned memory") || !reOOM.MatchString(l) {
		return
	}
	s := w.s
	seq := s.reqSeq()
	w.mu.Lock()
	held := false
	switch {
	case !w.in:
		w.in, w.first = true, seq
		w.gen++
		gen := w.gen
		// named moment: OOMHold, how close two requests' out-of-memory lines are a shrink signal
		host.Moment(s.cfg.OOMHold, func() {
			w.mu.Lock()
			if w.gen == gen {
				w.in = false
			}
			w.mu.Unlock()
		})
	case seq != w.first:
		held = true
		w.in = false
		w.gen++
	}
	w.mu.Unlock()
	s.say("mind.oom", "", map[string]any{"line": l, "held": held})
	go s.logf("llama-server reported out of memory and kept running: %s", l)
	s.card.event() // the room is read again on an out-of-memory line
	if held {
		s.wakeMind(func(c *causes) { c.oom, c.oomLine = true, l })
	}
}

// take ends the load: lines from now on count.
func (w *oomWatch) take() { w.live.Store(true) }

// ---- making room during a reply ----------------------------------------------------------

// drain is a shrink waiting for the replies in flight: new requests get "Loading model" (the
// phase is resizing) and the resize runs once the count of requests in flight reaches zero or
// the named moment DrainMax comes (the reactor, run/mind.go).
type drain struct {
	to  fit.Shape
	why string
	// rechoose: the drain is for the clients (run/clients.go): the shape is chosen afresh from
	// the merged wants once the replies end, or DrainMax comes and cuts them.
	rechoose bool
	late     atomic.Bool
	cancel   func() bool
}

func (s *Supervisor) startDrain(srv *server, to fit.Shape, why string) {
	d := &drain{to: to, why: why}
	// named moment: DrainMax, the longest a shrink waits for the replies in flight
	d.cancel = host.Moment(s.cfg.DrainMax, func() {
		d.late.Store(true)
		s.wakeMind(func(c *causes) { c.drain = true })
	})
	srv.drain = d
	s.logf("%s: finishing the replies in flight, then %d x %d (moe %d); new requests wait", why, to.Slots, to.CtxPerSlot, to.NCpuMoe)
	s.setPhase("resizing", fmt.Sprintf("Making room on the graphics card: finishing replies, then %d x %d", to.Slots, to.CtxPerSlot))
}

// startClientDrain holds new requests and waits for the replies in flight to end, then reshapes
// for the clients (rechoose). Only a local caller's grow drains (run/mind.go); DrainMax bounds
// it as it bounds a shrink: once it comes the replies still running are cut, each stream with its
// terminal event, so one client's long reply cannot hold the engine for the others.
func (s *Supervisor) startClientDrain(srv *server, why string) {
	d := &drain{why: why, rechoose: true}
	// named moment: DrainMax, the longest a client's grow waits for the replies in flight
	d.cancel = host.Moment(s.cfg.DrainMax, func() {
		d.late.Store(true)
		s.wakeMind(func(c *causes) { c.drain = true })
	})
	srv.drain = d
	s.logf("%s: finishing the replies in flight, then a new shape; new requests wait", why)
	s.setPhase("resizing", "Making room for a client: finishing replies")
}

// rechoose stops the model and starts it again with the shape chosen from the clients' merged
// wants: the contexts are kept across it (run/context.go), the door holds requests meanwhile.
// late: DrainMax came with replies still running; they are cut by the stop.
func (s *Supervisor) rechoose(ctx context.Context, srv *server, why string, late bool) {
	s.setPhase("resizing", "Reshaping for the clients")
	if !late && s.inflightNow() != 0 { // a request slipped in before the phase: looked at again when idle
		s.setPhase("ready", "Ready")
		return
	}
	s.countRestartCauses()
	s.logf("clients: %s: reshaping", why)
	s.countRestart()
	s.keepContexts(why, false)
	s.stopServerLocked()
	s.startLocked()
}

// needShort is how far the running shape's need passes the room's budget (MiB, <= 0: fits).
func needShort(srv *server, room fit.Room) (short, need, vram int, ok bool) {
	v, _, err := room.Budget()
	if err != nil {
		return 0, 0, 0, false
	}
	n, _, err := srv.fp.Need(srv.shape)
	if err != nil {
		return 0, 0, 0, false
	}
	need, vram = int(math.Ceil(n)), int(v)
	return need - vram, need, vram, true
}
