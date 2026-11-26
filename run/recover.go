package run

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
)

// Recovery when llama-server crashes, runs out of memory, or another app takes the card's
// memory while it runs (docs/ENGINE.md section 5, "At run time"):
//   - failed starts and crashes retry after a growing wait (Config.CrashBackoff), never in a
//     tight loop and never given up for good while the host app runs;
//   - CrashSteps quick crashes in a row at one shape are read as a memory problem: one step down;
//   - a shape known to fail is a ceiling the engine does not return to until it expires;
//   - an out-of-memory the server survives (failed requests) is watched in its log;
//   - a held shortfall during a reply stops new requests and shrinks once the reply ends.

// recovery is the crash and failure record. Guarded by s.mu, except ceil (opMu).
type recovery struct {
	fails      int    // failed starts and crashes in a row since the last stable run
	crashes    int    // quick crashes in a row at crashShape
	crashShape string // shapeKey of the shape that crashed
	lastErr    string
	retryAt    time.Time
	suspect    bool // the last step down came from crashes read as a memory problem
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
	RetryInS      int    `json:"retry_in_s"`     // seconds to the next try (0: trying now or running)
	SuspectMemory bool   `json:"suspect_memory"` // crashes were read as another app taking memory
	CeilingCtx    int    `json:"ceiling_ctx"`    // total ctx of the shape it will not return to (0: none)
}

func (c *Config) backoff(n int) time.Duration {
	if n < 1 {
		n = 1
	}
	if n > len(c.CrashBackoff) {
		n = len(c.CrashBackoff)
	}
	return c.CrashBackoff[n-1]
}

func (c *Config) slowest() time.Duration { return c.CrashBackoff[len(c.CrashBackoff)-1] }

// recoveryLocked is the status block; caller holds s.mu.
func (s *Supervisor) recoveryLocked() *RecoveryInfo {
	r := s.rec
	if r.fails == 0 && r.crashes == 0 {
		return nil
	}
	ri := &RecoveryInfo{Failures: r.fails, Crashes: r.crashes, LastError: r.lastErr, SuspectMemory: r.suspect}
	if s.phase == "error" && s.retryable {
		if d := r.retryAt.Sub(s.cfg.Now()); d > 0 {
			ri.RetryInS = int(math.Ceil(d.Seconds()))
		}
	}
	now := s.cfg.Now()
	for _, c := range r.ceil {
		if now.Before(c.until) {
			ri.CeilingCtx = c.shape.Slots * c.shape.CtxPerSlot
		}
	}
	return ri
}

// noteFailure records a failed start or crash and returns the wait before the next try.
// slow makes it the longest wait (a memory shortfall waiting for another app to let go).
func (s *Supervisor) noteFailure(err error, slow bool) time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rec.fails++
	s.rec.lastErr = err.Error()
	wait := s.cfg.backoff(s.rec.fails)
	if slow {
		wait = s.cfg.slowest()
	}
	s.rec.retryAt = s.cfg.Now().Add(wait)
	return wait
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
	return s.rec.crashes
}

// shapeKey names a layout by what llama-server is given (not the estimate it was priced at,
// which moves as the book calibrates).
func shapeKey(s fit.Shape) string {
	return fmt.Sprintf("%dx%d moe %d kv %s cpu %v", s.Slots, s.CtxPerSlot, s.NCpuMoe, s.KVType, s.CPUOnly)
}

// stable clears the record once a server has run CrashQuick without trouble.
func (s *Supervisor) stable(srv *server) {
	if s.cfg.Now().Sub(srv.startedAt) < s.cfg.CrashQuick {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.rec.fails != 0 || s.rec.crashes != 0 {
		s.rec.fails, s.rec.crashes, s.rec.crashShape, s.rec.suspect = 0, 0, "", false
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

// oomWatch counts out-of-memory lines in llama-server's output while it runs.
type oomWatch struct {
	n    atomic.Int32
	last atomic.Value // string
}

func (w *oomWatch) line(l string) {
	if strings.Contains(l, "pinned memory") || !reOOM.MatchString(l) {
		return
	}
	w.n.Add(1)
	w.last.Store(l)
}

func (w *oomWatch) take() (int, string) {
	n := int(w.n.Swap(0))
	l, _ := w.last.Load().(string)
	return n, l
}

// survivedOOM reads this tick's out-of-memory lines and reports a held run of them: lines in
// two ticks within OOMHold. One failed request is not a shrink signal; a second one is.
func (s *Supervisor) survivedOOM(srv *server) (bool, string) {
	if srv.oom == nil {
		return false, ""
	}
	n, last := srv.oom.take()
	now := s.cfg.Now()
	keep := srv.oomTicks[:0]
	for _, t := range srv.oomTicks {
		if now.Sub(t) < s.cfg.OOMHold {
			keep = append(keep, t)
		}
	}
	srv.oomTicks = keep
	if n == 0 {
		return false, ""
	}
	srv.oomTicks = append(srv.oomTicks, now)
	s.logf("llama-server reported out of memory and kept running (%d lines): %s", n, last)
	if len(srv.oomTicks) >= 2 {
		srv.oomTicks = nil
		return true, last
	}
	return false, ""
}

// ---- making room during a reply ----------------------------------------------------------

// drain is a shrink waiting for the replies in flight: new requests get "Loading model" (the
// phase is resizing) and the resize runs once the slots are idle or DrainMax has passed.
type drain struct {
	to    fit.Shape
	since time.Time
	why   string
}

func (s *Supervisor) startDrain(srv *server, to fit.Shape, why string) {
	srv.drain = &drain{to: to, since: s.cfg.Now(), why: why}
	s.logf("%s: finishing the replies in flight, then %d x %d (moe %d); new requests wait", why, to.Slots, to.CtxPerSlot, to.NCpuMoe)
	s.setPhase("resizing", fmt.Sprintf("Making room on the graphics card: finishing replies, then %d x %d", to.Slots, to.CtxPerSlot))
}

// tickDrain runs one watchdog tick of a drain.
func (s *Supervisor) tickDrain(ctx context.Context, srv *server, base string) {
	d := srv.drain
	if srv.proc.Exited() {
		srv.drain = nil
		s.handleExit(ctx, srv)
		return
	}
	idle := s.slotsIdle(base)
	late := s.cfg.Now().Sub(d.since) >= s.cfg.DrainMax
	if !idle && !late {
		return
	}
	srv.drain = nil
	if !idle {
		s.addError("replies still running after %s: making room now (%s)", s.cfg.DrainMax, d.why)
	}
	s.shrink(ctx, srv, d.to)
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
