package run

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"hermesengine/catalog"
	"hermesengine/lifecycle"
	"hermesengine/plan"
	"hermesengine/probe"
	"hermesengine/probe/gpusense"
)

type loadCtx struct {
	in   install
	key  string
	fp   fit.Footprint // the seed with the book's overlay
	seed fit.Footprint // the header's seed alone, which Calibrate holds every reading to
	gpu  probe.GPU
	hasG bool
	sum  gpusense.GPUSummary // the reading the room came from: the adapter llama-server is held to
}

func (s *Supervisor) llamaLog(trunc bool) (*lifecycle.LogSink, error) {
	return lifecycle.OpenLog(filepath.Join(s.dir, "logs", "llama-server.log"))
}

// room measures what the card has now. ourPIDs are excluded from "others".
func (s *Supervisor) room(ctx context.Context, hw probe.Hardware, ourPIDs []int) fit.Room {
	r, _ := s.roomSensed(ctx, hw, ourPIDs)
	return r
}

// roomSensed is room with the reading it was made from: the others and the host's share come
// from the sensed adapters (plan.MeasureReserve), on the adapter the engine runs on.
func (s *Supervisor) roomSensed(ctx context.Context, hw probe.Hardware, ourPIDs []int) (fit.Room, sensed) {
	sn := s.measure(ctx, hw, ourPIDs)
	r := plan.RoomForRun(hw, sn.sum, s.backendNow())
	if r.NoGPU {
		return r, sn
	}
	s.mu.Lock()
	id := s.st.Model
	s.mu.Unlock()
	if cat, err := s.catalog(); err == nil { // speed curves were measured on one card; others plan slower
		if m, ok := s.modelFor(cat, id); ok {
			r.SpeedScale = plan.SpeedScaleFor(r.GPUName, m.Footprint)
		}
	}
	if s.customEstimated() { // no measurement behind the footprint: a wider margin
		r = plan.WithCustomMargin(r)
	}
	return r, sn
}

func (s *Supervisor) loadContext(ctx context.Context) (loadCtx, fit.Room, fit.Wants, error) {
	in, ok := s.resolveInstall()
	if !ok {
		return loadCtx{}, fit.Room{}, fit.Wants{}, errors.New("not installed")
	}
	hw, err := s.cfg.Probe(ctx)
	if err != nil {
		return loadCtx{}, fit.Room{}, fit.Wants{}, err
	}
	room, sn := s.roomSensed(ctx, hw, nil)
	g, hasG := bestGPU(hw)
	if sn.gpu >= 0 { // the adapter the engine runs on, not the one with the most dedicated memory
		g, hasG = hw.GPUs[sn.gpu], true
	}
	key := fit.BookKey(in.model.SHA256, g.Name, in.backend)
	s.mu.Lock()
	kv := s.st.KVType
	s.mu.Unlock()
	seed, err := s.cfg.Seed(in.model, in.modelPath, kv)
	if err != nil {
		return loadCtx{}, fit.Room{}, fit.Wants{}, err
	}
	s.mu.Lock()
	was := s.st.Book.Entries[key]
	stands, lift, err := s.st.Book.LetGo(key, room, in.build)
	fp := s.st.Book.Footprint(key, seed)
	s.mu.Unlock()
	if err != nil {
		return loadCtx{}, fit.Room{}, fit.Wants{}, err
	}
	switch {
	case was.CapCtxTotal > 0 && !stands:
		s.logf("ctx cap %d let go: the room grew past what it missed by, or the engine build moved", was.CapCtxTotal)
		s.save()
	case stands && fp.CapCtxTotal != was.CapCtxTotal: // let go to an earlier cap that still stands
		s.logf("ctx cap %d let go; the earlier cap %d stands, lifted by a room above %d MiB or a build other than %s", was.CapCtxTotal, fp.CapCtxTotal, lift.RoomMiB+lift.MissMiB, lift.Build)
		s.save()
	case stands && was.Lift == nil: // an older book's cap now records the room and build it stands under
		s.logf("ctx cap %d stands; lifted by a room above %d MiB or a build other than %s", was.CapCtxTotal, lift.RoomMiB+lift.MissMiB, lift.Build)
		s.save()
	}
	lc := loadCtx{in: in, key: key, fp: fp, seed: seed, gpu: g, hasG: hasG, sum: sn.sum}
	return lc, room, s.wantsFor(key), nil
}

// startLocked chooses a shape and loads it. opMu must be held.
func (s *Supervisor) startLocked() error {
	if _, ok := s.resolveInstall(); !ok {
		if s.customMissingPhase() {
			return nil
		}
		// a failed or running install job owns the phase; do not paint over it
		s.mu.Lock()
		busy := s.jobRunning || s.phase == "error" || s.phase == "downloading" || s.phase == "installing"
		s.mu.Unlock()
		if !busy {
			s.setPhase("not_installed", "No model installed")
		}
		return nil
	}
	ctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	s.loadCancel = cancel
	s.mu.Unlock()
	defer func() {
		cancel()
		s.mu.Lock()
		s.loadCancel = nil
		s.mu.Unlock()
	}()
	s.setPhase("starting", "Choosing a shape")
	lc, room, w, err := s.loadContext(ctx)
	if err != nil {
		return s.failStart(err, true)
	}
	shape, err := fit.Choose(lc.fp, room, w)
	if err != nil { // another app may be holding the card: look again slowly, never give up for good
		return s.failSlow(fmt.Errorf("this model does not fit: %w", err))
	}
	if shape, err = s.belowCeiling(lc, shape, room, w); err != nil {
		return s.failSlow(err)
	}
	return s.runShape(ctx, lc, shape)
}

// failStart puts the engine in the error phase. With retry the watchdog tries again after the
// backoff for the failures in a row (Config.CrashBackoff); without, only a user action does.
func (s *Supervisor) failStart(err error, retry bool) error {
	return s.fail(err, retry, false)
}

// failSlow is a retried failure that waits the longest backoff: a memory shortfall, which another
// app letting go of the card ends, or nothing does.
func (s *Supervisor) failSlow(err error) error {
	return s.fail(err, true, true)
}

func (s *Supervisor) fail(err error, retry, slow bool) error {
	s.addError("%v", err)
	wait := s.noteFailure(err, slow)
	s.mu.Lock()
	msg := err.Error()
	if retry {
		msg = fmt.Sprintf("%s (failure %d in a row; trying again in %s)", msg, s.rec.fails, wait.Round(100*time.Millisecond))
	}
	s.setPhaseLocked("error", msg)
	s.retryable = retry
	s.retryAt = s.rec.retryAt
	s.mu.Unlock()
	return err
}

type launchResult struct {
	ok       bool
	oom      bool
	failedMB int
	err      error
}

// runShape loads a shape: one measured retry on a load OOM, then step down.
func (s *Supervisor) runShape(ctx context.Context, lc loadCtx, shape fit.Shape) error {
	for attempt := 0; attempt < 8; attempt++ {
		res := s.launch(ctx, lc, shape)
		if res.ok {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !res.oom {
			return s.failStart(res.err, true)
		}
		hw, err := s.cfg.Probe(ctx)
		if err != nil {
			return s.failStart(err, true)
		}
		room := s.room(ctx, hw, nil)
		w := s.wantsFor(lc.key)
		var next fit.Shape
		if attempt == 0 && res.failedMB > 0 {
			s.logf("load OOM (buffer %d MiB): measured retry", res.failedMB)
			next, err = fit.MissRetry(lc.fp, shape, res.failedMB, room, w)
		} else {
			s.logf("load OOM again: stepping down")
			next, err = fit.StepDown(lc.fp, shape, room, w)
			if err == nil {
				s.lowerCap(lc, shape, next.Slots*next.CtxPerSlot, room)
				s.setCeiling(lc.key, shape, "it ran out of memory while loading")
			}
		}
		if err != nil {
			return s.failSlow(fmt.Errorf("model server out of memory at every shape: %w", err))
		}
		shape = next
	}
	return s.failSlow(errors.New("model server kept running out of memory"))
}

// launch starts llama-server at a shape and waits until it is healthy.
func (s *Supervisor) launch(ctx context.Context, lc loadCtx, shape fit.Shape) launchResult {
	port := s.cfg.LlamaPort
	if port == 0 {
		var err error
		if port, err = freePort(); err != nil {
			return launchResult{err: err}
		}
	}
	alias := lc.in.model.ID
	var official *catalog.SamplerProfile
	if p, ok := lc.in.model.Sampler[catalog.ProfileOfficial]; ok {
		official = &p
	}
	known := knownFlags(lc.in.bin, s.cfg.ExtraEnv)
	args, dropped := BuildArgs(lc.in.modelPath, shape, "127.0.0.1", port, alias, known, official)
	if len(dropped) > 0 {
		s.logf("llama-server does not know %v: dropped", dropped)
	}
	pin := resolvePin(lc.in.bin, plan.AccelOf(lc.in.backend), lc.sum, s.cfg.ExtraEnv, known)
	args = append(args, pin.Args...)
	env := append(append([]string{}, s.cfg.ExtraEnv...), pin.Env...)
	s.mu.Lock()
	s.device = &pin
	s.mu.Unlock()
	s.logf("device: %s", pin.Note)
	s.logf("start: %s%s %s", strings.Join(append(pin.Env, ""), " "), lc.in.bin, strings.Join(args, " "))
	sink, err := s.llamaLog(false)
	if err != nil {
		return launchResult{err: err}
	}
	oomw := &oomWatch{}
	sink.OnLine(oomw.line)
	s.setPhase("starting", fmt.Sprintf("Starting llama-server (%d x %d)", shape.Slots, shape.CtxPerSlot))
	proc, err := s.cfg.Group.Launch(lifecycle.Spec{Name: "llama-server", Path: lc.in.bin, Args: args, Dir: filepath.Dir(lc.in.bin), Env: llamaEnv(lc.in.bin, env), Log: sink})
	if err != nil {
		sink.Close()
		return launchResult{err: err}
	}
	go func() { // the log stays on disk while the server runs, and is closed once it exits
		<-proc.Done()
		time.Sleep(50 * time.Millisecond)
		sink.Close()
	}()
	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	s.setPhase("loading", fmt.Sprintf("Loading model (%d slots x %d ctx)", shape.Slots, shape.CtxPerSlot))
	deadline := time.NewTimer(s.cfg.LoadTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(s.cfg.HealthPoll)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			proc.Stop(s.cfg.StopGrace)
			return launchResult{err: ctx.Err()}
		case <-deadline.C:
			proc.Stop(s.cfg.StopGrace)
			return launchResult{err: errors.New("model load timed out")}
		case <-proc.Done():
			time.Sleep(20 * time.Millisecond) // let the output copy finish
			tail := proc.Tail()
			oom, mb := ParseOOM(tail)
			return launchResult{oom: oom, failedMB: mb, err: fmt.Errorf("llama-server exited while loading: %s", lastLine(tail))}
		case <-tick.C:
			if s.healthy(url) {
				oomw.take() // what loading printed is not a run-time signal
				srv := &server{proc: proc, port: port, shape: shape, key: lc.key, fp: lc.fp, seed: lc.seed, startedAt: s.cfg.Now(), prevReserve: -1,
					reserve0: lc.sum.Reserve.OthersMiB + lc.sum.Reserve.HostMiB, oom: oomw}
				srv.nextGrow = s.cfg.Now().Add(s.cfg.GrowIdle)
				s.calibrate(ctx, srv)
				s.srv = srv
				s.ours.set(proc.Pid)
				s.mu.Lock()
				s.llamaURL = url
				basis := shape.VRAMBasis
				if srv.measured > 0 {
					basis = fit.Measured
				}
				s.shape = &ShapeInfo{Slots: shape.Slots, CtxPerSlot: shape.CtxPerSlot, NCpuMoe: shape.NCpuMoe, KVType: shape.KVType, VRAMMiB: firstPos(srv.measured, shape.VRAMMiB), VRAMBasis: basis}
				s.retryable = false
				s.tokS = 0
				s.mu.Unlock()
				s.setPhase("ready", "Ready")
				return launchResult{ok: true}
			}
		}
	}
}

func firstPos(a, b int) int {
	if a > 0 {
		return a
	}
	return b
}

func lastLine(t string) string {
	t = strings.TrimSpace(t)
	if i := strings.LastIndexByte(t, '\n'); i >= 0 {
		t = t[i+1:]
	}
	if len(t) > 200 {
		t = t[:200]
	}
	return t
}

func (s *Supervisor) healthy(base string) bool {
	resp, err := s.cfg.HTTP.Get(base + "/health")
	if err != nil {
		return false
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return resp.StatusCode == 200
}

// lowerCap records an out-of-memory of failed in the book (Book.LowerCap) and saves it; where the
// core refuses the room or the shape, nothing is recorded and the refusal is logged.
func (s *Supervisor) lowerCap(lc loadCtx, failed fit.Shape, ctxTotal int, room fit.Room) {
	s.mu.Lock()
	err := s.st.Book.LowerCap(lc.key, lc.fp, failed, ctxTotal, room, lc.in.build, time.Now())
	s.mu.Unlock()
	if err != nil {
		s.logf("ctx cap not lowered: %v", err)
		return
	}
	s.save()
}

// calibrate records the measured VRAM after a healthy load.
func (s *Supervisor) calibrate(ctx context.Context, srv *server) {
	mctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	m, err := s.cfg.Measure(mctx, srv.proc.Pid)
	if err != nil || m <= 0 {
		return
	}
	srv.measured = m
	s.mu.Lock()
	c, err := s.st.Book.Calibrate(srv.key, srv.seed, srv.shape, m, time.Now())
	s.mu.Unlock()
	if err != nil {
		s.logf("calibrate: measured %d MiB, not recorded: %v", m, err)
		return
	}
	s.logf("calibrate: measured %d MiB, %s", m, c)
	if c.Verdict != fit.CPUOnly { // the reading is kept for its shape whether or not it was taken
		s.save()
	}
}

// ---- watchdog -----------------------------------------------------------------------------

func (s *Supervisor) watchdog() {
	t := time.NewTicker(s.cfg.WatchEvery)
	defer t.Stop()
	for {
		select {
		case <-s.root.Done():
			return
		case <-t.C:
			s.tick()
		}
	}
}

// slotsIdle is true when /slots shows no slot processing and no proxied request runs.
func (s *Supervisor) slotsIdle(base string) bool {
	if s.inflightNow() != 0 {
		return false
	}
	resp, err := s.cfg.HTTP.Get(base + "/slots")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return true // /slots not served: rely on the proxy's own count
	}
	var slots []struct {
		Processing bool `json:"is_processing"`
	}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if json.Unmarshal(b, &slots) != nil {
		return true
	}
	for _, sl := range slots {
		if sl.Processing {
			return false
		}
	}
	return true
}

func (s *Supervisor) tick() {
	if !s.opMu.TryLock() {
		return
	}
	defer s.opMu.Unlock()
	s.mu.Lock()
	local := s.st.Mode == "local"
	phase := s.phase
	stopped := s.userStopped
	retry := s.retryable && !s.cfg.Now().Before(s.retryAt)
	reshape := s.reshape
	s.mu.Unlock()
	if !local {
		return
	}
	srv := s.srv
	if srv == nil {
		if phase == "error" && retry && !stopped && !s.cfg.NoAutoStart {
			s.startLocked()
		}
		return
	}
	ctx := s.root
	base := fmt.Sprintf("http://127.0.0.1:%d", srv.port)
	if srv.drain != nil && phase == "resizing" {
		s.tickDrain(ctx, srv, base)
		return
	}
	srv.drain = nil
	if phase != "ready" {
		return
	}
	// 1. the process died on its own
	if srv.proc.Exited() {
		s.handleExit(ctx, srv)
		return
	}
	// 2. health
	if s.healthy(base) {
		srv.misses = 0
	} else {
		srv.misses++
		if srv.misses >= s.cfg.HealthMisses {
			s.addError("llama-server missed %d health checks: restarting", srv.misses)
			s.restartSame(ctx, srv)
			return
		}
		return
	}
	s.readTokS(base)
	s.stable(srv)
	oomHeld, oomLine := s.survivedOOM(srv)
	// 3. idle tracking
	if s.slotsIdle(base) {
		srv.idleCount++
		if srv.idleSince.IsZero() {
			srv.idleSince = s.cfg.Now()
		}
	} else {
		srv.idleCount = 0
		srv.idleSince = time.Time{}
	}
	idle := srv.idleCount >= s.cfg.IdleChecks
	if reshape && idle {
		s.mu.Lock()
		s.reshape = false
		s.mu.Unlock()
		s.reload(ctx, srv, "KV type changed")
		return
	}
	// 4. make room for the game: read the adapters (the reserve is measured, never the card's
	// used memory, which holds our own server too), and shrink when the model's memory no longer
	// fits the budget the reserve leaves
	hw, err := s.cfg.Probe(ctx)
	if err != nil {
		return
	}
	room, sn := s.roomSensed(ctx, hw, []int{srv.proc.Pid})
	held := srv.heldReserve(sn.sum.Reserve.OthersMiB + sn.sum.Reserve.HostMiB)
	if room.NoGPU || srv.shape.CPUOnly {
		return
	}
	// 4a. the server keeps failing requests for want of memory: it is out of room whatever the
	// reading says. One step down, recorded; at idle now, else once the replies in flight end
	// (they are failing anyway, so the wait is short).
	if oomHeld {
		w := s.wantsFor(srv.key)
		next, err := fit.StepDown(srv.fp, srv.shape, room, w)
		if err != nil {
			s.addError("llama-server is out of memory at the smallest shape (%s)", oomLine)
			return
		}
		s.addError("llama-server keeps running out of memory at %d x %d (%s): shrinking", srv.shape.Slots, srv.shape.CtxPerSlot, oomLine)
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
		if !held {
			return // not on a reading that is still moving
		}
		if !idle && short < s.cfg.DrainMinMiB {
			return // a small shortfall waits for a pause between lines
		}
		w := s.wantsFor(srv.key)
		next, err := fit.StepDown(srv.fp, srv.shape, room, w)
		if err != nil {
			return
		}
		// straight to the shape that fits the room when one step is not enough: one restart,
		// not one per tick
		if c, err := fit.Choose(srv.fp, room, w); err == nil {
			if cn, _, err1 := srv.fp.Need(c); err1 == nil {
				if nn, _, err2 := srv.fp.Need(next); err2 == nil && cn < nn {
					next = c
				}
			}
		}
		r := sn.sum.Reserve
		s.logf("model needs %d MiB, the room leaves %d (others %d, host %d, margin %d): shrinking to %d x %d (moe %d)", need, vram, r.OthersMiB, r.HostMiB, r.MarginMiB, next.Slots, next.CtxPerSlot, next.NCpuMoe)
		if idle {
			s.resize(ctx, srv, next, false)
		} else {
			// mid-reply: llama-server holds its memory from load, so the reply in flight is
			// safe to finish; new requests wait while it does, then the model shrinks
			s.startDrain(srv, next, fmt.Sprintf("another app took memory (short %d MiB)", short))
		}
		return
	}
	// 5. grow back when idle for long enough, on a reserve that has held
	if held && idle && !srv.idleSince.IsZero() && s.cfg.Now().Sub(srv.idleSince) >= s.cfg.GrowIdle && !s.cfg.Now().Before(srv.nextGrow) {
		w := s.wantsFor(srv.key)
		if next, ok, err := fit.Grow(srv.fp, srv.shape, room, w); err == nil && ok && !s.atCeiling(srv.key, srv.fp, next) {
			s.logf("idle with room: growing to %d x %d (moe %d)", next.Slots, next.CtxPerSlot, next.NCpuMoe)
			s.resize(ctx, srv, next, true)
		}
	}
}

// heldReserve records this tick's reserve (others plus the host's share, MiB) and reports
// whether the last two reads agree: a change in the room reshapes the model only once held.
func (srv *server) heldReserve(total int) bool {
	held := srv.prevReserve >= 0 && total-srv.prevReserve <= heldTolMiB && srv.prevReserve-total <= heldTolMiB
	srv.prevReserve = total
	return held
}

func (s *Supervisor) countRestart() {
	s.mu.Lock()
	s.restarts++
	s.mu.Unlock()
}

func (s *Supervisor) reload(ctx context.Context, srv *server, why string) {
	s.restartSame(ctx, srv)
}

// restartSame restarts the server at its current shape through the normal choose path.
func (s *Supervisor) restartSame(ctx context.Context, srv *server) {
	s.countRestart()
	s.stopServerLocked()
	s.startLocked()
}

// handleExit deals with a server that died on its own. An OOM in its last words lowers the
// stored cap and restarts one step smaller at once. Anything else restarts after the backoff for
// the failures in a row, through the normal choose path (which reads the room again); after
// CrashSteps quick crashes at one shape it is read as a memory problem and steps down once.
func (s *Supervisor) handleExit(ctx context.Context, srv *server) {
	s.countRestart()
	tail := srv.proc.Tail()
	oom, _ := ParseOOM(tail)
	quick := s.cfg.Now().Sub(srv.startedAt) < s.cfg.CrashQuick
	why := lastLine(tail)
	s.srv = nil
	s.ours.set(0)
	s.mu.Lock()
	s.shape = nil
	s.llamaURL = ""
	s.mu.Unlock()
	s.setPhase("starting", "Restarting model server")
	cctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	s.loadCancel = cancel
	s.mu.Unlock()
	defer func() { cancel(); s.mu.Lock(); s.loadCancel = nil; s.mu.Unlock() }()
	lc, room, w, err := s.loadContext(cctx)
	if err != nil {
		s.failStart(err, true)
		return
	}
	shape := srv.shape
	if oom {
		next, err := fit.StepDown(lc.fp, shape, room, w)
		if err != nil {
			s.failSlow(fmt.Errorf("out of memory at the smallest shape: %w", err))
			return
		}
		s.addError("llama-server ran out of memory at %d x %d: lowering the cap", shape.Slots, shape.CtxPerSlot)
		s.lowerCap(lc, shape, next.Slots*next.CtxPerSlot, room)
		s.setCeiling(lc.key, shape, "it ran out of memory while serving")
		s.noteFailure(fmt.Errorf("out of memory at %d x %d", shape.Slots, shape.CtxPerSlot), false)
		if err := s.runShape(cctx, lc, next); err != nil {
			s.logf("restart after the out-of-memory failed: %v", err)
		}
		return
	}
	n := s.noteCrash(shape, quick, why)
	if n >= s.cfg.CrashSteps {
		// the same shape keeps dying soon after it starts, with no memory error in its last
		// words: most often another app took the card's memory and the driver killed it. The
		// sensed reserve now against the one the shape was chosen under says whether one was seen.
		now := lc.sum.Reserve.OthersMiB + lc.sum.Reserve.HostMiB
		grew := now - srv.reserve0
		next, err := fit.StepDown(lc.fp, shape, room, w)
		if err == nil {
			s.mu.Lock()
			s.rec.suspect = true
			s.rec.crashes = 0
			s.mu.Unlock()
			if grew > heldTolMiB {
				s.addError("llama-server crashed %d times in a row at %d x %d (%s); other apps took %d MiB since it started: stepping down and lowering the cap", n, shape.Slots, shape.CtxPerSlot, why, grew)
				s.lowerCap(lc, shape, next.Slots*next.CtxPerSlot, room)
			} else {
				s.addError("llama-server crashed %d times in a row at %d x %d (%s); no other app was seen taking memory, stepping down once anyway", n, shape.Slots, shape.CtxPerSlot, why)
			}
			s.setCeiling(lc.key, shape, fmt.Sprintf("it crashed %d times in a row", n))
			s.noteFailure(fmt.Errorf("crashed %d times at %d x %d: %s", n, shape.Slots, shape.CtxPerSlot, why), false)
			if err := s.runShape(cctx, lc, next); err != nil {
				s.logf("restart one step down failed: %v", err)
			}
			return
		}
		// the smallest shape crashes too: keep trying slowly
		s.failSlow(fmt.Errorf("llama-server keeps crashing at the smallest shape (%s), crash %d in a row", why, n))
		return
	}
	s.failStart(fmt.Errorf("llama-server exited (%s), crash %d in a row at %d x %d", why, n, shape.Slots, shape.CtxPerSlot), true)
}

// resize restarts at a new shape, only if nothing is being served. A failed grow goes back to
// the old shape; a failed shrink steps down from the shape it tried (shrink).
func (s *Supervisor) resize(ctx context.Context, srv *server, to fit.Shape, grow bool) {
	s.setPhase("resizing", fmt.Sprintf("Resizing to %d x %d", to.Slots, to.CtxPerSlot))
	// New requests now get 503; make sure nothing slipped in before the switch.
	if !s.slotsIdle(fmt.Sprintf("http://127.0.0.1:%d", srv.port)) {
		s.setPhase("ready", "Ready")
		return
	}
	if grow {
		s.growTo(ctx, srv, to)
		return
	}
	s.shrink(ctx, srv, to)
}

func (s *Supervisor) growTo(ctx context.Context, srv *server, to fit.Shape) {
	old := srv.shape
	s.countRestart()
	s.stopServerLocked()
	cctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	s.loadCancel = cancel
	s.mu.Unlock()
	defer func() { cancel(); s.mu.Lock(); s.loadCancel = nil; s.mu.Unlock() }()
	lc, room, _, err := s.loadContext(cctx)
	if err != nil {
		s.failStart(err, true)
		return
	}
	res := s.launch(cctx, lc, to)
	if res.ok {
		return
	}
	s.addError("grow to %d x %d failed (%v): back to the old shape", to.Slots, to.CtxPerSlot, res.err)
	if res.oom {
		s.lowerCap(lc, to, to.Slots*to.CtxPerSlot-1024*to.Slots, room)
	}
	s.setCeiling(lc.key, to, "growing to it failed")
	if s.runShape(cctx, lc, old) == nil && s.srv != nil {
		s.srv.nextGrow = s.cfg.Now().Add(s.cfg.GrowBackoff)
	}
}

// shrink restarts smaller because the running shape no longer fits beside the other apps (the
// old shape did not fail: it grows back once they let go). If the smaller one fails to load on
// an out-of-memory it is recorded and becomes a ceiling, and the engine steps down from it, never
// back up; any other failure retries through the backoff and the choose path, which reads the
// room again.
func (s *Supervisor) shrink(ctx context.Context, srv *server, to fit.Shape) {
	s.setPhase("resizing", fmt.Sprintf("Resizing to %d x %d", to.Slots, to.CtxPerSlot))
	s.countRestart()
	s.stopServerLocked()
	cctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	s.loadCancel = cancel
	s.mu.Unlock()
	defer func() { cancel(); s.mu.Lock(); s.loadCancel = nil; s.mu.Unlock() }()
	lc, room, w, err := s.loadContext(cctx)
	if err != nil {
		s.failStart(err, true)
		return
	}
	res := s.launch(cctx, lc, to)
	if res.ok || cctx.Err() != nil {
		return
	}
	if !res.oom {
		s.failStart(fmt.Errorf("shrink to %d x %d failed: %v", to.Slots, to.CtxPerSlot, res.err), true)
		return
	}
	next, err := fit.StepDown(lc.fp, to, room, w)
	if err != nil {
		s.failSlow(fmt.Errorf("shrink to %d x %d ran out of memory and no smaller shape fits: %w", to.Slots, to.CtxPerSlot, err))
		return
	}
	s.addError("shrink to %d x %d ran out of memory: stepping down to %d x %d", to.Slots, to.CtxPerSlot, next.Slots, next.CtxPerSlot)
	s.lowerCap(lc, to, next.Slots*next.CtxPerSlot, room)
	s.setCeiling(lc.key, to, "shrinking to it ran out of memory")
	if err := s.runShape(cctx, lc, next); err != nil {
		s.logf("step down after the failed shrink failed: %v", err)
	}
}

func (s *Supervisor) inflightNow() int32 {
	return loadInt32(&s.inflight)
}

// httpGetOK is used by tests and the API for quick liveness checks.
func httpGetOK(c *http.Client, url string) bool {
	resp, err := c.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode/100 == 2
}

// readTokS keeps the last decode rate from llama-server's /metrics.
func (s *Supervisor) readTokS(base string) {
	resp, err := s.cfg.HTTP.Get(base + "/metrics")
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return
	}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	for _, l := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(l, "llamacpp:predicted_tokens_seconds") {
			f := strings.Fields(l)
			if len(f) == 2 {
				if v, err := strconv.ParseFloat(f[1], 64); err == nil && v > 0 {
					s.mu.Lock()
					s.tokS = v
					s.mu.Unlock()
				}
			}
		}
	}
}
