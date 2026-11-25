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

	"hermesengine/fit"
	"hermesengine/lifecycle"
	"hermesengine/probe"
)

type loadCtx struct {
	in   install
	key  string
	fp   fit.Footprint
	gpu  probe.GPU
	hasG bool
}

func (s *Supervisor) llamaLog(trunc bool) (*lifecycle.LogSink, error) {
	return lifecycle.OpenLog(filepath.Join(s.dir, "logs", "llama-server.log"))
}

// room measures what the card has now. ourPIDs are excluded from "others".
func (s *Supervisor) room(ctx context.Context, hw probe.Hardware, ourPIDs []int) fit.Room {
	g, ok := bestGPU(hw)
	r := fit.Room{RAMAvail: hw.RAMAvailMiB, GameReserve: s.cfg.Profile.GameReserveMiB, NoGPU: !ok || g.VRAMTotalMiB == 0, Unified: g.UnifiedMemory}
	if ok {
		r.VRAMTotal = g.VRAMTotalMiB
		r.Margin = fit.DefaultMargin(g.VRAMTotalMiB)
		if o, err := s.cfg.Others(ctx, g, ourPIDs); err == nil {
			r.Others = o
		} else {
			r.Others = g.VRAMUsedMiB
		}
	}
	return r
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
	g, hasG := bestGPU(hw)
	key := fit.BookKey(in.model.SHA256, g.Name, in.backend)
	s.mu.Lock()
	kv := s.st.KVType
	s.mu.Unlock()
	seed, err := s.cfg.Seed(in.model, in.modelPath, kv)
	if err != nil {
		return loadCtx{}, fit.Room{}, fit.Wants{}, err
	}
	s.mu.Lock()
	fp := s.st.Book.Footprint(key, seed)
	s.mu.Unlock()
	lc := loadCtx{in: in, key: key, fp: fp, gpu: g, hasG: hasG}
	return lc, s.room(ctx, hw, nil), s.wantsFor(key), nil
}

// startLocked chooses a shape and loads it. opMu must be held.
func (s *Supervisor) startLocked() error {
	if _, ok := s.resolveInstall(); !ok {
		s.setPhase("not_installed", "No model installed")
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
	if err != nil {
		return s.failStart(fmt.Errorf("this model does not fit: %w", err), false)
	}
	return s.runShape(ctx, lc, shape)
}

func (s *Supervisor) failStart(err error, retry bool) error {
	s.addError("%v", err)
	s.mu.Lock()
	s.setPhaseLocked("error", err.Error())
	s.retryable = retry
	s.retryAt = s.cfg.Now().Add(6 * s.cfg.WatchEvery)
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
				s.mu.Lock()
				s.st.Book.LowerCap(lc.key, next.Slots*next.CtxPerSlot)
				s.mu.Unlock()
				s.save()
			}
		}
		if err != nil {
			return s.failStart(fmt.Errorf("model server out of memory at every shape: %w", err), false)
		}
		shape = next
	}
	return s.failStart(errors.New("model server kept running out of memory"), false)
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
	args, dropped := BuildArgs(lc.in.modelPath, shape, "127.0.0.1", port, alias, knownFlags(lc.in.bin, s.cfg.ExtraEnv))
	if len(dropped) > 0 {
		s.logf("llama-server does not know %v: dropped", dropped)
	}
	s.logf("start: %s %s", lc.in.bin, strings.Join(args, " "))
	sink, err := s.llamaLog(false)
	if err != nil {
		return launchResult{err: err}
	}
	s.setPhase("starting", fmt.Sprintf("Starting llama-server (%d x %d)", shape.Slots, shape.CtxPerSlot))
	proc, err := s.cfg.Group.Spawn(lifecycle.Spec{Name: "llama-server", Path: lc.in.bin, Args: args, Dir: filepath.Dir(lc.in.bin), Env: llamaEnv(lc.in.bin, s.cfg.ExtraEnv), Log: sink})
	if err != nil {
		sink.Close()
		return launchResult{err: err}
	}
	defer sink.Close()
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
				srv := &server{proc: proc, port: port, shape: shape, key: lc.key, fp: lc.fp, startedAt: s.cfg.Now()}
				srv.nextGrow = s.cfg.Now().Add(s.cfg.GrowIdle)
				s.calibrate(ctx, srv)
				s.srv = srv
				s.mu.Lock()
				s.llamaURL = url
				s.shape = &ShapeInfo{Slots: shape.Slots, CtxPerSlot: shape.CtxPerSlot, NCpuMoe: shape.NCpuMoe, KVType: shape.KVType, VRAMMiB: firstPos(srv.measured, shape.VRAMMiB)}
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
	ok := s.st.Book.Calibrate(srv.key, srv.fp, srv.shape, m)
	s.mu.Unlock()
	s.logf("calibrate: measured %d MiB, accepted=%v", m, ok)
	if ok {
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
	if phase != "ready" {
		return
	}
	ctx := s.root
	base := fmt.Sprintf("http://127.0.0.1:%d", srv.port)
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
	// 4. make room for the game
	hw, err := s.cfg.Probe(ctx)
	if err != nil {
		return
	}
	g, ok := bestGPU(hw)
	if !ok || g.VRAMTotalMiB == 0 {
		return
	}
	margin := fit.DefaultMargin(g.VRAMTotalMiB)
	free := g.VRAMTotalMiB - g.VRAMUsedMiB
	if free < margin {
		if !idle {
			return // never in the middle of a line
		}
		room := s.room(ctx, hw, []int{srv.proc.Pid})
		w := s.wantsFor(srv.key)
		next, err := fit.StepDown(srv.fp, srv.shape, room, w)
		if err != nil {
			return
		}
		s.logf("free VRAM %d MiB below margin %d: shrinking to %d x %d (moe %d)", free, margin, next.Slots, next.CtxPerSlot, next.NCpuMoe)
		s.resize(ctx, srv, next, false)
		return
	}
	// 5. grow back when idle for long enough
	if idle && !srv.idleSince.IsZero() && s.cfg.Now().Sub(srv.idleSince) >= s.cfg.GrowIdle && !s.cfg.Now().Before(srv.nextGrow) {
		room := s.room(ctx, hw, []int{srv.proc.Pid})
		w := s.wantsFor(srv.key)
		if next, ok := fit.Grow(srv.fp, srv.shape, room, w); ok {
			s.logf("idle with room: growing to %d x %d (moe %d)", next.Slots, next.CtxPerSlot, next.NCpuMoe)
			s.resize(ctx, srv, next, true)
		}
	}
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

// handleExit deals with a server that died on its own: an OOM lowers the stored
// cap and restarts smaller, anything else restarts at the same shape.
func (s *Supervisor) handleExit(ctx context.Context, srv *server) {
	s.countRestart()
	tail := srv.proc.Tail()
	oom, _ := ParseOOM(tail)
	s.srv = nil
	s.mu.Lock()
	s.shape = nil
	s.llamaURL = ""
	s.mu.Unlock()
	s.setPhase("starting", "Restarting model server")
	lc, room, w, err := s.loadContext(ctx)
	if err != nil {
		s.failStart(err, true)
		return
	}
	shape := srv.shape
	if oom {
		next, err := fit.StepDown(lc.fp, shape, room, w)
		if err != nil {
			s.failStart(fmt.Errorf("out of memory at the smallest shape: %w", err), false)
			return
		}
		s.addError("llama-server ran out of memory at %d x %d: lowering the cap", shape.Slots, shape.CtxPerSlot)
		s.mu.Lock()
		s.st.Book.LowerCap(lc.key, next.Slots*next.CtxPerSlot)
		s.mu.Unlock()
		s.save()
		shape = next
	} else {
		s.addError("llama-server exited (%s): restarting", lastLine(tail))
	}
	cctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	s.loadCancel = cancel
	s.mu.Unlock()
	defer func() { cancel(); s.mu.Lock(); s.loadCancel = nil; s.mu.Unlock() }()
	s.runShape(cctx, lc, shape)
}

// resize restarts at a new shape, only if nothing is being served. A failed grow
// goes back to the old shape.
func (s *Supervisor) resize(ctx context.Context, srv *server, to fit.Shape, grow bool) {
	old := srv.shape
	s.setPhase("resizing", fmt.Sprintf("Resizing to %d x %d", to.Slots, to.CtxPerSlot))
	// New requests now get 503; make sure nothing slipped in before the switch.
	if !s.slotsIdle(fmt.Sprintf("http://127.0.0.1:%d", srv.port)) {
		s.setPhase("ready", "Ready")
		return
	}
	s.countRestart()
	s.stopServerLocked()
	cctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	s.loadCancel = cancel
	s.mu.Unlock()
	defer func() { cancel(); s.mu.Lock(); s.loadCancel = nil; s.mu.Unlock() }()
	lc, _, _, err := s.loadContext(cctx)
	if err != nil {
		s.failStart(err, true)
		return
	}
	if grow {
		res := s.launch(cctx, lc, to)
		if res.ok {
			return
		}
		s.addError("grow to %d x %d failed (%v): back to the old shape", to.Slots, to.CtxPerSlot, res.err)
		if res.oom {
			s.mu.Lock()
			s.st.Book.LowerCap(lc.key, to.Slots*to.CtxPerSlot-1024*to.Slots)
			s.mu.Unlock()
			s.save()
		}
		if s.runShape(cctx, lc, old) == nil && s.srv != nil {
			s.srv.nextGrow = s.cfg.Now().Add(s.cfg.GrowBackoff)
		}
		return
	}
	s.runShape(cctx, lc, to)
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
