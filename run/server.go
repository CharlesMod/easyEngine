package run

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
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
	// the vision projector of the start being made, or of the server that runs: the fit is given
	// that much less of the card (run/vision.go withProjectorRoom)
	r = withProjectorRoom(r, s.visionRoomMiB())
	return r, sn
}

func (s *Supervisor) loadContext(ctx context.Context) (loadCtx, fit.Room, fit.Wants, error) {
	if s.srv == nil { // every load starts here (start, restart, reshape, yield's end): the staged pin runs from it
		s.promoteNext()
	}
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
	key := fit.BookKey(in.model.SHA256, g.Name, bookBackend(in.backend, in.tag))
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
	// The cancel is visible before the yielded check: a yield that begins after this point finds
	// it and cancels the load (yieldNow sets the yield, then reads loadCancel, both under s.mu), and
	// one that began before is seen by the check. No yield waits on opMu for a whole load.
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
	if s.yielded() != nil { // the card is yielded: the yield's end loads again
		return nil
	}
	s.checkPin() // a start is a word for a move that is still to be made (build.go)
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
	s.setPhase("starting", "Choosing a shape")
	s.clearExcl() // every client is in the merge again: the shape is chosen afresh
	if in, ok := s.resolveInstall(); ok {
		s.planVision(in) // before the room is read: the room is given less for a projector
	}
	lc, room, w, err := s.loadContext(ctx)
	if err != nil {
		return s.failStart(err, true)
	}
	shape, err := fit.Choose(lc.fp, room, w)
	for {
		switch {
		case (err != nil && errors.Is(err, fit.ErrTooBig) || err == nil && s.floorShort(shape.CtxPerSlot)) && s.visionTrying():
			// text is never refused for vision: the projector goes before any client is left out
			room = withoutProjectorRoom(room, s.visionRoomMiB())
			s.dropVision("the model does not fit this card with its vision projector: serving text only")
		case err != nil && errors.Is(err, fit.ErrTooBig):
			// the clients' merged wants do not fit even at the floor: one client is left out
			// (unserved, said) until the next start: remote before local, later before earlier
			if _, ok := s.dropLatest(); !ok {
				goto chosen
			}
		case err == nil && s.floorShort(shape.CtxPerSlot):
			// a remote client whose floor the shape does not reach is left out: its slots are not
			// sized for nothing
		default:
			goto chosen
		}
		w = s.wantsFor(lc.key)
		shape, err = fit.Choose(lc.fp, room, w)
	}
chosen:
	if err != nil { // another app may be holding the card: look again slowly, never give up for good
		return s.failSlow(fmt.Errorf("this model does not fit: %w", err))
	}
	if shape, err = s.belowCeiling(lc, shape, room, w); err != nil {
		return s.failSlow(err)
	}
	return s.runShape(ctx, lc, shape)
}

// failStart puts the engine in the error phase. With retry the first failure in a row is tried
// again at once and a later one waits for an event that could change the outcome (run/recover.go);
// without, only a user action does.
func (s *Supervisor) failStart(err error, retry bool) error {
	return s.fail(err, retry, false)
}

// failSlow is a retried failure with nothing left to try at once: a memory shortfall, which
// another app letting go of the card ends, or nothing does. It waits for an event.
func (s *Supervisor) failSlow(err error) error {
	return s.fail(err, true, true)
}

func (s *Supervisor) fail(err error, retry, slow bool) error {
	s.addError("%v", err)
	now := s.noteFailure(err, slow || !retry)
	s.mu.Lock()
	msg := err.Error()
	if retry {
		if now {
			msg = fmt.Sprintf("%s (failure %d in a row; trying again now)", msg, s.rec.fails)
		} else {
			msg = fmt.Sprintf("%s (failure %d in a row; waiting for %s)", msg, s.rec.fails, waitingWord)
		}
	}
	s.retryable = retry
	if !retry {
		s.rec.waitingFor = ""
	}
	s.setPhaseLocked("error", msg)
	s.mu.Unlock()
	// A start that failed for want of room waits for the room to change: the room watch is armed
	// while that wait stands (its answer can change what the engine does), and let go when the
	// start succeeds (mindUp keeps it for the model), the engine is stopped, or a failure of
	// another kind stands instead.
	if retry && slow {
		s.card.arm()
	} else if s.srv == nil {
		s.card.disarm()
	}
	if retry && now {
		s.wakeMind(func(c *causes) { c.retry = true })
	}
	return err
}

type launchResult struct {
	ok       bool
	oom      bool
	failedMB int
	err      error
	exited   bool      // the server exited while starting (not out of memory)
	opts     launchSet // the clients' options it launched with
}

// runShape loads a shape: one measured retry on a load OOM, then step down.
func (s *Supervisor) runShape(ctx context.Context, lc loadCtx, shape fit.Shape) error {
	bare := false // the one retry without the clients' extra_args was made
	for attempt := 0; attempt < 8; attempt++ {
		res := s.launch(ctx, lc, shape)
		if res.ok {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !res.oom && res.exited && !bare && s.startFailedWith(res.opts) {
			// it exited while starting with a client's extra_args: once more at once without them
			// (told not taken: the model server did not start with them)
			bare = true
			attempt--
			continue
		}
		if !res.oom {
			return s.failStart(res.err, true)
		}
		if s.visionTrying() {
			// out of memory with the projector: the same shape once more without it (text first)
			s.dropVision("the model server ran out of memory loading the vision projector: serving text only")
			attempt--
			continue
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
	args = withBefore(args, "--host", s.slotSaveArgs(known))
	// the clients' launch options (run/clients.go), checked against this build's --help
	help := helpOf(lc.in.bin, s.cfg.ExtraEnv)
	opts := s.clientOpts(help, shape, known)
	args = withOptions(args, opts.args, help)
	if opts.ramOff { // more than one owner of the slots: no RAM prompt cache shared among them
		args = withBefore(args, "--host", []string{"--cache-ram", "0"})
	}
	if va := s.visionArgs(); va != nil { // planVision checked the build knows --mmproj
		args = withBefore(args, "--host", va)
	}
	if len(dropped) > 0 {
		s.logf("llama-server does not know %v: dropped", dropped)
	}
	pin := resolvePin(lc.in.bin, plan.AccelOf(lc.in.backend), lc.sum, s.cfg.ExtraEnv, known)
	args = append(args, pin.Args...)
	env := append(append([]string{}, s.cfg.ExtraEnv...), pin.Env...)
	s.mu.Lock()
	s.device = &pin
	if pin.Adapter >= 0 {
		s.gpu.keep = pin.Adapter + 1
	}
	s.mu.Unlock()
	s.logf("device: %s", pin.Note)
	s.logf("start: %s%s %s", strings.Join(append(pin.Env, ""), " "), lc.in.bin, strings.Join(args, " "))
	sink, err := s.llamaLog(false)
	if err != nil {
		return launchResult{err: err}
	}
	s.setVMM("") // what an earlier server said of its buffers is not this one's
	// Ready is heard in the server's own lines (docs/EVENTS.md section 3): after the line that
	// says it listens, /health is asked once per following line until it answers. The lines are
	// the first word, not the only one: when the server has said nothing for ReadyQuiet after a
	// line, /health is asked once, listening line or not (a build worded otherwise, or one whose
	// last line came before /health turned 200); one ask per quiet stretch, armed again only by a
	// new line, never a cadence. A load that says nothing for LoadTimeout is asked once more and
	// then fails. A build that prints forever and never answers 200 is not bounded: each line is
	// a sign of life.
	oomw := &oomWatch{s: s}
	lines := make(chan struct{}, 1)
	ask := make(chan struct{}, 1)
	var listening atomic.Bool
	silent := make(chan struct{})
	// watchdog for silence: a load with no line from the server for LoadTimeout has hung
	quiet := host.NewSilence(s.cfg.LoadTimeout, func() { close(silent) })
	defer quiet.Stop()
	var rqMu sync.Mutex
	var rq *host.Silence // the ReadyQuiet watchdog of the quiet stretch now; nil once it fired
	armQuiet := func() {
		// watchdog for silence: ReadyQuiet, a server quiet that long after a line is asked /health once
		rq = host.NewSilence(s.cfg.ReadyQuiet, func() {
			rqMu.Lock()
			rq = nil
			rqMu.Unlock()
			select {
			case ask <- struct{}{}:
			default:
			}
		})
	}
	defer func() {
		rqMu.Lock()
		if rq != nil {
			rq.Stop()
		}
		rqMu.Unlock()
	}()
	sink.OnLine(func(l string) {
		quiet.Heard()
		s.Heard()
		oomw.line(l)
		if v := vmmLine(l); v != "" {
			s.setVMM(v)
		}
		if !listening.Load() && saysListening(l) {
			listening.Store(true)
		}
		rqMu.Lock()
		if rq == nil {
			armQuiet()
		} else {
			rq.Heard()
		}
		rqMu.Unlock()
		select {
		case lines <- struct{}{}:
		default:
		}
	})
	select { // a word from a request held during an earlier load is not about this one
	case <-s.loadAsk:
	default:
	}
	s.setPhase("starting", fmt.Sprintf("Starting llama-server (%d x %d)", shape.Slots, shape.CtxPerSlot))
	proc, err := s.cfg.Group.Launch(lifecycle.Spec{Name: "llama-server", Path: lc.in.bin, Args: args, Dir: filepath.Dir(lc.in.bin), Env: llamaEnv(lc.in.bin, env), Log: sink})
	if err != nil {
		sink.Close()
		return launchResult{err: err}
	}
	s.launching.set(proc.Pid)
	rqMu.Lock()
	if rq == nil { // a server that says nothing at all is asked once a quiet stretch too
		armQuiet()
	}
	rqMu.Unlock()
	go func() { // the log stays on disk while the server runs, and is closed once it exits
		<-proc.Done() // Done is closed once every line has reached the sink (lifecycle.Proc)
		sink.Close()
	}()
	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	s.setPhase("loading", fmt.Sprintf("Loading model (%d slots x %d ctx)", shape.Slots, shape.CtxPerSlot))
	for {
		select {
		case <-ctx.Done():
			proc.Stop(s.cfg.StopGrace)
			return launchResult{err: ctx.Err()}
		case <-silent:
			if !s.healthy(url) { // asked once before the load is called failed
				proc.Stop(s.cfg.StopGrace)
				return launchResult{err: fmt.Errorf("model load timed out: llama-server said nothing for %s", s.cfg.LoadTimeout)}
			}
		case <-proc.Done():
			tail := proc.Tail()
			oom, mb := ParseOOM(tail)
			return launchResult{oom: oom, failedMB: mb, exited: true, opts: opts, err: fmt.Errorf("llama-server exited while loading: %s", lastLine(tail))}
		case <-ask:
			if !s.healthy(url) {
				continue
			}
		case <-s.loadAsk: // a request held at the door is a word: /health is asked once on it
			if !s.healthy(url) {
				continue
			}
		case <-lines:
			if !listening.Load() || !s.healthy(url) {
				continue
			}
		}
		{
			oomw.take() // what loading printed is not a run-time signal
			s.mu.Lock()
			visSHA, visMiB := s.visionUpLocked()
			s.mu.Unlock()
			ck := ctxKeyOf(lc, shape)
			ck.Vision = visSHA // a saved slot goes back only onto a server with the same projector state
			srv := &server{proc: proc, port: port, shape: shape, key: lc.key, fp: lc.fp, seed: lc.seed, startedAt: s.cfg.Now(),
				reserve0: lc.sum.Reserve.OthersMiB + lc.sum.Reserve.HostMiB, oom: oomw, ctxKey: ck, build: lc.in.build, projMiB: visMiB}
			srv.nextGrow = s.cfg.Now()
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
			s.rec.waitingFor = ""
			s.tokS = 0
			s.mu.Unlock()
			// the clients' slots in this mind are set now, before the saved slots go back to them
			s.startMap(shape, opts)
			// the saved slots go back before the phase says ready: the door holds requests till then
			s.layBack(srv, url)
			s.mindUp(srv)
			s.setPhase("ready", "Ready")
			s.wg.Add(1)
			go func(in install) { // this build served ready: the builds it replaced go, off opMu
				defer s.wg.Done()
				s.pruneBuilds(in)
			}(lc.in)
			return launchResult{ok: true}
		}
	}
}

// listensRe matches the line llama-server says once it listens. Newer builds say it after the
// model loaded ("llama_server: listening on http://127.0.0.1:41745"); older ones say "main: HTTP
// server is listening, ..." before loading and "main: server is listening on http://... -
// starting the main loop" after it. Fixtures: run/testdata/ready-*.txt.
var listensRe = regexp.MustCompile(`(?i)\blistening on https?://|server is listening`)

func saysListening(l string) bool { return listensRe.MatchString(l) }

// recordLC is what a failure of srv is recorded under (lowerCap, setCeiling): srv's own book key,
// footprint and build, when the load after it (lc) runs another build (promoteNext selected a
// staged one in between); lc otherwise.
func recordLC(srv *server, lc loadCtx) loadCtx {
	if srv == nil || srv.key == "" || srv.key == lc.key {
		return lc
	}
	r := lc
	r.key, r.fp, r.in.build = srv.key, srv.fp, srv.build
	return r
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
	if srv.projMiB > 0 && m > srv.projMiB {
		// the book holds the model alone: the projector's load is the catalog's figure, which the
		// room already gives up (withProjectorRoom), so it is not learnt into the model's footprint
		m -= srv.projMiB
	}
	s.mu.Lock()
	c, err := s.st.Book.Calibrate(srv.key, srv.seed, srv.shape, m, time.Now())
	s.mu.Unlock()
	if err != nil {
		s.logf("calibrate: measured %d MiB, not recorded: %v", m, err)
		return
	}
	s.logf("calibrate: measured %d MiB, %s", m, c)
	s.save() // a reading of either kind (card or host) is kept for its shape, taken or not

}

func (s *Supervisor) countRestart() {
	s.mu.Lock()
	s.restarts++
	s.mu.Unlock()
}

func (s *Supervisor) reload(ctx context.Context, srv *server, why string) {
	s.letGoContexts(why) // a cache type change: no saved slot fits the new server
	s.restartSame(ctx, srv)
}

// restartSame restarts the server at its current shape through the normal choose path.
func (s *Supervisor) restartSame(ctx context.Context, srv *server) {
	s.countRestart()
	s.stopServerLocked()
	s.startLocked()
}

// handleExit deals with a server that died on its own. An OOM in its last words lowers the
// stored cap and restarts one step smaller at once. Anything else restarts at once through the
// normal choose path (which reads the room again); CrashSteps quick crashes at one shape are read
// as a memory problem and step down once, at once; with nothing left to try it waits for an event.
func (s *Supervisor) handleExit(ctx context.Context, srv *server) {
	s.countRestart()
	tail := srv.proc.Tail()
	oom, _ := ParseOOM(tail)
	quick := s.cfg.Now().Sub(srv.startedAt) < s.cfg.CrashQuick
	why := lastLine(tail)
	if err := srv.proc.Err(); err != nil { // how it ended says more than its last line when it was killed
		why = fmt.Sprintf("%v; last line: %s", err, why)
	}
	s.say("mind.exit", "", map[string]any{"pid": srv.proc.Pid, "why": why, "quick": quick, "oom": oom})
	s.mindDown(srv)
	s.srv = nil
	s.ours.set(0)
	s.launching.set(0)
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
		rec := recordLC(srv, lc) // under the build that ran out, not one the restart moved to
		s.lowerCap(rec, shape, next.Slots*next.CtxPerSlot, room)
		s.setCeiling(rec.key, shape, "it ran out of memory while serving")
		s.noteRestart(fmt.Errorf("out of memory at %d x %d", shape.Slots, shape.CtxPerSlot))
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
				s.lowerCap(recordLC(srv, lc), shape, next.Slots*next.CtxPerSlot, room)
			} else {
				s.addError("llama-server crashed %d times in a row at %d x %d (%s); no other app was seen taking memory, stepping down once anyway", n, shape.Slots, shape.CtxPerSlot, why)
			}
			s.setCeiling(recordLC(srv, lc).key, shape, fmt.Sprintf("it crashed %d times in a row", n))
			s.noteRestart(fmt.Errorf("crashed %d times at %d x %d: %s", n, shape.Slots, shape.CtxPerSlot, why))
			if err := s.runShape(cctx, lc, next); err != nil {
				s.logf("restart one step down failed: %v", err)
			}
			return
		}
		// the smallest shape crashes too: nothing left to try, wait for an event
		s.failSlow(fmt.Errorf("llama-server keeps crashing at the smallest shape (%s), crash %d in a row", why, n))
		return
	}
	// one restart at once, through the choose path
	err = fmt.Errorf("llama-server exited (%s), crash %d in a row at %d x %d", why, n, shape.Slots, shape.CtxPerSlot)
	s.addError("%v: restarting", err)
	s.noteRestart(err)
	s.startLocked()
}

// resize restarts at a new shape, only if nothing is being served. A failed grow goes back to
// the old shape; a failed shrink steps down from the shape it tried (shrink).
func (s *Supervisor) resize(ctx context.Context, srv *server, to fit.Shape, grow bool) {
	s.setPhase("resizing", fmt.Sprintf("Resizing to %d x %d", to.Slots, to.CtxPerSlot))
	// New requests now wait; make sure nothing slipped in before the switch.
	if s.inflightNow() != 0 {
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
	s.keepContexts("a grow", false)
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
	s.keepContexts("a shrink", false)
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

// readTokS keeps the last decode rate from llama-server's /metrics: read once when the count of
// requests in flight returns to zero.
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
					s.say("request", "", map[string]any{"what": "idle", "tok_s": v})
				}
			}
		}
	}
}
