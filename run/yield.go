package run

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
	"github.com/CharlesMod/easyEngine/yield"
)

// YieldConfig turns on yielding the card to other programs (serve --yield-to-games):
// docs/ENGINE.md section 6b. Zero values get the defaults below.
type YieldConfig struct {
	Policy    yield.Config // Grace 15s, Steady 60s, Floor 128 MiB, Strict false
	Except    []string     // launcher app ids that never yield the card (the supported host's)
	Ignore    []string     // process names that never yield the card
	MinNewMiB int          // MiB a newcomer must hold to count by memory alone (0: any)
	Every     time.Duration
	Drain     time.Duration // longest wait for replies in flight before the unload (20s)
	// Injectable for tests.
	OS    func() yield.OS
	Procs func(ctx context.Context, as []gpusense.Adapter) []gpusense.Process
}

func (y *YieldConfig) defaults() {
	if y.Policy.Grace <= 0 {
		y.Policy.Grace = 15 * time.Second
	}
	if y.Policy.Steady <= 0 {
		y.Policy.Steady = 60 * time.Second
	}
	if y.Policy.Floor <= 0 {
		y.Policy.Floor = 128
	}
	if y.Every <= 0 {
		y.Every = time.Second
	}
	if y.Drain <= 0 {
		y.Drain = 20 * time.Second
	}
	if y.OS == nil {
		y.OS = yield.NewOS
	}
	if y.Procs == nil {
		y.Procs = func(ctx context.Context, as []gpusense.Adapter) []gpusense.Process {
			ps, _ := gpusense.Procs(ctx, probe.OSSources{}, as)
			return ps
		}
	}
}

// YieldInfo is in status while the card is yielded (or the engine is draining to yield it).
type YieldInfo struct {
	Reason   string    `json:"reason"` // what the card is yielded to: "AppId 123 (game.exe)", "app (opened the GPU)"
	Kind     string    `json:"kind"`   // game | gpu
	Since    time.Time `json:"since"`
	Draining bool      `json:"draining"` // replies in flight are finishing; new requests already get 503
}

// ErrYielded is returned by Start while the card is yielded.
var ErrYielded = errors.New("the card is yielded to another program")

const (
	yieldPending = 200 * time.Millisecond // read again this soon after a device open
	yieldTick    = 250 * time.Millisecond // launcher games: new pids are looked at this often
)

func (s *Supervisor) yielded() *YieldInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.yieldInfo
}

// yieldView is what the loop needs from the supervisor in one lock.
type yieldView struct {
	eligible bool // local, installed, not stopped by a person
	loaded   bool // llama-server is starting, loading or serving
	info     *YieldInfo
	adapters []gpusense.Adapter
	engine   int // engine adapter index, -1 unknown
}

func (s *Supervisor) yieldState() yieldView {
	s.mu.Lock()
	defer s.mu.Unlock()
	v := yieldView{info: s.yieldInfo, engine: -1}
	switch s.phase {
	case "not_installed", "downloading", "installing", "removing", "external":
	default:
		v.eligible = s.st.Mode == "local" && !s.userStopped
	}
	switch s.phase {
	case "starting", "loading", "ready", "resizing":
		v.loaded = true
	}
	if sum := s.gpu.sum; sum != nil {
		v.adapters = sum.Adapters
		v.engine = sum.EngineAdapter
	}
	if s.device != nil && s.device.Adapter >= 0 {
		v.engine = s.device.Adapter
	}
	return v
}

// yieldExcluded builds the filter for one observation: the engine and its children, the
// host, ignored names, and anything under a launch the profile supports.
func (s *Supervisor) yieldExcluded(o yield.OS, except []yield.Launch) func(pid int, name string) bool {
	y := s.cfg.Yield
	self := os.Getpid()
	host, _ := s.hostInfo()
	fixed := map[int]bool{self: true}
	for _, p := range s.ours.pids() {
		fixed[p] = true
	}
	if p := int(s.launching.v.Load()); p > 0 {
		fixed[p] = true
	}
	if p := s.agent.status().Pid; p > 0 {
		fixed[p] = true
	}
	if host.PID > 0 {
		fixed[host.PID] = true
	}
	ignore := map[string]bool{}
	for _, n := range append(append([]string{}, builtinIgnore...), y.Ignore...) {
		ignore[normName(n)] = true
	}
	return func(pid int, name string) bool {
		if fixed[pid] || (name != "" && ignore[normName(name)]) || o.Under(pid, self) {
			return true
		}
		for _, l := range except {
			if l.PID > 0 && o.Under(pid, l.PID) {
				return true
			}
		}
		return false
	}
}

// builtinIgnore is never a reason to yield: nvidia-smi opens the card to read it (the engine,
// monitors and dashboards run it every second, and on a two-card machine one call outlives the
// 200 ms second look), but it allocates nothing a game would miss.
var builtinIgnore = []string{"nvidia-smi"}

func normName(n string) string {
	n = strings.ToLower(filepath.Base(strings.ReplaceAll(strings.TrimSpace(n), `\`, "/")))
	return strings.TrimSuffix(n, ".exe")
}

// yieldLoop watches for other programs that want the card (cfg.Yield set). Earliest signal
// first: a launcher starting a game (new pids every 250 ms; usually before the game touches the
// GPU), a process opening a GPU device (inotify wakes the loop at once on Linux; confirmed by a
// second look 200 ms later), memory on the engine's adapter (read every Every).
func (s *Supervisor) yieldLoop() {
	y := s.cfg.Yield
	o := y.OS()
	defer o.Close()
	tr := &yield.Tracker{Floor: y.Policy.Floor, MinNew: y.MinNewMiB}
	m := yield.NewMachine(y.Policy)
	exceptIDs := map[string]bool{}
	for _, id := range y.Except {
		exceptIDs[strings.TrimSpace(id)] = true
	}
	var (
		lastMem, lastOpen time.Time
		baseEngine        = -2 // the engine adapter the baseline was taken on (-1: none known)
		mem               []gpusense.Process
		games, except     []yield.Launch
		changes           []yield.Change
		openers           map[int]string
		next              = yieldTick
	)
	for s.root.Err() == nil {
		woke := o.Wait(s.root, next)
		if s.root.Err() != nil {
			return
		}
		now := s.cfg.Now()
		v := s.yieldState()
		if !v.eligible {
			if v.info != nil {
				s.clearYield()
			}
			m.Reset()
			tr = &yield.Tracker{Floor: y.Policy.Floor, MinNew: y.MinNewMiB}
			next = yieldTick
			continue
		}
		if v.info == nil && m.Yielded() { // a person's Stop/Start cleared it
			m.Reset()
			tr = &yield.Tracker{Floor: y.Policy.Floor, MinNew: y.MinNewMiB}
		}
		games, except = games[:0], except[:0]
		for _, l := range o.Launches() {
			if l.AppID != "" && exceptIDs[l.AppID] {
				except = append(except, l)
			} else {
				games = append(games, l)
			}
		}
		for _, a := range v.adapters {
			if a.Index == v.engine {
				o.Adapter(a.PCI)
			}
		}
		skip := s.yieldExcluded(o, except)
		pending := next == yieldPending
		fresh := false
		if woke || pending || now.Sub(lastOpen) >= 2*y.Every || lastOpen.IsZero() {
			openers = o.Openers(func(pid int) bool { return skip(pid, "") })
			lastOpen, fresh = now, true
		}
		if (pending || now.Sub(lastMem) >= y.Every*9/10 || lastMem.IsZero()) && v.engine >= 0 && len(v.adapters) > 0 {
			mem = y.Procs(s.root, v.adapters)
			o.Drain() // our own nvidia-smi opened the device
			lastMem, fresh = now, true
		}
		uses := map[int]*yield.Use{}
		for _, p := range mem {
			if p.Adapter != v.engine || skip(p.PID, p.Name) {
				continue
			}
			uses[p.PID] = &yield.Use{PID: p.PID, Name: p.Name, MiB: p.DedicatedMiB}
		}
		for pid, name := range openers {
			if skip(pid, name) {
				continue
			}
			if u := uses[pid]; u != nil {
				u.Opened = true
			} else {
				uses[pid] = &yield.Use{PID: pid, Name: name, Opened: true}
			}
		}
		cur := make([]yield.Use, 0, len(uses))
		for _, u := range uses {
			cur = append(cur, *u)
		}
		memKnown := v.engine >= 0 && len(v.adapters) > 0
		if memKnown && lastMem.IsZero() {
			continue // the first memory read has not run: no baseline without it
		}
		if v.engine != baseEngine && !m.Yielded() { // the adapter became known or changed: look afresh
			tr = &yield.Tracker{Floor: y.Policy.Floor, MinNew: y.MinNewMiB}
			baseEngine = v.engine
		}
		switch {
		case !tr.Baselined() || (!v.loaded && !m.Yielded()):
			tr.Reset(cur) // nothing of ours on the card: whatever runs now is the baseline
			changes = nil
		case fresh: // a newcomer counts on its second fresh read, never twice on one
			changes = tr.Changes(cur)
		}
		next = yieldTick
		if tr.Pending() {
			next = yieldPending
		}
		d := m.Step(yield.Input{Now: now, Games: games, Changes: changes})
		switch d.Action {
		case yield.Yield:
			s.yieldNow(d)
		case yield.Resume:
			if s.resumeYield(d.Beside) {
				m.Reset()
				tr.Reset(cur)
			} else {
				m.Hold(now)
			}
		}
	}
}

// yieldNow stops admitting requests at once, lets replies in flight finish (up to Drain),
// then unloads llama-server. The model book and recovery record are not touched: a yield is
// neither a crash nor a memory shortfall.
func (s *Supervisor) yieldNow(d yield.Decision) {
	start := s.cfg.Now()
	s.mu.Lock()
	s.yieldInfo = &YieldInfo{Reason: d.Reason, Kind: d.Kind, Since: start, Draining: true}
	s.mu.Unlock()
	s.logf("yield: the card goes to %s", d.Reason)
	end := time.Now().Add(s.cfg.Yield.Drain)
	for s.inflightNow() > 0 && time.Now().Before(end) && s.root.Err() == nil {
		time.Sleep(50 * time.Millisecond)
	}
	if n := s.inflightNow(); n > 0 {
		s.logf("yield: %d request(s) still running after %s: cut", n, s.cfg.Yield.Drain)
	}
	s.mu.Lock()
	lc := s.loadCancel
	s.mu.Unlock()
	if lc != nil {
		lc()
	}
	s.opMu.Lock()
	s.stopServerLocked()
	s.mu.Lock()
	still := s.yieldInfo != nil
	if still {
		s.yieldInfo.Draining = false
	}
	s.mu.Unlock()
	if still {
		s.setPhase("yielded", "Paused while "+d.Reason+" runs")
	}
	s.opMu.Unlock()
	s.logf("yield: unloaded in %s", s.cfg.Now().Sub(start).Round(time.Millisecond))
}

// resumeYield loads again. Beside newcomers that still hold memory, it loads only when a shape
// fits the room they leave; otherwise it stays yielded (false).
func (s *Supervisor) resumeYield(beside bool) bool {
	s.opMu.Lock()
	if beside {
		ctx, cancel := context.WithTimeout(s.root, time.Minute)
		lc, room, w, err := s.loadContext(ctx)
		cancel()
		if err == nil {
			_, err = fit.Choose(lc.fp, room, w)
		}
		if err != nil {
			s.opMu.Unlock()
			s.logf("yield: still no room beside the other program: %v", err)
			return false
		}
	}
	s.clearYieldLocked()
	s.opMu.Unlock()
	s.logf("yield: resuming%s", map[bool]string{true: " beside the programs still running", false: ""}[beside])
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.Start()
	}()
	return true
}

func (s *Supervisor) clearYield() {
	s.opMu.Lock()
	defer s.opMu.Unlock()
	s.clearYieldLocked()
}

// clearYieldLocked drops the yield; opMu held.
func (s *Supervisor) clearYieldLocked() {
	s.mu.Lock()
	was := s.yieldInfo != nil
	s.yieldInfo = nil
	s.mu.Unlock()
	if was && s.srv == nil {
		s.settle()
	}
}
