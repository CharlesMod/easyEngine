package run

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/host"
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
	// Every is the period of the one watchdog (host.Level) on what the system gives no word
	// about: other programs' memory on the card and, where there is no word for them, a
	// launcher's argv. It is armed only while a model is loaded or the card is yielded
	// (default 1 s).
	Every time.Duration
	// Drain is a named moment: the longest wait for replies in flight before the unload (20s).
	Drain time.Duration
	// Injectable for tests.
	OS    func() yield.OS
	Procs func(ctx context.Context, as []gpusense.Adapter) []gpusense.Process
	// Looked, when set, is called at the end of each look the reactor takes with a reading that
	// has held, with the engine adapter it was read on (-1: not known yet), the programs and the
	// games it decided on: a test waits on it to know the
	// engine has seen a change, never on a clock.
	Looked func(engine int, uses []yield.Use, games []yield.Launch)
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
	fixed := s.fixedPids()
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

// fixedPids are the pids that are the engine's or the host's own: the engine, its model server,
// the one being launched, its agent, and the host program.
func (s *Supervisor) fixedPids() map[int]bool {
	host, _ := s.hostInfo()
	fixed := map[int]bool{os.Getpid(): true}
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
	return fixed
}

// builtinIgnore is never a reason to yield: nvidia-smi opens the card to read it (the engine,
// monitors and dashboards run it every second, and on a two-card machine one call outlives the
// 200 ms second look), but it allocates nothing a game would miss.
var builtinIgnore = []string{"nvidia-smi"}

func normName(n string) string {
	n = strings.ToLower(filepath.Base(strings.ReplaceAll(strings.TrimSpace(n), `\`, "/")))
	return strings.TrimSuffix(n, ".exe")
}

// yieldSeen is one reading of other programs: what holds the engine's card, and the launchers'
// games where the system gives no word for them.
type yieldSeen struct {
	engine   int // the engine adapter the reading was taken on (-1: none known)
	uses     []yield.Use
	launches []yield.Launch // all launches, sorted by pid; empty where the OS says launches by a word
}

// sameSeen says two readings are one value: the same programs, and memory within tol MiB (a
// card's memory moves a little all the time; the floor is what counts).
func sameSeen(tol int) func(a, b yieldSeen) bool {
	return func(a, b yieldSeen) bool {
		if a.engine != b.engine || len(a.uses) != len(b.uses) || !reflect.DeepEqual(a.launches, b.launches) {
			return false
		}
		for i, u := range a.uses {
			v := b.uses[i]
			d := u.MiB - v.MiB
			if u.PID != v.PID || u.Name != v.Name || u.Opened != v.Opened || d > tol || -d > tol {
				return false
			}
		}
		return true
	}
}

// yieldReactor is the one goroutine that watches for other programs that want the card (cfg.Yield
// set; docs/EVENTS.md section 3, "Yield loop"). It waits on channels only. What wakes it:
//   - a word from the system: a program opening the engine adapter's device node (Linux,
//     host.Opens, no time cap), Steam's RunningAppID changing (Windows registry, macOS
//     registry.vdf), a program it follows ending (host.Exits);
//   - the hub: the phase, the yield, the model (a person's stop or start, a load, an unload);
//   - the level: host.Level reports a change of what has no word (a launcher's argv in /proc on
//     Linux, memory on the card on NVIDIA and Windows) once two reads in a row agree, so a short
//     probe never counts. It is armed only while a model is loaded or the card is yielded;
//   - the named moments Yield.Policy.Grace and Yield.Policy.Steady, armed for the time the
//     machine says is left (host.Moment).
type yieldReactor struct {
	s      *Supervisor
	y      *YieldConfig
	o      yield.OS
	except map[string]bool

	osMu  sync.Mutex // o is not safe for concurrent use: the level's reads and the reactor take turns
	level *host.Level[yieldSeen]
	wakeC chan struct{}
	armed atomic.Bool // the level is armed: a word is worth a read

	mu      sync.Mutex // held, hasHeld, heldNew, known
	held    yieldSeen  // the last reading that held
	hasHeld bool
	heldNew bool         // held has not been looked at
	known   map[int]bool // programs holding the device at the last look: only a new one is a word

	readEngine int // the engine adapter of the last read (-2: none yet)

	// the reactor goroutine's own
	tr         *yield.Tracker
	m          *yield.Machine
	changes    []yield.Change
	baseEngine int
	words      yield.Words
	wordsPCI   string
	wordsSet   atomic.Value // string: the nodes the watch was made on (the level's read compares)
	wordsStop  chan struct{}
	launchWord atomic.Bool // the OS says its launchers by a word: they are read when it comes
	stopMoment func() bool
	follows    map[int]func()
}

func newYieldReactor(s *Supervisor) *yieldReactor {
	y := s.cfg.Yield
	r := &yieldReactor{s: s, y: y, o: y.OS(), except: map[string]bool{}, wakeC: make(chan struct{}, 1),
		m: yield.NewMachine(y.Policy), baseEngine: -2, readEngine: -2, follows: map[int]func(){}}
	r.tr = &yield.Tracker{Floor: y.Policy.Floor, MinNew: y.MinNewMiB}
	for _, id := range y.Except {
		r.except[strings.TrimSpace(id)] = true
	}
	r.level = host.NewLevel(y.Every, r.read, sameSeen(max(y.Policy.Floor/4, 1)), r.changed)
	return r
}

// wake asks the reactor to look; wakes coalesce and never block.
func (r *yieldReactor) wake() {
	select {
	case r.wakeC <- struct{}{}:
	default:
	}
}

// yieldLoop runs the reactor until the supervisor ends.
func (s *Supervisor) yieldLoop() {
	r := newYieldReactor(s)
	defer r.close()
	w, err := s.hub.Wake("phase", "yield", "model", "room") // "room": the engine's adapter may be known or changed, and so its nodes
	if err != nil {
		s.logf("yield: %v", err)
		return
	}
	defer w.Close()
	r.wake()
	for s.root.Err() == nil {
		select {
		case <-s.root.Done():
			return
		case <-s.hub.Done():
			return
		case <-w.C():
		case <-r.wakeC:
		}
		r.step()
	}
}

// split sets the launches the profile supports apart from those that yield the card.
func (r *yieldReactor) split(ls []yield.Launch) (games, except []yield.Launch) {
	for _, l := range ls {
		if l.AppID != "" && r.except[l.AppID] {
			except = append(except, l)
		} else {
			games = append(games, l)
		}
	}
	return
}

// enginePCI is the PCI address of the engine's adapter, "" when not known.
func enginePCI(v yieldView) string {
	for _, a := range v.adapters {
		if a.Index == v.engine {
			return a.PCI
		}
	}
	return ""
}

// look reads the OS under its lock: the launches, the filter for this look and the programs
// holding the device. The caller holds osMu.
func (r *yieldReactor) lookLocked(v yieldView) (launches []yield.Launch, skip func(int, string) bool, openers map[int]string) {
	if pci := enginePCI(v); pci != "" {
		r.o.Adapter(pci)
	}
	if r.armed.Load() && r.nodesMoved() { // Adapter found other nodes than the watch holds
		r.wake()
	}
	launches = r.o.Launches()
	_, except := r.split(launches)
	skip = r.s.yieldExcluded(r.o, except)
	openers = r.o.Openers(func(pid int) bool { return skip(pid, "") })
	return
}

// wordLocked is lookLocked for an open word, which needs the filter and not the holders: the
// scan of who holds the device is the supervisor's, shared with the room's word. The caller holds
// osMu.
func (r *yieldReactor) wordLocked(v yieldView) (skip func(int, string) bool) {
	if pci := enginePCI(v); pci != "" {
		r.o.Adapter(pci)
	}
	if r.armed.Load() && r.nodesMoved() {
		r.wake()
	}
	_, except := r.split(r.o.Launches())
	return r.s.yieldExcluded(r.o, except)
}

// read is the level's reading: memory on the engine's adapter (the one thing that has no word on
// NVIDIA and Windows), the programs holding the device, and the launchers where there is no word
// for them.
func (r *yieldReactor) read() (yieldSeen, bool) {
	s := r.s
	if s.root.Err() != nil {
		return yieldSeen{}, false
	}
	v := s.yieldState()
	var mem []gpusense.Process
	if v.engine >= 0 && len(v.adapters) > 0 {
		// the memory read runs nvidia-smi, which opens the card's device: our own open, kept
		// from every Opens watch (the room's too) by the one gate
		host.Own(func() { mem = r.y.Procs(s.root, v.adapters) })
	}
	r.osMu.Lock()
	launches, skip, openers := r.lookLocked(v)
	r.osMu.Unlock()
	uses := map[int]*yield.Use{}
	for _, p := range mem {
		if p.Adapter != v.engine || skip(p.PID, p.Name) {
			continue
		}
		uses[p.PID] = &yield.Use{PID: p.PID, Name: p.Name, MiB: p.DedicatedMiB}
	}
	known := map[int]bool{}
	for pid, name := range openers {
		if skip(pid, name) {
			continue
		}
		known[pid] = true
		if u := uses[pid]; u != nil {
			u.Opened = true
		} else {
			uses[pid] = &yield.Use{PID: pid, Name: name, Opened: true}
		}
	}
	seen := yieldSeen{engine: v.engine}
	for _, u := range uses {
		seen.uses = append(seen.uses, *u)
	}
	sort.Slice(seen.uses, func(i, j int) bool { return seen.uses[i].PID < seen.uses[j].PID })
	if !r.launchWord.Load() {
		seen.launches = launches
		sort.Slice(seen.launches, func(i, j int) bool { return seen.launches[i].PID < seen.launches[j].PID })
	}
	r.mu.Lock()
	r.known = known
	newEngine := seen.engine != r.readEngine
	r.readEngine = seen.engine
	r.mu.Unlock()
	if newEngine { // the card is read on another adapter than before: one more read, so it holds at once
		r.level.Now()
	}
	return seen, true
}

// changed is a held change of what the level reads: the reactor looks at it.
func (r *yieldReactor) changed(v yieldSeen) {
	r.mu.Lock()
	r.held, r.hasHeld, r.heldNew = v, true, true
	r.mu.Unlock()
	r.wake()
}

// opened is the word that a program opened a device node. Only a program not seen holding the
// device is news: the engine's own reads (nvidia-smi) open it too, and a program that keeps it
// open must not set the level reading again and again.
func (r *yieldReactor) opened() {
	if !r.armed.Load() || r.s.root.Err() != nil {
		return
	}
	v := r.s.yieldState()
	ticket := r.s.holders.word()
	r.osMu.Lock()
	skip := r.wordLocked(v)
	r.osMu.Unlock()
	openers := r.s.holders.look(ticket, enginePCI(v)) // one scan serves this and the room's word
	fresh := false
	next := map[int]bool{}
	r.mu.Lock()
	for pid, name := range openers {
		if skip(pid, name) {
			continue
		}
		if !r.known[pid] {
			fresh = true
		}
		next[pid] = true
	}
	r.known = next
	r.mu.Unlock()
	if fresh {
		r.look()
	}
}

// look is a word that the level may have moved (a program opened the device, or one followed has
// ended): read it twice at once, so the change holds without waiting a period. A program that
// lives through two reads in a row is no probe.
func (r *yieldReactor) look() {
	if r.armed.Load() {
		r.level.Now()
		r.level.Now()
	}
	r.wake()
}

// startWords listens for the OS's words on the engine adapter's nodes, and for launchers.
func (r *yieldReactor) startWords(pci string) {
	r.stopWords()
	r.wordsPCI = pci
	w, ok := r.o.(yield.Watcher)
	if !ok {
		return
	}
	// The nodes are the OS's own (Adapter sets them on the level's read goroutine); the set is
	// noted so a watch made before they were known is made again when they are.
	pre := r.nodeSet()
	r.words = w.Watch()
	set := r.words.Set
	if set == "" {
		set = pre // no watch was made on them (or the OS names none): look again only if they move
	}
	r.wordsSet.Store(set)
	r.launchWord.Store(r.words.Launches != nil)
	done := make(chan struct{})
	r.wordsStop = done
	opens, launches := r.words.Opens, r.words.Launches // nil: never ready
	go func() {
		for {
			select {
			case <-done:
				return
			case <-r.s.root.Done():
				return
			case _, ok := <-opens:
				if !ok { // the watch has ended: no more words from it
					opens = nil
					continue
				}
				r.opened()
			case <-launches:
				r.wake()
			}
		}
	}()
}

// nodeSet is the set of nodes the OS would watch now ("" where it does not say).
func (r *yieldReactor) nodeSet() string {
	if n, ok := r.o.(yield.Noder); ok {
		return n.NodeSet()
	}
	return ""
}

// nodesMoved says the nodes the OS would watch are not those the watch was made on.
func (r *yieldReactor) nodesMoved() bool {
	n, ok := r.o.(yield.Noder)
	if !ok {
		return false
	}
	set, _ := r.wordsSet.Load().(string)
	return n.NodeSet() != set
}

func (r *yieldReactor) stopWords() {
	if r.wordsStop != nil {
		close(r.wordsStop)
		r.wordsStop = nil
	}
	r.words.Release()
	r.words = yield.Words{}
	r.launchWord.Store(false)
}

// arm starts the level and the words while a model is loaded or the card is yielded.
func (r *yieldReactor) arm(pci string) {
	if r.armed.Load() {
		if pci != r.wordsPCI || r.nodesMoved() { // the adapter became known or changed, or its nodes were found since: the nodes are others
			r.startWords(pci)
		}
		return
	}
	r.startWords(pci)
	r.armed.Store(true)
	r.level.Arm()
	r.level.Now() // a second read at once, so the baseline holds without waiting a period
}

// reset forgets what stood: a person's stop, or nothing of ours on the card. The level is let
// go too, so its next arming reports a fresh baseline.
func (r *yieldReactor) reset() {
	r.armed.Store(false)
	r.level.Disarm()
	r.stopWords()
	r.wordsPCI = ""
	r.m.Reset()
	r.tr = &yield.Tracker{Floor: r.y.Policy.Floor, MinNew: r.y.MinNewMiB}
	r.changes = nil
	r.baseEngine = -2
	r.mu.Lock()
	r.held, r.hasHeld, r.heldNew, r.known, r.readEngine = yieldSeen{}, false, false, nil, -2
	r.mu.Unlock()
	r.cancelMoment()
	r.followOnly(nil)
}

func (r *yieldReactor) cancelMoment() {
	if r.stopMoment != nil {
		r.stopMoment()
		r.stopMoment = nil
	}
}

// followOnly watches the given pids to their end (host.Exits); a pid no longer listed is let go.
// A pid is followed once while it is listed.
func (r *yieldReactor) followOnly(pids []int) {
	listed := map[int]bool{}
	for _, p := range pids {
		listed[p] = true
		if _, ok := r.follows[p]; ok {
			continue
		}
		ch, stop, err := host.Exits(p)
		if err != nil {
			r.follows[p] = nil
			continue
		}
		done := make(chan struct{})
		r.follows[p] = func() { stop(); close(done) }
		go func() {
			select {
			case <-ch:
				r.look()
			case <-done:
			case <-r.s.root.Done():
			}
		}()
	}
	for p, stop := range r.follows {
		if !listed[p] {
			if stop != nil {
				stop()
			}
			delete(r.follows, p)
		}
	}
}

func (r *yieldReactor) close() {
	r.level.Stop()
	r.armed.Store(false)
	r.stopWords()
	r.cancelMoment()
	r.followOnly(nil)
	r.o.Close()
}

// step looks at what stands and acts on it: the machine decides, the supervisor yields or
// resumes.
func (r *yieldReactor) step() {
	s := r.s
	v := s.yieldState()
	if !v.eligible {
		if v.info != nil {
			s.clearYield()
		}
		r.reset()
		return
	}
	if v.info == nil && r.m.Yielded() { // a person's Stop/Start cleared it
		r.reset()
	}
	if !v.loaded && !r.m.Yielded() { // nothing of ours on the card, nothing to give up: not watched
		r.reset()
		return
	}
	r.arm(enginePCI(v))

	r.mu.Lock()
	cur, have, fresh := r.held, r.hasHeld, r.heldNew
	r.heldNew = false
	r.mu.Unlock()
	if !have {
		return // the first reading has not held: no baseline without it
	}
	if v.engine != cur.engine { // the adapter became known or changed: the card is read on it now
		r.look()
	}
	launches := cur.launches
	if r.launchWord.Load() { // the OS said it: read them now
		r.osMu.Lock()
		launches = r.o.Launches()
		r.osMu.Unlock()
	}
	games, _ := r.split(launches)
	if fresh {
		if cur.engine != r.baseEngine && !r.m.Yielded() { // the adapter became known or changed: look afresh
			r.tr = &yield.Tracker{Floor: r.y.Policy.Floor, MinNew: r.y.MinNewMiB}
			r.baseEngine = cur.engine
		}
		if !r.tr.Baselined() {
			r.tr.Reset(cur.uses) // whatever runs when the engine starts to watch is the baseline
			r.changes = nil
		} else {
			r.changes = r.tr.Changes(cur.uses)
		}
		var pids []int
		for _, c := range r.changes {
			pids = append(pids, c.PID)
		}
		for _, g := range games {
			if g.PID > 0 {
				pids = append(pids, g.PID)
			}
		}
		r.followOnly(pids)
	}
	now := s.cfg.Now()
	d := r.m.Step(yield.Input{Now: now, Games: games, Changes: r.changes})
	switch d.Action {
	case yield.Yield:
		s.yieldNow(d)
	case yield.Resume:
		if s.resumeYield(d.Beside) {
			r.m.Reset()
			r.tr.Reset(cur.uses)
			r.changes = nil
		} else {
			r.m.Hold(now)
		}
	}
	r.cancelMoment()
	if left := r.m.Due(s.cfg.Now()); left > 0 {
		// named moment: the keeper's --yield-grace and --yield-steady, what is left of the one that runs
		r.stopMoment = host.Moment(left, r.wake)
	}
	if y := r.y; y.Looked != nil {
		y.Looked(cur.engine, cur.uses, games)
	}
}

// yieldNow stops admitting requests at once, lets replies in flight finish (up to Drain, a named
// moment), then unloads llama-server. The model book and recovery record are not touched: a yield
// is neither a crash nor a memory shortfall.
func (s *Supervisor) yieldNow(d yield.Decision) {
	start := s.cfg.Now()
	s.mu.Lock()
	s.yieldInfo = &YieldInfo{Reason: d.Reason, Kind: d.Kind, Since: start, Draining: true}
	s.refreshLocked()
	s.mu.Unlock()
	s.logf("yield: the card goes to %s", d.Reason)
	ctx, cancel := context.WithCancel(s.root)
	// named moment: Yield.Drain, the longest wait for replies in flight before the unload
	stop := host.Moment(s.cfg.Yield.Drain, cancel)
	s.waitIdle(ctx)
	stop()
	cancel()
	if n := s.inflightNow(); n > 0 && s.root.Err() == nil {
		s.logf("yield: %d request(s) still running after %s: cut", n, s.cfg.Yield.Drain)
	}
	s.mu.Lock()
	lc := s.loadCancel
	s.mu.Unlock()
	if lc != nil {
		lc()
	}
	s.opMu.Lock()
	s.mu.Lock()
	still := s.yieldInfo != nil
	if still {
		s.yieldInfo.Draining = false
		s.refreshLocked()
	}
	s.mu.Unlock()
	if still { // said before the stop: a phase that says ready while the server is stopped is a lie
		s.setPhase("yielded", "Paused while "+d.Reason+" runs")
	}
	if s.inflightNow() == 0 { // a reply cut by the drain's end leaves its slot half-written: nothing kept
		s.keepContexts("a yield to "+d.Reason, true)
	}
	s.stopServerLocked()
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
	s.refreshLocked()
	s.mu.Unlock()
	if was && s.srv == nil {
		s.settle()
	}
}
