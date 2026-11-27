// Package smiloop answers the engine's repeated nvidia-smi reads from one long-lived child per
// query instead of a process per read. Starting nvidia-smi initialises the driver library each
// time, which is most of what a read costs; the tool can repeat a query itself (`-lms <ms>`), so
// the child is started once and the latest whole block it printed is kept with the time it was
// read. A line arriving is the event; the child's period is the watchdog for a level that has no
// event (the card's memory and who holds it), and nothing on this side has a timer.
//
// Reads are opt-in twice: a user Holds the loop while it needs it (the child ends when the last
// user lets go, and with the engine: it is launched through lifecycle, so it dies with it), and a
// read asks for a sample by putting Sampled on its context. Any other read, a decision at a start
// or a grow, runs the one-shot as before. A read answers from the sample only while the sample is
// fresh (read after the last Stale and no older than a period and a half); else it is one-shot.
//
// Known limit, found on a real card (see the report of ee-sense): a looping nvidia-smi opens the
// card's device nodes again, a burst of four, in every period, at a time the engine cannot predict
// from the lines it prints. host.Own cannot bracket those opens, so every inotify watch of the
// nodes (host.Opens) hears them. The package therefore is not wired into run/ by itself.
package smiloop

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/lifecycle"
)

// Kind is one of the queries that can be looped.
type Kind int

const (
	CSV  Kind = iota // --query-gpu=<fields> --format=csv,noheader,nounits
	XML              // -q -x
	PIDS             // -q -d PIDS
)

func (k Kind) String() string { return [...]string{"csv", "xml", "pids"}[k] }

// KindOf says which loopable query args are (the arguments after "nvidia-smi"), and the field list
// of a csv one.
func KindOf(args []string) (Kind, string, bool) {
	switch {
	case len(args) == 2 && args[0] == "-q" && args[1] == "-x":
		return XML, "", true
	case len(args) == 3 && args[0] == "-q" && args[1] == "-d" && args[2] == "PIDS":
		return PIDS, "", true
	case len(args) == 2 && strings.HasPrefix(args[0], "--query-gpu=") && args[1] == "--format=csv,noheader,nounits":
		return CSV, strings.TrimPrefix(args[0], "--query-gpu="), true
	}
	return 0, "", false
}

// Config is what a Manager needs; the zero value is the engine's own.
type Config struct {
	Path    string                 // the nvidia-smi to run; empty: found on PATH when first asked
	Periods map[Kind]time.Duration // the child's period per query (default: pids 1 s, the others 5 s)
	Grace   time.Duration          // how long a child has to end when let go (default 2 s)
	Logf    func(format string, a ...any)
	// Heard is called after each whole block (tests wait on it; it must not call the manager).
	Heard func(Kind)
	// Ended is called after a child's end has been taken in (tests wait on it).
	Ended func(Kind)
}

// Manager owns the children. One per engine; Default is it.
type Manager struct {
	cfg   Config
	group *lifecycle.Group

	mu      sync.Mutex
	users   int
	readers map[string]*reader
	staleAt time.Time
	wg      sync.WaitGroup // the goroutines that wait for a child's end
	starts  atomic.Int64   // children started since the manager was made
}

// Starts is how many children this manager has started.
func (m *Manager) Starts() int { return int(m.starts.Load()) }

// NewManager makes a manager with no child running.
func NewManager(cfg Config) *Manager {
	if cfg.Grace <= 0 {
		cfg.Grace = 2 * time.Second
	}
	if cfg.Logf == nil {
		cfg.Logf = func(string, ...any) {}
	}
	return &Manager{cfg: cfg, group: lifecycle.NewGroup(), readers: map[string]*reader{}}
}

func (m *Manager) period(k Kind) time.Duration {
	if d := m.cfg.Periods[k]; d > 0 {
		return d
	}
	if k == PIDS {
		return time.Second
	}
	return 5 * time.Second
}

// Hold says a user needs the loop; the returned func lets go (once). The child of a query starts
// when it is first read, not here. When the last user lets go every child is ended.
func (m *Manager) Hold() (release func()) {
	m.mu.Lock()
	m.users++
	m.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			m.mu.Lock()
			m.users--
			var stop []*lifecycle.Proc
			if m.users == 0 {
				stop = m.stopAllLocked()
			}
			m.mu.Unlock()
			m.endAll(stop)
		})
	}
}

// Stale says something happened that makes every sample read before now out of date (a word that
// the card changed): the next reads are one-shot until a block newer than this arrives.
func (m *Manager) Stale() {
	m.mu.Lock()
	m.staleAt = time.Now()
	m.mu.Unlock()
}

// Close ends every child and waits until each has gone. The manager can be held again after.
func (m *Manager) Close() {
	m.mu.Lock()
	m.users = 0
	stop := m.stopAllLocked()
	m.mu.Unlock()
	m.endAll(stop)
	m.wg.Wait()
}

func (m *Manager) stopAllLocked() []*lifecycle.Proc {
	var ps []*lifecycle.Proc
	for _, r := range m.readers {
		r.mu.Lock()
		if r.proc != nil {
			ps = append(ps, r.proc)
			r.proc = nil
		}
		r.sample = nil
		r.fails, r.gaveUp = 0, false
		r.mu.Unlock()
	}
	return ps
}

func (m *Manager) endAll(ps []*lifecycle.Proc) {
	var wg sync.WaitGroup
	for _, p := range ps {
		wg.Add(1)
		go func() { defer wg.Done(); p.Stop(m.cfg.Grace) }()
	}
	wg.Wait()
}

// Read answers args from the loop's latest block when a user holds the loop and the sample is
// fresh; ok is false in every other case and the caller runs the one-shot. A read that finds the
// query's child not running starts it (once for each event that finds it dead, and never again
// after it died twice without proving itself).
func (m *Manager) Read(args []string) (out []byte, ok bool) {
	k, fields, loopable := KindOf(args)
	if !loopable {
		return nil, false
	}
	m.mu.Lock()
	if m.users == 0 {
		m.mu.Unlock()
		return nil, false
	}
	key := k.String() + "|" + fields
	r := m.readers[key]
	if r == nil {
		r = &reader{m: m, kind: k, fields: fields}
		m.readers[key] = r
	}
	staleAt := m.staleAt
	m.mu.Unlock()
	return r.read(staleAt)
}

type reader struct {
	m      *Manager
	kind   Kind
	fields string

	mu     sync.Mutex
	proc   *lifecycle.Proc
	gen    int
	p      parser
	blocks int // whole blocks this child has printed
	sample []byte
	at     time.Time
	fails  int
	gaveUp bool
}

func (r *reader) read(staleAt time.Time) ([]byte, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.gaveUp {
		return nil, false
	}
	if r.proc == nil {
		r.startLocked()
		if r.proc == nil {
			return nil, false
		}
	}
	if r.sample == nil || !r.at.After(staleAt) || time.Since(r.at) > r.m.period(r.kind)*3/2 {
		return nil, false
	}
	return append([]byte(nil), r.sample...), true
}

func (r *reader) args() []string {
	ms := strconv.FormatInt(r.m.period(r.kind).Milliseconds(), 10)
	switch r.kind {
	case XML:
		return []string{"-q", "-x", "-lms", ms}
	case PIDS:
		return []string{"-q", "-d", "PIDS", "-lms", ms}
	}
	return []string{"--query-gpu=count," + r.fields, "--format=csv,noheader,nounits", "-lms", ms}
}

// startLocked starts the child (r.mu held) inside host.Own, which is the engine's way of saying
// its own open of the card's nodes is not a stranger's. It covers the first open only: see the
// package comment.
func (r *reader) startLocked() {
	path := r.m.cfg.Path
	if path == "" {
		p, err := exec.LookPath("nvidia-smi")
		if err != nil {
			r.gaveUp = true
			return
		}
		path = p
	}
	switch r.kind {
	case XML:
		r.p = &xmlParser{}
	case PIDS:
		r.p = &pidsParser{}
	default:
		r.p = &csvParser{}
	}
	r.gen++
	gen := r.gen
	r.blocks = 0
	sink, _ := lifecycle.OpenLog("")
	sink.OnLine(func(line string) { r.line(gen, line) })
	var proc *lifecycle.Proc
	var err error
	host.Own(func() {
		proc, err = r.m.group.Launch(lifecycle.Spec{Path: path, Args: r.args(), Name: "nvidia-smi " + r.kind.String(), Log: sink})
	})
	if err != nil {
		r.m.cfg.Logf("nvidia-smi loop (%s): %v; reads stay one-shot", r.kind, err)
		r.gaveUp = true
		return
	}
	r.proc = proc
	r.m.starts.Add(1)
	r.m.wg.Add(1)
	go func() {
		defer r.m.wg.Done()
		<-proc.Done()
		r.ended(gen, proc)
	}()
}

// line takes one line of the child's output (under the log sink's lock: quick).
func (r *reader) line(gen int, line string) {
	r.mu.Lock()
	if gen != r.gen || r.p == nil {
		r.mu.Unlock()
		return
	}
	b, whole := r.p.feed(line)
	if whole {
		r.sample, r.at = b, time.Now()
		r.blocks++
	}
	heard := r.m.cfg.Heard
	r.mu.Unlock()
	if whole && heard != nil {
		heard(r.kind)
	}
}

// ended is the child's end: when it was ended on purpose it is already forgotten. Else the next
// read starts it again, once for that event; a child that died twice in a row without printing two
// blocks is not started again (reads stay one-shot), said once.
func (r *reader) ended(gen int, p *lifecycle.Proc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.m.cfg.Ended != nil {
		defer r.m.cfg.Ended(r.kind)
	}
	if r.proc != p {
		return // let go on purpose
	}
	r.proc = nil
	if r.blocks >= 2 {
		r.fails = 0
	} else {
		r.fails++
	}
	if r.fails >= 2 {
		r.gaveUp = true
		r.m.cfg.Logf("nvidia-smi loop (%s) ended twice in a row without proving itself; reads stay one-shot", r.kind)
	}
}

// ---- the engine's one manager, and the context mark ----

// Default is the manager the engine's reads use.
var Default = NewManager(Config{})

type sampledKey struct{}

// Sampled marks ctx as one whose reads may be answered from the loop's sample.
func Sampled(ctx context.Context) context.Context { return context.WithValue(ctx, sampledKey{}, true) }

// IsSampled says whether ctx allows a sample.
func IsSampled(ctx context.Context) bool { v, _ := ctx.Value(sampledKey{}).(bool); return v }

// ReadFor is Read for a context: a sample only where ctx is Sampled.
func ReadFor(ctx context.Context, args []string) ([]byte, bool) {
	if !IsSampled(ctx) {
		return nil, false
	}
	return Default.Read(args)
}
