package run

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// The room: the reserve other programs and the host hold on the engine's card, read when needed
// and never watched on a clock of its own (docs/EVENTS.md section 3, "Sense the card"). It is
// read at each start (loadContext), before a grow, on an out-of-memory line, when a stranger
// opens the card's device (host.Opens, Linux), when a program the last picture listed ends
// (host.Exits) and on the host's word (SetHost). Another living program's memory gives no event
// (NVIDIA, and everything on Windows and macOS), so while a model is loaded the reading is also
// kept by host.Level, the one watchdog for it, which reports a change only once two reads agree:
// a game still loading does not move the model.

// roomRead is one reading of the room.
type roomRead struct {
	room  fit.Room
	sum   gpusense.GPUSummary
	total int // the reserve: others plus the host's share, MiB
}

// sameRoom says two readings are one level: the reserve within heldTolMiB.
func sameRoom(a, b roomRead) bool {
	d := a.total - b.total
	return a.room.NoGPU == b.room.NoGPU && d <= heldTolMiB && -d <= heldTolMiB
}

// roomDevicesMu guards roomDevices: a read of the room (its own goroutine) hears the globs too.
var roomDevicesMu sync.RWMutex

// roomDevices are the globs of the device nodes whose opens the room watch hears (a test points
// them at a temp dir).
var roomDevices = []string{"/dev/nvidia[0-9]*", "/dev/dri/renderD*", "/dev/dri/card*"}

// Test seams (run/export_room_test.go): a test hears that a read of the room has ended, and that
// the goroutine reading the opens of an arming has ended, by those events and never by a clock.
var (
	roomReadDone atomic.Pointer[func()]
	roomOpensEnd atomic.Pointer[func()]
	roomWordDone atomic.Pointer[func()] // a word of the nodes has been looked at (read or not)
)

func hook(p *atomic.Pointer[func()]) {
	if f := p.Load(); f != nil {
		(*f)()
	}
}

type roomWatch struct {
	s     *Supervisor
	level *host.Level[roomRead]
	// baseline: the level's first report since arm is still to come (it is not a change)
	baseline atomic.Bool

	mu        sync.Mutex
	armed     bool
	opensStop func()
	exits     map[int]func() // strangers followed to their end, with what lets each go; nil: none
	stopped   bool
	listened  string // the nodes listened to, joined

	kmu   sync.Mutex
	known map[int]bool // the holders at the last look (not nvidia-smi, not ours)
}

func newRoomWatch(s *Supervisor) *roomWatch {
	rw := &roomWatch{s: s, exits: map[int]func(){}}
	rw.level = host.NewLevel(s.cfg.RoomEvery, rw.read, sameRoom, rw.changed)
	return rw
}

// read senses the room once (never under opMu: the reactor and the level call it outside it).
func (rw *roomWatch) read() (roomRead, bool) {
	s := rw.s
	defer hook(&roomReadDone)
	if s.root.Err() != nil {
		return roomRead{}, false
	}
	// The probe and the sense each run nvidia-smi, which opens the card's device: those opens are
	// ours, and host.Own keeps every Opens watch (this one and the yield level's) from hearing
	// them, however late the kernel's events are read.
	var (
		hw   probe.Hardware
		err  error
		room fit.Room
		sn   sensed
	)
	host.Own(func() {
		if hw, err = s.cfg.Probe(s.root); err != nil {
			return
		}
		room, sn = s.roomSensed(s.root, hw, s.launching.pids())
	})
	if err != nil {
		return roomRead{}, false
	}
	var strangers []int
	for _, p := range sn.pic.Processes {
		if !p.Ours && p.PID > 0 && p.PID != os.Getpid() {
			strangers = append(strangers, p.PID)
		}
	}
	rw.follow(strangers)
	return roomRead{room: room, sum: sn.sum, total: sn.sum.Reserve.OthersMiB + sn.sum.Reserve.HostMiB}, true
}

// changed is a held change of the room: the reactor looks at it.
func (rw *roomWatch) changed(r roomRead) {
	rw.s.say("room.held", "", map[string]int{"reserve_mib": r.total})
	if rw.baseline.Swap(false) {
		return // the first reading of an arming is where the room stands, not a move
	}
	rw.s.wakeMind(func(c *causes) { c.room = true })
}

// held is the last reading that held, while armed.
func (rw *roomWatch) held() (roomRead, bool) { return rw.level.Last() }

// arm keeps the reading while a model is loaded: the level's watchdog, and on Linux the opens of
// the card's device nodes.
func (rw *roomWatch) arm() {
	rw.mu.Lock()
	if rw.armed || rw.stopped {
		rw.mu.Unlock()
		return
	}
	rw.armed = true
	rw.baseline.Store(true)
	rw.level.Arm()
	rw.level.Now() // a second read at once, so the baseline holds without waiting a period
	rw.mu.Unlock()
	// The nodes come after the reads are on their way, and outside rw.mu: they take the
	// supervisor's lock, which disarm holds before rw.mu. renew does the same, so a set found
	// after a disarm is dropped there.
	paths := rw.roomPaths()
	rw.mu.Lock()
	defer rw.mu.Unlock()
	if rw.armed && !rw.stopped && rw.listened == "" {
		rw.listenLocked(paths)
	}
}

// roomPaths are the device nodes whose opens the room hears: the engine's adapter's (see
// yield/os_linux.go adapterNodes), every GPU node while the adapter is not known.
func (rw *roomWatch) roomPaths() []string {
	if ns, ok := rw.s.holders.nodes(enginePCI(rw.s.yieldState())); ok && len(ns) > 0 {
		return ns
	}
	var paths []string
	roomDevicesMu.RLock()
	globs := roomDevices
	roomDevicesMu.RUnlock()
	for _, g := range globs {
		m, _ := filepath.Glob(g)
		paths = append(paths, m...)
	}
	return paths
}

// listenLocked starts the goroutine that hears the opens of roomPaths; the caller holds rw.mu
// and has set armed. A listening already on foot is stopped first.
func (rw *roomWatch) listenLocked(paths []string) {
	if rw.opensStop != nil {
		rw.opensStop()
		rw.opensStop = nil
	}
	sort.Strings(paths)
	rw.listened = strings.Join(paths, ",")
	if len(paths) == 0 {
		return
	}
	ch, stop, err := host.Opens(paths)
	if err != nil { // host.ErrNoEvent off Linux: the level's watchdog keeps it alone
		return
	}
	done := make(chan struct{}) // disarm ends the goroutine below: it never outlives its arming
	var once sync.Once
	rw.opensStop = func() { stop(); once.Do(func() { close(done) }) }
	go func() {
		defer hook(&roomOpensEnd)
		rw.opened(true) // who holds the nodes now: what the next word is compared with
		for {
			select {
			case <-done:
				return
			case <-rw.s.root.Done():
				return
			case _, ok := <-ch:
				if !ok {
					return
				}
				rw.opened(false)
			}
		}
	}()
}

// renew listens again when the nodes of the engine's adapter are not those listened to: the
// event that says so is the room's own new reading (the engine's adapter, and so its pci, is
// known or has changed: measure), never a clock.
func (rw *roomWatch) renew() {
	paths := rw.roomPaths() // before rw.mu, as in arm
	sort.Strings(paths)
	rw.mu.Lock()
	defer rw.mu.Unlock()
	if !rw.armed || rw.stopped {
		return
	}
	if strings.Join(paths, ",") == rw.listened {
		return
	}
	rw.listenLocked(paths)
}

// disarm lets the reading go when no model is loaded.
func (rw *roomWatch) disarm() {
	rw.mu.Lock()
	defer rw.mu.Unlock()
	rw.armed = false
	rw.listened = ""
	rw.level.Disarm()
	rw.s.sayGone("room.held", "")
	if rw.opensStop != nil {
		rw.opensStop()
		rw.opensStop = nil
	}
}

// opened is a word from the card's device nodes: somebody opened one. That says a program touched
// the card, not that the room moved (a monitor that opens it, queries and exits leaves it as it
// was), so the holders of the nodes are looked at, once for every watch (run/holders.go), and the
// card is read only when a holder not ours has appeared or gone since the last look. A holder
// that grows without opening anything is the level's to find. baseline only takes the holders.
func (rw *roomWatch) opened(baseline bool) {
	h := rw.s.holders
	set := h.look(h.word(), enginePCI(rw.s.yieldState()))
	moved := rw.holdersMoved(set)
	if moved && !baseline {
		rw.event()
	}
	hook(&roomWordDone)
}

// holdersMoved says whether the holders differ from those at the last look, and keeps these.
// nvidia-smi is no holder (a monitor runs it every second or two: it allocates nothing a game
// would miss, and a change it makes is never the room's).
func (rw *roomWatch) holdersMoved(set map[int]string) bool {
	next := map[int]bool{}
	for pid, name := range set {
		if !ignoredHolder(name) {
			next[pid] = true
		}
	}
	rw.kmu.Lock()
	defer rw.kmu.Unlock()
	moved := len(next) != len(rw.known)
	for pid := range next {
		if !rw.known[pid] {
			moved = true
		}
	}
	rw.known = next
	return moved
}

func ignoredHolder(name string) bool {
	n := normName(name)
	for _, b := range builtinIgnore {
		if n == b {
			return true
		}
	}
	return false
}

// event is a word that the room may have moved (a stranger's open or end, the host's word, an
// out-of-memory line): read it now, and let a retry that waits on the room look again.
func (rw *roomWatch) event() {
	rw.level.Now()
	rw.s.wakeMind(func(c *causes) { c.room = true })
}

// follow watches the strangers the last picture listed to their end; one no longer listed is let
// go. A pid is followed once while it is listed (one that has already ended is not asked again).
func (rw *roomWatch) follow(pids []int) {
	rw.mu.Lock()
	defer rw.mu.Unlock()
	if rw.stopped {
		return
	}
	listed := map[int]bool{}
	for _, p := range pids {
		listed[p] = true
		if _, ok := rw.exits[p]; ok {
			continue
		}
		ch, stop, err := host.Exits(p)
		if err != nil {
			rw.exits[p] = nil
			continue
		}
		done := make(chan struct{})
		rw.exits[p] = func() { stop(); close(done) }
		go func() {
			select {
			case <-ch:
				rw.event()
			case <-done:
			case <-rw.s.root.Done():
			}
		}()
	}
	for p, stop := range rw.exits {
		if !listed[p] {
			if stop != nil {
				stop()
			}
			delete(rw.exits, p)
		}
	}
}

// stop ends the watch for good (Shutdown).
func (rw *roomWatch) stop() {
	rw.level.Stop()
	rw.mu.Lock()
	defer rw.mu.Unlock()
	rw.stopped = true
	if rw.opensStop != nil {
		rw.opensStop()
		rw.opensStop = nil
	}
	for p, stop := range rw.exits {
		if stop != nil {
			stop()
		}
		delete(rw.exits, p)
	}
}

// ownReads wraps the reads that run nvidia-smi or open the card (Probe, Others, Sense) in
// host.Own, wherever they are called from (a start, the room's read, a status ask): the engine's
// own opens of the card are never heard as a stranger's by any Opens watch (calls nest).
func ownReads(c *Config) {
	p, o, sn := c.Probe, c.Others, c.Sense
	c.Probe = func(ctx context.Context) (hw probe.Hardware, err error) {
		host.Own(func() { hw, err = p(ctx) })
		return
	}
	c.Others = func(ctx context.Context, g probe.GPU, ours []int) (n int, err error) {
		host.Own(func() { n, err = o(ctx, g, ours) })
		return
	}
	c.Sense = func(ctx context.Context, ours []int, hostPID int) (pic gpusense.Picture) {
		host.Own(func() { pic = sn(ctx, ours, hostPID) })
		return
	}
}
