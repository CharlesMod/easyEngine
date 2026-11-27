package run

import (
	"os"
	"path/filepath"
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

// roomDevices are the globs of the device nodes whose opens the room watch hears (a test points
// them at a temp dir).
var roomDevices = []string{"/dev/nvidia[0-9]*", "/dev/dri/renderD*", "/dev/dri/card*"}

// Test seams (run/export_room_test.go): a test hears that a read of the room has ended, and that
// the goroutine reading the opens of an arming has ended, by those events and never by a clock.
var (
	roomReadDone atomic.Pointer[func()]
	roomOpensEnd atomic.Pointer[func()]
)

func hook(p *atomic.Pointer[func()]) {
	if f := p.Load(); f != nil {
		(*f)()
	}
}

type roomWatch struct {
	s     *Supervisor
	level *host.Level[roomRead]

	mu        sync.Mutex
	armed     bool
	opensStop func()
	exits     map[int]func() // strangers followed to their end, with what lets each go; nil: none
	stopped   bool
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
func (rw *roomWatch) changed(roomRead) {
	rw.s.wakeMind(func(c *causes) { c.room = true })
}

// held is the last reading that held, while armed.
func (rw *roomWatch) held() (roomRead, bool) { return rw.level.Last() }

// arm keeps the reading while a model is loaded: the level's watchdog, and on Linux the opens of
// the card's device nodes.
func (rw *roomWatch) arm() {
	rw.mu.Lock()
	defer rw.mu.Unlock()
	if rw.armed || rw.stopped {
		return
	}
	rw.armed = true
	rw.level.Arm()
	rw.level.Now() // a second read at once, so the baseline holds without waiting a period
	var paths []string
	for _, g := range roomDevices {
		m, _ := filepath.Glob(g)
		paths = append(paths, m...)
	}
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
				rw.event()
			}
		}
	}()
}

// disarm lets the reading go when no model is loaded.
func (rw *roomWatch) disarm() {
	rw.mu.Lock()
	defer rw.mu.Unlock()
	rw.armed = false
	rw.level.Disarm()
	if rw.opensStop != nil {
		rw.opensStop()
		rw.opensStop = nil
	}
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
