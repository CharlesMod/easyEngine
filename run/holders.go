package run

// holders.go — who holds the card's device nodes open now, read once for every word that serves
// the room watch and the yield watch both.
//
// An open of a device node is a word that somebody touched the card, not that the room moved: a
// monitor that opens it, queries and exits leaves the card as it was. What moves the room is a
// holder that appears or goes, so each watch compares the holders it sees with those at its last
// look and reads the card only when they differ. The scan (every process's fd directory) is
// the same for both, so a word is scanned once however many watches are armed: a word takes a
// ticket as it arrives, and a scan that began after the ticket was taken saw what the word did,
// whoever asks for it.

import (
	"os"
	"sync"
	"sync/atomic"

	"github.com/CharlesMod/easyEngine/yield"
)

// holderScanHook, when set, replaces the scan (run/export_room_test.go).
var holderScanHook atomic.Pointer[func() map[int]string]

type holderScan struct {
	s       *Supervisor
	arrived atomic.Uint64 // words that have come

	mu    sync.Mutex
	o     yield.OS // the scan's own view of the OS, made when first needed
	upTo  uint64   // the scan in res began after this many words had arrived
	pci   string   // the adapter res was scanned for
	res   map[int]string
	scans int // scans made (a test counts them)
}

// word takes the ticket of a word that has just come.
func (h *holderScan) word() uint64 { return h.arrived.Add(1) }

// look gives the holders (pid -> name) as of the ticket's word or later, scanning only when no
// scan since has. The map is shared: callers do not change it. The holders are those of the
// engine's adapter once its pci is known (else any GPU node), and exclude the engine's own and
// what descends from it.
func (h *holderScan) look(ticket uint64, pci string) map[int]string {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.res != nil && h.upTo >= ticket && h.pci == pci {
		return h.res
	}
	began := h.arrived.Load()
	h.res = h.scan(pci)
	h.upTo, h.pci = began, pci
	h.scans++
	return h.res
}

func (h *holderScan) scan(pci string) map[int]string {
	if f := holderScanHook.Load(); f != nil {
		return (*f)()
	}
	if h.o == nil {
		if y := h.s.cfg.Yield; y != nil && y.OS != nil {
			h.o = y.OS()
		} else {
			h.o = yield.NewOS()
		}
	}
	if pci != "" {
		h.o.Adapter(pci)
	}
	self := os.Getpid()
	fixed := h.s.fixedPids()
	out := h.o.Openers(func(pid int) bool { return fixed[pid] || h.o.Under(pid, self) })
	if out == nil {
		out = map[int]string{}
	}
	return out
}
