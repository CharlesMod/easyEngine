package run

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// gpuReportTTL is how long GET /engine/v1/gpu serves a reading before it senses again.
const gpuReportTTL = 5 * time.Second

// heldTolMiB is how far two reads of the reserve may differ and still agree: a change reshapes
// the model only when two consecutive reads agree, so a game still loading does not move it.
const heldTolMiB = 64

// gpuState is what the host told the engine about itself and the last readings (under s.mu).
type gpuState struct {
	hostPID     int
	hostAdapter *plan.HostAdapter
	hostReserve int
	sum         *gpusense.GPUSummary // the last reading of the watchdog, a plan or a room
	rep         *gpusense.GPUReport  // GET /engine/v1/gpu's cache
	repAt       time.Time
	// keep is the adapter the model server last started on, +1 (0: none yet): the engine stays
	// on that card for the session (plan.MeasureReserveOn).
	keep int
}

// ourPID is the running llama-server's pid (0: none), readable without the op lock.
type ourPID struct{ v atomic.Int64 }

func (o *ourPID) set(p int) { o.v.Store(int64(p)) }
func (o *ourPID) pids() []int {
	if p := int(o.v.Load()); p > 0 {
		return []int{p}
	}
	return nil
}

// HostSet is POST /engine/v1/host. Zero fields keep what is held (a pid 0, a nil Adapter, a
// ReserveMiB 0).
type HostSet struct {
	PID        int               `json:"pid"`
	Adapter    *plan.HostAdapter `json:"adapter"`
	ReserveMiB int               `json:"reserve_mib"`
}

// ErrBadHost is a refused HostSet.
var ErrBadHost = errors.New("bad host")

// SetHost stores what the host says about itself. The room is read again at once with it (its
// word is an event, run/room.go); a reading cached for GET /engine/v1/gpu is dropped.
func (s *Supervisor) SetHost(h HostSet) error {
	switch {
	case h.PID < 0:
		return fmt.Errorf("%w: pid must not be negative", ErrBadHost)
	case h.PID > 0 && !lifecycle.PidAlive(h.PID):
		return fmt.Errorf("%w: no process %d", ErrBadHost, h.PID)
	case h.ReserveMiB < 0 || h.ReserveMiB > 1<<20:
		return fmt.Errorf("%w: reserve_mib out of range", ErrBadHost)
	}
	if a := h.Adapter; a != nil && (a.VendorID < 0 || a.MemoryMiB < 0 || a.VendorID > 0xffff || (a.Name == "" && a.VendorID == 0)) {
		return fmt.Errorf("%w: adapter needs a name or a vendor_id, and no negative figures", ErrBadHost)
	}
	s.mu.Lock()
	if h.PID > 0 {
		s.gpu.hostPID = h.PID
	}
	if h.Adapter != nil {
		a := *h.Adapter
		s.gpu.hostAdapter = &a
	}
	if h.ReserveMiB > 0 {
		s.gpu.hostReserve = h.ReserveMiB
	}
	s.gpu.rep = nil
	s.mu.Unlock()
	s.card.event() // the host's word: the room is read again
	return nil
}

// hostInfo is what is known of the host app and its estimate of its own use: the pid from
// POST /engine/v1/host, else the profile's, else the first parent pid; the estimate from the
// POST, else the profile's game_reserve_mib.
func (s *Supervisor) hostInfo() (plan.HostInfo, int) {
	s.mu.Lock()
	pid, ad, est := s.gpu.hostPID, s.gpu.hostAdapter, s.cfg.Profile.GameReserveMiB
	if pid == 0 {
		pid = s.cfg.Profile.HostPID
	}
	if ad == nil {
		ad = s.cfg.Profile.HostAdapter
	}
	if s.gpu.hostReserve > 0 {
		est = s.gpu.hostReserve
	}
	s.mu.Unlock()
	if pid == 0 && s.cfg.Parents != nil {
		if ps := s.cfg.Parents(); len(ps) > 0 {
			pid = ps[0]
		}
	}
	return plan.HostInfo{PID: pid, Adapter: ad}, est
}

// sensed is one reading: the picture, the reserve measured from it and the engine's GPU in
// hw.GPUs (-1: none).
type sensed struct {
	pic gpusense.Picture
	sum gpusense.GPUSummary
	gpu int
}

// measure senses the adapters and measures the reserve; ours are the engine's pids. Where the
// picture has no adapters (nothing could be sensed) the card's others come from cfg.Others, as
// they did before the engine sensed. The summary is kept for the status.
func (s *Supervisor) measure(ctx context.Context, hw probe.Hardware, ours []int) sensed {
	host, est := s.hostInfo()
	pic := s.cfg.Sense(ctx, ours, host.PID)
	s.mu.Lock()
	keep := s.gpu.keep - 1
	s.mu.Unlock()
	sum, gi := plan.MeasureReserveOn(pic, hw, host, est, keep)
	if len(pic.Adapters) == 0 && gi >= 0 {
		g := hw.GPUs[gi]
		if a := sum.Adapters[sum.EngineAdapter]; !a.Unified && !a.Integrated {
			o, err := s.cfg.Others(ctx, g, ours)
			if err != nil {
				o = g.VRAMUsedMiB
			}
			plan.SetOthers(&sum, o)
		}
	}
	s.mu.Lock()
	cp := sum
	s.gpu.sum = &cp
	s.say("room", "", cp) // a reading: said only when it changed
	s.mu.Unlock()
	return sensed{pic: pic, sum: sum, gpu: gi}
}

func (sn sensed) report() *gpusense.GPUReport {
	src := map[string]string{}
	for k, v := range sn.pic.Sources {
		src[k] = v
	}
	procs := sn.pic.Processes
	if procs == nil {
		procs = []gpusense.Process{}
	}
	return &gpusense.GPUReport{GPUSummary: sn.sum, Processes: procs, Sources: src}
}

// GPUReport is GET /engine/v1/gpu: the adapters, who holds memory on them and the reserve, read
// at most every gpuReportTTL unless refresh.
func (s *Supervisor) GPUReport(ctx context.Context, refresh bool) (gpusense.GPUReport, error) {
	s.mu.Lock()
	rep, at := s.gpu.rep, s.gpu.repAt
	s.mu.Unlock()
	if rep != nil && !refresh && s.cfg.Now().Sub(at) < gpuReportTTL {
		return *rep, nil
	}
	hw, err := s.Hardware(ctx, true)
	if err != nil {
		return gpusense.GPUReport{}, err
	}
	r := s.measure(ctx, hw, s.ours.pids()).report()
	s.mu.Lock()
	s.gpu.rep, s.gpu.repAt = r, s.cfg.Now()
	s.mu.Unlock()
	return *r, nil
}

// backendNow is the id of the llama.cpp build in use.
func (s *Supervisor) backendNow() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.st.Backend
}
