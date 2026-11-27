package plan

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/CharlesMod/easyEngine/core/fit"

	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// HostAdapter is what the host app says about the adapter it renders on.
type HostAdapter struct {
	Name      string `json:"name"`
	VendorID  int    `json:"vendor_id"`
	MemoryMiB int    `json:"memory_mib"`
}

// HostInfo is what the engine knows about the host app: its pid (0 unknown) and, if the host
// said, the adapter it renders on.
type HostInfo struct {
	PID     int
	Adapter *HostAdapter
}

// othersNoteMiB is the least other-apps use worth telling the user about.
const othersNoteMiB = 256

// integratedShareOfRAM is the largest share of the machine's RAM an integrated adapter's pool
// may take (Apple's unified memory is bounded the same way by the probe).
const integratedShareOfRAM = 70

var amdIntegratedName = regexp.MustCompile(`(?i)^\s*(amd\s+)?radeon(\(tm\))?\s+graphics\s*$`)

// adaptersFromHW describes hw's GPUs as adapters, for a machine nothing sensed per adapter: the
// card's used memory stands for the whole card ("card" basis). An Intel GPU that is not an Arc
// and AMD's plain "Radeon Graphics" are taken as integrated.
func adaptersFromHW(hw probe.Hardware) []gpusense.Adapter {
	var as []gpusense.Adapter
	for i, g := range hw.GPUs {
		a := gpusense.Adapter{Index: i, Vendor: g.Vendor, Name: g.Name, DedicatedMiB: g.VRAMTotalMiB,
			UsedMiB: g.VRAMUsedMiB, Unified: g.UnifiedMemory, Source: g.Source}
		switch g.Vendor {
		case "nvidia":
			a.VendorID = 0x10de
		case "amd":
			a.VendorID = 0x1002
			a.Integrated = amdIntegratedName.MatchString(g.Name)
		case "intel":
			a.VendorID = 0x8086
			a.Integrated = !strings.Contains(strings.ToLower(g.Name), "arc")
		}
		if g.UnifiedMemory {
			a.Integrated, a.UsedMiB = true, -1
		}
		as = append(as, a)
	}
	return as
}

func adapterByIndex(as []gpusense.Adapter, idx int) (gpusense.Adapter, bool) {
	for _, a := range as {
		if a.Index == idx {
			return a, true
		}
	}
	return gpusense.Adapter{}, false
}

// pickEngine is the adapter the engine runs on: the discrete one with the most dedicated
// memory (of equal cards the emptier); an integrated one only when there is no discrete adapter with memory.
func pickEngine(as []gpusense.Adapter) (gpusense.Adapter, bool) {
	var best gpusense.Adapter
	found := false
	for _, a := range as {
		if a.Integrated || a.Unified || a.DedicatedMiB <= 0 {
			continue
		}
		if !found || a.DedicatedMiB > best.DedicatedMiB || (a.DedicatedMiB == best.DedicatedMiB && a.UsedMiB >= 0 && best.UsedMiB > a.UsedMiB) {
			best, found = a, true // equal cards: the emptier one
		}
	}
	if found {
		return best, true
	}
	for _, a := range as {
		if !(a.Integrated || a.Unified) || a.DedicatedMiB+a.SharedMiB <= 0 {
			continue
		}
		if !found || a.DedicatedMiB+a.SharedMiB > best.DedicatedMiB+best.SharedMiB {
			best, found = a, true
		}
	}
	return best, found
}

// oursAdapter is the discrete adapter the engine's own processes hold memory on, if any.
func oursAdapter(pic gpusense.Picture, as []gpusense.Adapter) (gpusense.Adapter, bool) {
	best, held := gpusense.Adapter{}, 0
	for _, p := range pic.Processes {
		if !p.Ours || p.DedicatedMiB <= held {
			continue
		}
		if a, ok := adapterByIndex(as, p.Adapter); ok && !a.Integrated && !a.Unified && a.DedicatedMiB > 0 {
			best, held = a, p.DedicatedMiB
		}
	}
	return best, held > 0
}

// pooled reports whether the adapter's memory is system memory (unified or integrated), whose
// dedicated figure is never taken as VRAM.
func pooled(a gpusense.Adapter) bool { return a.Unified || a.Integrated }

// capacity is the memory in MiB a model may be planned against on adapter a: a discrete card's
// dedicated memory; for a pooled adapter what it may borrow (dedicated plus shared; half the RAM
// where the shared figure is unknown), bounded by integratedShareOfRAM of the RAM. An Apple
// adapter takes the probe's own figure where it has one.
func capacity(a gpusense.Adapter, hw probe.Hardware) int {
	if !pooled(a) {
		// A card the probe lists keeps the probe's total (nvidia-smi on NVIDIA), the figure the
		// footprints and the planner's thresholds were set against; Windows' DXGI reports a
		// little less for the same card (3965 of 4096 MiB on a 4 GB laptop part).
		if i := hwIndex(hw, a); i >= 0 && hw.GPUs[i].VRAMTotalMiB > 0 {
			return hw.GPUs[i].VRAMTotalMiB
		}
		return a.DedicatedMiB
	}
	if a.Unified {
		for _, g := range hw.GPUs {
			if g.UnifiedMemory && g.VRAMTotalMiB > 0 {
				return g.VRAMTotalMiB
			}
		}
	}
	pool := a.DedicatedMiB + a.SharedMiB
	if a.SharedMiB <= 0 {
		pool = a.DedicatedMiB + hw.RAMTotalMiB/2
	}
	if hw.RAMTotalMiB > 0 {
		pool = min(pool, hw.RAMTotalMiB*integratedShareOfRAM/100)
	}
	return pool
}

// MeasureReserve is the reserve the engine plans around, measured from a picture of the
// adapters (gpusense.Sense) instead of guessed. The engine runs on one adapter E (pickEngine).
//
//	others = what processes neither ours nor the host hold on E (basis "processes"), or the
//	         card's used memory less ours and the host's where that is more (driver contexts and
//	         processes the sense cannot see); where no
//	         process is listed on E, the card's used memory less ours ("card"); a pooled adapter
//	         (unified or integrated) counts the pool's memory the rest of the system holds
//	host   = 0 when the host renders on another adapter ("other-adapter"); else the larger of
//	         what the host measurably holds on E and estimateMiB ("measured" | "estimate")
//	margin = fit.DefaultMargin
//
// The host's adapter is where its pid holds memory (the most, dedicated plus shared), else the one host.Adapter names,
// else unknown (the estimate is held). An empty picture (nothing sensed) is read from hw's GPUs
// with the card's used memory standing for the whole card, as before the engine sensed.
// engineGPU is E's index in hw.GPUs, -1 when E is not one of them or there is no E (CPU).
func MeasureReserve(pic gpusense.Picture, hw probe.Hardware, host HostInfo, estimateMiB int) (gpusense.GPUSummary, int) {
	return MeasureReserveOn(pic, hw, host, estimateMiB, -1)
}

// MeasureReserveOn is MeasureReserve for an engine that has a card already: keep is the adapter
// index its model server last ran on (-1: none). The engine stays on the card its server holds
// memory on, else on keep when keep is as large as the card it would pick; it never moves to an
// equal card because another app filled its own (the pin follows this choice, plan/device.go).
func MeasureReserveOn(pic gpusense.Picture, hw probe.Hardware, host HostInfo, estimateMiB int, keep int) (gpusense.GPUSummary, int) {
	estimateMiB = max(estimateMiB, 0)
	as := pic.Adapters
	if len(as) == 0 {
		as = adaptersFromHW(hw)
		pic = gpusense.Picture{Adapters: as, Processes: pic.Processes, Sources: pic.Sources}
	}
	sum := gpusense.GPUSummary{Adapters: as, EngineAdapter: -1, HostAdapter: -1, HostIntegrated: "unknown"}
	sum.Reserve.OthersBasis, sum.Reserve.HostBasis = "none", "none"
	e, ok := pickEngine(as)
	if !ok {
		sum.HostAdapter, sum.HostIntegrated = hostAdapter(pic, host)
		return sum, -1
	}
	if oa, has := oursAdapter(pic, as); has {
		// the engine's server already runs there: that card stays the engine's, however full
		// another app makes it (of equal cards the emptier would flip to the other one)
		e = oa
	} else if k, has := adapterByIndex(as, keep); keep >= 0 && has && !pooled(k) && k.DedicatedMiB >= e.DedicatedMiB {
		e = k // between a stop and the next start (a resize, a crash) the server holds nothing
	}
	sum.EngineAdapter = e.Index
	total := capacity(e, hw)

	// others
	procs := pic.OnAdapter(e.Index)
	isHost := func(p gpusense.Process) bool { return p.Host || (host.PID > 0 && p.PID == host.PID) }
	ours, others := 0, 0
	for _, p := range procs {
		switch {
		case p.Ours:
			ours += p.DedicatedMiB
		case isHost(p):
		default:
			others += p.DedicatedMiB
		}
	}
	switch {
	case pooled(e):
		sum.Reserve.OthersBasis = "card"
		others = 0
		if hw.RAMAvailMiB > 0 {
			others = max(total-hw.RAMAvailMiB, 0) // what the rest of the system already holds
		}
	case len(procs) > 0:
		sum.Reserve.OthersBasis = "processes"
		if e.UsedMiB >= 0 { // the card's own count also holds what no listed process accounts for
			hostHeld := 0
			for _, p := range procs {
				if !p.Ours && isHost(p) {
					hostHeld += p.DedicatedMiB
				}
			}
			others = max(others, e.UsedMiB-ours-hostHeld)
		}
	case e.UsedMiB >= 0:
		sum.Reserve.OthersBasis = "card"
		others = max(e.UsedMiB-ours, 0)
	default:
		others = 0
	}

	// host
	measured := 0
	for _, p := range procs {
		if isHost(p) {
			measured += p.DedicatedMiB
		}
	}
	sum.HostAdapter, sum.HostIntegrated = hostAdapter(pic, host)
	hostMiB, basis := 0, "none"
	switch {
	case sum.HostAdapter >= 0 && sum.HostAdapter != e.Index:
		basis = "other-adapter"
	case measured > estimateMiB:
		hostMiB, basis = measured, "measured"
	case estimateMiB > 0:
		hostMiB, basis = estimateMiB, "estimate"
	}

	margin := fit.DefaultMargin(total)
	sum.Reserve.OthersMiB, sum.Reserve.HostMiB, sum.Reserve.HostBasis = others, hostMiB, basis
	sum.Reserve.MarginMiB, sum.Reserve.TotalMiB = margin, total
	refreshFree(&sum)
	return sum, hwIndex(hw, e)
}

// hostAdapter resolves the adapter the host renders on and whether it is integrated.
func hostAdapter(pic gpusense.Picture, host HostInfo) (int, string) {
	idx, held := -1, 0
	for _, p := range pic.Processes {
		// any memory counts: on an integrated adapter a renderer holds shared memory only
		if m := p.DedicatedMiB + p.SharedMiB; (p.Host || (host.PID > 0 && p.PID == host.PID)) && m > held {
			if _, ok := adapterByIndex(pic.Adapters, p.Adapter); ok {
				idx, held = p.Adapter, m
			}
		}
	}
	if idx < 0 && host.Adapter != nil && (host.Adapter.Name != "" || host.Adapter.VendorID != 0) {
		idx = gpusense.FindAdapter(pic.Adapters, host.Adapter.VendorID, host.Adapter.Name)
	}
	if idx < 0 {
		return -1, "unknown"
	}
	a, ok := adapterByIndex(pic.Adapters, idx)
	switch {
	case !ok:
		return -1, "unknown"
	case a.Integrated:
		return idx, "yes"
	}
	return idx, "no"
}

// refreshFree recomputes FreeForModelMiB and OthersNote from the reserve.
func refreshFree(sum *gpusense.GPUSummary) {
	r := sum.Reserve
	sum.FreeForModelMiB = max(r.TotalMiB-r.OthersMiB-r.HostMiB-r.MarginMiB, 0)
	sum.OthersNote = ""
	if r.OthersMiB >= othersNoteMiB {
		sum.OthersNote = fmt.Sprintf("Other apps are using %.1f GB of your graphics card", float64(r.OthersMiB)/1024)
	}
}

// SetOthers replaces the summary's others figure (the card basis) with a reading taken another
// way, for a caller that has a better source than the picture.
func SetOthers(sum *gpusense.GPUSummary, mib int) {
	if sum.EngineAdapter < 0 {
		return
	}
	sum.Reserve.OthersMiB, sum.Reserve.OthersBasis = max(mib, 0), "card"
	refreshFree(sum)
}

// hwIndex is a's place in hw.GPUs: the GPU with its vendor, name and index, else its vendor and
// name, else (NVIDIA) its index; -1 when hw does not list it.
func hwIndex(hw probe.Hardware, a gpusense.Adapter) int {
	same := func(g probe.GPU) bool {
		return g.Vendor == a.Vendor && strings.EqualFold(strings.TrimSpace(g.Name), strings.TrimSpace(a.Name))
	}
	for i, g := range hw.GPUs {
		if same(g) && g.Index == a.Index {
			return i
		}
	}
	for i, g := range hw.GPUs {
		if same(g) {
			return i
		}
	}
	for i, g := range hw.GPUs {
		if a.Vendor == "nvidia" && g.Vendor == "nvidia" && g.Index == a.Index {
			return i
		}
	}
	return -1
}

// RoomForSensed is the planning room for a measured reserve: the engine's adapter, its
// capacity, and the others, host and margin MeasureReserve found. backend is the llama.cpp
// build's id; an integrated adapter (not Apple's unified memory) is used only by a vulkan build,
// as a pool of system memory (Room.Unified), and otherwise the run is CPU-only: its dedicated
// figure is never taken as VRAM.
func RoomForSensed(hw probe.Hardware, sum gpusense.GPUSummary, backend string) fit.Room {
	r := RoomForRun(hw, sum, backend)
	if r.NoGPU {
		return r
	}
	if v, _, err := r.Budget(); err == nil && int(v) < minUsableVRAM {
		r.NoGPU = true // too little card to be worth using next to the game
	}
	return r
}

// RoomForRun is RoomForSensed for a server that is running or about to: the same room without
// the planner's rule that a card with under minUsableVRAM left is no GPU (the shape chosen
// answers for it).
func RoomForRun(hw probe.Hardware, sum gpusense.GPUSummary, backend string) fit.Room {
	r := fit.Room{RAMAvail: hw.RAMAvailMiB}
	e, ok := adapterByIndex(sum.Adapters, sum.EngineAdapter)
	if !ok || sum.Reserve.TotalMiB <= 0 {
		r.NoGPU = true
		return r
	}
	r.GPUName = e.Name
	r.VRAMTotal = sum.Reserve.TotalMiB
	r.Margin = sum.Reserve.MarginMiB
	r.Others, r.GameReserve = sum.Reserve.OthersMiB, sum.Reserve.HostMiB
	r.Unified = pooled(e)
	if e.Integrated && !e.Unified && !strings.HasPrefix(backend, "vulkan") {
		r.NoGPU = true
	}
	return r
}
