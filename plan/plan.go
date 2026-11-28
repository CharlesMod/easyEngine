// Package plan is the engine's planner of a whole catalog on the probed machine: for every
// model, whether it fits and in what shape, and which to recommend. It is the engine's side of
// the sizing core (core/fit, core/gguf): it joins the core to the catalog and the probe, opens
// model files and logs the core's notes. Nothing the core must stay free of (the machine, the
// clock, a log, a file) may move back into the core from here.
package plan

import (
	"errors"
	"fmt"
	"log"
	"math"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/core/gguf"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

const (
	mib = 1 << 20
	// minUsableVRAM: a GPU whose budget is below this is treated as no GPU by Plan.
	minUsableVRAM = 2048
)

// SeedFromSize is the footprint of a model not yet downloaded. With a measured footprint in the
// catalog (m.Footprint, from a real llama-server run) it is that block, so the verdict before
// download matches the one after. Without one it is the core's guess scaled by the file size
// (fit.SeedFromBytes). Fit is re-checked from the real header after download. A cache type the
// core does not know is a *fit.KVTypeError.
func SeedFromSize(m catalog.Model, kvType string) (fit.Footprint, error) {
	if kvType == "" {
		kvType = "q4_0"
	}
	if m.Footprint != nil {
		return FromMeasured(m.Footprint, kvType)
	}
	return fit.SeedFromBytes(m.Size, m.MoE, kvType)
}

// Seed is the core's fit.Seed with its notes logged, as the engine has always logged them.
func Seed(info gguf.Info, kvType string) (fit.Footprint, error) {
	fp, notes, err := fit.Seed(info, kvType)
	for _, n := range notes {
		log.Print(n)
	}
	return fp, err
}

// Verdict is one catalog model's answer for this machine.
type Verdict struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	// Codename is the tier's code name, shown before Name ("" when the catalog has none).
	Codename string `json:"codename"`
	// CodenameFull is the full code name (Campfire-26B); "" when the catalog has none.
	CodenameFull string `json:"codename_full"`
	// OptIn: a person picks this model; it is never Recommended.
	OptIn bool   `json:"opt_in,omitempty"`
	Size  int64  `json:"size"`
	Fits  string `json:"fits"` // yes | tight | no
	Why   string `json:"why"`
	// Limit is what stops a "no" model, one of the Limit* codes ("" when it fits). Why says it
	// in words; front ends may key on Limit.
	Limit      string    `json:"limit,omitempty"`
	Shape      fit.Shape `json:"shape"`
	EstVRAMMiB int       `json:"est_vram_mib"`
	EstRAMMiB  int       `json:"est_ram_mib"`
	// EstDecodeTPS is the expected decode speed per stream in tokens/s on this machine (0 = no
	// measured curve for this model).
	EstDecodeTPS float64 `json:"est_decode_tps"`
	// DiskNeedBytes is the free disk setting up the model still needs from the current state: 0
	// when installed, else DiskNeed.Total (filled by the supervisor). DiskNeed says what needs it.
	DiskNeedBytes int64     `json:"disk_need_bytes"`
	DiskNeed      *DiskNeed `json:"disk_need,omitempty"`
	// Estimated is true when the numbers rest on a guess rather than a measurement: a CPU-only
	// run (no CPU runs were ever measured) or a model with no measured footprint.
	Estimated  bool `json:"estimated"`
	Installed  bool `json:"installed"`
	ImportOnly bool `json:"import_only"` // the catalog has no url: only `import` can install it
	// Fallback is true for a model kept only as a last resort; FallbackNote says what to expect.
	Fallback     bool   `json:"fallback"`
	FallbackNote string `json:"fallback_note,omitempty"`
}

type PlanResult struct {
	Recommended string    `json:"recommended"`
	Backend     string    `json:"backend"`
	Models      []Verdict `json:"models"`
	// Custom is the verdict on the user's own GGUF file, or nil when none is set.
	Custom *CustomCheck `json:"custom"`
	// DiskFreeBytes is the free space at the data dir; -1 when unknown (filled by the supervisor).
	DiskFreeBytes int64 `json:"disk_free_bytes"`
	// GPU is the adapters, the processes on them and the reserve the verdicts were made around
	// (filled by the supervisor).
	GPU *gpusense.GPUReport `json:"gpu,omitempty"`
	// Device is how llama-server is (or would be) held to the planned adapter (PinDevice); the
	// supervisor replaces the prediction with the installed build's, or the running server's.
	Device *DevicePin `json:"device,omitempty"`
}

// Limit codes: what stops a model that does not fit (Verdict.Limit, CustomCheck.Limit).
const (
	LimitVRAM   = "vram"              // the card: too little graphics memory free
	LimitRAM    = "ram"               // system RAM: a CPU-only run, MoE expert layers moved off the card, or a GPU that borrows system memory
	LimitDisk   = "disk"              // too little free disk for the download
	LimitDense  = "dense_never_split" // a dense model too big for the card; dense models are never split between card and RAM
	LimitKVType = "kv_type"           // the cache type asked is unknown
	LimitModel  = "model"             // the model itself: its header lacks what the planner needs
	LimitBlock  = "vram_block"        // the card has the memory, but not in one piece large enough
)

// RoomFor derives the planning room from the probed hardware alone: the GPU with the most
// memory, all processes counted as others (nothing of ours is running at plan time), the host's
// room the given estimate. It is RoomForSensed over an empty picture; callers that sense the
// adapters use MeasureReserve and RoomForSensed.
func RoomFor(hw probe.Hardware, gameReserve int) fit.Room {
	sum, _ := MeasureReserve(gpusense.Picture{}, hw, HostInfo{}, gameReserve)
	return RoomForSensed(hw, sum, "vulkan")
}

// Plan answers, for every catalog model, whether it fits hw and in what shape, and recommends
// the highest-quality installable model that fits comfortably ("yes"), else the best "tight" one.
func Plan(c *catalog.Catalog, hw probe.Hardware, w fit.Wants, gameReserve int, installed func(catalog.Model) bool) PlanResult {
	return PlanWith(c, hw, w, gameReserve, installed, nil)
}

// PlanSensed is Plan over a measured reserve (MeasureReserve) in place of the fixed one.
func PlanSensed(c *catalog.Catalog, hw probe.Hardware, w fit.Wants, sum gpusense.GPUSummary, installed func(catalog.Model) bool) PlanResult {
	return PlanSensedWith(c, hw, w, sum, installed, nil)
}

// PlanWith is Plan with a header source: for models whose file is on disk, info returns the
// GGUF Info so the footprint is exact instead of size-guessed.
func PlanWith(c *catalog.Catalog, hw probe.Hardware, w fit.Wants, gameReserve int, installed func(catalog.Model) bool, info func(catalog.Model) (gguf.Info, bool)) PlanResult {
	sum, _ := MeasureReserve(gpusense.Picture{}, hw, HostInfo{}, gameReserve)
	return PlanSensedWith(c, hw, w, sum, installed, info)
}

// PlanSensedWith is PlanWith over a measured reserve.
func PlanSensedWith(c *catalog.Catalog, hw probe.Hardware, w fit.Wants, sum gpusense.GPUSummary, installed func(catalog.Model) bool, info func(catalog.Model) (gguf.Info, bool)) PlanResult {
	return PlanSensedDisk(c, hw, w, sum, installed, info, nil)
}

// PlanSensedDisk is PlanSensedWith with the disk the whole setup of each model needs from the
// current state (the installer's own count, run.Supervisor.setupNeed); nil counts the model's
// download alone. hw.DiskFreeMiB is the free disk (0 = unknown: no disk verdict).
func PlanSensedDisk(c *catalog.Catalog, hw probe.Hardware, w fit.Wants, sum gpusense.GPUSummary, installed func(catalog.Model) bool, info func(catalog.Model) (gguf.Info, bool), disk func(catalog.Model) DiskNeed) PlanResult {
	w = w.Norm()
	var res PlanResult
	if bs := c.Backends(hw); len(bs) > 0 {
		res.Backend = bs[0].ID
	}
	room := RoomForSensed(hw, sum, res.Backend)
	pin := PinDevice(sum, AccelOf(res.Backend), nil)
	res.Device = &pin
	bestYes, bestTight := [2]int{-1, -1}, [2]int{-1, -1} // [0] the models, [1] the last resorts
	for _, m := range c.Models {
		v := Verdict{ID: m.ID, Name: m.Name, Codename: m.Codename, CodenameFull: m.CodenameFull, OptIn: m.OptIn, Size: m.Size, ImportOnly: m.URL == "", Fallback: m.Fallback, FallbackNote: m.FallbackNote}
		if installed != nil {
			v.Installed = installed(m)
		}
		var fp fit.Footprint
		if info != nil && v.Installed {
			if in, ok := info(m); ok {
				// An unknown cache type leaves fp empty: Choose below says so.
				fp, _ = Seed(in, w.KVType)
				if m.Footprint != nil { // the header gives the sizes; the measurement gives the speed
					mf, _ := FromMeasured(m.Footprint, w.KVType)
					fp.TPS = mf.TPS
				}
			}
		}
		if fp.FixedMiB == 0 {
			fp, _ = SeedFromSize(m, w.KVType)
		}
		rm := room
		rm.SpeedScale = SpeedScaleFor(room.GPUName, m.Footprint)
		sh, err := fit.Choose(fp, rm, w)
		v.Estimated = room.NoGPU || (m.Footprint == nil && fp.Estimated)
		var need DiskNeed
		switch {
		case disk != nil:
			need = disk(m)
		case !v.Installed:
			need = ModelOnlyNeed(m)
		}
		switch {
		case err != nil:
			v.Fits = "no"
			v.Limit, v.Why = whyNot(fp, room, w, err)
		case need.Downloads() && hw.DiskFreeMiB > 0 && need.Short(int64(hw.DiskFreeMiB)<<20):
			v.Fits, v.Shape, v.Limit = "no", sh, LimitDisk
			free := int64(hw.DiskFreeMiB) << 20
			v.Why = fmt.Sprintf("not enough free disk: %s, %s free (%s short)", need.Explain(), GB(free), GB(need.Total-free))
		default:
			v.Shape, v.EstVRAMMiB, v.EstRAMMiB, v.EstDecodeTPS = sh, sh.VRAMMiB, sh.RAMMiB, sh.EstTPS
			v.Fits, v.Why = verdictFor(fp, sh, w)
		}
		res.Models = append(res.Models, v)
		if v.Fits == "no" || (!v.Installed && v.ImportOnly) || v.OptIn {
			continue // cannot be recommended: no fit, not installable from here, or a person's pick
		}
		k := 0 // a fallback model is a candidate only when nothing else is (lastResort)
		if lastResort(v) {
			k = 1
		}
		if v.Fits == "yes" && (bestYes[k] < 0 || m.Quality > c.Models[bestYes[k]].Quality) {
			bestYes[k] = len(res.Models) - 1
		}
		if v.Fits == "tight" && (bestTight[k] < 0 || m.Quality > c.Models[bestTight[k]].Quality) {
			bestTight[k] = len(res.Models) - 1
		}
	}
	switch {
	case room.NoGPU:
		res.Recommended = cpuOnlyPick(res.Models, room)
	case bestYes[0] >= 0:
		res.Recommended = res.Models[bestYes[0]].ID
	case bestTight[0] >= 0:
		res.Recommended = res.Models[bestTight[0]].ID
	case bestYes[1] >= 0:
		res.Recommended = res.Models[bestYes[1]].ID
	case bestTight[1] >= 0:
		res.Recommended = res.Models[bestTight[1]].ID
	}
	return res
}

// lastResort says whether a verdict may be recommended only when no other is a candidate: a
// fallback model (today the E2B) is offered where nothing else can run, and never ahead of a
// model that can, even one that fits tight or runs on the processor.
func lastResort(v Verdict) bool { return v.Fallback }

// cpuComfort is the share of free RAM (after the OS reserve) a CPU-only model may take and still
// count as comfortable: the rest is for the game and the OS caches.
const cpuComfort = 0.6

// cpuOnlyPick is the recommendation without a usable GPU. Nothing was ever measured on a CPU, so
// the choice is conservative: the smallest installable model that fits in RAM, preferring one
// whose need stays within cpuComfort of the free RAM, not the best-quality one. A last resort (a
// fallback model) is picked only when no other model fits at all, comfortable or not. The
// verdicts are marked Estimated.
func cpuOnlyPick(vs []Verdict, room fit.Room) string {
	_, free, _ := room.Budget() // RoomFor's room: every field read
	for _, last := range []bool{false, true} {
		best, bestComfy := -1, -1
		for i, v := range vs {
			if v.Fits == "no" || (!v.Installed && v.ImportOnly) || v.OptIn || lastResort(v) != last {
				continue
			}
			if best < 0 || v.Size < vs[best].Size {
				best = i
			}
			if float64(v.EstRAMMiB) <= cpuComfort*free && (bestComfy < 0 || v.Size < vs[bestComfy].Size) {
				bestComfy = i
			}
		}
		if bestComfy >= 0 {
			return vs[bestComfy].ID
		}
		if best >= 0 {
			return vs[best].ID
		}
	}
	return ""
}

func verdictFor(fp fit.Footprint, s fit.Shape, w fit.Wants) (string, string) {
	desc := fmt.Sprintf("%d slots x %d ctx", s.Slots, s.CtxPerSlot)
	switch {
	case s.CPUOnly:
		return "tight", desc + ", CPU only (no usable GPU; estimate: no CPU speed was measured, expect it slow)"
	case s.EstTPS > 0 && w.MinTPS > 0 && s.EstTPS < w.MinTPS:
		return "tight", fmt.Sprintf("%s, %d of %d expert layers in RAM: about %.0f tok/s, under the %.0f floor (slow)", desc, s.NCpuMoe, fp.ExpertLayers, s.EstTPS, w.MinTPS)
	case s.Slots < w.Slots:
		return "tight", fmt.Sprintf("only %d of %d slots, %d ctx each", s.Slots, w.Slots, s.CtxPerSlot)
	case s.CtxPerSlot < 2*w.CtxFloor:
		return "tight", desc + " (close to the floor)"
	case fp.MoE && s.NCpuMoe*2 > fp.ExpertLayers:
		return "tight", fmt.Sprintf("%s, %d of %d expert layers in RAM (slower)", desc, s.NCpuMoe, fp.ExpertLayers)
	case s.NCpuMoe > 0:
		return "yes", fmt.Sprintf("%s, %d of %d expert layers in RAM", desc, s.NCpuMoe, fp.ExpertLayers)
	}
	return "yes", desc + ", all on the GPU"
}

// whyNot says what stops a model Choose refused with err: the limit code and the words. Only
// fit.ErrTooBig is a matter of room; the core's other refusals (an unknown cache type, a header
// without what the planner needs) are passed on as they are. For a model too big, the least shape
// (minimum slots at the floor ctx) is priced: a dense model, or one kept whole, needs it all on
// the card; an MoE model needs the card to hold at least its non-expert part, and then system
// RAM for every expert layer the card cannot hold (the fewest moved that the card takes).
func whyNot(fp fit.Footprint, room fit.Room, w fit.Wants, err error) (string, string) {
	if errors.As(err, new(*fit.KVTypeError)) {
		return LimitKVType, err.Error()
	}
	if !errors.Is(err, fit.ErrTooBig) {
		return LimitModel, err.Error()
	}
	vram, ram, err := room.Budget()
	if err != nil {
		return LimitModel, err.Error()
	}
	least := fit.Shape{Slots: w.MinSlots, CtxPerSlot: w.CtxFloor, KVType: w.KVType, CPUOnly: room.NoGPU}
	need := func(n int) (float64, float64, error) {
		s := least
		s.NCpuMoe = n
		return fp.Need(s)
	}
	cardNeed, hostNeed, err := need(0)
	if err != nil {
		return LimitModel, err.Error()
	}
	floor := fmt.Sprintf("%d slots x %d ctx", least.Slots, least.CtxPerSlot)
	if least.Slots == 1 {
		floor = fmt.Sprintf("1 slot x %d ctx", least.CtxPerSlot)
	}
	vram, ram = max(vram, 0), max(ram, 0)
	switch {
	case room.NoGPU:
		why := "no usable GPU"
		switch {
		case room.VRAMTotal > 0 && room.Unified:
			why = "the integrated GPU is used only by a Vulkan build"
		case room.VRAMTotal > 0:
			why = fmt.Sprintf("the card has under %d MiB left after the game's room, too little to use", minUsableVRAM)
		}
		return LimitRAM, fmt.Sprintf("needs about %d MiB of system RAM to run on the CPU at %s, %d MiB available after the OS reserve (%s)", ceilMiB(hostNeed), floor, int(ram), why)
	case room.Unified:
		return LimitRAM, fmt.Sprintf("needs about %d MiB of the system memory the GPU shares, %d MiB of it free after the game's room and the OS", ceilMiB(cardNeed), int(vram))
	case !fp.MoE:
		if cardNeed > vram {
			return LimitDense, fmt.Sprintf("needs about %d MiB of VRAM, %d MiB free after the game's room; dense models are never split between card and RAM", ceilMiB(cardNeed), int(vram))
		}
	case w.Whole:
		if cardNeed > vram {
			return LimitVRAM, fmt.Sprintf("needs about %d MiB of VRAM kept whole on the card, %d MiB free after the game's room", ceilMiB(cardNeed), int(vram))
		}
	default: // MoE: the fewest expert layers moved that the card can take, and the RAM they need
		moved := -1
		for n := 0; n <= fp.ExpertLayers; n++ {
			c, h, err := need(n)
			if err != nil {
				return LimitModel, err.Error()
			}
			if c <= vram {
				moved, cardNeed, hostNeed = n, c, h
				break
			}
			cardNeed = c
		}
		if moved < 0 {
			return LimitVRAM, fmt.Sprintf("needs about %d MiB of VRAM even with every expert layer in system RAM, %d MiB free after the game's room", ceilMiB(cardNeed), int(vram))
		}
		if hostNeed > ram {
			return LimitRAM, fmt.Sprintf("needs about %d MiB of system RAM for the %d of %d expert layers the card cannot hold, %d MiB available after the OS reserve (the card has room for the rest)", ceilMiB(hostNeed), moved, fp.ExpertLayers, int(ram))
		}
	}
	if room.LargestBlock > 0 {
		return LimitBlock, fmt.Sprintf("the card has the memory but not in one piece: its largest free block is %d MiB", room.LargestBlock)
	}
	return LimitVRAM, fmt.Sprintf("no shape fits the card at %s (%d MiB of VRAM free after the game's room)", floor, int(vram))
}

func ceilMiB(f float64) int { return int(math.Ceil(f)) }
