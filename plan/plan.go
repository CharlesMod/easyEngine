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

	"hermesengine/catalog"
	"hermesengine/probe"
	"hermesengine/probe/gpusense"
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
	ID         string    `json:"id"`
	Name       string    `json:"name"`
	Size       int64     `json:"size"`
	Fits       string    `json:"fits"` // yes | tight | no
	Why        string    `json:"why"`
	Shape      fit.Shape `json:"shape"`
	EstVRAMMiB int       `json:"est_vram_mib"`
	EstRAMMiB  int       `json:"est_ram_mib"`
	// EstDecodeTPS is the expected decode speed per stream in tokens/s on this machine (0 = no
	// measured curve for this model).
	EstDecodeTPS float64 `json:"est_decode_tps"`
	// DiskNeedBytes is the free disk the model still needs: 0 when installed, else its size minus a
	// partial download plus the headroom the engine keeps (filled by the supervisor).
	DiskNeedBytes int64 `json:"disk_need_bytes"`
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
}

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
	w = w.Norm()
	var res PlanResult
	if bs := c.Backends(hw); len(bs) > 0 {
		res.Backend = bs[0].ID
	}
	room := RoomForSensed(hw, sum, res.Backend)
	bestYes, bestTight := -1, -1
	for _, m := range c.Models {
		v := Verdict{ID: m.ID, Name: m.Name, Size: m.Size, ImportOnly: m.URL == "", Fallback: m.Fallback, FallbackNote: m.FallbackNote}
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
		switch {
		case errors.As(err, new(*fit.KVTypeError)):
			v.Fits, v.Why = "no", err.Error()
		case err != nil:
			v.Fits, v.Why = "no", whyNot(fp, room, w)
		case !v.Installed && hw.DiskFreeMiB > 0 && int(m.Size/mib)+1024 > hw.DiskFreeMiB:
			v.Fits, v.Shape = "no", sh
			v.Why = fmt.Sprintf("needs %d MiB of free disk, %d MiB available", m.Size/mib+1024, hw.DiskFreeMiB)
		default:
			v.Shape, v.EstVRAMMiB, v.EstRAMMiB, v.EstDecodeTPS = sh, sh.VRAMMiB, sh.RAMMiB, sh.EstTPS
			v.Fits, v.Why = verdictFor(fp, sh, w)
		}
		res.Models = append(res.Models, v)
		if v.Fits == "no" || (!v.Installed && v.ImportOnly) {
			continue // cannot be recommended: no fit, or not installable from here
		}
		if v.Fits == "yes" && (bestYes < 0 || m.Quality > c.Models[bestYes].Quality) {
			bestYes = len(res.Models) - 1
		}
		if v.Fits == "tight" && (bestTight < 0 || m.Quality > c.Models[bestTight].Quality) {
			bestTight = len(res.Models) - 1
		}
	}
	switch {
	case room.NoGPU:
		res.Recommended = cpuOnlyPick(res.Models, room)
	case bestYes >= 0:
		res.Recommended = res.Models[bestYes].ID
	case bestTight >= 0:
		res.Recommended = res.Models[bestTight].ID
	}
	return res
}

// cpuComfort is the share of free RAM (after the OS reserve) a CPU-only model may take and still
// count as comfortable: the rest is for the game and the OS caches.
const cpuComfort = 0.6

// cpuOnlyPick is the recommendation without a usable GPU. Nothing was ever measured on a CPU, so
// the choice is conservative: the smallest installable model that fits in RAM, preferring one
// whose need stays within cpuComfort of the free RAM, not the best-quality one. The verdicts are
// marked Estimated.
func cpuOnlyPick(vs []Verdict, room fit.Room) string {
	_, free, _ := room.Budget() // RoomFor's room: every field read
	best, bestComfy := -1, -1
	for i, v := range vs {
		if v.Fits == "no" || (!v.Installed && v.ImportOnly) {
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

func whyNot(fp fit.Footprint, room fit.Room, w fit.Wants) string {
	vram, ram, err := room.Budget()
	if err != nil {
		return err.Error()
	}
	least := fit.Shape{Slots: w.MinSlots, CtxPerSlot: w.CtxFloor, KVType: w.KVType, CPUOnly: room.NoGPU}
	if fp.MoE && !room.Unified && !room.NoGPU {
		least.NCpuMoe = fp.ExpertLayers
	}
	cardNeed, hostNeed, err := fp.Need(least)
	if err != nil {
		return err.Error()
	}
	if room.NoGPU {
		return fmt.Sprintf("needs about %d MiB of RAM, %d MiB available after the OS reserve", ceilMiB(hostNeed), int(ram))
	}
	need, have := cardNeed, int(vram)
	kind := "dense models cannot be split between card and RAM"
	if fp.MoE {
		kind = "even with every expert layer in RAM"
	}
	return fmt.Sprintf("needs about %d MiB of VRAM, %d MiB free after the game's room (%s)", ceilMiB(need), have, kind)
}

func ceilMiB(f float64) int { return int(math.Ceil(f)) }
