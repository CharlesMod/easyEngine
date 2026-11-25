package fit

import (
	"fmt"

	"hermesengine/catalog"
	"hermesengine/gguf"
	"hermesengine/probe"
)

// SeedFromSize is the footprint of a model not yet downloaded: the catalog size and a guess
// at the KV cost scaled by size (about 0.0006 MiB per token per GiB of weights at q4_0, with a
// 25 % safety factor). Fit is re-checked from the real header after download.
func SeedFromSize(m catalog.Model, kvType string) Footprint {
	if kvType == "" {
		kvType = "q4_0"
	}
	sizeMiB := float64(m.Size) / mib
	gib := sizeMiB / 1024
	fp := Footprint{
		FixedMiB:    sizeMiB + cudaCtxMiB,
		PerTokenMiB: 0.0006 * gib * 1.25 * kvBytes(kvType) / kvBytes("q4_0"),
		ScratchMiB:  scratchSeed,
	}
	if m.MoE {
		const layers = 30 // typical; the header gives the real count after download
		fp.MoE = true
		fp.ExpertLayers = layers
		fp.ExpertMiBPerLayer = 0.75 * sizeMiB / layers
	}
	return fp
}

// Verdict is one catalog model's answer for this machine.
type Verdict struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Size       int64  `json:"size"`
	Fits       string `json:"fits"` // yes | tight | no
	Why        string `json:"why"`
	Shape      Shape  `json:"shape"`
	EstVRAMMiB int    `json:"est_vram_mib"`
	EstRAMMiB  int    `json:"est_ram_mib"`
	Installed  bool   `json:"installed"`
	ImportOnly bool   `json:"import_only"` // the catalog has no url: only `import` can install it
}

type PlanResult struct {
	Recommended string    `json:"recommended"`
	Backend     string    `json:"backend"`
	Models      []Verdict `json:"models"`
}

// RoomFor derives the planning room from the probed hardware: the GPU with the most memory,
// all processes counted as others (nothing of ours is running at plan time).
func RoomFor(hw probe.Hardware, gameReserve int) Room {
	r := Room{GameReserve: gameReserve, RAMAvail: hw.RAMAvailMiB}
	best := -1
	for i, g := range hw.GPUs {
		if g.VRAMTotalMiB > 0 && (best < 0 || g.VRAMTotalMiB > hw.GPUs[best].VRAMTotalMiB) {
			best = i
		}
	}
	if best < 0 {
		r.NoGPU = true
		return r
	}
	g := hw.GPUs[best]
	r.VRAMTotal = g.VRAMTotalMiB
	r.Margin = DefaultMargin(g.VRAMTotalMiB)
	if g.UnifiedMemory {
		r.Unified = true
		if o := g.VRAMTotalMiB - hw.RAMAvailMiB; o > 0 { // what the rest of the system already holds
			r.Others = o
		}
	} else {
		r.Others = g.VRAMUsedMiB
	}
	if int(budgetOf(r)) < minUsableVRAM {
		r.NoGPU = true // too little card to be worth using next to the game
	}
	return r
}

// Plan answers, for every catalog model, whether it fits hw and in what shape, and recommends
// the highest-quality installable model that fits comfortably ("yes"), else the best "tight" one.
func Plan(c *catalog.Catalog, hw probe.Hardware, w Wants, gameReserve int, installed func(catalog.Model) bool) PlanResult {
	return PlanWith(c, hw, w, gameReserve, installed, nil)
}

// PlanWith is Plan with a header source: for models whose file is on disk, info returns the
// GGUF Info so the footprint is exact instead of size-guessed.
func PlanWith(c *catalog.Catalog, hw probe.Hardware, w Wants, gameReserve int, installed func(catalog.Model) bool, info func(catalog.Model) (gguf.Info, bool)) PlanResult {
	w = w.norm()
	room := RoomFor(hw, gameReserve)
	var res PlanResult
	if bs := c.Backends(hw); len(bs) > 0 {
		res.Backend = bs[0].ID
	}
	bestYes, bestTight := -1, -1
	for _, m := range c.Models {
		v := Verdict{ID: m.ID, Name: m.Name, Size: m.Size, ImportOnly: m.URL == ""}
		if installed != nil {
			v.Installed = installed(m)
		}
		var fp Footprint
		if info != nil && v.Installed {
			if in, ok := info(m); ok {
				fp = Seed(in, w.KVType)
			}
		}
		if fp.FixedMiB == 0 {
			fp = SeedFromSize(m, w.KVType)
		}
		sh, err := Choose(fp, room, w)
		switch {
		case err != nil:
			v.Fits, v.Why = "no", whyNot(fp, room, w)
		case !v.Installed && hw.DiskFreeMiB > 0 && int(m.Size/mib)+1024 > hw.DiskFreeMiB:
			v.Fits, v.Shape = "no", sh
			v.Why = fmt.Sprintf("needs %d MiB of free disk, %d MiB available", m.Size/mib+1024, hw.DiskFreeMiB)
		default:
			v.Shape, v.EstVRAMMiB, v.EstRAMMiB = sh, sh.VRAMMiB, sh.RAMMiB
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
	case bestYes >= 0:
		res.Recommended = res.Models[bestYes].ID
	case bestTight >= 0:
		res.Recommended = res.Models[bestTight].ID
	}
	return res
}

func verdictFor(fp Footprint, s Shape, w Wants) (string, string) {
	desc := fmt.Sprintf("%d slots x %d ctx", s.Slots, s.CtxPerSlot)
	switch {
	case s.CPUOnly:
		return "tight", desc + ", CPU only (no usable GPU; slow)"
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

func whyNot(fp Footprint, room Room, w Wants) string {
	if room.NoGPU {
		need := fp.allHost(w.MinSlots, w.CtxFloor)
		return fmt.Sprintf("needs about %d MiB of RAM, %d MiB available after the OS reserve", ceilMiB(need), room.RAMAvail-osReserveMiB)
	}
	need := fp.vram(w.MinSlots, w.CtxFloor, 0)
	if fp.MoE && !room.Unified {
		need = fp.vram(w.MinSlots, w.CtxFloor, fp.ExpertLayers)
	}
	have := int(budgetOf(room))
	kind := "dense models cannot be split between card and RAM"
	if fp.MoE {
		kind = "even with every expert layer in RAM"
	}
	return fmt.Sprintf("needs about %d MiB of VRAM, %d MiB free after the game's room (%s)", ceilMiB(need), have, kind)
}
