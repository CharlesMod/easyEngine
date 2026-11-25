// STUB: replaced by eng/fetch
package fit

import (
	"errors"
	"fmt"
	"strconv"
	"time"

	"hermesengine/catalog"
	"hermesengine/gguf"
	"hermesengine/probe"
)

type Wants struct {
	Slots, MinSlots, CtxMax, CtxFloor int
	KVType                            string
}
type Room struct {
	VRAMTotal, Others, GameReserve, Margin, RAMAvail int
	NoGPU, Unified                                   bool
}
type Footprint struct {
	FixedMiB, PerTokenMiB, ScratchMiB, ExpertMiBPerLayer float64
	ExpertLayers                                         int
	MoE                                                  bool
}
type Shape struct {
	Slots      int    `json:"slots"`
	CtxPerSlot int    `json:"ctx_per_slot"`
	NCpuMoe    int    `json:"n_cpu_moe"`
	KVType     string `json:"kv_type"`
	VRAMMiB    int    `json:"vram_mib"`
	RAMMiB     int    `json:"ram_mib"`
	CPUOnly    bool   `json:"cpu_only"`
}

var ErrTooBig = errors.New("model too big for this machine")

func Seed(info gguf.Info, kv string) Footprint {
	return Footprint{FixedMiB: float64(info.FileSize) / (1 << 20), PerTokenMiB: 0.05, ScratchMiB: 400}
}
func SeedFromSize(m catalog.Model, kv string) Footprint {
	return Footprint{FixedMiB: float64(m.Size) / (1 << 20), PerTokenMiB: 0.06, ScratchMiB: 500, MoE: m.MoE}
}
func DefaultMargin(total int) int {
	if m := total * 4 / 100; m > 512 {
		return m
	}
	return 512
}

func vram(fp Footprint, s Shape) float64 {
	return fp.FixedMiB + fp.PerTokenMiB*float64(s.Slots*s.CtxPerSlot) + fp.ScratchMiB - fp.ExpertMiBPerLayer*float64(s.NCpuMoe)
}
func budget(r Room) int { return r.VRAMTotal - r.Others - r.GameReserve - r.Margin }

func try(fp Footprint, r Room, s Shape) (Shape, bool) {
	if r.NoGPU {
		s.CPUOnly = true
		return s, true
	}
	v := vram(fp, s)
	if v <= float64(budget(r)) {
		s.VRAMMiB = int(v)
		return s, true
	}
	return s, false
}

func Choose(fp Footprint, r Room, w Wants) (Shape, error) {
	for slots := w.Slots; slots >= w.MinSlots && slots >= 1; slots-- {
		maxN := 0
		if fp.MoE {
			maxN = fp.ExpertLayers
		}
		for n := 0; n <= maxN; n++ {
			for ctx := w.CtxMax; ctx >= w.CtxFloor; ctx -= 1024 {
				if s, ok := try(fp, r, Shape{Slots: slots, CtxPerSlot: ctx, NCpuMoe: n, KVType: w.KVType}); ok {
					return s, nil
				}
			}
		}
	}
	return Shape{}, ErrTooBig
}

func MissRetry(fp Footprint, cur Shape, failedBufMiB int, r Room, w Wants) (Shape, error) {
	if fp.PerTokenMiB <= 0 {
		return StepDown(fp, cur, r, w)
	}
	drop := int(float64(failedBufMiB)/fp.PerTokenMiB/float64(cur.Slots))/1024*1024 + 1024
	n := cur
	n.CtxPerSlot -= drop
	if n.CtxPerSlot < w.CtxFloor {
		return StepDown(fp, cur, r, w)
	}
	return n, nil
}

func StepDown(fp Footprint, cur Shape, r Room, w Wants) (Shape, error) {
	n := cur
	n.CtxPerSlot -= 8192
	if n.CtxPerSlot >= w.CtxFloor {
		return n, nil
	}
	n = cur
	n.CtxPerSlot = w.CtxFloor
	if fp.MoE && n.NCpuMoe < fp.ExpertLayers {
		n.NCpuMoe++
		return n, nil
	}
	if cur.Slots > w.MinSlots && cur.Slots > 1 {
		n = cur
		n.Slots--
		return n, nil
	}
	return cur, ErrTooBig
}

func Grow(fp Footprint, cur Shape, r Room, w Wants) (Shape, bool) {
	n := cur
	n.CtxPerSlot += 8192
	if n.CtxPerSlot > w.CtxMax {
		return cur, false
	}
	if s, ok := try(fp, r, n); ok {
		return s, true
	}
	return cur, false
}

func Args(s Shape) []string {
	ngl := "999"
	if s.CPUOnly {
		ngl = "0"
	}
	a := []string{"-ngl", ngl}
	if s.NCpuMoe > 0 {
		a = append(a, "--n-cpu-moe", strconv.Itoa(s.NCpuMoe))
	}
	return append(a, "-np", strconv.Itoa(s.Slots), "-c", strconv.Itoa(s.Slots*s.CtxPerSlot), "-ctk", s.KVType, "-ctv", s.KVType)
}

type Calib struct {
	FixedMiB    float64
	CapCtxTotal int
	Loads       int
	Updated     time.Time
}
type Book struct{ Entries map[string]Calib }

func BookKey(sha, gpu, backend string) string { return sha + "|" + gpu + "|" + backend }
func (b *Book) Footprint(key string, seed Footprint) Footprint {
	if c, ok := b.Entries[key]; ok && c.FixedMiB > 0 {
		seed.FixedMiB = c.FixedMiB
	}
	return seed
}
func (b *Book) Calibrate(key string, seed Footprint, s Shape, measured int) bool {
	fixed := float64(measured) - seed.PerTokenMiB*float64(s.Slots*s.CtxPerSlot) - seed.ScratchMiB + seed.ExpertMiBPerLayer*float64(s.NCpuMoe)
	if fixed < seed.FixedMiB*0.85 || fixed > seed.FixedMiB*1.15 {
		return false
	}
	if b.Entries == nil {
		b.Entries = map[string]Calib{}
	}
	c := b.Entries[key]
	c.FixedMiB, c.Loads, c.Updated = fixed, c.Loads+1, time.Now()
	b.Entries[key] = c
	return true
}
func (b *Book) LowerCap(key string, ctxTotal int) {
	if b.Entries == nil {
		b.Entries = map[string]Calib{}
	}
	c := b.Entries[key]
	if c.CapCtxTotal == 0 || ctxTotal < c.CapCtxTotal {
		c.CapCtxTotal = ctxTotal
	}
	c.Updated = time.Now()
	b.Entries[key] = c
}

type Verdict struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Size       int64  `json:"size"`
	Fits       string `json:"fits"`
	Why        string `json:"why"`
	Shape      Shape  `json:"shape"`
	EstVRAMMiB int    `json:"est_vram_mib"`
	EstRAMMiB  int    `json:"est_ram_mib"`
	Installed  bool   `json:"installed"`
	ImportOnly bool   `json:"import_only"`
}
type PlanResult struct {
	Recommended string    `json:"recommended"`
	Backend     string    `json:"backend"`
	Models      []Verdict `json:"models"`
}

func Plan(c *catalog.Catalog, hw probe.Hardware, w Wants, gameReserve int, installed func(catalog.Model) bool) PlanResult {
	room := Room{RAMAvail: hw.RAMAvailMiB, GameReserve: gameReserve, NoGPU: len(hw.GPUs) == 0}
	if len(hw.GPUs) > 0 {
		room.VRAMTotal = hw.GPUs[0].VRAMTotalMiB
		room.Others = hw.GPUs[0].VRAMUsedMiB
		room.Margin = DefaultMargin(room.VRAMTotal)
	}
	var pr PlanResult
	if bs := c.Backends(hw); len(bs) > 0 {
		pr.Backend = bs[0].ID
	}
	for _, m := range c.Models {
		v := Verdict{ID: m.ID, Name: m.Name, Size: m.Size, Installed: installed != nil && installed(m), ImportOnly: m.URL == ""}
		s, err := Choose(SeedFromSize(m, w.KVType), room, w)
		if err != nil {
			v.Fits, v.Why = "no", err.Error()
		} else {
			v.Fits, v.Shape, v.EstVRAMMiB = "yes", s, s.VRAMMiB
			v.Why = fmt.Sprintf("%d x %d", s.Slots, s.CtxPerSlot)
			if pr.Recommended == "" {
				pr.Recommended = m.ID
			}
		}
		pr.Models = append(pr.Models, v)
	}
	return pr
}
