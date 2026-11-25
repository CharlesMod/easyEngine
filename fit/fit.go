// Package fit is the ctx balancer's planner: pure functions that turn a model footprint, the
// room on the machine and what the game wants into an explicit llama-server shape. It is a
// generic port of the hive's measure-then-try launcher (hive-mind-llm-run): the machine
// constants became numbers read from the GGUF header plus a calibration book.
package fit

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"hermesengine/gguf"
)

// ErrTooBig means no shape fits the room, even at the floor context, minimum slots and every
// movable expert layer in RAM. Dense models are never split between card and RAM.
var ErrTooBig = errors.New("fit: model does not fit this machine")

const (
	alignTokens   = 1024
	osReserveMiB  = 2048 // host RAM kept for the OS when moving experts to RAM
	cudaCtxMiB    = 300  // CUDA context and runtime, seeded; calibrated after the first load
	scratchSeed   = 1100 // compute buffers at -ub 512, seeded from the reference build's measured footprint
	runtimeMiB    = 40   // scratch a request takes beyond what load allocates (hive RUNTIME_MIB)
	minUsableVRAM = 2048 // a GPU whose budget is below this is treated as no GPU by Plan
)

// Wants is what the game asks for.
type Wants struct {
	Slots    int    `json:"slots"`
	MinSlots int    `json:"min_slots"`
	CtxMax   int    `json:"ctx_max"` // per slot
	CtxFloor int    `json:"ctx_floor"`
	KVType   string `json:"kv_type"`
}

// Room is what the machine offers, all in MiB.
type Room struct {
	VRAMTotal   int  `json:"vram_total"`
	Others      int  `json:"others"` // VRAM held by processes that are not ours
	GameReserve int  `json:"game_reserve"`
	Margin      int  `json:"margin"`
	RAMAvail    int  `json:"ram_avail"`
	NoGPU       bool `json:"no_gpu"`  // no usable GPU: CPU-only runs are allowed
	Unified     bool `json:"unified"` // Apple: VRAM is RAM, no expert offload
}

// Footprint models the memory of (model, device, backend):
//
//	vram = Fixed + PerToken*ctxTotal + SWAPerToken*min(ctxPerSlot, SWAWindow)*slots + Scratch - ExpertPerLayer*nCpuMoe
type Footprint struct {
	FixedMiB          float64 `json:"fixed_mib"`
	PerTokenMiB       float64 `json:"per_token_mib"` // full-attention KV per token of total context
	ScratchMiB        float64 `json:"scratch_mib"`
	ExpertMiBPerLayer float64 `json:"expert_mib_per_layer"`
	ExpertLayers      int     `json:"expert_layers"`
	MoE               bool    `json:"moe"`

	SWAPerTokenMiB float64 `json:"swa_per_token_mib"` // sliding-window KV per cached token
	SWAWindow      int     `json:"swa_window"`        // tokens kept per slot by those layers
	CapCtxTotal    int     `json:"cap_ctx_total"`     // runtime-OOM cap from the book; 0 = none
}

// Shape is a concrete llama-server configuration.
type Shape struct {
	Slots      int    `json:"slots"`
	CtxPerSlot int    `json:"ctx_per_slot"`
	NCpuMoe    int    `json:"n_cpu_moe"`
	KVType     string `json:"kv_type"`
	VRAMMiB    int    `json:"vram_mib"`
	RAMMiB     int    `json:"ram_mib"`
	CPUOnly    bool   `json:"cpu_only"`
}

// CtxTotal is the -c value.
func (s Shape) CtxTotal() int { return s.Slots * s.CtxPerSlot }

func (w Wants) norm() Wants {
	if w.Slots < 1 {
		w.Slots = 1
	}
	if w.MinSlots < 1 {
		w.MinSlots = 1
	}
	if w.MinSlots > w.Slots {
		w.MinSlots = w.Slots
	}
	if w.CtxMax <= 0 {
		w.CtxMax = 65536
	}
	if w.CtxFloor <= 0 {
		w.CtxFloor = 8192
	}
	w.CtxMax = alignDown(w.CtxMax)
	w.CtxFloor = alignDown(w.CtxFloor)
	if w.CtxFloor < alignTokens {
		w.CtxFloor = alignTokens
	}
	if w.CtxFloor > w.CtxMax {
		w.CtxMax = w.CtxFloor
	}
	if w.KVType == "" {
		w.KVType = "q4_0"
	}
	return w
}

func alignDown(n int) int {
	if n < 0 {
		return 0
	}
	return n / alignTokens * alignTokens
}

// kvBytes is bytes per KV element for a llama.cpp cache type.
func kvBytes(t string) float64 {
	switch t {
	case "f32":
		return 4
	case "f16", "bf16":
		return 2
	case "q8_0":
		return 8.5 / 8
	case "q5_1":
		return 6.0 / 8
	case "q5_0":
		return 5.5 / 8
	case "q4_1":
		return 5.0 / 8
	case "q4_0", "iq4_nl":
		return 4.5 / 8
	}
	return 2 // unknown: assume f16
}

// DefaultMargin is max(512 MiB, 4 % of the card).
func DefaultMargin(vramTotal int) int {
	m := vramTotal * 4 / 100
	if m < 512 {
		m = 512
	}
	return m
}

const mib = 1 << 20

// Seed builds a footprint from a GGUF header.
func Seed(info gguf.Info, kvType string) Footprint {
	if kvType == "" {
		kvType = "q4_0"
	}
	b := kvBytes(kvType)
	layers := info.Layers
	if len(layers) != info.NLayer || len(layers) == 0 {
		layers = nil
		for i := 0; i < info.NLayer; i++ {
			l := gguf.Layer{HasKV: true, KVHeads: info.NKVHead, HeadDimK: info.HeadDimK, HeadDimV: info.HeadDimV}
			if i < info.SWALayers && info.SlidingWindow > 0 {
				l.SWA = true
				if info.HeadDimKSWA > 0 {
					l.HeadDimK, l.HeadDimV = info.HeadDimKSWA, info.HeadDimVSWA
				}
			}
			layers = append(layers, l)
		}
	}
	var full, swa float64
	for _, l := range layers {
		if !l.HasKV {
			continue
		}
		e := float64(l.KVHeads * (l.HeadDimK + l.HeadDimV))
		if l.SWA && info.SlidingWindow > 0 {
			swa += e
		} else {
			full += e
		}
	}
	fp := Footprint{
		FixedMiB:       float64(info.DenseBytes+info.ExpertBytes)/mib + cudaCtxMiB,
		PerTokenMiB:    full * b / mib,
		SWAPerTokenMiB: swa * b / mib,
		SWAWindow:      info.SlidingWindow,
		ScratchMiB:     scratchSeed,
	}
	if info.NExpert > 0 && info.ExpertLayers > 0 && info.ExpertBytes > 0 {
		fp.MoE = true
		fp.ExpertLayers = info.ExpertLayers
		fp.ExpertMiBPerLayer = float64(info.ExpertBytes) / mib / float64(info.ExpertLayers)
	}
	return fp
}

// kvAt returns the KV cache MiB for a layout.
func (fp Footprint) kvAt(slots, ctxPerSlot int) float64 {
	kv := fp.PerTokenMiB * float64(slots*ctxPerSlot)
	if fp.SWAPerTokenMiB > 0 {
		w := ctxPerSlot
		if fp.SWAWindow > 0 && fp.SWAWindow < w {
			w = fp.SWAWindow
		}
		kv += fp.SWAPerTokenMiB * float64(w*slots)
	}
	return kv
}

// vram is the GPU use of a layout with nCpuMoe expert layers moved to RAM.
func (fp Footprint) vram(slots, ctxPerSlot, nCpuMoe int) float64 {
	return fp.FixedMiB + fp.kvAt(slots, ctxPerSlot) + fp.ScratchMiB - fp.ExpertMiBPerLayer*float64(nCpuMoe)
}

// allHost is the RAM use of a CPU-only run.
func (fp Footprint) allHost(slots, ctxPerSlot int) float64 {
	return fp.FixedMiB + fp.kvAt(slots, ctxPerSlot) + fp.ScratchMiB
}

func ceilMiB(f float64) int { return int(math.Ceil(f)) }

func (fp Footprint) shape(w Wants, slots, ctx, n int, cpuOnly bool) Shape {
	s := Shape{Slots: slots, CtxPerSlot: ctx, NCpuMoe: n, KVType: w.KVType, CPUOnly: cpuOnly}
	if cpuOnly {
		s.NCpuMoe = 0
		s.RAMMiB = ceilMiB(fp.allHost(slots, ctx))
		return s
	}
	s.VRAMMiB = ceilMiB(fp.vram(slots, ctx, n))
	s.RAMMiB = ceilMiB(fp.ExpertMiBPerLayer * float64(n))
	return s
}

// ctxCeiling is the largest 1024-aligned ctx per slot allowed by wants and the book's cap.
func (fp Footprint) ctxCeiling(w Wants, slots int) int {
	c := w.CtxMax
	if fp.CapCtxTotal > 0 {
		if cc := alignDown(fp.CapCtxTotal / slots); cc < c {
			c = cc
		}
	}
	return c
}

// maxCtx finds the largest aligned ctx per slot in [floor, ceiling] with ok(ctx) true; ok must
// be monotone (true for small ctx).
func maxCtx(floor, ceiling int, ok func(ctx int) bool) (int, bool) {
	if ceiling < floor || !ok(floor) {
		return 0, false
	}
	lo, hi := floor/alignTokens, ceiling/alignTokens // lo ok
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if ok(mid * alignTokens) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo * alignTokens, true
}

func budgetOf(room Room) float64 {
	return float64(room.VRAMTotal - room.Others - room.GameReserve - room.Margin)
}

// Choose picks the shape, in the order of ENGINE.md section 5: all on GPU at the largest ctx;
// ctx shrinks toward the floor; MoE experts move to RAM a layer at a time (then ctx grows back
// as far as it fits); then fewer slots; else ErrTooBig. Dense models are never split between
// card and RAM; with no usable GPU they run CPU-only.
func Choose(fp Footprint, room Room, w Wants) (Shape, error) {
	w = w.norm()
	budget := budgetOf(room)
	ram := float64(room.RAMAvail - osReserveMiB)
	for slots := w.Slots; slots >= w.MinSlots; slots-- {
		ceil := fp.ctxCeiling(w, slots)
		if room.NoGPU {
			ctx, ok := maxCtx(w.CtxFloor, ceil, func(c int) bool { return fp.allHost(slots, c) <= ram })
			if ok {
				return fp.shape(w, slots, ctx, 0, true), nil
			}
			continue
		}
		maxN := 0
		if fp.MoE && !room.Unified {
			maxN = fp.ExpertLayers
		}
		for n := 0; n <= maxN; n++ {
			if fp.ExpertMiBPerLayer*float64(n) > ram {
				break
			}
			ctx, ok := maxCtx(w.CtxFloor, ceil, func(c int) bool { return fp.vram(slots, c, n) <= budget })
			if ok {
				return fp.shape(w, slots, ctx, n, false), nil
			}
		}
	}
	return Shape{}, ErrTooBig
}

// better reports whether a is a larger shape than b: more slots, then fewer CPU-MoE layers,
// then more ctx.
func better(a, b Shape) bool {
	if a.Slots != b.Slots {
		return a.Slots > b.Slots
	}
	if a.NCpuMoe != b.NCpuMoe {
		return a.NCpuMoe < b.NCpuMoe
	}
	return a.CtxPerSlot > b.CtxPerSlot
}

// MissRetry is the hive's miss_retry(): after a load OOM it re-plans from the failed buffer. The
// failed allocation bounds how much ctx must go (its size in tokens plus runtime scratch);
// the room is the freshly measured one. The result is always smaller than cur; when the floor
// is reached it moves on to StepDown.
func MissRetry(fp Footprint, cur Shape, failedBufMiB int, room Room, w Wants) (Shape, error) {
	w = w.norm()
	byBuf := cur.CtxPerSlot
	if failedBufMiB > 0 && fp.PerTokenMiB > 0 && cur.Slots > 0 {
		drop := int(math.Ceil((float64(failedBufMiB) + runtimeMiB) / (fp.PerTokenMiB * float64(cur.Slots))))
		byBuf = cur.CtxPerSlot - drop
	}
	cap := byBuf
	if cap > cur.CtxPerSlot-alignTokens {
		cap = cur.CtxPerSlot - alignTokens
	}
	cap = alignDown(cap)
	if cap < w.CtxFloor {
		cap = w.CtxFloor
	}
	w2 := w
	if cap < w2.CtxMax {
		w2.CtxMax = cap
	}
	if s, err := Choose(fp, room, w2); err == nil && better(cur, s) {
		return s, nil
	}
	return StepDown(fp, cur, room, w)
}

// StepDown makes the shape one notch smaller: ctx per slot down by 1/16 (at least 1024) toward
// the floor, then one more expert layer to RAM (MoE), then one slot fewer at the floor ctx.
func StepDown(fp Footprint, cur Shape, room Room, w Wants) (Shape, error) {
	w = w.norm()
	w.KVType = firstNonEmpty(cur.KVType, w.KVType)
	if cur.CtxPerSlot > w.CtxFloor {
		step := cur.CtxPerSlot / 16
		if step < alignTokens {
			step = alignTokens
		}
		ctx := alignDown(cur.CtxPerSlot - step)
		if ctx < w.CtxFloor {
			ctx = w.CtxFloor
		}
		return fp.shape(w, cur.Slots, ctx, cur.NCpuMoe, cur.CPUOnly), nil
	}
	ram := float64(room.RAMAvail - osReserveMiB)
	if !cur.CPUOnly && fp.MoE && !room.Unified && cur.NCpuMoe < fp.ExpertLayers &&
		fp.ExpertMiBPerLayer*float64(cur.NCpuMoe+1) <= ram {
		return fp.shape(w, cur.Slots, cur.CtxPerSlot, cur.NCpuMoe+1, false), nil
	}
	if cur.Slots > w.MinSlots {
		return fp.shape(w, cur.Slots-1, w.CtxFloor, cur.NCpuMoe, cur.CPUOnly), nil
	}
	return Shape{}, ErrTooBig
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// Grow returns the planner's best shape for the current room and true when it is larger than
// cur (more slots, fewer CPU-MoE layers or more ctx). The caller restarts at it when idle and
// falls back to cur if the bigger load fails.
func Grow(fp Footprint, cur Shape, room Room, w Wants) (Shape, bool) {
	s, err := Choose(fp, room, w)
	if err != nil || !better(s, cur) {
		return cur, false
	}
	return s, true
}

// Args renders the shape-dependent llama-server flags: -ngl, -np, -c, --n-cpu-moe, -ctk, -ctv.
func Args(s Shape) []string {
	ngl := "999"
	if s.CPUOnly {
		ngl = "0"
	}
	a := []string{"-ngl", ngl, "-np", strconv.Itoa(s.Slots), "-c", strconv.Itoa(s.CtxTotal())}
	if s.NCpuMoe > 0 && !s.CPUOnly {
		a = append(a, "--n-cpu-moe", strconv.Itoa(s.NCpuMoe))
	}
	kv := s.KVType
	if kv == "" {
		kv = "q4_0"
	}
	return append(a, "-ctk", kv, "-ctv", kv)
}

func (s Shape) String() string {
	if s.CPUOnly {
		return fmt.Sprintf("%dx%d CPU-only (ram %d MiB)", s.Slots, s.CtxPerSlot, s.RAMMiB)
	}
	return fmt.Sprintf("%dx%d n-cpu-moe %d (vram %d MiB, ram %d MiB)", s.Slots, s.CtxPerSlot, s.NCpuMoe, s.VRAMMiB, s.RAMMiB)
}

// ---- calibration book ----

// Calib is what one healthy load taught us about a (model, GPU, backend).
type Calib struct {
	FixedMiB    float64   `json:"fixed_mib"`
	CapCtxTotal int       `json:"cap_ctx_total"`
	Loads       int       `json:"loads"`
	Updated     time.Time `json:"updated"`
}

// Book is the calibration book kept in state.json.
type Book struct {
	Entries map[string]Calib `json:"entries"`
}

// BookKey keys the book by model sha256, GPU name and backend id.
func BookKey(modelSHA, gpuName, backendID string) string {
	return modelSHA + "|" + gpuName + "|" + backendID
}

// Footprint overlays what the book knows (calibrated fixed load, runtime-OOM cap) on a seed.
func (b *Book) Footprint(key string, seed Footprint) Footprint {
	if b == nil {
		return seed
	}
	if e, ok := b.Entries[key]; ok {
		if e.FixedMiB > 0 {
			seed.FixedMiB = e.FixedMiB
		}
		seed.CapCtxTotal = e.CapCtxTotal
	}
	return seed
}

// Calibrate records the fixed load implied by a healthy load that held measuredMiB of VRAM at
// shape s. Like the hive's calibrate() it accepts the reading only within +-15 % of the seed's
// fixed load (a wild reading means others changed during the load) and reports whether it did.
func (b *Book) Calibrate(key string, seed Footprint, s Shape, measuredMiB int) bool {
	if s.CPUOnly || measuredMiB <= 0 || seed.FixedMiB <= 0 {
		return false
	}
	implied := float64(measuredMiB) - (seed.vram(s.Slots, s.CtxPerSlot, s.NCpuMoe) - seed.FixedMiB)
	if math.Abs(implied-seed.FixedMiB) > 0.15*seed.FixedMiB {
		return false
	}
	if b.Entries == nil {
		b.Entries = map[string]Calib{}
	}
	e := b.Entries[key]
	e.FixedMiB = implied
	e.Loads++
	e.Updated = time.Now()
	b.Entries[key] = e
	return true
}

// LowerCap records a runtime OOM: the next start must use less total ctx than ctxTotal.
func (b *Book) LowerCap(key string, ctxTotal int) {
	if b.Entries == nil {
		b.Entries = map[string]Calib{}
	}
	e := b.Entries[key]
	if ctxTotal > 0 && (e.CapCtxTotal == 0 || ctxTotal < e.CapCtxTotal) {
		e.CapCtxTotal = ctxTotal
	}
	e.Updated = time.Now()
	b.Entries[key] = e
}
