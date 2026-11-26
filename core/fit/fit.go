// Package fit is the ctx balancer's planner: pure functions that turn a model footprint, the
// room on the machine and what the game wants into an explicit llama-server shape. It is a
// generic port of the hive's measure-then-try launcher (hive-mind-llm-run): the machine
// constants became numbers read from the GGUF header plus a calibration book.
//
// It is part of the sizing core (docs/ENGINE.md section 13): it imports only pure standard
// library packages, never reads the clock (a stamp is passed in), never logs (notes are
// returned) and never opens a file or runs a process. It must never import a package of the
// engine's root module.
package fit

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"time"

	"github.com/CharlesMod/easyEngine/core/gguf"
)

// ErrTooBig means no shape fits the room, even at the floor context, minimum slots and every
// movable expert layer in RAM (none, with Wants.Whole). Dense models are never split between card
// and RAM.
var ErrTooBig = errors.New("fit: model does not fit this machine")

// ErrNoCtxTrain means Wants.CtxFromModel was asked and the footprint carries no trained context
// (its header did not say it, or it was not built from a header). It is never guessed.
var ErrNoCtxTrain = errors.New("fit: the model's trained context is not known")

// ErrCtxTrainUnder means Wants.CtxFromModel was asked and the model's trained context is under
// the core's least context per slot (1024 tokens): no shape is planned above what it was trained for.
var ErrCtxTrainUnder = errors.New("fit: the model's trained context is under the least context per slot (1024)")

// ErrNoKVType means Wants.KVTypes was given and the footprint says no cache type its per-token
// costs were priced at (one built from measured numbers), so it cannot be priced at another.
var ErrNoKVType = errors.New("fit: the footprint names no cache type to price other types from")

// ErrNoWeights means Room.LargestBlock was given and the footprint does not know its weights on
// the card (Footprint.WeightsMiB), so its largest single allocation cannot be told.
var ErrNoWeights = errors.New("fit: the footprint cannot tell its largest allocation (no weights figure)")

// UnreadError means the caller marked a room field as not read (Room.Unread): it is never taken
// as zero, and no shape is chosen until it is read.
type UnreadError struct{ Field string }

func (e *UnreadError) Error() string { return "fit: room field " + e.Field + " was not read" }

// KVTypeError means a cache type name the core has no price for. It is never priced as another
// type; the empty name is not unknown, it means the default (q4_0).
type KVTypeError struct{ Name string }

func (e *KVTypeError) Error() string { return fmt.Sprintf("fit: unknown cache type %q", e.Name) }

// Basis says where a memory figure came from. Predicted: worked out from the header's seed and
// the constants above (or from the seed corrected where the book took a reading). Measured: read
// from a load of this model, card and backend that came up. "" is not said: a figure built by hand.
type Basis string

const (
	Predicted Basis = "predicted"
	Measured  Basis = "measured"
)

const (
	alignTokens  = 1024
	ubatchTokens = 512 // -ub; a sliding-window layer caches window+ub tokens per slot
)

// ConstName names one of the measured constants a footprint is built from. They are the same for
// every model: none is a per-model fudge. What was read on one engine build and one kind of card
// may differ on another, so they come as a set (Consts) a caller may give.
type ConstName string

const (
	// BaseOverheadMiB: the card context and the compute buffers that do not depend on the
	// vocabulary, taken before any buffer of the model's.
	BaseOverheadMiB ConstName = "base_overhead_mib"
	// VocabMiBPer1k: output (logits) buffers, MiB per 1000 vocabulary entries at -ub 512.
	VocabMiBPer1k ConstName = "vocab_mib_per_1k"
	// RuntimeMiB: what the first requests take beyond what load allocates.
	RuntimeMiB ConstName = "runtime_mib"
	// MaskMiBPer1k: attention-mask and flash-attention scratch, MiB per 1000 tokens of total ctx.
	MaskMiBPer1k ConstName = "mask_mib_per_1k"
	// DequantBytes: a quantised cache is expanded to f16 for the widest layer, bytes per element
	// per token.
	DequantBytes ConstName = "dequant_bytes"
	// OffloadMiB: card memory of the expert-offload copy buffers once any expert layer is in RAM.
	OffloadMiB ConstName = "offload_mib"
	// ExpertRAMFactor: host RAM taken per expert layer moved off the card, against its tensor bytes.
	ExpertRAMFactor ConstName = "expert_ram_factor"
)

// constNames is every constant a set must hold, in the order a lacking one is named.
var constNames = []ConstName{BaseOverheadMiB, VocabMiBPer1k, RuntimeMiB, MaskMiBPer1k, DequantBytes, OffloadMiB, ExpertRAMFactor}

// Const is one measured constant and a word of where its reading came from.
type Const struct {
	Value float64 `json:"value"`
	From  string  `json:"from"`
}

// Consts is a set of the measured constants, read with one engine build on one kind of card. A
// figure built with a set read on another build or card than the one it runs on is a prediction.
type Consts struct {
	// Backend is the engine build the set was read with, as the caller names its builds (its
	// backend id and release).
	Backend string `json:"backend"`
	// Card is the kind of card the set was read on.
	Card string              `json:"card"`
	Of   map[ConstName]Const `json:"of"`
}

// DefaultConsts is the set written into the core: read on an RTX 5080 (16303 MiB) with llama.cpp
// b12680 at -ub 512, -fa on, --no-kv-unified (docs/FOOTPRINTS.md; tools/fitcheck reproduces them
// from the sweep CSV). It is the set wherever a caller gives none. Each call answers a new map.
func DefaultConsts() Consts {
	const sweep = "RTX 5080 at b12680, fitted over the sweep's 69 rows (docs/FOOTPRINTS.md)"
	return Consts{Backend: "cuda@b12680", Card: "RTX 5080, 16303 MiB", Of: map[ConstName]Const{
		BaseOverheadMiB: {135, sweep},
		VocabMiBPer1k:   {1.2, sweep},
		RuntimeMiB:      {58, "RTX 5080 at b12680: read as 52 to 62 MiB over the first requests"},
		MaskMiBPer1k:    {0.44, sweep},
		DequantBytes:    {2.0, "the f16 width; the sweep's quantised-cache rows agree"},
		OffloadMiB:      {180, "RTX 5080 at b12680: read as 90 to 230 MiB once any expert layer was in RAM"},
		ExpertRAMFactor: {1.25, "RTX 5080 at b12680: read as 1.02 to 1.23, taken above the highest"},
	}}
}

// ConstError means a set of constants lacks one (Name), or its word of where it was read
// (NoFrom), or names no backend or card (Name "backend", "card"). It is never priced as zero.
type ConstError struct {
	Name   ConstName
	NoFrom bool
}

func (e *ConstError) Error() string {
	if e.NoFrom {
		return fmt.Sprintf("fit: the set of constants gives %s no word of where it was read", e.Name)
	}
	return fmt.Sprintf("fit: the set of constants lacks %s", e.Name)
}

// DefaultMinTPS is the decode floor (tokens/s per stream) used when the game names none: below
// about 20 tok/s a spoken line of 60 tokens takes more than 3 s to generate.
const DefaultMinTPS = 20

// OSReserveMiB is the host RAM kept for the OS when moving experts to RAM or running CPU-only,
// where Room.Reserve is unset.
const OSReserveMiB = 2048

// Wants is what the game asks for.
type Wants struct {
	Slots    int    `json:"slots"`
	MinSlots int    `json:"min_slots"`
	CtxMax   int    `json:"ctx_max"` // per slot
	CtxFloor int    `json:"ctx_floor"`
	KVType   string `json:"kv_type"`
	// MinTPS is the decode speed floor in tokens/s per stream (default 20, negative = no floor).
	// Where the footprint has a measured speed curve, the planner keeps the expert layers moved to
	// RAM few enough to stay above it, before it keeps max ctx.
	MinTPS float64 `json:"min_tps"`

	// Whole: no layer and no expert off the card. Where only a split shape fits, ErrTooBig.
	// Unset: expert layers may move to RAM (section 5, step 3).
	Whole bool `json:"whole"`
	// KVTypes is the cache types, best first: the shape takes the first type at which CtxMax fits
	// at Slots; where none does, the last type with the most ctx that fits down to the floor.
	// Each type is priced from the footprint (Footprint.KVType), never tried. When set, KVType is
	// not read. Unset: KVType alone, as given.
	KVTypes []string `json:"kv_types,omitempty"`
	// CtxFromModel: CtxMax is the model's trained context (Footprint.CtxTrain, from the header),
	// and a floor above it comes down to it. Where it is unknown, ErrNoCtxTrain. Unset: CtxMax.
	CtxFromModel bool `json:"ctx_from_model"`
}

// Room is what the machine offers, all in MiB.
type Room struct {
	VRAMTotal   int `json:"vram_total"`
	Others      int `json:"others"` // VRAM held by processes that are not ours
	GameReserve int `json:"game_reserve"`
	// Margin is VRAM left free: 0 means DefaultMargin(VRAMTotal), negative means none.
	Margin int `json:"margin"`
	// RAMAvail is the free host RAM; for a pool that is one node of a processor's memory (NoGPU),
	// that node's free memory, and nothing else is counted.
	RAMAvail int  `json:"ram_avail"`
	NoGPU    bool `json:"no_gpu"`  // no usable GPU (or a processor node): CPU-only runs, VRAM fields not read
	Unified  bool `json:"unified"` // Apple: VRAM is RAM, no expert offload
	// Reserve is host RAM kept for the OS: 0 means OSReserveMiB, negative means none.
	Reserve int `json:"reserve"`
	// LargestBlock is the largest single allocation the card can still give (MiB); a shape's
	// largest allocation (Footprint.Largest) must fit inside it. 0: not known, not bounded.
	LargestBlock int `json:"largest_block"`
	// Unread names the fields (by their json names) the caller could not read: none is taken as
	// zero, and every entry that reads a room answers *UnreadError naming the first (settle).
	Unread []string `json:"unread,omitempty"`
	// SpeedScale is how fast this machine decodes compared with the machine the speed curve was
	// measured on (1 = the same card). 0 means 1. The caller sets it per model (the engine's
	// planner from where the catalog's curve was measured).
	SpeedScale float64 `json:"speed_scale"`
	GPUName    string  `json:"gpu_name"`
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

	ComputePerTokenMiB float64 `json:"compute_per_token_mib"` // scratch that grows with total ctx (mask, f16 expansion of a quantised KV)
	MoESchedMiB        float64 `json:"moe_sched_mib"`         // extra GPU use when any expert layer is on the CPU
	ExpertRAMPerLayer  float64 `json:"expert_ram_per_layer"`  // host RAM per expert layer moved; 0 = ExpertMiBPerLayer
	HostMiB            float64 `json:"host_mib"`              // weights that stay in RAM even when the model is on the card (they count in a CPU-only run)

	// Speed on the machine the footprint was measured on (zero when unknown).
	TPS *TPSCurve `json:"tps,omitempty"`

	StatePerSlotMiB float64 `json:"state_per_slot_mib"` // recurrent (linear-attention/SSM) state per slot; fixed, not per token
	Estimated       bool    `json:"estimated"`          // built from a conservative guess, not a known architecture
	// Basis is of FixedMiB, the one term a reading corrects: Measured where the book took a
	// reading for this model, card and backend (Book.Footprint), Predicted from Seed and
	// SeedFromBytes. The per-token terms are always predicted. A reading passed over leaves it
	// as it was.
	Basis Basis `json:"basis,omitempty"`

	SWAPerTokenMiB float64 `json:"swa_per_token_mib"` // sliding-window KV per cached token
	SWAWindow      int     `json:"swa_window"`        // tokens kept per slot by those layers
	CapCtxTotal    int     `json:"cap_ctx_total"`     // runtime-OOM cap from the book; 0 = none

	// KVType is the cache type the per-token terms are priced at; "" when they cannot be priced
	// at another (measured numbers). DequantPerTokenMiB is the part of ComputePerTokenMiB a
	// quantised cache adds (the f16 expansion of the widest layer), whatever KVType is.
	KVType             string  `json:"kv_type"`
	DequantPerTokenMiB float64 `json:"dequant_per_token_mib"`
	// WeightsMiB is the weights on the card with no expert moved (one allocation); 0 = unknown.
	WeightsMiB float64 `json:"weights_mib"`
	// CtxTrain is the model's trained context from its header; 0 = unknown.
	CtxTrain int `json:"ctx_train"`
	// Consts is the set of measured constants the figures above were built with: Seed's is
	// DefaultConsts, SeedWith's the one given. nil (a footprint built by hand, or by an older
	// build) is the default set. Every entry prices the figures alone, never a constant read
	// beside them, so a price never takes one constant from each of two sets; the set is the
	// figures' word of where they came from (Shape.ConstsBackend, Shape.ConstsCard) and keys the
	// book's fixed part (Book.Footprint). A set lacking a constant is *ConstError from every entry.
	Consts *Consts `json:"consts,omitempty"`
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
	// EstTPS is the expected decode speed per stream (tokens/s) on this machine, 0 when no
	// measured curve exists for the model.
	EstTPS float64 `json:"est_decode_tps"`
	// VRAMBasis is of VRAMMiB: Predicted from the planner; Measured from Book.Mark, where the book
	// holds a reading of this very shape and VRAMMiB is that reading. RAMMiB is always predicted:
	// no reading of host memory is taken.
	VRAMBasis Basis `json:"vram_basis,omitempty"`
	// ConstsBackend and ConstsCard name the set of constants the predicted figures were built
	// with (Consts.Backend, Consts.Card); where they are not the build and card it runs on, the
	// figures were measured elsewhere.
	ConstsBackend string `json:"consts_backend,omitempty"`
	ConstsCard    string `json:"consts_card,omitempty"`
}

// CtxTotal is the -c value.
func (s Shape) CtxTotal() int { return s.Slots * s.CtxPerSlot }

// Norm is w with every default filled in and every bound made consistent: what Choose plans with.
func (w Wants) Norm() Wants {
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
	if w.MinTPS == 0 {
		w.MinTPS = DefaultMinTPS
	}
	return w
}

func alignDown(n int) int {
	if n < 0 {
		return 0
	}
	return n / alignTokens * alignTokens
}

// kvBytes is bytes per KV element for a llama.cpp cache type, and false for a name it does not
// know; "" is not a name here (the callers put the default in first).
func kvBytes(t string) (float64, bool) {
	switch t {
	case "f32":
		return 4, true
	case "f16", "bf16":
		return 2, true
	case "q8_0":
		return 8.5 / 8, true
	case "q5_1":
		return 6.0 / 8, true
	case "q5_0":
		return 5.5 / 8, true
	case "q4_1":
		return 5.0 / 8, true
	case "q4_0", "iq4_nl":
		return 4.5 / 8, true
	}
	return 0, false
}

// q4Bytes is kvBytes("q4_0"), the default type the size guesses are stated at.
const q4Bytes = 4.5 / 8

// knownKV is nil where every name given is a known cache type or empty, else a *KVTypeError
// naming the first that is not.
func knownKV(names ...string) error {
	for _, t := range names {
		if _, ok := kvBytes(t); t != "" && !ok {
			return &KVTypeError{Name: t}
		}
	}
	return nil
}

// KVBytes is the bytes per element of a llama.cpp KV cache type; "" is the default (q4_0), and a
// name the core does not know is a *KVTypeError.
func KVBytes(t string) (float64, error) {
	if t == "" {
		t = "q4_0"
	}
	b, ok := kvBytes(t)
	if !ok {
		return 0, &KVTypeError{Name: t}
	}
	return b, nil
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

// Seed builds a footprint from a GGUF header, priced at cache type kvType ("" is q4_0; a name
// the core does not know is a *KVTypeError), with the default set of constants (DefaultConsts).
// notes says what it had to guess (an architecture not fully understood, no KV dimensions in the
// header), one line each, for the caller to log.
func Seed(info gguf.Info, kvType string) (fp Footprint, notes []string, err error) {
	return SeedWith(info, kvType, DefaultConsts())
}

// SeedWith is Seed with the set of constants c, read on the caller's own build and card; a set
// lacking a constant is *ConstError. The footprint carries the set (Footprint.Consts).
func SeedWith(info gguf.Info, kvType string, c Consts) (fp Footprint, notes []string, err error) {
	if kvType == "" {
		kvType = "q4_0"
	}
	c.Of = maps.Clone(c.Of)
	st, err := settle(&Footprint{KVType: kvType, Consts: &c}, nil, nil, nil)
	if err != nil {
		return Footprint{}, nil, err
	}
	k := st.fp.c
	b, _ := kvBytes(kvType)
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
	var full, swa, maxFull float64
	for _, l := range layers {
		if !l.HasKV {
			continue
		}
		e := float64(l.KVHeads * (l.HeadDimK + l.HeadDimV))
		if l.SWA && info.SlidingWindow > 0 {
			swa += e
		} else {
			full += e
			maxFull = math.Max(maxFull, e)
		}
	}
	dequant := k.dequant * maxFull / mib
	compute := k.mask / 1024
	if quantised(kvType) {
		compute += dequant
	}
	fp = Footprint{
		FixedMiB:       float64(info.DenseBytes+info.ExpertBytes-info.HostBytes)/mib + k.base + float64(info.VocabSize)/1000*k.vocab,
		HostMiB:        float64(info.HostBytes) / mib,
		PerTokenMiB:    full * b / mib,
		SWAPerTokenMiB: swa * b / mib,
		SWAWindow:      info.SlidingWindow + ubatchTokens,
		ScratchMiB:     k.runtime,

		ComputePerTokenMiB: compute,
		StatePerSlotMiB:    float64(info.StateBytesPerSlot) / mib,
		Estimated:          info.Estimated,
		Basis:              Predicted,

		KVType:             kvType,
		DequantPerTokenMiB: dequant,
		WeightsMiB:         float64(info.DenseBytes+info.ExpertBytes-info.HostBytes) / mib,
		CtxTrain:           info.CtxTrain,
		Consts:             &c,
	}
	if info.Estimated {
		notes = append(notes, fmt.Sprintf("fit: %s: architecture not fully understood (%s); using a conservative KV estimate", info.Arch, info.Note))
	}
	if fp.PerTokenMiB+fp.SWAPerTokenMiB <= 0 && fp.StatePerSlotMiB <= 0 {
		// Nothing in the header says what the cache costs: never plan with zero. Scale with the
		// weights like SeedFromBytes does.
		gib := float64(info.DenseBytes+info.ExpertBytes) / mib / 1024
		if gib < 1 {
			gib = 1
		}
		fp.PerTokenMiB = 0.0006 * gib * 1.25 * b / q4Bytes
		fp.Estimated = true
		notes = append(notes, fmt.Sprintf("fit: %s: header gave no KV dimensions; guessing %.4f MiB per token from the weight size", info.Arch, fp.PerTokenMiB))
	}
	if info.NExpert > 0 && info.ExpertLayers > 0 && info.ExpertBytes > 0 {
		fp.MoE = true
		fp.ExpertLayers = info.ExpertLayers
		fp.ExpertMiBPerLayer = float64(info.ExpertBytes) / mib / float64(info.ExpertLayers)
		fp.ExpertRAMPerLayer = fp.ExpertMiBPerLayer * k.expertRAM
		fp.MoESchedMiB = k.offload
	}
	return fp, notes, nil
}

// SeedFromBytes is the footprint of a model known only by its file size: about 0.0006 MiB per
// token per GiB of weights at q4_0 with a 25 % safety factor, 30 expert layers for a MoE. It is
// marked Estimated; the header gives the real footprint once the file is on disk. kvType is as
// for Seed, and the constants are the default set.
func SeedFromBytes(size int64, moe bool, kvType string) (Footprint, error) {
	return SeedFromBytesWith(size, moe, kvType, DefaultConsts())
}

// SeedFromBytesWith is SeedFromBytes with the set of constants c; a set lacking a constant is
// *ConstError. The footprint carries the set.
func SeedFromBytesWith(size int64, moe bool, kvType string, c Consts) (Footprint, error) {
	if kvType == "" {
		kvType = "q4_0"
	}
	c.Of = maps.Clone(c.Of)
	st, err := settle(&Footprint{KVType: kvType, Consts: &c}, nil, nil, nil)
	if err != nil {
		return Footprint{}, err
	}
	k := st.fp.c
	b, _ := kvBytes(kvType)
	sizeMiB := float64(size) / mib
	gib := sizeMiB / 1024
	fp := Footprint{
		FixedMiB:    sizeMiB + k.base + 200*k.vocab, // a typical 200k vocabulary
		PerTokenMiB: 0.0006 * gib * 1.25 * b / q4Bytes,
		ScratchMiB:  k.runtime,
		Estimated:   true,
		Basis:       Predicted,
		KVType:      kvType,
		WeightsMiB:  sizeMiB,
		Consts:      &c,
	}
	if moe {
		const layers = 30 // typical; the header gives the real count after download
		fp.MoE = true
		fp.ExpertLayers = layers
		fp.ExpertMiBPerLayer = 0.75 * sizeMiB / layers
		fp.ExpertRAMPerLayer = fp.ExpertMiBPerLayer * k.expertRAM
		fp.MoESchedMiB = k.offload
	}
	return fp, nil
}

// quantised reports whether a cache type is expanded to f16 for the widest layer.
func quantised(t string) bool { return t != "f16" && t != "bf16" && t != "f32" }

// ---- the one checked place ----
//
// A Footprint, a Room and a Wants as a caller hands them are never priced or read for a budget
// directly: every exported entry that prices memory or reads a room first passes them to settle,
// and takes every number from what it answers. The raw prices are methods of fig and the budgets
// fields of roomFig, and only settle (and price, which settle and fig.at call) builds either.

// fig is a footprint's figures priced at one known cache type (its KVType; "" where the figures
// were measured and are taken as given). Every memory figure the package answers is read from one.
type fig struct {
	Footprint
	src Footprint // the footprint as the entry was given it, which at prices another type from
	c   constFig  // the footprint's own set of constants, read whole: the only constants a price takes
}

// constFig is a set of constants every one of which was given, by value.
type constFig struct {
	backend, card                                           string
	base, vocab, runtime, mask, dequant, offload, expertRAM float64
}

// sameSet reports whether two sets hold the same build, card and constants (nil is the default
// set); a word of where a reading came from does not change a price, so it is not compared.
func sameSet(a, b *Consts) bool {
	d := DefaultConsts()
	if a == nil {
		a = &d
	}
	if b == nil {
		b = &d
	}
	return a.Backend == b.Backend && a.Card == b.Card &&
		maps.EqualFunc(a.Of, b.Of, func(x, y Const) bool { return x.Value == y.Value })
}

// cloneSet is a copy of a set sharing no map with it: a book handed to a writer keeps its own.
func cloneSet(c *Consts) *Consts {
	if c == nil {
		return nil
	}
	n := *c
	n.Of = maps.Clone(c.Of)
	return &n
}

// readConsts reads a set whole: nil is the default set, and a set lacking a constant (or its word
// of where it was read, or its backend or card) is *ConstError naming it. Only settle calls it.
func readConsts(set *Consts) (constFig, error) {
	if set == nil {
		d := DefaultConsts()
		set = &d
	}
	switch {
	case set.Backend == "":
		return constFig{}, &ConstError{Name: "backend"}
	case set.Card == "":
		return constFig{}, &ConstError{Name: "card"}
	}
	v := make(map[ConstName]float64, len(constNames))
	for _, n := range constNames {
		c, ok := set.Of[n]
		if !ok {
			return constFig{}, &ConstError{Name: n}
		}
		if c.From == "" {
			return constFig{}, &ConstError{Name: n, NoFrom: true}
		}
		v[n] = c.Value
	}
	return constFig{backend: set.Backend, card: set.Card,
		base: v[BaseOverheadMiB], vocab: v[VocabMiBPer1k], runtime: v[RuntimeMiB], mask: v[MaskMiBPer1k],
		dequant: v[DequantBytes], offload: v[OffloadMiB], expertRAM: v[ExpertRAMFactor]}, nil
}

// roomFig is a room every field of which was read, with its budgets in MiB.
type roomFig struct {
	Room
	budget float64 // card a shape may take: VRAMTotal less Others, GameReserve and the margin
	ram    float64 // host RAM a CPU-only run or the moved expert layers may take: RAMAvail less the reserve
}

// settled is what one entry prices from: the footprint priced at the type its numbers are taken
// at, the room read whole, the wants resolved and the type the shape runs at. A part the entry did
// not give is zero.
type settled struct {
	fp   fig
	room roomFig
	w    Wants
	kv   string // the cache type the shape runs at (runKV); "" with no shape
}

// runKV is the cache type a shape is run at: its own, or for "" the default q4_0. Args renders it
// and settle prices it, so a shape naming no type has that one meaning everywhere.
func runKV(s Shape) string {
	if s.KVType == "" {
		return "q4_0"
	}
	return s.KVType
}

// settle is the one checked place. Each argument is nil where the entry takes none (with wants,
// a footprint is always given). In this order it refuses: a room with a field marked unread
// (*UnreadError naming the first: never taken as zero); with wants, a largest block the
// footprint cannot be held to (ErrNoWeights); wants it cannot resolve (ErrNoCtxTrain,
// ErrCtxTrainUnder, *KVTypeError for KVType or any of KVTypes); a shape whose cache type has no
// price (*KVTypeError); a type the footprint cannot be priced at (ErrNoKVType, *KVTypeError); a
// footprint whose set of constants lacks one (*ConstError: nil is the default set).
//
// The figures are priced at the type the shape runs at where a shape is given (runKV: a shape's
// "" is the default q4_0, with or without wants, as Args renders it), else at the wants' (the
// first of KVTypes), else at the footprint's own: never at the seed's in place of the shape's. A
// footprint that names no type (measured numbers) is taken as given, except with a list of types,
// which it cannot be priced at (ErrNoKVType). The resolved wants are never rewritten from a shape:
// the type the shape runs at is answered beside them (settled.kv).
func settle(fp *Footprint, s *Shape, room *Room, w *Wants) (settled, error) {
	var st settled
	if room != nil {
		if len(room.Unread) > 0 {
			return st, &UnreadError{Field: room.Unread[0]}
		}
		if w != nil && room.LargestBlock > 0 && !room.NoGPU && fp.WeightsMiB <= 0 {
			return st, ErrNoWeights
		}
		m := room.Margin
		if m == 0 {
			m = DefaultMargin(room.VRAMTotal)
		}
		r := room.Reserve
		if r == 0 {
			r = OSReserveMiB
		}
		st.room = roomFig{Room: *room,
			budget: float64(room.VRAMTotal - room.Others - room.GameReserve - max(m, 0)),
			ram:    float64(room.RAMAvail - max(r, 0))}
	}
	if w != nil {
		rw, err := resolve(*fp, *w)
		if err != nil {
			return st, err
		}
		st.w = rw
	}
	if s != nil {
		if err := knownKV(s.KVType); err != nil {
			return st, err
		}
	}
	if fp == nil {
		return st, nil
	}
	t := fp.KVType
	switch {
	case s != nil:
		st.kv = runKV(*s)
		t = st.kv
	case w != nil:
		t = st.w.KVType
	}
	if fp.KVType == "" && len(st.w.KVTypes) == 0 {
		t = "" // measured figures: taken as given
	}
	f, err := price(*fp, t)
	if err != nil {
		return st, err
	}
	if f.c, err = readConsts(fp.Consts); err != nil {
		return st, err
	}
	st.fp = f
	return st, nil
}

// price is fp priced at cache type t: the per-token cache scales with the bytes per element and
// the f16 expansion comes or goes with quantisation. Worked out from the footprint, never tried.
// Only settle and fig.at call it.
func price(fp Footprint, t string) (fig, error) {
	src := fp
	if err := knownKV(t, fp.KVType); err != nil {
		return fig{}, err
	}
	if t != fp.KVType {
		if fp.KVType == "" {
			return fig{}, ErrNoKVType
		}
		bt, _ := kvBytes(t)
		bf, _ := kvBytes(fp.KVType)
		r := bt / bf
		fp.PerTokenMiB *= r
		fp.SWAPerTokenMiB *= r
		if quantised(fp.KVType) {
			fp.ComputePerTokenMiB -= fp.DequantPerTokenMiB
		}
		if quantised(t) {
			fp.ComputePerTokenMiB += fp.DequantPerTokenMiB
		}
		fp.KVType = t
	}
	return fig{Footprint: fp, src: src}, nil
}

// at is the same footprint priced at another type of a list, with the same set of constants.
func (fp fig) at(t string) (fig, error) {
	f, err := price(fp.src, t)
	f.c = fp.c
	return f, err
}

// capRoom is the room a cap is measured against: the card's budget, or host RAM with no GPU.
func (room roomFig) capRoom() int {
	if room.NoGPU {
		return int(room.ram)
	}
	return int(room.budget)
}

// kvAt returns the KV cache MiB for a layout.
func (fp fig) kvAt(slots, ctxPerSlot int) float64 {
	kv := (fp.PerTokenMiB+fp.ComputePerTokenMiB)*float64(slots*ctxPerSlot) + fp.StatePerSlotMiB*float64(slots)
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
func (fp fig) vram(slots, ctxPerSlot, nCpuMoe int) float64 {
	v := fp.FixedMiB + fp.kvAt(slots, ctxPerSlot) + fp.ScratchMiB - fp.ExpertMiBPerLayer*float64(nCpuMoe)
	if nCpuMoe > 0 {
		v += fp.MoESchedMiB
	}
	return v
}

// largest is the largest single allocation on the card of a layout, each of these taken as one
// allocation: the weights left on the card, the cache (full-attention, sliding-window and
// recurrent state together) and the compute buffers (the fixed load beyond the weights plus the
// scratch that grows with ctx). It needs WeightsMiB.
func (fp fig) largest(slots, ctxPerSlot, nCpuMoe int) float64 {
	weights := fp.WeightsMiB - fp.ExpertMiBPerLayer*float64(nCpuMoe)
	cache := fp.kvAt(slots, ctxPerSlot) - fp.ComputePerTokenMiB*float64(slots*ctxPerSlot)
	compute := math.Max(fp.FixedMiB-fp.WeightsMiB, 0) + fp.ComputePerTokenMiB*float64(slots*ctxPerSlot)
	return math.Max(weights, math.Max(cache, compute))
}

// expertRAM is the host RAM taken by nCpuMoe expert layers.
func (fp fig) expertRAM(nCpuMoe int) float64 {
	per := fp.ExpertRAMPerLayer
	if per <= 0 {
		per = fp.ExpertMiBPerLayer
	}
	return per * float64(nCpuMoe)
}

// allHost is the RAM use of a CPU-only run.
func (fp fig) allHost(slots, ctxPerSlot int) float64 {
	return fp.FixedMiB + fp.HostMiB + fp.kvAt(slots, ctxPerSlot) + fp.ScratchMiB
}

// Need is the predicted memory in MiB of shape s, priced at its own cache type: card memory and
// the host memory of the expert layers it moves, or for a CPU-only shape no card and the whole
// run's host memory. It refuses as settle does (an unknown type, a type the footprint cannot be
// priced at).
func (fp Footprint) Need(s Shape) (vram, ram float64, err error) {
	st, err := settle(&fp, &s, nil, nil)
	if err != nil {
		return 0, 0, err
	}
	if s.CPUOnly {
		return 0, st.fp.allHost(s.Slots, s.CtxPerSlot), nil
	}
	return st.fp.vram(s.Slots, s.CtxPerSlot, s.NCpuMoe), st.fp.expertRAM(s.NCpuMoe), nil
}

// Largest is the predicted largest single allocation in MiB of shape s on the card (see
// Room.LargestBlock), priced at its own cache type; ErrNoWeights where the footprint does not
// know its weights, and settle's refusals.
func (fp Footprint) Largest(s Shape) (float64, error) {
	st, err := settle(&fp, &s, nil, nil)
	if err != nil {
		return 0, err
	}
	if fp.WeightsMiB <= 0 {
		return 0, ErrNoWeights
	}
	return st.fp.largest(s.Slots, s.CtxPerSlot, s.NCpuMoe), nil
}

func ceilMiB(f float64) int { return int(math.Ceil(f)) }

func (fp fig) shape(w Wants, slots, ctx, n int, cpuOnly bool) Shape {
	s := Shape{Slots: slots, CtxPerSlot: ctx, NCpuMoe: n, KVType: w.KVType, CPUOnly: cpuOnly, VRAMBasis: Predicted,
		ConstsBackend: fp.c.backend, ConstsCard: fp.c.card}
	if cpuOnly {
		s.NCpuMoe = 0
		s.RAMMiB = ceilMiB(fp.allHost(slots, ctx))
		return s
	}
	s.VRAMMiB = ceilMiB(fp.vram(slots, ctx, n))
	s.RAMMiB = ceilMiB(fp.expertRAM(n))
	return s
}

// ctxCeiling is the largest 1024-aligned ctx per slot allowed by wants and the book's cap.
func (fp fig) ctxCeiling(w Wants, slots int) int {
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

// Budget is the card and host memory in MiB a shape may take in this room: the card less
// others, the game's reserve and the margin (Room.Margin, or DefaultMargin where it is 0); RAMAvail
// less the reserve (Room.Reserve, or OSReserveMiB where it is 0). A field marked unread is
// *UnreadError, never taken as zero.
func (room Room) Budget() (vram, ram float64, err error) {
	st, err := settle(nil, nil, &room, nil)
	if err != nil {
		return 0, 0, err
	}
	return st.room.budget, st.room.ram, nil
}

// kvTypes is a list of cache types taken once: each type at its first place, repeats and empty
// names dropped. Every reading of Wants.KVTypes goes through it (resolve).
func kvTypes(list []string) []string {
	var types []string
	for _, t := range list {
		if t != "" && !slices.Contains(types, t) {
			types = append(types, t)
		}
	}
	return types
}

// need is the predicted memory a shape is sized against: card memory, or host memory for a
// CPU-only shape. fp must be priced at s's cache type.
func (fp fig) need(s Shape) float64 {
	if s.CPUOnly {
		return fp.allHost(s.Slots, s.CtxPerSlot)
	}
	return fp.vram(s.Slots, s.CtxPerSlot, s.NCpuMoe)
}

// resolve fills the wants from the footprint (the trained context) and then every default. Only
// settle calls it.
func resolve(fp Footprint, w Wants) (Wants, error) {
	if w.CtxFromModel {
		if fp.CtxTrain <= 0 {
			return w, ErrNoCtxTrain
		}
		if fp.CtxTrain < alignTokens {
			return w, ErrCtxTrainUnder
		}
		w.CtxMax = fp.CtxTrain
		if w.CtxFloor <= 0 || w.CtxFloor > fp.CtxTrain {
			w.CtxFloor = min(8192, fp.CtxTrain)
		}
	}
	w = w.Norm()
	w.KVTypes = kvTypes(w.KVTypes)
	if len(w.KVTypes) > 0 {
		w.KVType = w.KVTypes[0]
	}
	if err := knownKV(w.KVType); err != nil {
		return w, err
	}
	return w, knownKV(w.KVTypes...)
}

// Choose picks the shape, in the order of ENGINE.md section 5: all on GPU at the largest ctx;
// ctx per slot down to half; MoE only: as many expert layers in RAM as RAM allows, then the
// most ctx that fits (fewer CPU layers on equal ctx); ctx down to the floor; then fewer slots;
// else ErrTooBig. Dense models are never split between
// card and RAM; with no usable GPU they run CPU-only. When the footprint has a measured speed
// curve the CPU layers stop at the Wants.MinTPS floor (scaled by Room.SpeedScale) even if that
// costs ctx; if nothing reaches the floor the best shape is returned, with EstTPS under it.
// With Wants.KVTypes it does all this at each type in turn (see there).
func Choose(fp Footprint, room Room, w Wants) (Shape, error) {
	st, err := settle(&fp, nil, &room, &w)
	if err != nil {
		return Shape{}, err
	}
	w, rm := st.w, st.room
	if len(w.KVTypes) == 0 {
		return chooseAt(st.fp, rm, w, rm.budget, rm.ram)
	}
	at := func(t string) (fig, Shape, error) {
		ft, err := st.fp.at(t)
		if err != nil {
			return ft, Shape{}, err
		}
		wt := w
		wt.KVType = t
		s, err := chooseAt(ft, rm, wt, rm.budget, rm.ram)
		return ft, s, err
	}
	last := len(w.KVTypes) - 1
	for _, t := range w.KVTypes[:last] {
		ft, s, err := at(t)
		if errors.Is(err, ErrNoKVType) {
			return Shape{}, err
		}
		if err == nil && s.Slots == w.Slots && s.CtxPerSlot >= ft.ctxCeiling(w, w.Slots) {
			return s, nil // the most context asked fits at this type
		}
	}
	_, s, err := at(w.KVTypes[last])
	return s, err
}

// chooseAt is Choose at the one type w.KVType, with fp priced at it, within budget MiB of card
// and ram MiB of host memory (the room's own, or less on the walk down: StepDown).
func chooseAt(fp fig, room roomFig, w Wants, budget, ram float64) (Shape, error) {
	s, err := choose(fp, room, w, budget, ram, true)
	if errors.Is(err, ErrTooBig) && fp.TPS != nil && w.MinTPS > 0 && !room.NoGPU {
		// Nothing fits above the speed floor. A slow shape still runs, so offer it (the verdict
		// says it is slow) rather than refusing a model that does fit.
		s, err = choose(fp, room, w, budget, ram, false)
	}
	if err == nil && !s.CPUOnly {
		s.EstTPS = fp.estTPS(s.NCpuMoe, room.SpeedScale)
	}
	return s, err
}

// estTPS is the measured speed with n layers on the CPU, scaled for this machine; 0 if unknown.
func (fp fig) estTPS(n int, scale float64) float64 {
	t, ok := fp.TPS.At(n)
	if !ok {
		return 0
	}
	if scale <= 0 {
		scale = 1
	}
	return math.Round(t*scale*10) / 10
}

func choose(fp fig, room roomFig, w Wants, budget, ram float64, enforceSpeed bool) (Shape, error) {
	block := float64(room.LargestBlock)
	for slots := w.Slots; slots >= w.MinSlots; slots-- {
		ceil := fp.ctxCeiling(w, slots)
		if room.NoGPU {
			ctx, ok := maxCtx(w.CtxFloor, ceil, func(c int) bool { return fp.allHost(slots, c) <= ram })
			if ok {
				return fp.shape(w, slots, ctx, 0, true), nil
			}
			continue
		}
		okAt := func(n int) func(int) bool {
			return func(c int) bool {
				return fp.vram(slots, c, n) <= budget && (block <= 0 || fp.largest(slots, c, n) <= block)
			}
		}
		// (1) all on GPU at the largest ctx.
		if ctx, ok := maxCtx(ceil, ceil, okAt(0)); ok {
			return fp.shape(w, slots, ctx, 0, false), nil
		}
		// (2) ctx per slot down to half the largest.
		half := alignDown(ceil / 2)
		if half < w.CtxFloor {
			half = w.CtxFloor
		}
		if ctx, ok := maxCtx(half, ceil, okAt(0)); ok {
			return fp.shape(w, slots, ctx, 0, false), nil
		}
		// (3) MoE only: expert layers to RAM, as many as RAM allows, then the most ctx that fits;
		// equal ctx goes to the fewer CPU layers.
		if fp.MoE && !room.Unified && !w.Whole {
			maxN := fp.ExpertLayers
			if per := fp.expertRAM(1); per > 0 {
				if byRAM := int(ram / per); byRAM < maxN {
					maxN = byRAM
				}
			}
			bestN, bestCtx := 0, 0
			for n := 1; n <= maxN; n++ {
				if enforceSpeed && fp.TPS != nil && w.MinTPS > 0 && fp.estTPS(n, room.SpeedScale) < w.MinTPS {
					break // each further layer on the CPU is slower still
				}
				if ctx, ok := maxCtx(w.CtxFloor, ceil, okAt(n)); ok && ctx > bestCtx {
					bestN, bestCtx = n, ctx
				}
			}
			if bestN > 0 {
				return fp.shape(w, slots, bestCtx, bestN, false), nil
			}
		}
		// (4) ctx down to the floor (dense models, or MoE with nothing found above).
		if ctx, ok := maxCtx(w.CtxFloor, ceil, okAt(0)); ok {
			return fp.shape(w, slots, ctx, 0, false), nil
		}
		// (5) fewer slots: next loop turn.
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
//
// With Wants.KVTypes (taken once, each type at its first place) it re-plans at cur's type, and
// at that type's floor StepDown moves on to the next type, never to a shape larger than cur.
func MissRetry(fp Footprint, cur Shape, failedBufMiB int, room Room, w Wants) (Shape, error) {
	st, err := settle(&fp, &cur, &room, &w)
	if err != nil {
		return Shape{}, err
	}
	f, w := st.fp, st.w // f is priced at the type cur runs at, st.kv
	byBuf := cur.CtxPerSlot
	if perTok := f.PerTokenMiB + f.ComputePerTokenMiB; failedBufMiB > 0 && perTok > 0 && cur.Slots > 0 {
		drop := int(math.Ceil((float64(failedBufMiB) + f.ScratchMiB) / (perTok * float64(cur.Slots))))
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
	w2.CtxFromModel = false // w is resolved: CtxMax already holds the trained context, and the cap must stand
	w2.KVType = st.kv       // the re-plan keeps the type cur runs at
	if len(w2.KVTypes) > 0 {
		w2.KVTypes = []string{st.kv}
	}
	if cap < w2.CtxMax {
		w2.CtxMax = cap
	}
	if s, err := Choose(f.Footprint, room, w2); err == nil && better(cur, s) {
		return s, nil
	}
	return StepDown(f.Footprint, cur, room, w)
}

// StepDown makes the shape one notch smaller: ctx per slot down by 1/16 (at least 1024) toward
// the floor, then one more expert layer to RAM (MoE, not with Wants.Whole), then one slot fewer
// at the floor ctx. It keeps the cache type cur runs at (runKV: a shape naming no type runs at the
// default q4_0, whatever type the wants name); with Whole no step has an expert off the card.
// With Wants.KVTypes (taken once, each type at its first place and repeats dropped), at the floor
// with MinSlots at a type that is not the last, it moves to the next type in the list with the
// largest shape there that needs no more memory than cur (predicted at cur's type) and is never
// better than it (no more slots, no more ctx per slot); where none fits, on to the type after;
// ErrTooBig comes only at the last type's floor. A shape at a type the wants do not list (with no
// list, any type; so a shape naming no type under wants at f16 walks down at q4_0) keeps that type
// to its floor and ends at ErrTooBig there, never moving to a type it was not running. It never
// plans the room afresh (Choose alone does): the load that failed disproved that plan.
//
// The walk is monotone: each step strictly lowers the pair (types left after the shape's in the
// list, then the predicted memory the shape is sized against, Footprint.need), compared in that
// order, so feeding each answer back ends at ErrTooBig in a bounded number of steps for any list.
// One step does not lower it: the first expert layer moved to RAM where a layer weighs less than
// Footprint.MoESchedMiB; those moves are bounded by ExpertLayers, so the walk still ends.
func StepDown(fp Footprint, cur Shape, room Room, w Wants) (Shape, error) {
	st, err := settle(&fp, &cur, &room, &w)
	if err != nil {
		return Shape{}, err
	}
	f, w := st.fp, st.w // f is priced at the type cur runs at, st.kv
	w.KVType = st.kv    // every step keeps that type until the move to the next type of the list
	if w.Whole {
		cur.NCpuMoe = 0
	}
	if cur.CtxPerSlot > w.CtxFloor {
		step := cur.CtxPerSlot / 16
		if step < alignTokens {
			step = alignTokens
		}
		ctx := alignDown(cur.CtxPerSlot - step)
		if ctx < w.CtxFloor {
			ctx = w.CtxFloor
		}
		return f.shape(w, cur.Slots, ctx, cur.NCpuMoe, cur.CPUOnly), nil
	}
	ram := st.room.ram
	if !cur.CPUOnly && f.MoE && !st.room.Unified && !w.Whole && cur.NCpuMoe < f.ExpertLayers &&
		f.expertRAM(cur.NCpuMoe+1) <= ram {
		return f.shape(w, cur.Slots, cur.CtxPerSlot, cur.NCpuMoe+1, false), nil
	}
	if cur.Slots > w.MinSlots {
		return f.shape(w, cur.Slots-1, w.CtxFloor, cur.NCpuMoe, cur.CPUOnly), nil
	}
	i := slices.Index(w.KVTypes, st.kv) // w.KVTypes holds each type once (resolve)
	if i < 0 {
		return Shape{}, ErrTooBig
	}
	// The next type's shape: no more slots or ctx per slot than cur, no more memory than cur needs.
	w.Slots = min(w.Slots, cur.Slots)
	w.MinSlots = min(w.MinSlots, w.Slots)
	w.CtxMax = min(w.CtxMax, alignDown(cur.CtxPerSlot))
	budget, hostRAM := st.room.budget, ram
	if cur.CPUOnly {
		hostRAM = min(hostRAM, f.need(cur))
	} else {
		budget = min(budget, f.need(cur))
	}
	for _, t := range w.KVTypes[i+1:] {
		ft, err := f.at(t)
		if err != nil {
			return Shape{}, err
		}
		w.KVType = t
		s, err := chooseAt(ft, st.room, w, budget, hostRAM)
		if !errors.Is(err, ErrTooBig) {
			return s, err
		}
	}
	return Shape{}, ErrTooBig
}

// Grow returns the planner's best shape for the current room and true when it is larger than
// cur (more slots, fewer CPU-MoE layers or more ctx). The caller restarts at it when idle and
// falls back to cur if the bigger load fails. Where cur, the room or the wants are refused
// (settle) or Choose answers an error, cur, false and that error. A better cache type alone is
// not larger.
func Grow(fp Footprint, cur Shape, room Room, w Wants) (Shape, bool, error) {
	if _, err := settle(&fp, &cur, &room, &w); err != nil {
		return cur, false, err
	}
	s, err := Choose(fp, room, w)
	if err != nil {
		return cur, false, err
	}
	if !better(s, cur) {
		return cur, false, nil
	}
	return s, true, nil
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
	kv := runKV(s)
	return append(a, "-ctk", kv, "-ctv", kv)
}

func (s Shape) String() string {
	if s.CPUOnly {
		return fmt.Sprintf("%dx%d CPU-only (ram %d MiB)", s.Slots, s.CtxPerSlot, s.RAMMiB)
	}
	return fmt.Sprintf("%dx%d n-cpu-moe %d (vram %d MiB, ram %d MiB)", s.Slots, s.CtxPerSlot, s.NCpuMoe, s.VRAMMiB, s.RAMMiB)
}

// ---- calibration book ----

// Calib is what the loads of one (model, GPU, backend) taught us. The four first fields are the
// book as older builds wrote it, kept with their names and meaning so either build reads the
// other's state; an older build drops the rest when it writes.
type Calib struct {
	FixedMiB    float64   `json:"fixed_mib"` // the fixed part from the last reading taken; 0: none taken
	CapCtxTotal int       `json:"cap_ctx_total"`
	Loads       int       `json:"loads"` // readings taken
	Updated     time.Time `json:"updated"`
	// Lift is what lets CapCtxTotal go (Book.LetGo). Every cap this build lowers carries one; a
	// cap an older build lowered has none until the first LetGo records it.
	Lift *CapLift `json:"cap_lift,omitempty"`
	// Under is the caps this one was lowered beneath, latest last, each with its own lift: where
	// LetGo lets CapCtxTotal go it falls back to the latest of these that still stands.
	Under []HeldCap `json:"cap_under,omitempty"`
	// Readings is the card memory each shape that came up held, one per shape, whether or not
	// its reading was taken for the fixed part (Book.Mark).
	Readings []Reading `json:"readings,omitempty"`
	// Consts is the set of constants of the seed FixedMiB was worked out against (the reading less
	// that seed's other figures); nil, as an older build wrote it, is the default set.
	Consts *Consts `json:"consts,omitempty"`
}

// CapLift is what a cap was measured under: the room (Room.Budget, or Room.RAMBudget for a run
// with no GPU), how much less the cap's shape was predicted to need than the shape that ran out
// of memory (0 where that cannot be told), and the engine build. A room larger than RoomMiB by
// more than MissMiB (with a miss of 0: any larger room), or another build, lets the cap go.
type CapLift struct {
	RoomMiB int    `json:"room_mib"`
	MissMiB int    `json:"miss_mib"`
	NoGPU   bool   `json:"no_gpu,omitempty"`
	Build   string `json:"build"`
}

// HeldCap is an earlier cap kept beneath a lower one, with the lift it was measured under.
type HeldCap struct {
	CapCtxTotal int     `json:"cap_ctx_total"`
	Lift        CapLift `json:"cap_lift"`
}

// lifted reports whether room and build, as they stand now, let a cap measured under l go.
func (l CapLift) lifted(room roomFig, build string) bool {
	return l.Build != build || (l.NoGPU == room.NoGPU && room.capRoom() > l.RoomMiB+l.MissMiB)
}

// Reading is the card memory one shape held when its load came up, read by its process.
type Reading struct {
	Slots      int    `json:"slots"`
	CtxPerSlot int    `json:"ctx_per_slot"`
	NCpuMoe    int    `json:"n_cpu_moe"`
	KVType     string `json:"kv_type"`
	VRAMMiB    int    `json:"vram_mib"`
}

// of is whether the reading was taken from this shape. A cache type is compared as the type it
// runs at (runKV), so a shape that names none and one that names the default are one shape.
func (r Reading) of(s Shape) bool {
	return !s.CPUOnly && r.Slots == s.Slots && r.CtxPerSlot == s.CtxPerSlot && r.NCpuMoe == s.NCpuMoe && runKV(Shape{KVType: r.KVType}) == runKV(s)
}

// Book is the calibration book kept in state.json.
type Book struct {
	Entries map[string]Calib `json:"entries"`
}

// BookKey keys the book by model sha256, GPU name and backend id.
func BookKey(modelSHA, gpuName, backendID string) string {
	return modelSHA + "|" + gpuName + "|" + backendID
}

// Footprint overlays what the book knows (the fixed load from a reading taken, marked Measured;
// the runtime-OOM cap) on a seed. The fixed load is laid only on a seed of the set of constants it
// was worked out under (Calib.Consts): on another set's seed it would join one set's fixed part to
// the other's figures, and predict for the very shape read a figure the reading does not hold.
// There the seed's fixed part stands, Predicted, until a reading is taken against it.
func (b *Book) Footprint(key string, seed Footprint) Footprint {
	if b == nil {
		return seed
	}
	if e, ok := b.Entries[key]; ok {
		if e.FixedMiB > 0 && sameSet(e.Consts, seed.Consts) {
			seed.FixedMiB = e.FixedMiB
			seed.Basis = Measured
		}
		seed.CapCtxTotal = e.CapCtxTotal
	}
	return seed
}

// Mark is s with what the book holds of it: where key has a reading of this very shape, VRAMMiB
// is that reading and VRAMBasis Measured. Otherwise s as given.
func (b *Book) Mark(key string, s Shape) Shape {
	if b == nil {
		return s
	}
	for _, r := range b.Entries[key].Readings {
		if r.of(s) {
			s.VRAMMiB, s.VRAMBasis = r.VRAMMiB, Measured
		}
	}
	return s
}

// CalibVerdict is what Calibrate did with a reading.
type CalibVerdict string

const (
	// Taken: within 15 % of the seed's fixed part, so the book's fixed part is now the implied one.
	Taken CalibVerdict = "taken"
	// OutsideGate: more than 15 % off the seed's fixed part (others may have changed during the
	// load, or the seed is wrong for this model); the fixed part stays predicted.
	OutsideGate CalibVerdict = "outside_gate"
	// NoSeed: the seed has no fixed part to hold the reading to.
	NoSeed CalibVerdict = "no_seed"
	// NoReading: no reading was given (0 MiB or less).
	NoReading CalibVerdict = "no_reading"
	// CPUOnly: a CPU-only shape's card use says nothing of the fixed part on the card.
	CPUOnly CalibVerdict = "cpu_only"
	// NotSeed: the footprint offered is the book's overlay (Basis Measured), not the header's
	// seed, so there is no seed to hold the reading to; the fixed part does not move.
	NotSeed CalibVerdict = "not_seed"
)

// Calibration is what Calibrate did: its verdict, the fixed part the reading implies (0 where
// there was no card reading) and the header seed's fixed part it was held to (0 for NotSeed).
type Calibration struct {
	Verdict         CalibVerdict `json:"verdict"`
	ImpliedFixedMiB float64      `json:"implied_fixed_mib"`
	SeedFixedMiB    float64      `json:"seed_fixed_mib"`
}

// Taken reports whether the reading became the book's fixed part.
func (c Calibration) Taken() bool { return c.Verdict == Taken }

// OffMiB is how far the seed's fixed part is from what the reading implies (implied less seed).
func (c Calibration) OffMiB() float64 { return c.ImpliedFixedMiB - c.SeedFixedMiB }

func (c Calibration) String() string {
	if c.Verdict == NoReading || c.Verdict == CPUOnly {
		return string(c.Verdict)
	}
	if c.Verdict == NotSeed {
		return fmt.Sprintf("%s (implied fixed %.0f MiB; the book's figure was offered, not the header's seed)", c.Verdict, c.ImpliedFixedMiB)
	}
	return fmt.Sprintf("%s (implied fixed %.0f MiB, seed %.0f MiB, off %+.0f MiB)", c.Verdict, c.ImpliedFixedMiB, c.SeedFixedMiB, c.OffMiB())
}

// Calibrate records a healthy load that held measuredMiB of VRAM at shape s, and says what it did.
// seed is the header's seed (Basis Predicted), never the book's overlay: every reading is held to
// it, so a run of readings each near the last cannot walk the fixed part away from the seed. The
// overlay offered is NotSeed and moves nothing. The reading is kept for its shape (Book.Mark)
// whatever the gate says. Like the hive's calibrate() it becomes the book's fixed part only
// within +-15 % of the seed's (a wild reading means others changed during the load, or the seed
// is wrong here: the caller sees by how much). now is the stamp the entry is given. The shape is
// priced at its own cache type (settle), and a shape or seed settle refuses is that error, with
// nothing recorded.
func (b *Book) Calibrate(key string, seed Footprint, s Shape, measuredMiB int, now time.Time) (Calibration, error) {
	st, err := settle(&seed, &s, nil, nil)
	if err != nil {
		return Calibration{}, err
	}
	c := Calibration{SeedFixedMiB: seed.FixedMiB}
	switch {
	case s.CPUOnly:
		c.Verdict = CPUOnly
		return c, nil
	case measuredMiB <= 0:
		c.Verdict = NoReading
		return c, nil
	}
	c.ImpliedFixedMiB = float64(measuredMiB) - (st.fp.vram(s.Slots, s.CtxPerSlot, s.NCpuMoe) - st.fp.FixedMiB)
	switch {
	case seed.Basis == Measured:
		c.Verdict, c.SeedFixedMiB = NotSeed, 0
	case seed.FixedMiB <= 0:
		c.Verdict = NoSeed
	case math.Abs(c.ImpliedFixedMiB-seed.FixedMiB) > 0.15*seed.FixedMiB:
		c.Verdict = OutsideGate
	default:
		c.Verdict = Taken
	}
	if b.Entries == nil {
		b.Entries = map[string]Calib{}
	}
	e := b.Entries[key]
	// A new slice every time: a copy of the book handed to a writer may still hold the old one.
	rs := make([]Reading, 0, len(e.Readings)+1)
	for _, r := range e.Readings {
		if !r.of(s) {
			rs = append(rs, r)
		}
	}
	e.Readings = append(rs, Reading{Slots: s.Slots, CtxPerSlot: s.CtxPerSlot, NCpuMoe: s.NCpuMoe, KVType: s.KVType, VRAMMiB: measuredMiB})
	if c.Taken() {
		e.FixedMiB = c.ImpliedFixedMiB
		e.Consts = cloneSet(seed.Consts)
		e.Loads++
	}
	e.Updated = now
	b.Entries[key] = e
	return c, nil
}

// LowerCap records an out-of-memory of shape failed in room with engine build build: the next
// start must use less total ctx than ctxTotal. Where the cap moves it carries what lifts it
// (CapLift): the room, the miss (the memory fp predicts failed needs above the same slots at
// ctxTotal) and the build. Where the miss cannot be told (no slots, a type fp cannot be priced
// at, or nothing between them, as when the walk kept the ctx and moved experts) the miss is 0 and
// any larger room lifts it. A standing cap that carries no lift (an older book's) is given this
// room and build. now is the stamp the entry is given. The failed shape is priced at its own cache
// type (settle); a room, shape or footprint settle refuses is that error, and nothing is recorded.
func (b *Book) LowerCap(key string, fp Footprint, failed Shape, ctxTotal int, room Room, build string, now time.Time) error {
	st, err := settle(&fp, &failed, &room, nil)
	if err != nil {
		return err
	}
	rm, f := st.room, st.fp
	if b.Entries == nil {
		b.Entries = map[string]Calib{}
	}
	e := b.Entries[key]
	if ctxTotal > 0 && (e.CapCtxTotal == 0 || ctxTotal < e.CapCtxTotal) {
		if e.CapCtxTotal > 0 {
			// The cap moved from keeps its own lift beneath; a new slice, as a copy of the book
			// handed to a writer may still hold the old one.
			l := CapLift{RoomMiB: rm.capRoom(), NoGPU: rm.NoGPU, Build: build}
			if e.Lift != nil {
				l = *e.Lift
			}
			e.Under = append(append(make([]HeldCap, 0, len(e.Under)+1), e.Under...), HeldCap{e.CapCtxTotal, l})
		}
		e.CapCtxTotal = ctxTotal
		miss := 0
		if failed.Slots > 0 {
			miss = max(0, ceilMiB(f.kvAt(failed.Slots, failed.CtxPerSlot)-f.kvAt(failed.Slots, ctxTotal/failed.Slots)))
		}
		e.Lift = &CapLift{RoomMiB: rm.capRoom(), MissMiB: miss, NoGPU: rm.NoGPU, Build: build}
	}
	if e.CapCtxTotal > 0 && e.Lift == nil {
		e.Lift = &CapLift{RoomMiB: rm.capRoom(), NoGPU: rm.NoGPU, Build: build}
	}
	e.Updated = now
	b.Entries[key] = e
	return nil
}

// LetGo lets the cap of key go where room, as it stands now, is larger than the room it was
// measured under by more than its miss (the same kind of room: card, or host with no GPU), or
// build is another than the one it was measured with. It answers whether a cap still stands and,
// where one does, what would lift it, never nothing. A cap an older book holds recorded nothing it
// was measured under: the first LetGo records room and build as they stand then (miss 0), so from
// there any larger room or another build lifts it, and until then it stands as it always did.
// A cap let go takes its lift with it and the latest earlier cap (Calib.Under) that still stands
// takes its place, with its own lift, so a shape that ran out of memory in a room never comes back
// in that room; those lifted go too. With no cap left: false, nil. The fixed part and the readings
// stay. A room with a field marked unread is *UnreadError (settle), and nothing is let go or
// recorded.
func (b *Book) LetGo(key string, room Room, build string) (stands bool, lift *CapLift, err error) {
	st, err := settle(nil, nil, &room, nil)
	if err != nil {
		return false, nil, err
	}
	rm := st.room
	if b == nil {
		return false, nil, nil
	}
	e, ok := b.Entries[key]
	if !ok || e.CapCtxTotal <= 0 {
		return false, nil, nil
	}
	l := e.Lift
	if l == nil {
		l = &CapLift{RoomMiB: rm.capRoom(), NoGPU: rm.NoGPU, Build: build}
		e.Lift = l
		b.Entries[key] = e
	}
	if !l.lifted(rm, build) {
		c := *l
		return true, &c, nil
	}
	e.CapCtxTotal, e.Lift = 0, nil
	for i := len(e.Under) - 1; i >= 0; i-- {
		if h := e.Under[i]; !h.Lift.lifted(rm, build) {
			c := h.Lift
			e.CapCtxTotal, e.Lift = h.CapCtxTotal, &c
			e.Under = e.Under[:i:i] // capped: a later LowerCap never writes into a slice a copy holds
			b.Entries[key] = e
			return true, &h.Lift, nil
		}
	}
	e.Under = nil
	b.Entries[key] = e
	return false, nil, nil
}
