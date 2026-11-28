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
// and RAM, unless the footprint carries Split (WithSplit): then every layer on the CPU and the
// processor alone were tried too.
var ErrTooBig = errors.New("fit: model does not fit this machine")

// ErrSplitMoE means WithSplit was asked of a mixture-of-experts model: its experts leave the card
// by --n-cpu-moe, never by whole layers. Since the dense split (in no core tag yet).
var ErrSplitMoE = errors.New("fit: a mixture-of-experts model is never split by whole layers")

// ErrNoBlocks means WithSplit was given a header without the bytes of each block
// (gguf.Info.BlockBytes, one per layer, not all of them empty). Since the dense split (in no core tag yet).
var ErrNoBlocks = errors.New("fit: the header gives no bytes per block")

// ErrSplitRecurrent means WithSplit was given a header with recurrent (linear-attention or SSM)
// layers: their fixed per-slot state would have to be placed by layer as llama.cpp places it, which
// the split does not do yet, so such a model is never split. Since the dense split (in no core tag yet).
var ErrSplitRecurrent = errors.New("fit: a model with recurrent layers is never split by whole layers")

// ErrNoSplit means a shape with layers on the CPU (Shape.CPULayers) was priced from a footprint
// that does not allow it (no Split) or that has fewer layers than the shape moves. It is never
// priced as a shape on the card. Since the dense split (in no core tag yet).
var ErrNoSplit = errors.New("fit: the shape moves layers the footprint cannot split")

// SplitShortError means the card holds the model with every layer on the CPU but the output head
// (the least split, Footprint.Split) and the machine's memory does not hold those layers, and the
// processor alone does not fit either: NeedMiB is the host memory of the least split at the least
// shape (minimum slots at the floor context), FreeMiB what stands for it (RAMAvail less the
// reserve, or less where Room.CardHostFree holds the start), Basis the basis of NeedMiB. It is
// ErrTooBig to errors.Is. Only a footprint with Split answers it. Since the dense split (in no core tag yet).
type SplitShortError struct {
	NeedMiB int   `json:"need_mib"`
	FreeMiB int   `json:"free_mib"`
	Basis   Basis `json:"basis"`
}

func (e *SplitShortError) Error() string {
	return fmt.Sprintf("fit: the card holds the model with its layers on the CPU, and they need about %d MiB of the machine's memory (%s), %d MiB free after the OS reserve", e.NeedMiB, e.Basis, e.FreeMiB)
}

// Is makes a SplitShortError ErrTooBig to errors.Is.
func (e *SplitShortError) Is(target error) bool { return target == ErrTooBig }

// ErrNoCtxTrain means Wants.CtxFromModel was asked and the footprint carries no trained context
// (its header did not say it, or it was not built from a header). It is never guessed.
var ErrNoCtxTrain = errors.New("fit: the model's trained context is not known")

// ErrCtxTrainUnder means Wants.CtxFromModel was asked and the model's trained context is under
// the core's least context per slot (1024 tokens): no shape is planned above what it was trained for.
var ErrCtxTrainUnder = errors.New("fit: the model's trained context is under the least context per slot (1024)")

// ErrNoKVType means Wants.KVTypes was given and the footprint says no cache type its per-token
// costs were priced at (one built from measured numbers), so it cannot be priced at another.
var ErrNoKVType = errors.New("fit: the footprint names no cache type to price other types from")

// ErrNoWeights means Room.LargestBlock was given as a size (above 0) and the footprint does not know its weights on
// the card (Footprint.WeightsMiB), so its largest single allocation cannot be told.
var ErrNoWeights = errors.New("fit: the footprint cannot tell its largest allocation (no weights figure)")

// ErrNoHostFigure means Room.CardHostFree was given and the footprint cannot tell the machine's
// memory a start on the card takes (Footprint.HostBaseMiB is 0: a footprint from measured numbers,
// from SeedFromBytes, which reads no header, or seeded with a set lacking HostBaseMiB), so the
// start cannot be held to it. Since core v0.1.4. The figure is never taken as zero.
var ErrNoHostFigure = errors.New("fit: the footprint cannot tell the host memory of a start on the card")

// HostShortError means a start on the card takes more of the machine's memory than the room has
// free for it (Room.CardHostFree less the reserve): NeedMiB is the least host memory any start of
// the model on the card takes (no expert layer moved, at any ctx: Shape.HostMiB), FreeMiB what
// stands, Basis the basis of NeedMiB (Predicted from every planning entry). It is ErrTooBig to
// errors.Is, so a caller that only tells too big from the rest reads it as before; errors.As
// gives the figures. Since core v0.1.4.
type HostShortError struct {
	NeedMiB int   `json:"need_mib"`
	FreeMiB int   `json:"free_mib"`
	Basis   Basis `json:"basis"`
}

func (e *HostShortError) Error() string {
	return fmt.Sprintf("fit: a start on the card needs about %d MiB of the machine's memory (%s), %d MiB free after the OS reserve", e.NeedMiB, e.Basis, e.FreeMiB)
}

// Is makes a HostShortError ErrTooBig to errors.Is.
func (e *HostShortError) Is(target error) bool { return target == ErrTooBig }

// UnreadError means the caller marked a room field as not read (Room.Unread): it is never taken
// as zero, and no shape is chosen until it is read.
type UnreadError struct{ Field string }

func (e *UnreadError) Error() string { return "fit: room field " + e.Field + " was not read" }

// KVTypeError means a cache type name the core has no price for. It is never priced as another
// type; the empty name is not unknown, it means the default (q4_0).
type KVTypeError struct{ Name string }

func (e *KVTypeError) Error() string { return fmt.Sprintf("fit: unknown cache type %q", e.Name) }

// FigureError means a footprint's per-token figure (Name) is one no footprint can hold (Rule): a
// price from it would fall as ctx grows. It is never clamped and never priced; figures that hold
// lift it.
type FigureError struct{ Name, Rule string }

func (e *FigureError) Error() string { return "fit: the footprint's " + e.Name + " " + e.Rule }

// Basis says where a memory figure came from. Predicted: worked out from the header's seed and
// the constants above (or from the seed corrected where the book took a reading). Measured: read
// from a load of this model, card and backend that came up. "" is not said: a figure built by hand.
type Basis string

const (
	Predicted Basis = "predicted"
	Measured  Basis = "measured"
	// Unknown: the core cannot tell the figure, and it is 0 only as a placeholder, never a size.
	// Only Shape.HostBasis takes it (since core v0.1.4).
	Unknown Basis = "unknown"
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
	// HostBaseMiB: the machine's memory a start on the card holds beyond the weights the header
	// says stay in RAM (Footprint.HostMiB) and the expert layers moved there: the runtime, the
	// input embedding and the host compute buffers, read as the process's resident memory. A set
	// may lack it (it is not in constNames): the host memory of a card start is then not known
	// (Footprint.HostBaseMiB 0). It has no part in the card's fixed part, so the book does not key
	// on it (sameSet). Since core v0.1.4.
	HostBaseMiB ConstName = "host_base_mib"
)

// constNames is every constant a set must hold, in the order a lacking one is named. HostBaseMiB
// is not among them: a set may lack it.
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
		HostBaseMiB:     {1536, "RTX 5080 at b12680: a card start's resident memory less its host-resident weights, read as 974 to 1520 MiB over the four Gemma E-series sweeps (rss_used less host_mib, every row), taken above the highest"},
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
	CtxMax   int    `json:"ctx_max"`   // per slot
	CtxFloor int    `json:"ctx_floor"` // per slot; held exactly, on the step of context or not (Norm)
	KVType   string `json:"kv_type"`
	// MinTPS is the decode speed floor in tokens/s per stream (default 20, negative = no floor).
	// Where the footprint has a measured speed curve, the planner keeps the expert layers moved to
	// RAM few enough to stay above it, before it keeps max ctx.
	MinTPS float64 `json:"min_tps"`

	// Whole: no layer and no expert off the card. Where only a split shape fits, ErrTooBig.
	// Unset: expert layers may move to RAM (section 5, step 3).
	Whole bool `json:"whole"`
	// KVTypes is the cache types, best first: the shape takes the first type at which CtxMax fits
	// at Slots; where none does, the last type with the most ctx that fits down to the floor. The
	// core orders the list itself, most bytes per element first (Norm), so any order means the same.
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
	// Negative: the card was read and has no free block left, so no shape on it can be held and
	// every entry that bounds a shape by the block answers ErrTooBig (see roomFig.noBlock). Not
	// read with NoGPU; read with Unified as on any card, the card's own memory being the RAM.
	LargestBlock int `json:"largest_block"`
	// Unread names the fields (by their json names) the caller could not read: none is taken as
	// zero, and every entry that reads a room answers *UnreadError naming the first (settle).
	Unread []string `json:"unread,omitempty"`
	// SpeedScale is how fast this machine decodes compared with the machine the speed curve was
	// measured on (1 = the same card). 0 means 1. The caller sets it per model (the engine's
	// planner from where the catalog's curve was measured).
	SpeedScale float64 `json:"speed_scale"`
	GPUName    string  `json:"gpu_name"`
	// CardHostFree is the machine's free memory (MiB) a start on the card is held to: the start's
	// host memory (Shape.HostMiB) must fit in it less the reserve (Reserve), or the entries that plan
	// (Choose, MissRetry, StepDown, Grow) answer *HostShortError (ErrTooBig to errors.Is), and no
	// expert layer is moved past it. Negative: read, and none free. A footprint that cannot tell the
	// host memory (Footprint.HostBaseMiB 0) is ErrNoHostFigure where it is given. Not read with
	// NoGPU (a CPU-only run is held to RAMAvail). 0: not read, and a card start is held to nothing
	// but RAMAvail for its expert layers, as before. Since core v0.1.4.
	CardHostFree int `json:"card_host_free,omitempty"`
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
	// HostBaseMiB is the machine's memory a start on the card holds beyond HostMiB and the moved
	// expert layers (the set's HostBaseMiB, which Seed and SeedWith carry over). 0: not known (a
	// footprint built by hand, from measured numbers or by an older build; SeedFromBytes, which
	// reads no header and so cannot tell HostMiB either; a set lacking the constant): the host
	// memory of a card start is then Unknown (Shape.HostBasis), never priced as zero. Since core v0.1.4.
	HostBaseMiB float64 `json:"host_base_mib,omitempty"`

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
	// quantised cache adds on a card (the f16 expansion of the widest layer), whatever KVType is;
	// a shape that lays nothing on a card is priced without it (fig.perToken), and the figures
	// always keep it.
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
	// Split, where set, allows a dense model's layers to leave the card (llama-server -ngl), and
	// says what each one weighs: it is both the permission and the figures, so neither can be had
	// without the other. nil (every footprint before the dense split, and every one the caller does not
	// mark): never split, and every entry plans exactly as before. Only WithSplit sets it from a
	// header; a MoE footprint never carries it (*FigureError). Since the dense split (in no core tag yet).
	Split *Split `json:"split,omitempty"`
}

// Split is a dense footprint's layers as llama.cpp moves them with -ngl N: the last N-1 repeating
// blocks and the output head stay on the card, the first blocks go to the CPU with their KV cache
// (llama.cpp b12651 src/llama-model.cpp 1816-1841, src/llama-kv-cache.cpp 214-225). Each slice is
// one entry per block, block 0 first. Since the dense split (in no core tag yet).
type Split struct {
	// LayerMiB is the weights of each block (gguf.Info.BlockBytes).
	LayerMiB []float64 `json:"layer_mib"`
	// FullShare and SWAShare are each block's part of Footprint.PerTokenMiB and SWAPerTokenMiB (a
	// block without its own cache has none; each sums to 1, or 0 where the footprint has no such
	// cache). Shares, not MiB, so the cache is priced at any type as the footprint is.
	FullShare []float64 `json:"full_share"`
	SWAShare  []float64 `json:"swa_share"`
	// OutputMiB is the output head, on the card at every split (it is inside FixedMiB already);
	// kept so a reading can be held to it. Not priced.
	OutputMiB float64 `json:"output_mib"`
	// SchedMiB is the card memory any split takes beyond its layers: CPU-layer weights copied to the
	// card for batch work (op offload, ggml-backend.cpp 1394-1402) and the second graph split.
	// HostMiB is the machine's memory any split takes beyond its layers and their cache: the CPU
	// compute buffer and pinned staging. HostFactor is the machine's memory per MiB of a block on the
	// CPU (0 = 1).
	SchedMiB   float64 `json:"sched_mib"`
	HostMiB    float64 `json:"host_mib"`
	HostFactor float64 `json:"host_factor"`
	// Basis is of SchedMiB, HostMiB and HostFactor: Predicted from WithSplit's seed margins, which
	// no card has measured yet (splitSchedSeedMiB, splitHostSeedMiB).
	Basis Basis `json:"basis,omitempty"`
}

// The seed margins of a split, before any card has measured one: the card memory and the
// machine's memory a split takes beyond its layers and their cache. Generous on purpose (the
// op-offload copies depend on graph splits no header tells); measured, they become a set's
// constants. See docs in ~/.local/share/ee-densesplit/DESIGN.md section 3.
const (
	splitSchedSeedMiB = 256
	splitHostSeedMiB  = 512
)

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
	// holds a card reading of this very shape and VRAMMiB is that reading.
	VRAMBasis Basis `json:"vram_basis,omitempty"`
	// RAMBasis is of RAMMiB: Predicted from the planner; Measured from Book.Mark only for a
	// CPU-only shape, where the book holds a host reading of this very shape and RAMMiB is that
	// reading. A card shape's RAMMiB (its expert layers in RAM) is never read, so always predicted.
	// "" (a shape built by hand, or by an older build) says nothing.
	RAMBasis Basis `json:"ram_basis,omitempty"`
	// HostMiB is the machine's memory the start holds in all, as its process's resident memory: for
	// a card shape, the footprint's HostBaseMiB, its host-resident weights (HostMiB) and the expert
	// layers moved to RAM (RAMMiB), the same at any ctx; for a CPU-only shape, RAMMiB. HostBasis is
	// of it: Predicted from the planner; Measured from Book.Mark, where the book holds a reading of
	// this very shape that carries the process's resident memory (Book.CalibrateCard; any host
	// reading of a CPU-only shape); Unknown where the footprint cannot tell it (Footprint.HostBaseMiB
	// 0), and HostMiB is then 0. "" (a shape built by hand, or by an older build) says nothing.
	// Room.CardHostFree holds a card start to it. Since core v0.1.4.
	HostMiB   int   `json:"host_mib,omitempty"`
	HostBasis Basis `json:"host_basis,omitempty"`
	// ConstsBackend and ConstsCard name the set of constants the predicted figures were built
	// with (Consts.Backend, Consts.Card); where they are not the build and card it runs on, the
	// figures were measured elsewhere.
	ConstsBackend string `json:"consts_backend,omitempty"`
	ConstsCard    string `json:"consts_card,omitempty"`
	// CPULayers is how many repeating layers (the first ones) run on the CPU with their cache; 0:
	// every layer on the card, as every shape before the dense split. Layers is the model's repeating
	// layer count, set where CPULayers is above 0 and on a processor-alone shape of a footprint
	// with Split: Args renders -ngl from it. Only a footprint with Split plans them. Since the dense split (in no
	// core tag yet).
	CPULayers int `json:"cpu_layers,omitempty"`
	Layers    int `json:"layers,omitempty"`
}

// CtxTotal is the -c value.
func (s Shape) CtxTotal() int { return s.Slots * s.CtxPerSlot }

// Norm is w with every default filled in and every bound made consistent: what Choose plans with.
// KVTypes is taken once and best first (each type once, empty names dropped, most bytes per
// element first; since core v0.1.4).
// CtxMax is taken down to the step of context (1024 tokens); CtxFloor is kept exactly as asked,
// on the step or not (never under 1024), and where it is above CtxMax, CtxMax is the floor. So a
// shape planned from it never holds less context per slot than the floor asked.
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
	w.KVTypes = kvTypes(w.KVTypes)
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
	var full, swa, maxFull float64
	for _, l := range seedLayers(info) {
		e, isSWA := layerKV(info, l)
		if isSWA {
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
		HostBaseMiB:    k.hostBase,
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

// seedLayers is the header's layers, or where it does not give one per block, layers built from
// its scalars (the first SWALayers sliding-window).
func seedLayers(info gguf.Info) []gguf.Layer {
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
	return layers
}

// layerKV is one layer's cache elements per token and whether they are sliding-window ones; 0
// for a layer without its own cache.
func layerKV(info gguf.Info, l gguf.Layer) (float64, bool) {
	if !l.HasKV {
		return 0, false
	}
	return float64(l.KVHeads * (l.HeadDimK + l.HeadDimV)), l.SWA && info.SlidingWindow > 0
}

// WithSplit is fp allowed to move its layers off the card (Footprint.Split), with each block's
// weights and part of the cache read from the header info fp was seeded from, and the seed
// margins (Basis Predicted, not measured on a card). It is the only way a footprint is split: a
// caller marks the models it may split, and every other footprint plans as before. A MoE model is
// ErrSplitMoE; a header without the bytes of each block ErrNoBlocks; figures no footprint can
// hold *FigureError. fp itself is not changed. Since the dense split (in no core tag yet).
func WithSplit(fp Footprint, info gguf.Info) (Footprint, error) {
	if fp.MoE || info.NExpert > 0 || info.ExpertBytes > 0 {
		return Footprint{}, ErrSplitMoE
	}
	if info.StateBytesPerSlot > 0 || slices.ContainsFunc(info.Layers, func(l gguf.Layer) bool { return l.Recurrent }) {
		return Footprint{}, ErrSplitRecurrent // its per-slot state is not placed by layer here
	}
	if info.NLayer <= 0 || len(info.BlockBytes) != info.NLayer || !slices.ContainsFunc(info.BlockBytes, func(b int64) bool { return b > 0 }) {
		return Footprint{}, ErrNoBlocks // no figures, or none naming a byte: never blocks taken as weightless
	}
	layers := seedLayers(info)
	sp := Split{LayerMiB: make([]float64, info.NLayer), FullShare: make([]float64, info.NLayer),
		SWAShare: make([]float64, info.NLayer), OutputMiB: float64(info.OutputBytes) / mib,
		SchedMiB: splitSchedSeedMiB, HostMiB: splitHostSeedMiB, HostFactor: 1, Basis: Predicted}
	var full, swa float64
	for i, l := range layers {
		e, isSWA := layerKV(info, l)
		if isSWA {
			sp.SWAShare[i], swa = e, swa+e
		} else {
			sp.FullShare[i], full = e, full+e
		}
	}
	for i := range sp.LayerMiB {
		sp.LayerMiB[i] = float64(info.BlockBytes[i]) / mib
		if full > 0 {
			sp.FullShare[i] /= full
		}
		if swa > 0 {
			sp.SWAShare[i] /= swa
		}
	}
	fp.Split = &sp
	if err := coherent(fp); err != nil {
		return Footprint{}, err
	}
	return fp, nil
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
	hostBase                                                float64 // 0: the set lacks HostBaseMiB
}

// sameSet reports whether two sets hold the same build, card and constants (nil is the default
// set); a word of where a reading came from does not change a price, so it is not compared. Only
// the constants every set holds (constNames) are compared: the others (HostBaseMiB) have no part in
// the card's fixed part the book keys on them, and a book an older build wrote lacks them.
func sameSet(a, b *Consts) bool {
	d := DefaultConsts()
	if a == nil {
		a = &d
	}
	if b == nil {
		b = &d
	}
	if a.Backend != b.Backend || a.Card != b.Card {
		return false
	}
	for _, n := range constNames {
		x, okx := a.Of[n]
		y, oky := b.Of[n]
		if okx != oky || x.Value != y.Value {
			return false
		}
	}
	return true
}

// bookSet is the set as the book keeps it beside a fixed part: a copy holding only the constants
// every set holds (constNames), so an older build reading it finds the set it knows.
func bookSet(c *Consts) *Consts {
	if c == nil {
		return nil
	}
	n := *c
	n.Of = make(map[ConstName]Const, len(constNames))
	for _, k := range constNames {
		if v, ok := c.Of[k]; ok {
			n.Of[k] = v
		}
	}
	return &n
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
	var hostBase float64
	if c, ok := set.Of[HostBaseMiB]; ok { // a set may lack it; where it holds it, it holds its word
		if c.From == "" {
			return constFig{}, &ConstError{Name: HostBaseMiB, NoFrom: true}
		}
		hostBase = max(c.Value, 0)
	}
	return constFig{backend: set.Backend, card: set.Card,
		base: v[BaseOverheadMiB], vocab: v[VocabMiBPer1k], runtime: v[RuntimeMiB], mask: v[MaskMiBPer1k],
		dequant: v[DequantBytes], offload: v[OffloadMiB], expertRAM: v[ExpertRAMFactor], hostBase: hostBase}, nil
}

// roomFig is a room every field of which was read, with its budgets in MiB.
type roomFig struct {
	Room
	budget float64 // card a shape may take: VRAMTotal less Others, GameReserve and the margin
	ram    float64 // host RAM a CPU-only run or the moved expert layers may take: RAMAvail less the reserve
	// host is the machine's memory a start on the card may hold in all (CardHostFree less the
	// reserve) where held is true: CardHostFree given, on a card.
	host float64
	held bool
}

// hostOK reports whether a card start with n expert layers moved fits the machine's free memory
// the room holds it to (always, where the room holds it to none).
func (fp fig) hostOK(room roomFig, n int) bool {
	return !room.held || fp.cardHost(n) <= room.host
}

// hostShort is the refusal of a room whose free memory no card start of fp fits: the least such
// start, with no expert moved. nil where one fits or the room holds starts to nothing.
func (fp fig) hostShort(room roomFig) error {
	if fp.hostOK(room, 0) {
		return nil
	}
	return &HostShortError{NeedMiB: ceilMiB(fp.cardHost(0)), FreeMiB: int(math.Floor(room.host)), Basis: Predicted}
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
// footprint cannot be held to (ErrNoWeights), then a free host memory it cannot be held to
// (ErrNoHostFigure: Room.CardHostFree given, Footprint.HostBaseMiB 0); wants it cannot resolve (ErrNoCtxTrain,
// ErrCtxTrainUnder, *KVTypeError for KVType or any of KVTypes); a shape whose cache type has no
// price (*KVTypeError); a type the footprint cannot be priced at (ErrNoKVType, *KVTypeError);
// figures no footprint can hold (*FigureError, see coherent); a footprint whose set of constants
// lacks one (*ConstError: nil is the default set).
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
		// A negative block (none free) is not held to the weights: it bounds every layout on the
		// card whether or not the footprint knows its weights, so Choose answers it ErrTooBig
		// (roomFig.noBlock), never ErrNoWeights, after the wants' and figures' own refusals.
		if w != nil && room.LargestBlock > 0 && !room.NoGPU && fp.WeightsMiB <= 0 {
			return st, ErrNoWeights
		}
		if w != nil && room.CardHostFree != 0 && !room.NoGPU && fp.HostBaseMiB <= 0 {
			return st, ErrNoHostFigure
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
			ram:    float64(room.RAMAvail - max(r, 0)),
			host:   float64(room.CardHostFree - max(r, 0)),
			held:   room.CardHostFree != 0 && !room.NoGPU}
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
	if s != nil && !s.CPUOnly && s.CPULayers != 0 && (fp.Split == nil || s.CPULayers < 0 || s.CPULayers > len(fp.Split.LayerMiB)) {
		return st, ErrNoSplit
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
	if err := coherent(*fp); err != nil {
		return st, err
	}
	if f.c, err = readConsts(fp.Consts); err != nil {
		return st, err
	}
	st.fp = f
	return st, nil
}

// coherent refuses figures no footprint can hold. The invariant, which price keeps at every type:
// each per-token figure a price is made from (cache, sliding-window cache, compute, expansion) is a
// finite number not below zero, with or without a type; and for figures held at a quantised type
// the expansion is not above the compute that holds it (at another type the compute holds none of
// it). Zero is a model with no such term. It prices nothing; only settle calls it.
func coherent(fp Footprint) error {
	for _, f := range []struct {
		name string
		v    float64
	}{{"PerTokenMiB", fp.PerTokenMiB}, {"SWAPerTokenMiB", fp.SWAPerTokenMiB},
		{"ComputePerTokenMiB", fp.ComputePerTokenMiB}, {"DequantPerTokenMiB", fp.DequantPerTokenMiB}} {
		if math.IsNaN(f.v) || math.IsInf(f.v, 0) {
			return &FigureError{Name: f.name, Rule: "is not a finite number"}
		}
		if f.v < 0 {
			return &FigureError{Name: f.name, Rule: "is below zero: a token never costs less than none"}
		}
	}
	if fp.KVType != "" && quantised(fp.KVType) && fp.DequantPerTokenMiB > fp.ComputePerTokenMiB {
		return &FigureError{Name: "DequantPerTokenMiB", Rule: "is above ComputePerTokenMiB: at a quantised type the compute holds the expansion"}
	}
	return coherentSplit(fp)
}

// coherentSplit refuses split figures no footprint can hold: a split of a MoE model, no layers,
// a share or weight that is not a finite number at or above zero, shares of a cache summing past
// all of it, or layers weighing more than the weights. nil without Split.
func coherentSplit(fp Footprint) error {
	sp := fp.Split
	if sp == nil {
		return nil
	}
	if fp.MoE {
		return &FigureError{Name: "Split", Rule: "is set on a mixture-of-experts footprint: its experts leave the card by --n-cpu-moe"}
	}
	n := len(sp.LayerMiB)
	if n == 0 {
		return &FigureError{Name: "Split.LayerMiB", Rule: "is empty: a split moves layers"}
	}
	if len(sp.FullShare) != n || len(sp.SWAShare) != n {
		return &FigureError{Name: "Split.FullShare", Rule: "is not one share per layer"}
	}
	bad := func(v float64) bool { return math.IsNaN(v) || math.IsInf(v, 0) || v < 0 }
	var weights float64
	for _, f := range []struct {
		name string
		vs   []float64
	}{{"Split.LayerMiB", sp.LayerMiB}, {"Split.FullShare", sp.FullShare}, {"Split.SWAShare", sp.SWAShare}} {
		var sum float64
		for _, v := range f.vs {
			if bad(v) {
				return &FigureError{Name: f.name, Rule: "holds a figure that is not a finite number at or above zero"}
			}
			sum += v
		}
		if f.name == "Split.LayerMiB" {
			weights = sum
		} else if sum > 1+1e-9 {
			return &FigureError{Name: f.name, Rule: "sums past the whole cache"}
		}
	}
	for _, f := range []struct {
		name string
		v    float64
	}{{"Split.OutputMiB", sp.OutputMiB}, {"Split.SchedMiB", sp.SchedMiB}, {"Split.HostMiB", sp.HostMiB}, {"Split.HostFactor", sp.HostFactor}} {
		if bad(f.v) {
			return &FigureError{Name: f.name, Rule: "is not a finite number at or above zero"}
		}
	}
	held := fp.WeightsMiB
	if held <= 0 {
		held = fp.FixedMiB
	}
	if weights > held+1e-6 {
		return &FigureError{Name: "Split.LayerMiB", Rule: "weighs more than the footprint's weights on the card"}
	}
	return nil
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

// noBlock reports whether the card was read and has no free block (Room.LargestBlock negative):
// then no shape that lays anything on it is held. A NoGPU room has no card and its block is not
// read. A Unified room is a card like any other here: its block is the largest buffer the GPU can
// still be given out of the shared memory, and a caller that read none has said that no layout
// on the GPU can be held, the same fact whether that memory is also the host's or not.
func (room roomFig) noBlock() bool { return !room.NoGPU && room.LargestBlock < 0 }

// capRoom is the room a cap is measured against: the card's budget, or host RAM with no GPU.
func (room roomFig) capRoom() int {
	if room.NoGPU {
		return int(room.ram)
	}
	return int(room.budget)
}

// perToken is the price in MiB of one token of total context for a shape, cpuOnly where it lays
// nothing on a card: the full-attention cache, and the compute scratch that grows with ctx. It is
// the one place that price is said; every price, rate and count of tokens takes it from here. A
// quantised cache's f16 expansion is laid only on a card, so a CPU-only shape is priced without it;
// measured figures (no type) are taken as given. It must never alter the figures, which keep the
// term for any card shape priced from them.
func (fp fig) perToken(cpuOnly bool) (cache, compute float64) {
	compute = fp.ComputePerTokenMiB
	if cpuOnly && fp.KVType != "" && quantised(fp.KVType) {
		compute -= fp.DequantPerTokenMiB
	}
	return fp.PerTokenMiB, compute
}

// kvAt returns the KV cache MiB for a layout, cpuOnly where it lays nothing on a card.
func (fp fig) kvAt(slots, ctxPerSlot int, cpuOnly bool) float64 {
	cache, compute := fp.perToken(cpuOnly)
	kv := (cache+compute)*float64(slots*ctxPerSlot) + fp.StatePerSlotMiB*float64(slots)
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
	v := fp.FixedMiB + fp.kvAt(slots, ctxPerSlot, false) + fp.ScratchMiB - fp.ExpertMiBPerLayer*float64(nCpuMoe)
	if nCpuMoe > 0 {
		v += fp.MoESchedMiB
	}
	return v
}

// largest is the largest single allocation on the card of a layout, each of these taken as one
// allocation: the weights left on the card, the cache (full-attention, sliding-window and
// recurrent state together) and the compute buffers (the fixed load beyond the weights plus the
// scratch that grows with ctx). It needs WeightsMiB, and prices a card layout only.
func (fp fig) largest(slots, ctxPerSlot, nCpuMoe int) float64 {
	weights := fp.WeightsMiB - fp.ExpertMiBPerLayer*float64(nCpuMoe)
	_, grows := fp.perToken(false)
	scratch := grows * float64(slots*ctxPerSlot)
	cache := fp.kvAt(slots, ctxPerSlot, false) - scratch
	compute := math.Max(fp.FixedMiB-fp.WeightsMiB, 0) + scratch
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

// allHost is the RAM use of a CPU-only run, which lays nothing on a card.
func (fp fig) allHost(slots, ctxPerSlot int) float64 {
	return fp.FixedMiB + fp.HostMiB + fp.kvAt(slots, ctxPerSlot, true) + fp.ScratchMiB
}

// layers is the count of repeating layers a split footprint can move; 0 without Split.
func (fp fig) layers() int {
	if fp.Split == nil {
		return 0
	}
	return len(fp.Split.LayerMiB)
}

// splitOn reports whether a plan in room under w may move fp's layers off the card: a footprint
// with Split, on a card that is not a pool of system memory, with the model not asked whole.
func (fp fig) splitOn(room roomFig, w Wants) bool {
	return fp.Split != nil && !fp.MoE && !room.NoGPU && !room.Unified && !w.Whole
}

// cpuWeights is the weights of the first c layers, the ones -ngl leaves on the CPU.
func (fp fig) cpuWeights(c int) float64 {
	var v float64
	for i := 0; i < c && i < fp.layers(); i++ {
		v += fp.Split.LayerMiB[i]
	}
	return v
}

// cpuFullShare and cpuSWAShare are the first c layers' parts of the full-attention and
// sliding-window cache.
func (fp fig) cpuFullShare(c int) float64 {
	var v float64
	for i := 0; i < c && i < fp.layers(); i++ {
		v += fp.Split.FullShare[i]
	}
	return v
}

func (fp fig) cpuSWAShare(c int) float64 {
	var v float64
	for i := 0; i < c && i < fp.layers(); i++ {
		v += fp.Split.SWAShare[i]
	}
	return v
}

// cpuCache is the cache the first c layers keep on the CPU at a layout: their share of the
// full-attention and sliding-window cache. The compute scratch stays on the card.
func (fp fig) cpuCache(slots, ctxPerSlot, c int) float64 {
	full, swa := fp.cpuFullShare(c), fp.cpuSWAShare(c)
	kv := fp.PerTokenMiB * full * float64(slots*ctxPerSlot)
	if fp.SWAPerTokenMiB > 0 {
		w := ctxPerSlot
		if fp.SWAWindow > 0 && fp.SWAWindow < w {
			w = fp.SWAWindow
		}
		kv += fp.SWAPerTokenMiB * swa * float64(w*slots)
	}
	return kv
}

// splitVRAM is the card memory of a layout with c layers on the CPU: the whole layout less those
// layers and their cache, plus the split's own card cost. c 0 is vram with no expert moved.
func (fp fig) splitVRAM(slots, ctxPerSlot, c int) float64 {
	v := fp.vram(slots, ctxPerSlot, 0)
	if c <= 0 {
		return v
	}
	return v - fp.cpuWeights(c) - fp.cpuCache(slots, ctxPerSlot, c) + fp.Split.SchedMiB
}

// splitRAM is the machine's memory the c layers on the CPU take: their weights (times
// HostFactor), their cache and the split's own host cost. It grows with ctx. 0 for c 0.
func (fp fig) splitRAM(slots, ctxPerSlot, c int) float64 {
	if c <= 0 || fp.Split == nil {
		return 0
	}
	f := fp.Split.HostFactor
	if f <= 0 {
		f = 1
	}
	return f*fp.cpuWeights(c) + fp.cpuCache(slots, ctxPerSlot, c) + fp.Split.HostMiB
}

// splitLargest is largest for a layout with c layers on the CPU: the weights and cache left on the
// card, and the compute buffers.
func (fp fig) splitLargest(slots, ctxPerSlot, c int) float64 {
	weights := fp.WeightsMiB - fp.cpuWeights(c)
	_, grows := fp.perToken(false)
	scratch := grows * float64(slots*ctxPerSlot)
	cache := fp.kvAt(slots, ctxPerSlot, false) - scratch - fp.cpuCache(slots, ctxPerSlot, c)
	compute := math.Max(fp.FixedMiB-fp.WeightsMiB, 0) + scratch
	return math.Max(weights, math.Max(cache, compute))
}

// splitHostRoom is the machine's memory the layers on the CPU may take: ram, and where the room
// holds a card start to its free memory, no more than it leaves beside the start's own.
func (fp fig) splitHostRoom(room roomFig, ram float64) float64 {
	if room.held {
		return math.Min(ram, room.host-fp.cardHost(0))
	}
	return ram
}

// cardHost is the machine's memory a start on the card holds with nCpuMoe expert layers moved:
// the process's base, the host-resident weights and the moved layers. Only where hostKnown.
func (fp fig) cardHost(nCpuMoe int) float64 {
	return fp.HostBaseMiB + fp.HostMiB + fp.expertRAM(nCpuMoe)
}

// hostKnown reports whether the footprint can tell the host memory of a card start.
func (fp fig) hostKnown() bool { return fp.HostBaseMiB > 0 }

// HostNeed is the predicted machine memory in MiB a start of shape s holds, priced at its own
// cache type: for a card shape the footprint's HostBaseMiB, its host-resident weights and the
// expert layers s moves to RAM (Shape.HostMiB); for a CPU-only shape the whole run's host memory
// (as Need). ErrNoHostFigure where the footprint cannot tell it for a card shape
// (Footprint.HostBaseMiB 0), never 0; and settle's refusals. Since core v0.1.4.
func (fp Footprint) HostNeed(s Shape) (float64, error) {
	st, err := settle(&fp, &s, nil, nil)
	if err != nil {
		return 0, err
	}
	if s.CPUOnly {
		return st.fp.allHost(s.Slots, s.CtxPerSlot), nil
	}
	if !st.fp.hostKnown() {
		return 0, ErrNoHostFigure
	}
	return st.fp.cardHost(s.NCpuMoe) + st.fp.splitRAM(s.Slots, s.CtxPerSlot, s.CPULayers), nil
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
	if s.CPULayers > 0 {
		return st.fp.splitVRAM(s.Slots, s.CtxPerSlot, s.CPULayers), st.fp.splitRAM(s.Slots, s.CtxPerSlot, s.CPULayers), nil
	}
	return st.fp.vram(s.Slots, s.CtxPerSlot, s.NCpuMoe), st.fp.expertRAM(s.NCpuMoe), nil
}

// Largest is the predicted largest single allocation in MiB of shape s on the card (see
// Room.LargestBlock), priced at its own cache type; none (0) for a CPU-only shape, which lays
// nothing on the card; ErrNoWeights where the footprint does not know its weights, and settle's
// refusals.
func (fp Footprint) Largest(s Shape) (float64, error) {
	st, err := settle(&fp, &s, nil, nil)
	if err != nil {
		return 0, err
	}
	if fp.WeightsMiB <= 0 {
		return 0, ErrNoWeights
	}
	if s.CPUOnly {
		return 0, nil
	}
	if s.CPULayers > 0 {
		return st.fp.splitLargest(s.Slots, s.CtxPerSlot, s.CPULayers), nil
	}
	return st.fp.largest(s.Slots, s.CtxPerSlot, s.NCpuMoe), nil
}

func ceilMiB(f float64) int { return int(math.Ceil(f)) }

func (fp fig) shape(w Wants, slots, ctx, n int, cpuOnly bool) Shape {
	s := Shape{Slots: slots, CtxPerSlot: ctx, NCpuMoe: n, KVType: w.KVType, CPUOnly: cpuOnly, VRAMBasis: Predicted, RAMBasis: Predicted,
		ConstsBackend: fp.c.backend, ConstsCard: fp.c.card}
	if cpuOnly {
		s.NCpuMoe = 0
		s.RAMMiB = ceilMiB(fp.allHost(slots, ctx))
		s.HostMiB, s.HostBasis = s.RAMMiB, Predicted
		s.Layers = fp.layers() // a split footprint's processor alone: every layer off the card (better)
		return s
	}
	s.VRAMMiB = ceilMiB(fp.vram(slots, ctx, n))
	s.RAMMiB = ceilMiB(fp.expertRAM(n))
	s.HostBasis = Unknown
	if fp.hostKnown() {
		s.HostMiB, s.HostBasis = ceilMiB(fp.cardHost(n)), Predicted
	}
	return s
}

// splitShape is the shape with c layers on the CPU (c 0: shape on the card with no expert moved):
// its card memory, the machine's memory of those layers (RAMMiB) and of the whole start (HostMiB).
func (fp fig) splitShape(w Wants, slots, ctx, c int) Shape {
	s := fp.shape(w, slots, ctx, 0, false)
	if c <= 0 {
		return s
	}
	s.CPULayers, s.Layers = c, fp.layers()
	s.VRAMMiB = ceilMiB(fp.splitVRAM(slots, ctx, c))
	s.RAMMiB = ceilMiB(fp.splitRAM(slots, ctx, c))
	if fp.hostKnown() {
		s.HostMiB = ceilMiB(fp.cardHost(0) + fp.splitRAM(slots, ctx, c))
	}
	return s
}

// ctxCeiling is the largest ctx per slot allowed by wants and the book's cap: aligned, or the
// floor itself where CtxMax is the floor; under the floor where the cap is (no shape then).
func (fp fig) ctxCeiling(w Wants, slots int) int {
	c := w.CtxMax
	if fp.CapCtxTotal > 0 {
		if cc := alignDown(fp.CapCtxTotal / slots); cc < c {
			c = cc
		}
	}
	return c
}

// maxCtx finds the largest aligned ctx per slot in [floor, ceiling] with ok(ctx) true, or the
// floor itself where ok holds there and at no aligned ctx above it (a floor off the step). It
// never answers under the floor. ok must be monotone (true for small ctx).
func maxCtx(floor, ceiling int, ok func(ctx int) bool) (int, bool) {
	if ceiling < floor || !ok(floor) {
		return 0, false
	}
	lo, hi := (floor+alignTokens-1)/alignTokens, ceiling/alignTokens // the aligned ctx in [floor, ceiling]
	if lo > hi || !ok(lo*alignTokens) {
		return floor, true
	}
	for lo < hi { // lo ok
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

// kvTypes is a list of cache types taken once and best first: repeats and empty names dropped,
// then ordered by bytes per element, most first (a stable order: types of equal bytes, as f16 and
// bf16, keep the caller's order; a name the core does not know goes last, and resolve refuses it).
// Every reading of Wants.KVTypes goes through it (Norm, resolve). A smaller type after a larger one
// could only ever be planned as the last type's fallback, with less context than the smaller type
// gives, so the order a caller passes never changes what fits; the walk down and the least shape a
// refusal is worded from (the last type's floor) are those of the list best first.
func kvTypes(list []string) []string {
	var types []string
	for _, t := range list {
		if t != "" && !slices.Contains(types, t) {
			types = append(types, t)
		}
	}
	slices.SortStableFunc(types, func(a, b string) int {
		x, _ := kvBytes(a)
		y, _ := kvBytes(b)
		switch {
		case x > y:
			return -1
		case x < y:
			return 1
		}
		return 0
	})
	return types
}

// need is the predicted memory a shape is sized against: card memory, or host memory for a
// CPU-only shape. fp must be priced at s's cache type.
func (fp fig) need(s Shape) float64 {
	if s.CPUOnly {
		return fp.allHost(s.Slots, s.CtxPerSlot)
	}
	if s.CPULayers > 0 {
		return fp.splitVRAM(s.Slots, s.CtxPerSlot, s.CPULayers)
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
	w = w.Norm() // the list taken once, best first (kvTypes)
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
// card and RAM, unless the footprint carries Split (WithSplit): then, where nothing holds the model
// whole on the card at any slot count, layers move to the CPU and at last the processor alone
// (chooseSplit). With no usable GPU they run CPU-only. When the footprint has a measured speed
// curve the CPU layers stop at the Wants.MinTPS floor (scaled by Room.SpeedScale) even if that
// costs ctx; if nothing reaches the floor the best shape is returned, with EstTPS under it.
// With Wants.KVTypes it does all this at each type in turn (see there). No shape it answers holds
// less context per slot than Wants.CtxFloor: above the floor ctx moves by the step (1024 tokens),
// and a floor off the step is answered as itself where no aligned ctx at or above it fits.
// Where the room holds card starts to the machine's free memory (Room.CardHostFree), a model none
// of whose card starts fits it is *HostShortError (ErrTooBig to errors.Is), and expert layers move
// to RAM only as far as it holds them.
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
		if err == nil && s.Slots == w.Slots && s.CtxPerSlot >= ft.ctxCeiling(w, w.Slots) &&
			(rm.NoGPU || (!s.CPUOnly && s.CPULayers == 0)) {
			return s, nil // the most context asked fits at this type (whole on the card, where there is one)
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
	if err == nil && !s.CPUOnly && s.CPULayers == 0 {
		s.EstTPS = fp.estTPS(s.NCpuMoe, room.SpeedScale)
	}
	return s, err // a split's speed was never measured: its EstTPS is 0, unknown
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
	if room.noBlock() {
		if fp.splitOn(room, w) { // a split footprint may still run on the processor alone: it needs no block
			return fp.processorAlone(w, ram)
		}
		return Shape{}, ErrTooBig // no free block on the card: no layout on it holds
	}
	if !room.NoGPU {
		if err := fp.hostShort(room); err != nil {
			return Shape{}, err // the machine's free memory holds no start on the card, at any ctx
		}
	}
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
				return fp.vram(slots, c, n) <= budget && (block == 0 || fp.largest(slots, c, n) <= block)
			}
		}
		// (1) all on GPU at the largest ctx, never under the floor (a cap may put ceil there).
		if ctx, ok := maxCtx(max(ceil, w.CtxFloor), ceil, okAt(0)); ok {
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
				if room.held { // and no more than the machine's free memory holds beside the rest
					if byHost := int((room.host - fp.cardHost(0)) / per); byHost < maxN {
						maxN = byHost
					}
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
	return fp.chooseSplit(room, w, budget, ram)
}

// chooseSplit is Choose's last two steps, for a footprint with Split only (else ErrTooBig, as
// before): where no shape holds the model whole on the card at any slot count, (6) the fewest
// layers on the CPU that hold half the largest ctx (the bar of step 2) at some slot count, else the
// fewest that hold the floor; at that count the most slots that hold the bar, at the most ctx they
// hold; (7) the processor alone at the most slots and ctx host RAM holds. Each layer on the CPU
// costs a dense model speed, so the fewest come before slots. A split is held to the card budget,
// the largest block, host RAM (ram) and the machine's free memory where the room holds starts to
// it; the processor alone to ram. Where the card holds the least split and the machine's memory
// does not, *SplitShortError.
//
// Each candidate (bar, layers, slots) that holds in a room holds in every room with more card or
// more host memory, and they are tried in one fixed order, so more room never answers a later
// candidate: with one bar (half the ceiling at or under the floor) more card never moves more
// layers; across bars it may, to reach half the ctx where less card held only the floor.
func (fp fig) chooseSplit(room roomFig, w Wants, budget, ram float64) (Shape, error) {
	if !fp.splitOn(room, w) {
		return Shape{}, ErrTooBig
	}
	n := fp.layers()
	block := float64(room.LargestBlock)
	host := fp.splitHostRoom(room, ram)
	card := func(slots, ctx, c int) bool {
		return fp.splitVRAM(slots, ctx, c) <= budget && (block == 0 || fp.splitLargest(slots, ctx, c) <= block)
	}
	bar := func(slots int, half bool) (int, int) {
		ceil := fp.ctxCeiling(w, slots)
		if !half {
			return w.CtxFloor, ceil
		}
		h := alignDown(ceil / 2)
		if h < w.CtxFloor {
			h = w.CtxFloor
		}
		return h, ceil
	}
	for _, half := range []bool{true, false} {
		for c := 1; c <= n; c++ {
			for slots := w.Slots; slots >= w.MinSlots; slots-- {
				lo, ceil := bar(slots, half)
				ok := func(ctx int) bool { return card(slots, ctx, c) && fp.splitRAM(slots, ctx, c) <= host }
				if ctx, held := maxCtx(lo, ceil, ok); held {
					return fp.splitShape(w, slots, ctx, c), nil
				}
			}
		}
	}
	if s, err := fp.processorAlone(w, ram); err == nil {
		return s, nil
	}
	if card(w.MinSlots, w.CtxFloor, n) {
		return Shape{}, &SplitShortError{NeedMiB: ceilMiB(fp.splitRAM(w.MinSlots, w.CtxFloor, n)),
			FreeMiB: int(math.Floor(math.Max(host, 0))), Basis: Predicted}
	}
	return Shape{}, ErrTooBig
}

// processorAlone is the processor-alone shape at the most slots and ctx ram holds, or ErrTooBig.
func (fp fig) processorAlone(w Wants, ram float64) (Shape, error) {
	for slots := w.Slots; slots >= w.MinSlots; slots-- {
		ceil := fp.ctxCeiling(w, slots)
		if ctx, ok := maxCtx(w.CtxFloor, ceil, func(c int) bool { return fp.allHost(slots, c) <= ram }); ok {
			return fp.shape(w, slots, ctx, 0, true), nil
		}
	}
	return Shape{}, ErrTooBig
}

// better reports whether a is a larger shape than b: less of the model off the card (offCard; the
// same for every shape of a footprint without Split), then more slots, then fewer CPU-MoE layers,
// then more ctx.
func better(a, b Shape) bool {
	if oa, ob := offCard(a), offCard(b); oa != ob {
		return oa < ob // whole on the card, then fewer layers on the CPU, then the processor alone
	}
	if a.Slots != b.Slots {
		return a.Slots > b.Slots
	}
	if a.NCpuMoe != b.NCpuMoe {
		return a.NCpuMoe < b.NCpuMoe
	}
	return a.CtxPerSlot > b.CtxPerSlot
}

// offCard ranks how much of a split footprint's model a shape runs off the card: 0 all on it (and
// every shape of a footprint without Split, which never moves a layer), its CPULayers, or Layers+1
// for the processor alone of a split footprint.
func offCard(s Shape) int {
	if s.CPUOnly {
		if s.Layers > 0 {
			return s.Layers + 1
		}
		return 0
	}
	return s.CPULayers
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
	cache, compute := f.perToken(cur.CPUOnly)
	if cur.CPULayers > 0 && !cur.CPUOnly {
		cache *= 1 - f.cpuFullShare(cur.CPULayers) // a split's failed buffer is the card's: its share of each token
	}
	if perTok := cache + compute; failedBufMiB > 0 && perTok > 0 && cur.Slots > 0 {
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
// A room whose card has no free block (Room.LargestBlock negative) is ErrTooBig at once for a
// shape on the card: no smaller layout there holds either. Likewise a room whose free machine
// memory (Room.CardHostFree) holds no start on the card is *HostShortError at once, and no step
// moves an expert layer past it.
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
	split := f.splitOn(st.room, w)
	if !cur.CPUOnly && st.room.noBlock() {
		if split { // the processor alone needs no block on the card: no more slots or ctx than cur
			ws := w
			ws.Slots = min(w.Slots, cur.Slots)
			ws.MinSlots = min(w.MinSlots, ws.Slots)
			ws.CtxMax = min(w.CtxMax, max(cur.CtxPerSlot, w.CtxFloor))
			return f.processorAlone(ws, st.room.ram)
		}
		return Shape{}, ErrTooBig // no free block on the card: no smaller layout on it holds either
	}
	if !cur.CPUOnly {
		if err := f.hostShort(st.room); err != nil {
			return Shape{}, err // no start on the card fits the machine's free memory, smaller or not
		}
	}
	if cur.CPULayers > 0 && !cur.CPUOnly && !split {
		// A split where the room or the wants no longer allow one (Whole, a unified pool, no card):
		// cur whole would need more card than cur, so walk the whole layout down from cur's slots and
		// ctx to the first step the room holds; never a shape it does not hold.
		return f.wholeWithin(fp, cur, room, w, st.room)
	}
	if w.Whole {
		cur.NCpuMoe = 0
	}
	if cur.CPUOnly {
		cur.CPULayers = 0 // a caller's processor-alone shape moves no layer: it lays none on the card
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
		if cur.CPULayers > 0 {
			return f.splitShape(w, cur.Slots, ctx, cur.CPULayers), nil
		}
		return f.shape(w, cur.Slots, ctx, cur.NCpuMoe, cur.CPUOnly), nil
	}
	ram := st.room.ram
	if cur.CPULayers > 0 { // a split at the floor: the next layer that needs less card, else the processor alone
		if s, ok := f.splitBelow(st.room, w, cur, f.need(cur), ram); ok {
			return s, nil
		}
	}
	if !cur.CPUOnly && f.MoE && !st.room.Unified && !w.Whole && cur.NCpuMoe < f.ExpertLayers &&
		f.expertRAM(cur.NCpuMoe+1) <= ram && f.hostOK(st.room, cur.NCpuMoe+1) {
		return f.shape(w, cur.Slots, cur.CtxPerSlot, cur.NCpuMoe+1, false), nil
	}
	if cur.Slots > w.MinSlots {
		if cur.CPULayers > 0 {
			return f.splitShape(w, cur.Slots-1, w.CtxFloor, cur.CPULayers), nil
		}
		return f.shape(w, cur.Slots-1, w.CtxFloor, cur.NCpuMoe, cur.CPUOnly), nil
	}
	// At the floor with the least slots, whole on the card: the first split that needs less card.
	startSplit := func() (Shape, error) {
		if split && !cur.CPUOnly && cur.CPULayers == 0 {
			if s, ok := f.splitBelow(st.room, w, cur, f.need(cur), ram); ok {
				return s, nil
			}
		}
		return Shape{}, ErrTooBig
	}
	i := slices.Index(w.KVTypes, st.kv) // w.KVTypes holds each type once (resolve)
	if i < 0 {
		return startSplit()
	}
	// The next type's shape: no more slots or ctx per slot than cur, no more memory than cur needs.
	w.Slots = min(w.Slots, cur.Slots)
	w.MinSlots = min(w.MinSlots, w.Slots)
	c := alignDown(cur.CtxPerSlot)
	if c < w.CtxFloor {
		c = cur.CtxPerSlot // cur at a floor off the step: the next type holds that floor exactly
	}
	w.CtxMax = min(w.CtxMax, c)
	budget, hostRAM := st.room.budget, ram
	if cur.CPUOnly {
		hostRAM = min(hostRAM, f.need(cur))
		if f.Split != nil && !st.room.NoGPU {
			budget = -1 // the processor alone of a split: no step lays anything back on the card
		}
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
	return startSplit()
}

// wholeWithin is StepDown for a split cur in a room or under wants that allow none: the walk of
// cur's slots and ctx with every layer on the card (on the processor alone with no card), down to
// the first shape the room holds, or the walk's refusal. Each step is priced at its own type.
func (f fig) wholeWithin(fp Footprint, cur Shape, room Room, w Wants, rm roomFig) (Shape, error) {
	s := f.shape(w, cur.Slots, cur.CtxPerSlot, 0, rm.NoGPU)
	for i := 0; ; i++ {
		st, err := settle(&fp, &s, &room, nil)
		if err != nil {
			return Shape{}, err
		}
		if st.fp.holds(st.room, s) {
			return s, nil
		}
		if i > 1<<16 {
			return Shape{}, ErrTooBig // the walk is bounded; this is never reached
		}
		if s, err = StepDown(fp, s, room, w); err != nil {
			return Shape{}, err
		}
	}
}

// holds reports whether room holds shape s of fp (priced at s's type): the processor alone in
// host RAM; on the card its need within the budget, its largest allocation within the block and
// its start within the machine's free memory where held, and a split's layers within host RAM.
func (fp fig) holds(room roomFig, s Shape) bool {
	if s.CPUOnly {
		return fp.allHost(s.Slots, s.CtxPerSlot) <= room.ram
	}
	if room.noBlock() || fp.need(s) > room.budget || !fp.hostOK(room, s.NCpuMoe) {
		return false
	}
	if s.CPULayers > 0 {
		if fp.splitRAM(s.Slots, s.CtxPerSlot, s.CPULayers) > fp.splitHostRoom(room, room.ram) {
			return false
		}
		return room.LargestBlock == 0 || fp.splitLargest(s.Slots, s.CtxPerSlot, s.CPULayers) <= float64(room.LargestBlock)
	}
	return room.LargestBlock == 0 || fp.largest(s.Slots, s.CtxPerSlot, s.NCpuMoe) <= float64(room.LargestBlock)
}

// splitBelow is the walk's next split from cur (at its slots and ctx): the fewest layers on the CPU
// above cur's whose card need is under below and whose layers host RAM holds; past the last layer,
// the processor alone where ram holds it. false where neither holds.
func (fp fig) splitBelow(room roomFig, w Wants, cur Shape, below, ram float64) (Shape, bool) {
	host := fp.splitHostRoom(room, ram)
	for c := cur.CPULayers + 1; c <= fp.layers(); c++ {
		if fp.splitRAM(cur.Slots, cur.CtxPerSlot, c) > host {
			break // each further layer takes more of it
		}
		if fp.splitVRAM(cur.Slots, cur.CtxPerSlot, c) < below {
			return fp.splitShape(w, cur.Slots, cur.CtxPerSlot, c), true
		}
	}
	if fp.allHost(cur.Slots, cur.CtxPerSlot) <= ram {
		return fp.shape(w, cur.Slots, cur.CtxPerSlot, 0, true), true
	}
	return Shape{}, false
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
	switch {
	case s.CPUOnly:
		ngl = "0"
	case s.CPULayers > 0:
		// -ngl N keeps the last N-1 blocks and the output head on the card (llama.cpp b12651
		// src/llama-model.cpp 1816-1841): CPULayers on the CPU is -ngl Layers-CPULayers+1.
		ngl = strconv.Itoa(max(s.Layers-s.CPULayers, 0) + 1)
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
	if s.CPULayers > 0 {
		return fmt.Sprintf("%dx%d %d/%d layers on CPU (vram %d MiB, ram %d MiB)", s.Slots, s.CtxPerSlot, s.CPULayers, s.Layers, s.VRAMMiB, s.RAMMiB)
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
	// Readings is the card memory each shape on a card that came up held, one per shape, whether
	// or not its reading was taken for the fixed part (Book.Mark).
	Readings []Reading `json:"readings,omitempty"`
	// HostReadings is the host memory each CPU-only shape that came up held, one per shape
	// (Book.Mark). They are kept apart from Readings so an older build, which takes every reading
	// there as card memory, never reads one as a card reading: it drops them when it writes.
	HostReadings []Reading `json:"host_readings,omitempty"`
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

// Reading is the memory one shape held when its load came up, read by its process: for a shape on
// a card, its card memory (VRAMMiB); for a CPU-only shape (CPUOnly), its host memory (RAMMiB), with
// no expert layer counted (a CPU-only run moves none). A reading without CPUOnly, as every older
// build wrote it, is a card reading. A card reading may carry the process's resident memory
// beside its card memory (HostMiB, Book.CalibrateCard; 0: not read). Since core v0.1.4.
type Reading struct {
	Slots      int    `json:"slots"`
	CtxPerSlot int    `json:"ctx_per_slot"`
	NCpuMoe    int    `json:"n_cpu_moe"`
	KVType     string `json:"kv_type"`
	VRAMMiB    int    `json:"vram_mib"`
	CPUOnly    bool   `json:"cpu_only,omitempty"`
	RAMMiB     int    `json:"ram_mib,omitempty"`
	HostMiB    int    `json:"host_mib,omitempty"`
	// CPULayers is the card shape's layers on the CPU (Shape.CPULayers): a reading is of one
	// split only. Since the dense split (in no core tag yet).
	CPULayers int `json:"cpu_layers,omitempty"`
}

// of is whether the reading was taken from this shape: a card reading only of a card shape and a
// host reading only of a CPU-only one, never across. A cache type is compared as the type it runs
// at (runKV), so a shape that names none and one that names the default are one shape; a CPU-only
// shape's expert layers are not compared (it runs none off a card, Args).
func (r Reading) of(s Shape) bool {
	return r.CPUOnly == s.CPUOnly && r.Slots == s.Slots && r.CtxPerSlot == s.CtxPerSlot &&
		(s.CPUOnly || (r.NCpuMoe == s.NCpuMoe && r.CPULayers == s.CPULayers)) && runKV(Shape{KVType: r.KVType}) == runKV(s)
}

// readingsOf is where the book keeps the readings of shapes like s: card or host.
func (e *Calib) readingsOf(s Shape) *[]Reading {
	if s.CPUOnly {
		return &e.HostReadings
	}
	return &e.Readings
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

// Mark is s with what the book holds of it: where key has a reading of this very shape, for a
// card shape VRAMMiB is that reading and VRAMBasis Measured; for a CPU-only shape RAMMiB is that
// reading (its host memory) and RAMBasis Measured. The machine's memory of the start (HostMiB) is
// that reading too, HostBasis Measured: a CPU-only shape's host reading, or the process's resident
// memory a card reading carries (Book.CalibrateCard); a card reading without it leaves HostMiB as
// given. Otherwise s as given. It writes only a measured figure, never a predicted one over it.
func (b *Book) Mark(key string, s Shape) Shape {
	if b == nil {
		return s
	}
	e := b.Entries[key]
	for _, r := range *e.readingsOf(s) {
		switch {
		case !r.of(s):
		case s.CPUOnly:
			s.RAMMiB, s.RAMBasis = r.RAMMiB, Measured
			s.HostMiB, s.HostBasis = r.RAMMiB, Measured
		default:
			s.VRAMMiB, s.VRAMBasis = r.VRAMMiB, Measured
			if r.HostMiB > 0 {
				s.HostMiB, s.HostBasis = r.HostMiB, Measured
			}
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
	// CPUOnly: the shape ran on the processor alone. The reading is its host memory, kept for its
	// shape (Book.Mark gives it as RAMMiB, Measured); it says nothing of the fixed part on the
	// card, which is not calibrated from it.
	CPUOnly CalibVerdict = "cpu_only"
	// SplitRead: the shape ran with layers on the CPU (Shape.CPULayers). Its reading is kept for its
	// shape (Book.Mark), but the fixed part is not calibrated from it: what it implies rests on the
	// split's seed margins, never measured. Since the dense split (in no core tag yet).
	SplitRead CalibVerdict = "split"
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
	switch c.Verdict {
	case NoReading:
		return string(c.Verdict)
	case CPUOnly:
		return string(c.Verdict) + " (host memory kept for its shape; the card's fixed part not calibrated)"
	case SplitRead:
		return string(c.Verdict) + " (kept for its shape; the card's fixed part not calibrated from a split)"
	}
	if c.Verdict == NotSeed {
		return fmt.Sprintf("%s (implied fixed %.0f MiB; the book's figure was offered, not the header's seed)", c.Verdict, c.ImpliedFixedMiB)
	}
	return fmt.Sprintf("%s (implied fixed %.0f MiB, seed %.0f MiB, off %+.0f MiB)", c.Verdict, c.ImpliedFixedMiB, c.SeedFixedMiB, c.OffMiB())
}

// Calibrate records a healthy load that held measuredMiB at shape s, and says what it did: card
// memory for a shape on a card, host memory for a CPU-only shape. A CPU-only reading is kept for
// its shape alone (verdict CPUOnly, no implied fixed part, the book's fixed part untouched). A
// reading of 0 MiB or less keeps nothing (NoReading), whatever the shape.
//
// seed is the header's seed (Basis Predicted), never the book's overlay: every reading is held to
// it, so a run of readings each near the last cannot walk the fixed part away from the seed. The
// overlay offered is NotSeed and moves nothing. The reading is kept for its shape (Book.Mark)
// whatever the gate says. Like the hive's calibrate() it becomes the book's fixed part only
// within +-15 % of the seed's (a wild reading means others changed during the load, or the seed
// is wrong here: the caller sees by how much). now is the stamp the entry is given. The shape is
// priced at its own cache type (settle), and a shape or seed settle refuses is that error, with
// nothing recorded.
func (b *Book) Calibrate(key string, seed Footprint, s Shape, measuredMiB int, now time.Time) (Calibration, error) {
	return b.calibrate(key, seed, s, measuredMiB, 0, now)
}

// CalibrateCard is Calibrate for a shape on a card with the process's resident memory read beside
// its card memory: vramMiB is held to the seed and kept exactly as Calibrate keeps measuredMiB, and
// hostMiB is kept beside it in the same reading (Reading.HostMiB; 0 or less: not read), so
// Book.Mark gives the shape's HostMiB as Measured. For a CPU-only shape the reading is its host
// memory, hostMiB, and vramMiB is not read: it is Calibrate with measuredMiB hostMiB. A card
// reading of 0 MiB or less keeps nothing (NoReading), the host memory with it. Since core v0.1.4.
func (b *Book) CalibrateCard(key string, seed Footprint, s Shape, vramMiB, hostMiB int, now time.Time) (Calibration, error) {
	if s.CPUOnly {
		return b.calibrate(key, seed, s, hostMiB, 0, now)
	}
	return b.calibrate(key, seed, s, vramMiB, max(hostMiB, 0), now)
}

func (b *Book) calibrate(key string, seed Footprint, s Shape, measuredMiB, hostMiB int, now time.Time) (Calibration, error) {
	st, err := settle(&seed, &s, nil, nil)
	if err != nil {
		return Calibration{}, err
	}
	c := Calibration{SeedFixedMiB: seed.FixedMiB}
	if measuredMiB <= 0 {
		c.Verdict = NoReading
		return c, nil
	}
	if !s.CPUOnly {
		c.ImpliedFixedMiB = float64(measuredMiB) - (st.fp.need(s) - st.fp.FixedMiB)
	}
	switch {
	case s.CPUOnly: // host memory: nothing of the card's fixed part to hold to the seed
		c.Verdict = CPUOnly
	case s.CPULayers > 0: // kept for its shape; the split's margins are not measured
		c.Verdict = SplitRead
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
	old := e.readingsOf(s)
	rs := make([]Reading, 0, len(*old)+1)
	for _, r := range *old {
		if !r.of(s) {
			rs = append(rs, r)
		}
	}
	r := Reading{Slots: s.Slots, CtxPerSlot: s.CtxPerSlot, NCpuMoe: s.NCpuMoe, KVType: s.KVType, VRAMMiB: measuredMiB, HostMiB: hostMiB, CPULayers: s.CPULayers}
	if s.CPUOnly {
		r = Reading{Slots: s.Slots, CtxPerSlot: s.CtxPerSlot, KVType: s.KVType, CPUOnly: true, RAMMiB: measuredMiB}
	}
	*old = append(rs, r)
	if c.Taken() {
		e.FixedMiB = c.ImpliedFixedMiB
		e.Consts = bookSet(seed.Consts) // the constants the fixed part was worked out under, alone
		e.Loads++
	}
	e.Updated = now
	b.Entries[key] = e
	return c, nil
}

// LowerCap records an out-of-memory of shape failed in room with engine build build: the next
// start must use less total ctx than ctxTotal. A processor-alone shape of a footprint with Split
// in a room with a card records nothing: its miss was of host memory, and the cap would bound
// every plan on the card (the walk down answers it instead). Where the cap moves it carries what lifts it
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
	if f.Split != nil && failed.CPUOnly && !rm.NoGPU {
		return nil // a split footprint's processor alone ran out of host memory: no cap on the card's plans
	}
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
			under := failed
			under.CtxPerSlot = ctxTotal / failed.Slots
			miss = max(0, ceilMiB(f.need(failed)-f.need(under)))
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
