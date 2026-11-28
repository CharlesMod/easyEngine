package fit

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/core/gguf"
)

// e4bInfo is a header shaped like Gemma 4 E4B (42 blocks, every sixth full attention, the last 18
// sharing an earlier block's cache, a tied output head, the per-layer token table kept in RAM).
// The block weights are made up to the file's scale (about 3.5 GB), not read from the real file.
func e4bInfo() gguf.Info {
	const n, shared = 42, 18
	in := gguf.Info{Arch: "gemma4", NLayer: n, NKVHead: 2, HeadDimK: 512, HeadDimV: 512, SlidingWindow: 512,
		HeadDimKSWA: 256, HeadDimVSWA: 256, VocabSize: 262144, CtxTrain: 131072,
		HostBytes: 912 << 20, OutputTied: true, OutputBytes: 360 << 20}
	for i := 0; i < n; i++ {
		l := gguf.Layer{HasKV: i < n-shared, SWA: i%6 != 5, KVHeads: 2, HeadDimK: 256, HeadDimV: 256}
		b := int64(50 << 20)
		if !l.SWA {
			l.HeadDimK, l.HeadDimV, b = 512, 512, 60<<20
		}
		in.Layers = append(in.Layers, l)
		in.BlockBytes = append(in.BlockBytes, b)
		in.DenseBytes += b
	}
	in.DenseBytes += in.HostBytes + in.OutputBytes
	return in
}

// e4bSplit is the E4B-like footprint allowed to split, as a caller with the permission builds it.
func e4bSplit(t *testing.T) (plain, split Footprint) {
	t.Helper()
	plain = seed(e4bInfo(), "q4_0")
	split, err := WithSplit(plain, e4bInfo())
	if err != nil {
		t.Fatal(err)
	}
	return plain, split
}

func cardRoom(vram, ram int) Room { return Room{VRAMTotal: vram, Others: 300, RAMAvail: ram} }

// WithSplit takes each block's weights and part of the cache from the header, with the seed
// margins marked Predicted; the footprint given is left as it was. A mixture-of-experts model, a
// header without block bytes, and figures no footprint can hold are refused, never planned.
func TestWithSplitReadsTheHeader(t *testing.T) {
	plain, fp := e4bSplit(t)
	if plain.Split != nil {
		t.Fatal("WithSplit changed the footprint it was given")
	}
	sp, in := fp.Split, e4bInfo()
	if sp == nil || len(sp.LayerMiB) != in.NLayer {
		t.Fatalf("split %+v", sp)
	}
	var full, swa float64
	for i := range sp.LayerMiB {
		if sp.LayerMiB[i] != float64(in.BlockBytes[i])/mib {
			t.Errorf("block %d: %.1f MiB, header %d bytes", i, sp.LayerMiB[i], in.BlockBytes[i])
		}
		if !in.Layers[i].HasKV && (sp.FullShare[i] != 0 || sp.SWAShare[i] != 0) {
			t.Errorf("block %d shares a cache it does not own: %v %v", i, sp.FullShare[i], sp.SWAShare[i])
		}
		full += sp.FullShare[i]
		swa += sp.SWAShare[i]
	}
	if math.Abs(full-1) > 1e-9 || math.Abs(swa-1) > 1e-9 {
		t.Errorf("shares sum to %v and %v, want 1 each", full, swa)
	}
	if sp.SchedMiB != splitSchedSeedMiB || sp.HostMiB != splitHostSeedMiB || sp.HostFactor != 1 || sp.Basis != Predicted ||
		sp.OutputMiB != 360 {
		t.Errorf("margins %+v", sp)
	}

	if _, err := WithSplit(seed(refInfo(), "q4_0"), refInfo()); !errors.Is(err, ErrSplitMoE) {
		t.Errorf("MoE header: %v", err)
	}
	moe := plain
	moe.MoE = true
	if _, err := WithSplit(moe, e4bInfo()); !errors.Is(err, ErrSplitMoE) {
		t.Errorf("MoE footprint: %v", err)
	}
	none := e4bInfo()
	none.BlockBytes = nil
	if _, err := WithSplit(plain, none); !errors.Is(err, ErrNoBlocks) {
		t.Errorf("no block bytes: %v", err)
	}
	none.BlockBytes = make([]int64, none.NLayer)
	if _, err := WithSplit(plain, none); !errors.Is(err, ErrNoBlocks) {
		t.Errorf("all blocks empty: %v", err)
	}
	heavy := e4bInfo()
	for i := range heavy.BlockBytes {
		heavy.BlockBytes[i] = 1 << 30
	}
	var fe *FigureError
	if _, err := WithSplit(plain, heavy); !errors.As(err, &fe) || fe.Name != "Split.LayerMiB" {
		t.Errorf("blocks heavier than the weights: %v", err)
	}
	// A split laid on a MoE footprint by hand is refused by every entry, never planned.
	bad := seed(refInfo(), "q4_0")
	bad.Split = fp.Split
	if _, err := Choose(bad, cardRoom(8192, 64000), game); !errors.As(err, &fe) || fe.Name != "Split" {
		t.Errorf("MoE with a split: %v", err)
	}
	for _, mod := range []func(*Split){
		func(s *Split) { s.FullShare = s.FullShare[1:] },
		func(s *Split) { s.LayerMiB[3] = math.NaN() },
		func(s *Split) { s.SWAShare[0] = 2 },
		func(s *Split) { s.SchedMiB = -1 },
		func(s *Split) { s.LayerMiB = nil },
	} {
		c := *fp.Split
		c.LayerMiB = append([]float64(nil), c.LayerMiB...)
		c.SWAShare = append([]float64(nil), c.SWAShare...)
		mod(&c)
		f := fp
		f.Split = &c
		if _, err := Choose(f, cardRoom(4096, 64000), game); !errors.As(err, &fe) {
			t.Errorf("incoherent split %+v planned: %v", c, err)
		}
	}
}

// splitRooms is a sweep of cards from too small for anything to roomy, with and without host
// memory to spare, and the room kinds that never split (no card, a unified pool) or bound it (a
// largest block, the machine's free memory held).
func splitRooms() map[string]Room {
	rs := map[string]Room{}
	for v := 1024; v <= 8192; v += 256 {
		for _, ram := range []int{3000, 6000, 64000} {
			rs[fmt.Sprintf("v%d-r%d", v, ram)] = cardRoom(v, ram)
		}
	}
	rs["nogpu"] = Room{NoGPU: true, RAMAvail: 16000}
	rs["unified"] = Room{VRAMTotal: 3072, Others: 300, RAMAvail: 16000, Unified: true}
	rs["block"] = Room{VRAMTotal: 3584, Others: 300, RAMAvail: 64000, LargestBlock: 1500}
	rs["held"] = Room{VRAMTotal: 3072, Others: 300, RAMAvail: 64000, CardHostFree: 5000}
	rs["game"] = Room{VRAMTotal: 4096, Others: 900, GameReserve: 500, RAMAvail: 16000, Reserve: 4096}
	return rs
}

func splitWants() map[string]Wants {
	whole := game
	whole.Whole = true
	listed := game
	listed.KVTypes = []string{"f16", "q8_0", "q4_0"}
	return map[string]Wants{
		"game":   game,
		"whole":  whole,
		"listed": listed,
		"one":    {Slots: 1, MinSlots: 1, CtxMax: 32768, CtxFloor: 4096, KVType: "q4_0"},
		"bar":    {Slots: 2, MinSlots: 1, CtxMax: 8192, CtxFloor: 4096, KVType: "q4_0"},
	}
}

// Split only where the model is not held whole: wherever the footprint without Split plans a shape,
// the same footprint with Split plans that very shape (a processor-alone shape in a room with no
// card says its layer count, nothing else); a room with no card or a unified pool, and wants of a
// whole model, answer as without Split. Where the footprint without Split is refused on a card,
// the split answers a split, the processor alone, or a refusal.
func TestSplitOnlyWhereWholeFails(t *testing.T) {
	plain, fp := e4bSplit(t)
	n := len(fp.Split.LayerMiB)
	splits, cpus := 0, 0
	for rn, r := range splitRooms() {
		for wn, w := range splitWants() {
			a, errA := Choose(plain, r, w)
			b, errB := Choose(fp, r, w)
			never := r.NoGPU || r.Unified || w.Whole
			switch {
			case errA == nil:
				if b.CPUOnly && b.Layers == n {
					b.Layers = 0
				}
				if errB != nil || !reflect.DeepEqual(a, b) {
					t.Errorf("%s %s: held whole as %v, with Split %v (%v)", rn, wn, a, b, errB)
				}
			case never:
				if errB == nil || errB.Error() != errA.Error() {
					t.Errorf("%s %s: never split, but answered %v (%v), not %v", rn, wn, b, errB, errA)
				}
			case errB == nil && b.CPULayers > 0 && !b.CPUOnly:
				splits++
			case errB == nil && b.CPUOnly && b.Layers == n:
				cpus++
			case errB != nil && errors.Is(errB, ErrTooBig):
			default:
				t.Errorf("%s %s: refused whole (%v), split answered %v (%v)", rn, wn, errA, b, errB)
			}
		}
	}
	if splits == 0 || cpus == 0 {
		t.Errorf("the sweep planned %d splits and %d processor-alone shapes: it proves nothing", splits, cpus)
	}
}

// order is where a shape stands in chooseSplit's fixed order of candidates (whole first, then half
// the ctx at each layer count, then the floor at each, then the processor alone; a refusal last).
func order(fp Footprint, w Wants, s Shape, err error) int {
	n := len(fp.Split.LayerMiB)
	switch {
	case err != nil:
		return 2*n + 2
	case s.CPUOnly:
		return 2*n + 1
	case s.CPULayers == 0:
		return 0
	}
	f, _ := price(fp, runKV(s))
	half := alignDown(f.ctxCeiling(w, s.Slots) / 2)
	if half < w.CtxFloor {
		half = w.CtxFloor
	}
	if s.CtxPerSlot >= half {
		return s.CPULayers
	}
	return n + s.CPULayers
}

// More room never plans a later candidate: sweeping the card up in 32 MiB steps (and, on a small
// card, host RAM up in 256 MiB steps), the place of the answer in chooseSplit's order never moves
// back. Where one bar is all there is (half the ceiling at or under the floor), that is the
// literal rule: more free VRAM never plans more CPU layers, and more host RAM never refuses what
// less accepted.
func TestSplitMonotone(t *testing.T) {
	_, fp := e4bSplit(t)
	n := len(fp.Split.LayerMiB)
	for wn, w := range splitWants() {
		if w.Whole {
			continue
		}
		rw := w
		rw.KVTypes = nil // one type: the order is of one price
		for _, ram := range []int{3000, 6000, 64000} {
			last := -1
			for v := 512; v <= 8192; v += 32 {
				s, err := Choose(fp, cardRoom(v, ram), rw)
				o := order(fp, rw, s, err)
				if last >= 0 && o > last {
					t.Errorf("%s ram %d: at %d MiB of card the answer stands %d in the order, %d at less (%v %v)", wn, ram, v, o, last, s, err)
				}
				last = o
			}
		}
		for _, v := range []int{2048, 3072, 4096} {
			last, accepted := -1, false
			for ram := 2048; ram <= 16384; ram += 256 {
				s, err := Choose(fp, cardRoom(v, ram), rw)
				if accepted && err != nil {
					t.Errorf("%s card %d: refused at %d MiB of RAM what less accepted: %v", wn, v, ram, err)
				}
				accepted = accepted || err == nil
				o := order(fp, rw, s, err)
				if last >= 0 && o > last {
					t.Errorf("%s card %d: at %d MiB of RAM the answer stands %d in the order, %d at less (%v)", wn, v, ram, o, last, s)
				}
				last = o
			}
		}
	}
	// One bar: the layers on the CPU (and the processor alone, past them) never grow with the card.
	one := Wants{Slots: 2, MinSlots: 1, CtxMax: 8192, CtxFloor: 4096, KVType: "q4_0"}
	last := math.MaxInt
	moved := map[int]bool{}
	for v := 512; v <= 8192; v += 32 {
		s, err := Choose(fp, cardRoom(v, 64000), one)
		c := s.CPULayers
		switch {
		case err != nil:
			c = n + 2
		case s.CPUOnly:
			c = n + 1
		}
		if c > last {
			t.Errorf("at %d MiB of card %d layers on the CPU, %d at less", v, c, last)
		}
		last = c
		moved[c] = true
	}
	if len(moved) < 5 {
		t.Errorf("the sweep saw %d layer counts: it proves nothing", len(moved))
	}
}

// Every shape planned with Split is held to the room: its card memory to the budget, its largest
// allocation to the largest block, its layers' host memory to RAM (and the machine's free memory
// where held), the processor alone to RAM. Each figure the shape says is the price of the shape.
func TestSplitHoldsTheRoom(t *testing.T) {
	_, fp := e4bSplit(t)
	seen := 0
	for rn, r := range splitRooms() {
		for wn, w := range splitWants() {
			s, err := Choose(fp, r, w)
			if err != nil || (!s.CPUOnly && s.CPULayers == 0) {
				continue
			}
			seen++
			vram, ram, err := r.Budget()
			if err != nil {
				t.Fatal(err)
			}
			card, host, err := fp.Need(s)
			if err != nil {
				t.Fatal(err)
			}
			if (!s.CPUOnly && card > vram) || host > ram {
				t.Errorf("%s %s: %v needs %.0f card and %.0f host, budget %.0f and %.0f", rn, wn, s, card, host, vram, ram)
			}
			if s.CPUOnly {
				if s.RAMMiB != int(math.Ceil(host)) {
					t.Errorf("%s %s: says %d MiB, priced %.1f", rn, wn, s.RAMMiB, host)
				}
				continue
			}
			if s.VRAMMiB != int(math.Ceil(card)) || s.RAMMiB != int(math.Ceil(host)) {
				t.Errorf("%s %s: says %d/%d MiB, priced %.1f/%.1f", rn, wn, s.VRAMMiB, s.RAMMiB, card, host)
			}
			if r.LargestBlock > 0 {
				if l, err := fp.Largest(s); err != nil || l > float64(r.LargestBlock) {
					t.Errorf("%s %s: largest %.0f over the block %d (%v)", rn, wn, l, r.LargestBlock, err)
				}
			}
			hn, err := fp.HostNeed(s)
			if err != nil || s.HostMiB != int(math.Ceil(hn)) || s.HostBasis != Predicted {
				t.Errorf("%s %s: host %d %s, priced %.1f (%v)", rn, wn, s.HostMiB, s.HostBasis, hn, err)
			}
			if r.CardHostFree > 0 && hn > float64(r.CardHostFree-OSReserveMiB) {
				t.Errorf("%s %s: host %.0f over the machine's free %d", rn, wn, hn, r.CardHostFree-OSReserveMiB)
			}
			if s.EstTPS != 0 {
				t.Errorf("%s %s: a split was given a speed, %.1f, no card measured", rn, wn, s.EstTPS)
			}
		}
	}
	if seen < 20 {
		t.Errorf("only %d split shapes: the sweep proves nothing", seen)
	}
}

// A split's speed is never guessed: even with a measured curve, a split shape says 0 (unknown).
func TestSplitSpeedUnknown(t *testing.T) {
	_, fp := e4bSplit(t)
	fp.TPS = &TPSCurve{Points: []TPSPoint{{0, 60}, {10, 30}}}
	s, err := Choose(fp, cardRoom(3072, 64000), Wants{Slots: 1, MinSlots: 1, CtxMax: 8192, CtxFloor: 4096, KVType: "q4_0", MinTPS: -1})
	if err != nil || s.CPULayers == 0 || s.EstTPS != 0 {
		t.Errorf("%v est %.1f (%v)", s, s.EstTPS, err)
	}
}

// The walk down from a split ends, and no step is a larger shape than the one before: from each
// planned shape, StepDown is fed back until it refuses (ErrTooBig to errors.Is), each step held to
// RAM, a split never laid back on the card once the processor alone is reached.
func TestSplitStepDownWalk(t *testing.T) {
	_, fp := e4bSplit(t)
	n := len(fp.Split.LayerMiB)
	walked := 0
	for rn, r := range splitRooms() {
		for wn, w := range splitWants() {
			cur, err := Choose(fp, r, w)
			if err != nil || r.NoGPU {
				continue
			}
			_, ram, _ := r.Budget()
			for i := 0; ; i++ {
				if i > 64*n {
					t.Fatalf("%s %s: the walk did not end", rn, wn)
				}
				next, err := StepDown(fp, cur, r, w)
				if err != nil {
					if !errors.Is(err, ErrTooBig) {
						t.Errorf("%s %s: walk ended with %v", rn, wn, err)
					}
					break
				}
				walked++
				if better(next, cur) {
					t.Errorf("%s %s: step from %v to the larger %v", rn, wn, cur, next)
				}
				if cur.CPUOnly && !next.CPUOnly {
					t.Errorf("%s %s: from the processor alone back to the card: %v", rn, wn, next)
				}
				if next.CPULayers > 0 {
					if _, h, _ := fp.Need(next); h > ram {
						t.Errorf("%s %s: step %v needs %.0f MiB of RAM, %.0f free", rn, wn, next, h, ram)
					}
				}
				cur = next
			}
		}
	}
	if walked < 100 {
		t.Errorf("%d steps walked: the sweep proves nothing", walked)
	}
	// From whole at the least shape, the next step is the first split needing less card.
	plain, _ := e4bSplit(t)
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 4096, CtxFloor: 4096, KVType: "q4_0"}
	whole, err := Choose(plain, cardRoom(8192, 64000), w)
	if err != nil {
		t.Fatal(err)
	}
	next, err := StepDown(fp, whole, cardRoom(8192, 64000), w)
	if err != nil || next.CPULayers == 0 || float64(next.VRAMMiB) >= raw(fp).need(whole) {
		t.Errorf("from whole %v to %v (%v)", whole, next, err)
	}
}

// Where the card holds every layer on the CPU and RAM does not hold those layers (nor the processor
// alone), the refusal says it is the machine's memory: *SplitShortError, ErrTooBig to errors.Is.
func TestSplitShortSaysRAM(t *testing.T) {
	_, fp := e4bSplit(t)
	f := raw(fp)
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 4096, CtxFloor: 4096, KVType: "q4_0"}
	least := f.splitVRAM(1, 4096, len(fp.Split.LayerMiB))
	r := cardRoom(int(math.Ceil(least))+300+DefaultMargin(2048)+16, OSReserveMiB+300)
	r.Margin = DefaultMargin(2048)
	_, err := Choose(fp, r, w)
	var se *SplitShortError
	if !errors.As(err, &se) || !errors.Is(err, ErrTooBig) || se.FreeMiB != 300 || se.NeedMiB <= 300 || se.Basis != Predicted {
		t.Fatalf("%v", err)
	}
	if !strings.Contains(se.Error(), "machine's memory") {
		t.Errorf("%q", se.Error())
	}
	r.VRAMTotal = 600 // no card for even the least split: not the machine's memory's fault
	if _, err := Choose(fp, r, w); errors.As(err, &se) || !errors.Is(err, ErrTooBig) {
		t.Errorf("tiny card: %v", err)
	}
}

// Args renders the layer count as llama.cpp b12651 reads -ngl: src/llama-model.cpp 1816-1841 puts
// layer il on the CPU when il < n_layer+1-ngl (i_gpu_start), and the output head (the virtual
// layer n_layer) on the card whenever ngl >= 1. So CPULayers c of Layers L is -ngl L-c+1: c = L is
// -ngl 1 (the head alone on the card), and the processor alone is -ngl 0 as before.
func TestSplitArgs(t *testing.T) {
	const L = 42
	for c := 1; c <= L; c++ {
		got := Args(Shape{Slots: 2, CtxPerSlot: 4096, CPULayers: c, Layers: L})
		want := []string{"-ngl", fmt.Sprint(L - c + 1), "-np", "2", "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("c %d: %v", c, got)
		}
		iGPUStart := max(L+1-(L-c+1), 0) // llama-model.cpp 1816
		if iGPUStart != c {
			t.Errorf("c %d: llama.cpp would start the card at layer %d", c, iGPUStart)
		}
	}
	if got := Args(Shape{Slots: 1, CtxPerSlot: 4096, CPUOnly: true, Layers: L}); got[1] != "0" {
		t.Errorf("processor alone: %v", got)
	}
	if got := Args(Shape{Slots: 1, CtxPerSlot: 4096, Layers: L}); got[1] != "999" {
		t.Errorf("whole: %v", got)
	}
}

// The new fields are omitted at their zero values, so a shape, footprint or reading without a split
// writes the JSON it wrote before; with one, they survive the round trip.
func TestSplitJSON(t *testing.T) {
	plain, fp := e4bSplit(t)
	for _, v := range []any{Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0"}, plain, Reading{Slots: 1, CtxPerSlot: 4096, VRAMMiB: 3000}} {
		j, _ := json.Marshal(v)
		for _, k := range []string{"cpu_layers", "\"layers\"", "\"split\""} {
			if strings.Contains(string(j), k) {
				t.Errorf("%T writes %s at its zero value: %s", v, k, j)
			}
		}
	}
	s, err := Choose(fp, cardRoom(3072, 64000), game)
	if err != nil || s.CPULayers == 0 {
		t.Fatalf("%v %v", s, err)
	}
	var s2 Shape
	var fp2 Footprint
	if j, _ := json.Marshal(s); json.Unmarshal(j, &s2) != nil || !reflect.DeepEqual(s, s2) {
		t.Errorf("shape lost in JSON: %v %v", s, s2)
	}
	if j, _ := json.Marshal(fp); json.Unmarshal(j, &fp2) != nil || !reflect.DeepEqual(fp.Split, fp2.Split) {
		t.Errorf("split lost in JSON")
	}
}

// A reading of a split is kept for that split alone and never calibrates the card's fixed part
// (the split's margins are not measured): Mark gives it to the same shape, not to another layer
// count; the book's fixed part stays as it was.
func TestSplitReadingKeptForItsShape(t *testing.T) {
	_, fp := e4bSplit(t)
	s, err := Choose(fp, cardRoom(3072, 64000), game)
	if err != nil || s.CPULayers == 0 {
		t.Fatalf("%v %v", s, err)
	}
	var b Book
	c, err := b.CalibrateCard("k", fp, s, s.VRAMMiB-40, 6000, stamp)
	if err != nil || c.Verdict != SplitRead || c.Taken() || !strings.Contains(c.String(), "split") {
		t.Fatalf("%v %v", c, err)
	}
	if e := b.Entries["k"]; e.FixedMiB != 0 || e.Loads != 0 || len(e.Readings) != 1 || e.Readings[0].CPULayers != s.CPULayers {
		t.Errorf("book %+v", e)
	}
	if m := b.Mark("k", s); m.VRAMMiB != s.VRAMMiB-40 || m.VRAMBasis != Measured || m.HostMiB != 6000 {
		t.Errorf("mark %v", m)
	}
	other := raw(fp).splitShape(Wants{KVType: s.KVType}, s.Slots, s.CtxPerSlot, s.CPULayers+1)
	if m := b.Mark("k", other); m.VRAMBasis != Predicted {
		t.Errorf("another split marked from this reading: %v", m)
	}
}

// A shape that moves layers is priced only from a footprint that allows it and has those layers:
// every pricing entry refuses it otherwise (ErrNoSplit), never prices it as a shape on the card.
func TestSplitShapeNeedsSplit(t *testing.T) {
	plain, fp := e4bSplit(t)
	s := Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0", CPULayers: 3, Layers: 42}
	over := s
	over.CPULayers = 43
	for _, c := range []struct {
		fp Footprint
		s  Shape
	}{{plain, s}, {fp, over}} {
		if _, _, err := c.fp.Need(c.s); !errors.Is(err, ErrNoSplit) {
			t.Errorf("Need %v: %v", c.s, err)
		}
		if _, err := c.fp.HostNeed(c.s); !errors.Is(err, ErrNoSplit) {
			t.Errorf("HostNeed: %v", err)
		}
		if _, err := c.fp.Largest(c.s); !errors.Is(err, ErrNoSplit) {
			t.Errorf("Largest: %v", err)
		}
		if _, err := StepDown(c.fp, c.s, cardRoom(4096, 64000), game); !errors.Is(err, ErrNoSplit) {
			t.Errorf("StepDown: %v", err)
		}
	}
	// Each layer moved takes card memory off and lays host memory on.
	f := raw(fp)
	for c := 2; c <= 42; c++ {
		if f.splitVRAM(2, 8192, c) >= f.splitVRAM(2, 8192, c-1) || f.splitRAM(2, 8192, c) <= f.splitRAM(2, 8192, c-1) {
			t.Errorf("layer %d: card %.1f -> %.1f, host %.1f -> %.1f", c, f.splitVRAM(2, 8192, c-1), f.splitVRAM(2, 8192, c),
				f.splitRAM(2, 8192, c-1), f.splitRAM(2, 8192, c))
		}
	}
}
