package fit

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/CharlesMod/easyEngine/core/gguf"
)

// handSplit is a small footprint whose prices are worked out on paper below. Measured figures (no
// cache type), so every price is taken as given:
//
//	fixed 1000, cache 0.1 + compute 0.02 MiB per token, scratch 50, weights 800, host weights 100,
//	host base 500; four layers of 100, 200, 300 and 150 MiB holding 1/2, 1/4, 1/4 and none of the
//	cache; a split costs 30 MiB more on the card and 40 MiB on the host; 1.5 MiB of host per MiB of
//	layer moved.
//
// At 1 slot x 1000 tokens (cache on the card 0.1*1000 = 100, compute 0.02*1000 = 20):
//
//	c  card                                   host of the layers              start's host (600 + layers)
//	0  1000 + 120 + 50            = 1170       0                               600
//	1  1170 - 100 - 50  + 30      = 1050       1.5*100 + 50  + 40   = 240      840
//	3  1170 - 600 - 100 + 30      =  500       1.5*600 + 100 + 40   = 1040     1640
//	4  1170 - 750 - 100 + 30      =  350       1.5*750 + 100 + 40   = 1265     1865
//
// The largest allocation at c 1: weights left 800-100 = 700; cache left 120-20-50 = 50; compute
// (1000-800) + 20 = 220: 700.
func handSplit() Footprint {
	return Footprint{FixedMiB: 1000, PerTokenMiB: 0.1, ComputePerTokenMiB: 0.02, ScratchMiB: 50, WeightsMiB: 800,
		HostMiB: 100, HostBaseMiB: 500, Basis: Predicted,
		Split: &Split{LayerMiB: []float64{100, 200, 300, 150}, FullShare: []float64{0.5, 0.25, 0.25, 0},
			SWAShare: []float64{0, 0, 0, 0}, SchedMiB: 30, HostMiB: 40, HostFactor: 1.5, Basis: Predicted}}
}

func handShape(c int) Shape {
	s := Shape{Slots: 1, CtxPerSlot: 1000, KVType: "q4_0"}
	if c > 0 {
		s.CPULayers, s.Layers = c, 4
	}
	return s
}

// The prices of a split, against the table worked out on paper (handSplit): the card loses the
// first layers and their cache and gains the split's own cost; the host gains the layers (times
// HostFactor), their cache and the split's own cost.
func TestSplitPricesByHand(t *testing.T) {
	fp := handSplit()
	for _, c := range []struct {
		c                int
		card, ram, start float64
		ngl              string
	}{{0, 1170, 0, 600, "999"}, {1, 1050, 240, 840, "4"}, {3, 500, 1040, 1640, "2"}, {4, 350, 1265, 1865, "1"}} {
		s := handShape(c.c)
		card, ram, err := fp.Need(s)
		if err != nil || math.Abs(card-c.card) > 1e-9 || math.Abs(ram-c.ram) > 1e-9 {
			t.Errorf("c %d: card %.3f ram %.3f, want %.0f and %.0f (%v)", c.c, card, ram, c.card, c.ram, err)
		}
		if h, err := fp.HostNeed(s); err != nil || math.Abs(h-c.start) > 1e-9 {
			t.Errorf("c %d: start's host %.3f, want %.0f (%v)", c.c, h, c.start, err)
		}
		if a := Args(s); a[1] != c.ngl {
			t.Errorf("c %d: -ngl %s, want %s", c.c, a[1], c.ngl)
		}
	}
	if l, err := fp.Largest(handShape(1)); err != nil || math.Abs(l-700) > 1e-9 {
		t.Errorf("largest at c 1: %.3f, want 700 (%v)", l, err)
	}
	// The shape the planner builds says those very figures.
	s := raw(fp).splitShape(Wants{KVType: "q4_0"}, 1, 1000, 1)
	if s.VRAMMiB != 1050 || s.RAMMiB != 240 || s.HostMiB != 840 || s.CPULayers != 1 || s.Layers != 4 {
		t.Errorf("planned %+v", s)
	}
}

// A split reading implies the fixed part from the split's own price (need), never from the whole
// layout's: a reading of 1060 MiB at c 1 (priced 1050 with fixed 1000) implies 1010.
func TestSplitCalibrateImpliesFromTheSplit(t *testing.T) {
	var b Book
	c, err := b.CalibrateCard("k", handSplit(), handShape(1), 1060, 900, stamp)
	if err != nil || c.Verdict != SplitRead || math.Abs(c.ImpliedFixedMiB-1010) > 1e-9 {
		t.Errorf("%+v %v", c, err)
	}
}

// Fix 1: StepDown never answers a shape the room does not hold. A split where the wants ask the
// model whole, or the room is a unified pool, walks the whole layout down from the split's slots
// and ctx to the first one the room holds (hand prices, budget 1400: whole at 4096 needs
// 1000 + 0.12*4096 + 50 = 1541.5, at 3072 1418.6, at 2048 1295.8), or refuses; with no card it is
// the processor alone. The split itself (c 1 at 4096) needs 980 + 0.07*4096 = 1266.7.
func TestStepDownNeverOutgrowsTheRoom(t *testing.T) {
	fp := handSplit()
	cur := Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0", CPULayers: 1, Layers: 4}
	room := Room{VRAMTotal: 1400, Margin: -1, RAMAvail: 64000}
	unified := room
	unified.Unified = true
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 8192, CtxFloor: 1024, KVType: "q4_0"}
	whole := w
	whole.Whole = true
	for _, c := range []struct {
		name string
		r    Room
		w    Wants
	}{{"whole", room, whole}, {"unified", unified, w}} {
		s, err := StepDown(fp, cur, c.r, c.w)
		if err != nil || s.CPULayers != 0 || s.CPUOnly || s.CtxPerSlot != 2048 {
			t.Errorf("%s: %v (%v), want whole at 2048", c.name, s, err)
			continue
		}
		if v, _, _ := fp.Need(s); v > 1400 {
			t.Errorf("%s: %v needs %.1f MiB of a 1400 MiB card", c.name, s, v)
		}
	}
	floor := whole
	floor.CtxFloor = 4096 // no whole layout down to the floor fits: a refusal, never a shape too big
	if s, err := StepDown(fp, cur, room, floor); !errors.Is(err, ErrTooBig) {
		t.Errorf("whole at the floor: %v (%v)", s, err)
	}
	if s, err := StepDown(fp, cur, Room{NoGPU: true, RAMAvail: 64000}, w); err != nil || !s.CPUOnly {
		t.Errorf("no card: %v (%v)", s, err)
	}
	// The E4B-like footprint: every split planned, stepped down with Whole or in a unified pool,
	// gives a shape the room holds or a refusal.
	_, e4b := e4bSplit(t)
	seen := 0
	for v := 1024; v <= 8192; v += 128 {
		r := cardRoom(v, 64000)
		cur, err := Choose(e4b, r, game)
		if err != nil || cur.CPULayers == 0 {
			continue
		}
		u := r
		u.Unified = true
		wh := game
		wh.Whole = true
		for _, c := range []struct {
			r Room
			w Wants
		}{{r, wh}, {u, game}} {
			s, err := StepDown(e4b, cur, c.r, c.w)
			if err != nil {
				continue
			}
			seen++
			budget, _, _ := c.r.Budget()
			if card, _, _ := e4b.Need(s); s.CPULayers != 0 || card > budget {
				t.Errorf("card %d: %v stepped to %v, %.0f MiB of %.0f", v, cur, s, card, budget)
			}
		}
	}
	_ = seen // most of these rooms hold no whole layout: a refusal is the right answer there
}

// Fix 2: a header with recurrent layers (a fixed per-slot state the split does not place by layer)
// is never split.
func TestWithSplitRefusesRecurrent(t *testing.T) {
	in := hybridInfo()
	in.BlockBytes = make([]int64, in.NLayer)
	for i := range in.BlockBytes {
		in.BlockBytes[i] = 50 << 20
	}
	if _, err := WithSplit(seed(in, "q4_0"), in); !errors.Is(err, ErrSplitRecurrent) {
		t.Errorf("hybrid: %v", err)
	}
	state := e4bInfo()
	state.StateBytesPerSlot = 1 << 20
	if _, err := WithSplit(seed(e4bInfo(), "q4_0"), state); !errors.Is(err, ErrSplitRecurrent) {
		t.Errorf("state per slot: %v", err)
	}
}

// Fix 3: after an out-of-memory of a split, the context dropped is sized by the card's part of each
// token (c 1 holds half the cache: 0.1*0.5 + 0.02 = 0.07 MiB), not the whole model's (0.12). A
// failed buffer of 200 MiB plus 50 scratch drops ceil(250/0.07) = 3572 tokens from 8192: 4620,
// aligned 4096. (The whole model's price would drop 2084 to 6108, 6144, and plan 5120.)
func TestMissRetrySplitDropsByTheCardsShare(t *testing.T) {
	fp := handSplit()
	room := Room{VRAMTotal: 1400, Margin: -1, RAMAvail: 64000}
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 8192, CtxFloor: 4096, KVType: "q4_0"}
	cur := Shape{Slots: 1, CtxPerSlot: 8192, KVType: "q4_0", CPULayers: 1, Layers: 4}
	s, err := MissRetry(fp, cur, 200, room, w)
	if err != nil || s.CtxPerSlot != 4096 || s.CPULayers != 1 {
		t.Errorf("%v (%v), want c 1 at 4096", s, err)
	}
}

// Fix 4a: a card with no free block holds no layout on it, but the processor alone needs none: a
// split footprint runs there; without Split it is refused as before.
func TestNoBlockLeavesTheProcessor(t *testing.T) {
	plain, fp := e4bSplit(t)
	r := cardRoom(8192, 64000)
	r.LargestBlock = -1
	if _, err := Choose(plain, r, game); !errors.Is(err, ErrTooBig) {
		t.Errorf("without Split: %v", err)
	}
	s, err := Choose(fp, r, game)
	if err != nil || !s.CPUOnly || s.Layers != 42 {
		t.Fatalf("with Split: %v (%v)", s, err)
	}
	cur := raw(fp).splitShape(game, 2, 8192, 10)
	d, err := StepDown(fp, cur, r, game)
	if err != nil || !d.CPUOnly || d.Slots > 2 || d.CtxPerSlot > 8192 {
		t.Errorf("walk from a split with no block: %v (%v)", d, err)
	}
}

// Fix 4b: an out-of-memory of a split footprint's processor alone, in a room with a card, lowers no
// cap: the cap would bound every plan on the card. Without Split, or with no card, it records the
// cap as before.
func TestProcessorAloneMissLeavesTheCardsCap(t *testing.T) {
	plain, fp := e4bSplit(t)
	cpu := Shape{Slots: 1, CtxPerSlot: 8192, KVType: "q4_0", CPUOnly: true, Layers: 42}
	var b Book
	if err := b.LowerCap("k", fp, cpu, 4096, cardRoom(4096, 16000), "b1", stamp); err != nil || len(b.Entries) != 0 {
		t.Errorf("split, card room: %v %+v", err, b.Entries)
	}
	if err := b.LowerCap("k", fp, cpu, 4096, Room{NoGPU: true, RAMAvail: 16000}, "b1", stamp); err != nil || b.Entries["k"].CapCtxTotal != 4096 {
		t.Errorf("split, no card: %v %+v", err, b.Entries)
	}
	cpu.Layers = 0
	var p Book
	if err := p.LowerCap("k", plain, cpu, 4096, cardRoom(4096, 16000), "b1", stamp); err != nil || p.Entries["k"].CapCtxTotal != 4096 {
		t.Errorf("without Split: %v %+v", err, p.Entries)
	}
}

// At the floor a split moves one more layer before it drops a slot (hand prices, 2 slots x 1024:
// c 1 needs 1123.4 MiB of card, c 2 872.2).
func TestStepDownMovesOneMoreLayer(t *testing.T) {
	fp := handSplit()
	w := Wants{Slots: 2, MinSlots: 1, CtxMax: 4096, CtxFloor: 1024, KVType: "q4_0"}
	cur := Shape{Slots: 2, CtxPerSlot: 1024, KVType: "q4_0", CPULayers: 1, Layers: 4}
	s, err := StepDown(fp, cur, Room{VRAMTotal: 10000, Margin: -1, RAMAvail: 64000}, w)
	if err != nil || s.CPULayers != 2 || s.Slots != 2 || s.CtxPerSlot != 1024 {
		t.Errorf("%v (%v), want 2x1024 with 2 layers on the CPU", s, err)
	}
}

// With a list of cache types, a split at the first type never wins over the model whole at a later
// one: wherever the last type holds the model whole, the list answers it whole. The ctx is fixed
// (floor = ceiling) so there is a band of cards where f16 holds only as a split at the full ctx
// while q4_0 holds whole, the band where the guard decides.
func TestKVListPrefersWholeAtALaterType(t *testing.T) {
	_, fp := e4bSplit(t)
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 16384, CtxFloor: 16384, KVTypes: []string{"f16", "q4_0"}}
	at := func(k string, whole bool) Wants {
		x := w
		x.KVTypes, x.KVType, x.Whole = nil, k, whole
		return x
	}
	mattered := 0
	for v := 2048; v <= 8192; v += 8 {
		r := cardRoom(v, 64000)
		if _, err := Choose(fp, r, at("q4_0", true)); err != nil {
			continue
		}
		s, err := Choose(fp, r, w)
		if err != nil || s.CPULayers != 0 || s.CPUOnly {
			t.Errorf("card %d: whole at q4_0 holds, the list answered %v (%v)", v, s, err)
		}
		if f, err := Choose(fp, r, at("f16", false)); err == nil && f.CPULayers > 0 && f.CtxPerSlot >= 16384 {
			mattered++ // f16 alone would split at the full ctx: the guard decides here
		}
	}
	if mattered == 0 {
		t.Error("no room where f16 splits at the full ctx and q4_0 holds whole: the test proves nothing")
	}
}

// Grow lays a split back on the card when the room holds it whole, even at less ctx: less of the
// model off the card is the larger shape.
func TestGrowBackOntoTheCard(t *testing.T) {
	_, fp := e4bSplit(t)
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 32768, CtxFloor: 4096, KVType: "q4_0"}
	var cur Shape
	for v := 1024; v <= 8192; v += 64 {
		if s, err := Choose(fp, cardRoom(v, 64000), w); err == nil && s.CPULayers > 0 && s.CtxPerSlot == 32768 {
			cur = s
			break
		}
	}
	if cur.CPULayers == 0 {
		t.Fatal("no split at the full ctx to grow from")
	}
	for v := 1024; v <= 16384; v += 64 {
		r := cardRoom(v, 64000)
		s, err := Choose(fp, r, w)
		if err != nil || s.CPULayers != 0 || s.CPUOnly || s.CtxPerSlot >= cur.CtxPerSlot {
			continue
		}
		g, grew, err := Grow(fp, cur, r, w)
		if err != nil || !grew || !reflect.DeepEqual(g, s) {
			t.Errorf("card %d: from %v Grow answered %v %v (%v), want %v", v, cur, g, grew, err, s)
		}
		return
	}
	t.Fatal("no room holds the model whole at less ctx than the split: the test proves nothing")
}

// From the processor alone the walk to the next cache type never lays the model back on the card,
// though the card would hold it at the smaller type.
func TestStepDownFromProcessorAloneStaysOff(t *testing.T) {
	_, fp := e4bSplit(t)
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 4096, CtxFloor: 4096, KVTypes: []string{"q8_0", "q4_0"}}
	r := cardRoom(8192, 64000)
	if s, err := Choose(fp, r, Wants{Slots: 1, MinSlots: 1, CtxMax: 4096, CtxFloor: 4096, KVType: "q4_0"}); err != nil || s.CPUOnly {
		t.Fatalf("the card does not hold q4_0: %v %v", s, err)
	}
	cur := Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q8_0", CPUOnly: true, Layers: 42}
	s, err := StepDown(fp, cur, r, w)
	if err == nil && !s.CPUOnly {
		t.Errorf("from the processor alone to the card: %v", s)
	}
}

var _ = gguf.Layer{} // the header type the fixtures above are built from
