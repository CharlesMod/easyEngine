package fit

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// otherCard is a set read on another card with another build: the base a start takes before any
// buffer read as 168 to 220 MiB there, against 135 in the default.
func otherCard() Consts {
	c := DefaultConsts()
	c.Backend, c.Card = "cuda@b9999", "4 GB card, sm_61"
	c.Of[BaseOverheadMiB] = Const{Value: 200, From: "read as 168 to 220 MiB at start"}
	c.Of[RuntimeMiB] = Const{Value: 2000, From: "a test's figure"}
	return c
}

// The measured constants are one set a caller may give, keyed by build and card. The default set
// is today's figures, so a seed with none given is the seed as it always was; a set given moves
// the figures it prices, and the footprint and every shape priced from it say which set, through
// JSON. A footprint built by hand (no set) prices at the default and its shapes say so.
func TestConstsByBuildAndCard(t *testing.T) {
	def := DefaultConsts()
	for _, n := range constNames {
		if c, ok := def.Of[n]; !ok || c.From == "" {
			t.Errorf("the default set lacks %s or its word of where it was read", n)
		}
	}
	ref := seed(refInfo(), "q4_0")
	with, _, err := SeedWith(refInfo(), "q4_0", DefaultConsts())
	if j1, _ := json.Marshal(ref); err != nil || string(j1) != string(mustJSON(with)) {
		t.Errorf("Seed and SeedWith the default differ: %v", err)
	}
	if ref.Consts == nil || ref.Consts.Backend != def.Backend || ref.Consts.Card != def.Card {
		t.Errorf("a seed does not say its set: %+v", ref.Consts)
	}

	other, _, err := SeedWith(refInfo(), "q4_0", otherCard())
	if err != nil || math.Abs(other.FixedMiB-ref.FixedMiB-65) > 1e-9 {
		t.Fatalf("the other card's base moved the fixed part by %.1f MiB (%v), want 65", other.FixedMiB-ref.FixedMiB, err)
	}
	r := room(16303, 600, 24000)
	s, err := Choose(other, r, game)
	if err != nil || s.ConstsBackend != "cuda@b9999" || s.ConstsCard != "4 GB card, sm_61" {
		t.Fatalf("a shape priced from the other set says %q %q (%v)", s.ConstsBackend, s.ConstsCard, err)
	}
	var fp2 Footprint
	var s2 Shape
	if json.Unmarshal(mustJSON(other), &fp2) != nil || fp2.Consts == nil || fp2.Consts.Card != other.Consts.Card ||
		fp2.Consts.Of[BaseOverheadMiB] != other.Consts.Of[BaseOverheadMiB] {
		t.Errorf("the footprint's set lost in JSON: %+v", fp2.Consts)
	}
	if json.Unmarshal(mustJSON(s), &s2) != nil || s2 != s {
		t.Errorf("the shape's set lost in JSON: %+v", s2)
	}

	byHand := ref
	byHand.Consts = nil
	if d, err := Choose(byHand, r, game); err != nil || d.ConstsBackend != def.Backend || d.ConstsCard != def.Card {
		t.Errorf("a footprint with no set: %v %v, want the default's", d, err)
	}

	// The walk down prices the miss from the footprint's own figures, which its set built: a seed of
	// the set whose runtime is 2000 MiB drops more context than one of 58.
	cur := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	a, err1 := MissRetry(ref, cur, 500, r, game)
	b, err2 := MissRetry(other, cur, 500, r, game)
	if err1 != nil || err2 != nil || b.CtxTotal() >= a.CtxTotal() || b.ConstsCard != other.Consts.Card {
		t.Errorf("MissRetry with the default %v (%v), with its own set %v (%v)", a, err1, b, err2)
	}
	// A set laid by hand on figures another set built moves no price: the figures are what is
	// priced, every entry alike, so no price takes one constant from each.
	relabel := ref
	relabel.Consts = other.Consts
	c, err := MissRetry(relabel, cur, 500, r, game)
	if err != nil || c.Slots != a.Slots || c.CtxPerSlot != a.CtxPerSlot || c.NCpuMoe != a.NCpuMoe {
		t.Errorf("MissRetry on the default's figures under another set's name: %v (%v), the figures give %v", c, err, a)
	}
}

// A fixed part the book worked out from a reading is laid only on a seed of the set it was worked
// out under: on a seed of another set it would take that set's base and vocabulary off a reading
// priced with the first set's runtime, mask and dequant, and predict for the very shape that was
// read a figure the reading does not hold. Calibrated under set B, the overlay on a B seed gives
// back the reading, and the set survives the book's JSON.
func TestBookOverlaysOnlyItsOwnSet(t *testing.T) {
	setB := DefaultConsts()
	setB.Of[RuntimeMiB] = Const{Value: 400, From: "re-read"}
	setB.Of[MaskMiBPer1k] = Const{Value: 2, From: "re-read"}
	seedA := seed(refInfo(), "q4_0")
	seedB, _, err := SeedWith(refInfo(), "q4_0", setB)
	if err != nil {
		t.Fatal(err)
	}
	s := Shape{Slots: 2, CtxPerSlot: 16384, KVType: "q4_0"}
	const read = 10626

	var b Book
	if c, err := b.Calibrate("k", seedA, s, read, stamp); err != nil || !c.Taken() {
		t.Fatalf("calibrate under the default: %v %v", c, err)
	}
	if fp := b.Footprint("k", seedA); fp.Basis != Measured {
		t.Errorf("the book's fixed part not laid on a seed of its own set: %+v", fp.Basis)
	}
	if fp := b.Footprint("k", seedB); fp.Basis != Predicted || fp.FixedMiB != seedB.FixedMiB {
		v, _, _ := fp.Need(s)
		t.Errorf("a fixed part worked out under the default laid on a seed of set B: basis %s, %.0f MiB predicted for the shape read as %d", fp.Basis, v, read)
	}

	var nb Book
	if c, err := nb.Calibrate("k", seedB, s, read, stamp); err != nil || !c.Taken() {
		t.Fatalf("calibrate under set B: %v %v", c, err)
	}
	var back Book
	if json.Unmarshal(mustJSON(nb), &back) != nil {
		t.Fatal("the book did not read back")
	}
	fp := back.Footprint("k", seedB)
	v, _, err := fp.Need(s)
	if err != nil || fp.Basis != Measured || math.Abs(v-read) > 1e-6 {
		t.Errorf("calibrated and overlaid under set B: basis %s, %.1f MiB for the shape read as %d (%v)", fp.Basis, v, read, err)
	}
	if fp := back.Footprint("k", seedA); fp.Basis != Predicted {
		t.Errorf("a fixed part worked out under set B laid on a default seed")
	}
}

// A set that lacks a constant, or a constant's word of where it was read, is a typed error naming
// it, never a zero.
func TestConstsLackingRefused(t *testing.T) {
	for _, n := range constNames {
		c := DefaultConsts()
		delete(c.Of, n)
		var ce *ConstError
		if _, _, err := SeedWith(refInfo(), "", c); !errors.As(err, &ce) || ce.Name != n {
			t.Errorf("lacking %s: %v", n, err)
		}
		c = DefaultConsts()
		c.Of[n] = Const{Value: c.Of[n].Value}
		if _, err := SeedFromBytesWith(10<<30, true, "", c); !errors.As(err, &ce) || ce.Name != n {
			t.Errorf("%s with no word of where it was read: %v", n, err)
		}
	}
}

func mustJSON(v any) []byte {
	j, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return j
}
