package fit

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
	"time"
)

// Every figure says whether it is predicted or measured. The footprint is measured only where the
// book took a reading for its key; a shape is measured only where the book holds a reading of that
// very shape (whatever the gate said of it). A reading passed over never marks the footprint, and
// no reading marks another shape.
func TestEveryFigureMarked(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	read := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	other := Shape{Slots: 2, CtxPerSlot: 32768, KVType: "q4_0"}
	cpu := Shape{Slots: 1, CtxPerSlot: 8192, KVType: "q4_0", CPUOnly: true}
	cases := []struct {
		name     string
		at       Shape // the shape the reading was taken at
		measured int   // MiB the load held; 0: no reading
		ask      Shape // the shape asked about afterwards
		wantFP   Basis
		wantS    Basis
		wantVRAM int // 0: the planner's own figure
	}{
		{"no reading", read, 0, read, Predicted, Predicted, 0},
		{"taken: the shape read", read, 12500, read, Measured, Measured, 12500},
		{"taken: another shape", read, 12500, other, Measured, Predicted, 0},
		{"passed over: the shape read", read, 16000, read, Predicted, Measured, 16000},
		{"passed over: another shape", read, 16000, other, Predicted, Predicted, 0},
		{"cpu-only: nothing read of the card", cpu, 12000, cpu, Predicted, Predicted, 0},
	}
	for _, c := range cases {
		var b Book
		if c.measured > 0 {
			b.Calibrate("k", ref, c.at, c.measured, stamp)
		}
		fp := b.Footprint("k", ref)
		if fp.Basis != c.wantFP {
			t.Errorf("%s: footprint %q, want %q", c.name, fp.Basis, c.wantFP)
		}
		planned := fp.shape(Wants{KVType: c.ask.KVType}, c.ask.Slots, c.ask.CtxPerSlot, c.ask.NCpuMoe, c.ask.CPUOnly)
		if planned.VRAMBasis != Predicted {
			t.Errorf("%s: the planner's shape says %q", c.name, planned.VRAMBasis)
		}
		got := b.Mark("k", planned)
		want := c.wantVRAM
		if want == 0 {
			want = planned.VRAMMiB
		}
		if got.VRAMBasis != c.wantS || got.VRAMMiB != want {
			t.Errorf("%s: shape %q %d MiB, want %q %d MiB", c.name, got.VRAMBasis, got.VRAMMiB, c.wantS, want)
		}
		// The mark survives JSON for both.
		var fp2 Footprint
		var s2 Shape
		if j, _ := json.Marshal(fp); json.Unmarshal(j, &fp2) != nil || fp2.Basis != fp.Basis {
			t.Errorf("%s: footprint basis lost in JSON", c.name)
		}
		if j, _ := json.Marshal(got); json.Unmarshal(j, &s2) != nil || s2.VRAMBasis != got.VRAMBasis {
			t.Errorf("%s: shape basis lost in JSON", c.name)
		}
	}
	if s, _ := SeedFromBytes(10<<30, false, "q4_0"); s.Basis != Predicted {
		t.Errorf("SeedFromBytes says %q", s.Basis)
	}
}

// A calibration says what it did: taken, or passed over with the figure the reading implies and
// why. The book's fixed part moves only when taken.
func TestCalibrationSaysWhatItDid(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	sh := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	rest := ref.vram(3, 65536, 0) - ref.FixedMiB // what the shape adds to the fixed part
	cases := []struct {
		name     string
		s        Shape
		seed     Footprint
		measured int
		want     CalibVerdict
		implied  float64 // 0: none
		moves    bool    // the book's fixed part is now the implied figure
	}{
		{"within the gate", sh, ref, 12500, Taken, 12500 - rest, true},
		{"too high", sh, ref, 16000, OutsideGate, 16000 - rest, false},
		{"too low", sh, ref, 8000, OutsideGate, 8000 - rest, false},
		{"no reading", sh, ref, 0, NoReading, 0, false},
		{"cpu-only", Shape{Slots: 1, CtxPerSlot: 8192, CPUOnly: true}, ref, 12000, CPUOnly, 0, false},
		{"no fixed part in the seed", sh, Footprint{PerTokenMiB: 0.01, KVType: "q4_0"}, 12000, NoSeed, 12000 - 0.01*3*65536, false},
	}
	for _, c := range cases {
		var b Book
		got := b.Calibrate("k", c.seed, c.s, c.measured, stamp)
		if got.Verdict != c.want || math.Abs(got.ImpliedFixedMiB-c.implied) > 1e-6 || got.SeedFixedMiB != c.seed.FixedMiB {
			t.Errorf("%s: %+v, want %s implied %.1f", c.name, got, c.want, c.implied)
		}
		if got.Taken() != (c.want == Taken) || (c.implied != 0 && math.Abs(got.OffMiB()-(c.implied-c.seed.FixedMiB)) > 1e-6) {
			t.Errorf("%s: taken %v off %.1f", c.name, got.Taken(), got.OffMiB())
		}
		e := b.Entries["k"]
		if moved := e.FixedMiB != 0; moved != c.moves || (moved && math.Abs(e.FixedMiB-c.implied) > 1e-6) {
			t.Errorf("%s: book fixed %.1f", c.name, e.FixedMiB)
		}
	}
}

// A cap lowered by an out-of-memory carries the room and the build it was measured with; LetGo
// lets it go where the room grew by more than the miss or the build is another, and says what
// would lift it while it stands. With the same room and build it stands exactly as before.
func TestCapNamesWhatLiftsIt(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	failed := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	at := room(16303, 3000, 24000)
	cases := []struct {
		name   string
		old    bool               // a cap from a book written before caps carried their room
		grow   func(miss int) int // MiB the room grew by since
		build  string             // the build now
		stands bool
	}{
		{"same room, same build", false, func(int) int { return 0 }, "b1", true},
		{"smaller room", false, func(int) int { return -500 }, "b1", true},
		{"grew by the miss exactly", false, func(m int) int { return m }, "b1", true},
		{"grew by more than the miss", false, func(m int) int { return m + 1 }, "b1", false},
		{"another build", false, func(int) int { return 0 }, "b2", false},
		{"an older book's cap, room grown", true, func(int) int { return 2000 }, "b2", true},
	}
	for _, c := range cases {
		var b Book
		b.LowerCap("k", ref, failed, 150000, at, "b1", stamp)
		b.LowerCap("k", ref, failed, 170000, at, "b1", stamp) // never raised
		if c.old {
			e := b.Entries["k"]
			e.Lift = nil
			b.Entries["k"] = e
		}
		lift := b.Entries["k"].Lift
		miss := 0
		if lift != nil {
			miss = lift.MissMiB
			if lift.RoomMiB != int(at.Budget()) || lift.Backend != "b1" || miss <= 0 {
				t.Fatalf("%s: lift %+v", c.name, lift)
			}
		}
		now := at
		now.Others -= c.grow(miss)
		stands, l := b.LetGo("k", now, c.build)
		cap := b.Footprint("k", ref).CapCtxTotal
		if stands != c.stands || (stands && cap != 150000) || (!stands && cap != 0) {
			t.Errorf("%s: stands %v cap %d", c.name, stands, cap)
		}
		if stands && !c.old && (l == nil || *l != *lift) {
			t.Errorf("%s: a standing cap says %v, want %v", c.name, l, lift)
		}
		if c.old && l != nil {
			t.Errorf("%s: an older cap names %v", c.name, l)
		}
	}
	var b Book
	if stands, l := b.LetGo("k", at, "b1"); stands || l != nil {
		t.Error("no cap, yet one stands")
	}
}

// An unknown cache type name is a typed error from every entry that takes one; the known names
// and the empty name (the default) are priced as before.
func TestUnknownCacheTypeRefused(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	r := room(16303, 600, 24000)
	cur := Shape{Slots: 3, CtxPerSlot: 32768, KVType: "q4_0"}
	entries := []struct {
		name string
		call func(kv string) error
	}{
		{"Seed", func(kv string) error { _, _, err := Seed(refInfo(), kv); return err }},
		{"SeedFromBytes", func(kv string) error { _, err := SeedFromBytes(10<<30, true, kv); return err }},
		{"KVBytes", func(kv string) error { _, err := KVBytes(kv); return err }},
		{"Choose, one type", func(kv string) error { w := game; w.KVType = kv; _, err := Choose(ref, r, w); return err }},
		{"Choose, the list", func(kv string) error {
			w := game
			w.KVTypes = []string{"q8_0", kv}
			_, err := Choose(ref, r, w)
			return err
		}},
		{"StepDown, the shape's type", func(kv string) error { c := cur; c.KVType = kv; _, err := StepDown(ref, c, r, game); return err }},
		{"StepDown, the list", func(kv string) error {
			w := game
			w.KVTypes = []string{"q4_0", kv}
			_, err := StepDown(ref, cur, r, w)
			return err
		}},
		{"MissRetry, the shape's type", func(kv string) error { c := cur; c.KVType = kv; _, err := MissRetry(ref, c, 500, r, game); return err }},
	}
	for _, e := range entries {
		var ke *KVTypeError
		if err := e.call("q3_k"); !errors.As(err, &ke) || ke.Name != "q3_k" {
			t.Errorf("%s: unknown type answered %v", e.name, err)
		}
		for _, kv := range []string{"", "q8_0"} {
			if err := e.call(kv); err != nil {
				t.Errorf("%s(%q): %v", e.name, kv, err)
			}
		}
	}
	known := map[string]float64{"f32": 4, "f16": 2, "bf16": 2, "q8_0": 8.5 / 8, "q5_1": 6.0 / 8, "q5_0": 5.5 / 8,
		"q4_1": 5.0 / 8, "q4_0": 4.5 / 8, "iq4_nl": 4.5 / 8, "": 4.5 / 8}
	for kv, want := range known {
		if got, err := KVBytes(kv); err != nil || got != want {
			t.Errorf("KVBytes(%q) = %v %v, want %v", kv, got, err, want)
		}
	}
}

// A book written before this change still loads and means what it meant; a book written after is
// read by the older reader (the Calib of before) with every field it knew.
func TestBookJSONBothWays(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	old := []byte(`{"entries":{"k":{"fixed_mib":11900.5,"cap_ctx_total":150000,"loads":2,"updated":"2026-01-02T03:04:05Z"}}}`)
	var b Book
	if err := json.Unmarshal(old, &b); err != nil {
		t.Fatal(err)
	}
	fp := b.Footprint("k", ref)
	if fp.FixedMiB != 11900.5 || fp.CapCtxTotal != 150000 || fp.Basis != Measured {
		t.Errorf("old book read as %+v", fp)
	}
	if stands, _ := b.LetGo("k", room(16303, 0, 24000), "another"); !stands {
		t.Error("an old cap was let go: nothing it recorded can lift it")
	}

	// The older reader: Calib as it was before this change.
	type oldCalib struct {
		FixedMiB    float64   `json:"fixed_mib"`
		CapCtxTotal int       `json:"cap_ctx_total"`
		Loads       int       `json:"loads"`
		Updated     time.Time `json:"updated"`
	}
	var nb Book
	nb.Calibrate("k", ref, Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}, 12500, stamp)
	nb.LowerCap("k", ref, Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}, 150000, room(16303, 600, 24000), "b1", stamp)
	j, err := json.Marshal(nb)
	if err != nil {
		t.Fatal(err)
	}
	var ob struct {
		Entries map[string]oldCalib `json:"entries"`
	}
	if err := json.Unmarshal(j, &ob); err != nil {
		t.Fatal(err)
	}
	e, o := nb.Entries["k"], ob.Entries["k"]
	if o.FixedMiB != e.FixedMiB || o.CapCtxTotal != 150000 || o.Loads != 1 || !o.Updated.Equal(stamp) {
		t.Errorf("the older reader saw %+v of %+v", o, e)
	}
}
