package fit

import (
	"encoding/json"
	"testing"
)

// A CPU-only load that came up teaches the book its host memory: Mark gives that very shape its
// RAMMiB measured, and no other shape (another ctx, slots or type, or the same sizes on a card);
// a card reading never marks a CPU-only shape. The card's fixed part is not calibrated from it,
// and the calibration says so. A reading of nothing keeps nothing; a second replaces the first.
func TestAHostReadingMarksItsShape(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	cpu := Shape{Slots: 1, CtxPerSlot: 8192, KVType: "q4_0", CPUOnly: true}
	plan := func(s Shape) Shape {
		return ref.shape(Wants{KVType: s.KVType}, s.Slots, s.CtxPerSlot, s.NCpuMoe, s.CPUOnly)
	}
	var b Book
	c, err := b.Calibrate("k", ref, cpu, 11000, stamp)
	if err != nil || c.Verdict != CPUOnly || c.Taken() || c.ImpliedFixedMiB != 0 {
		t.Fatalf("calibrated %+v %v", c, err)
	}
	if e := b.Entries["k"]; e.FixedMiB != 0 || e.Loads != 0 || b.Footprint("k", ref).Basis != Predicted {
		t.Errorf("a host reading moved the card's fixed part: %+v", e)
	}
	marked := func(bk *Book, s Shape) (int, Basis) { m := bk.Mark("k", plan(s)); return m.RAMMiB, m.RAMBasis }
	if ram, basis := marked(&b, cpu); ram != 11000 || basis != Measured {
		t.Errorf("the shape read: %d %q, want 11000 measured", ram, basis)
	}
	if m := b.Mark("k", plan(cpu)); m.VRAMMiB != 0 || m.VRAMBasis != Predicted {
		t.Errorf("a host reading marked the card figure: %+v", m)
	}
	card := cpu
	card.CPUOnly = false
	for _, o := range []Shape{
		{Slots: 1, CtxPerSlot: 16384, KVType: "q4_0", CPUOnly: true},
		{Slots: 2, CtxPerSlot: 8192, KVType: "q4_0", CPUOnly: true},
		{Slots: 1, CtxPerSlot: 8192, KVType: "f16", CPUOnly: true},
		card,
	} {
		m := b.Mark("k", plan(o))
		if p := plan(o); m != p || m.RAMBasis != Predicted {
			t.Errorf("another shape %v marked: %+v", o, m)
		}
	}
	// A card reading of the same sizes marks the card shape alone, and leaves the host one.
	b.Calibrate("k", ref, card, 5000, stamp)
	if m := b.Mark("k", plan(card)); m.VRAMMiB != 5000 || m.VRAMBasis != Measured || m.RAMBasis != Predicted {
		t.Errorf("the card shape: %+v", m)
	}
	if ram, basis := marked(&b, cpu); ram != 11000 || basis != Measured {
		t.Errorf("the host reading after a card one: %d %q", ram, basis)
	}
	var cardOnly Book
	cardOnly.Calibrate("k", ref, card, 5000, stamp)
	if ram, basis := marked(&cardOnly, cpu); basis != Predicted || ram != plan(cpu).RAMMiB {
		t.Errorf("a card reading marked a CPU-only shape: %d %q", ram, basis)
	}
	// A second reading replaces the first; nothing read keeps nothing.
	b.Calibrate("k", ref, cpu, 11500, stamp)
	if ram, _ := marked(&b, cpu); ram != 11500 {
		t.Errorf("a second reading: %d, want 11500", ram)
	}
	var none Book
	if c, _ := none.Calibrate("k", ref, cpu, 0, stamp); c.Verdict != NoReading || len(none.Entries) != 0 {
		t.Errorf("a reading of nothing: %+v, book %+v", c, none)
	}
	// The book with a host reading round-trips through JSON.
	j, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	var rb Book
	if err := json.Unmarshal(j, &rb); err != nil {
		t.Fatal(err)
	}
	if ram, basis := marked(&rb, cpu); ram != 11500 || basis != Measured {
		t.Errorf("after JSON: %d %q", ram, basis)
	}
	if m := rb.Mark("k", plan(card)); m.VRAMMiB != 5000 || m.VRAMBasis != Measured {
		t.Errorf("after JSON, the card shape: %+v", m)
	}
}

// A book written before host readings were kept still means what it meant: its readings are card
// readings, marking the card shape alone. And a book with a host reading, read by that older
// reader, holds no card reading it did not take.
func TestAnOlderBookReadsAsCardReadings(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	old := []byte(`{"entries":{"k":{"fixed_mib":0,"cap_ctx_total":0,"loads":0,"updated":"2026-01-02T03:04:05Z",` +
		`"readings":[{"slots":1,"ctx_per_slot":8192,"n_cpu_moe":0,"kv_type":"q4_0","vram_mib":5000}]}}}`)
	var b Book
	if err := json.Unmarshal(old, &b); err != nil {
		t.Fatal(err)
	}
	card := ref.shape(Wants{KVType: "q4_0"}, 1, 8192, 0, false)
	cpu := ref.shape(Wants{KVType: "q4_0"}, 1, 8192, 0, true)
	if m := b.Mark("k", card); m.VRAMMiB != 5000 || m.VRAMBasis != Measured {
		t.Errorf("the older book's card reading: %+v", m)
	}
	if m := b.Mark("k", cpu); m != cpu {
		t.Errorf("the older book's card reading marked a CPU-only shape: %+v", m)
	}

	var nb Book
	nb.Calibrate("k", ref, Shape{Slots: 1, CtxPerSlot: 8192, KVType: "q4_0", CPUOnly: true}, 11000, stamp)
	j, err := json.Marshal(nb)
	if err != nil {
		t.Fatal(err)
	}
	type oldReading struct {
		Slots      int    `json:"slots"`
		CtxPerSlot int    `json:"ctx_per_slot"`
		NCpuMoe    int    `json:"n_cpu_moe"`
		KVType     string `json:"kv_type"`
		VRAMMiB    int    `json:"vram_mib"`
	}
	var ob struct {
		Entries map[string]struct {
			Readings []oldReading `json:"readings"`
		} `json:"entries"`
	}
	if err := json.Unmarshal(j, &ob); err != nil {
		t.Fatal(err)
	}
	if rs := ob.Entries["k"].Readings; len(rs) != 0 {
		t.Errorf("the older reader takes a host reading for a card one: %+v", rs)
	}
}
