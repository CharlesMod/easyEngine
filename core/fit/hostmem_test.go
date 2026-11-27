package fit

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// gemmaLike is the dense model with weights the header says stay in RAM (as Gemma E-series
// per_layer_token_embd).
func gemmaLike() Footprint {
	in := denseInfo()
	in.DenseBytes = 4 << 30
	in.HostBytes = 912 << 20
	in.DenseBytes += in.HostBytes
	return seed(in, "q4_0")
}

// sameRun is whether two shapes run alike and are priced alike on the card: what the engine reads.
func sameRun(a, b Shape) bool {
	return a.Slots == b.Slots && a.CtxPerSlot == b.CtxPerSlot && a.NCpuMoe == b.NCpuMoe && a.KVType == b.KVType &&
		a.CPUOnly == b.CPUOnly && a.VRAMMiB == b.VRAMMiB && a.RAMMiB == b.RAMMiB && a.EstTPS == b.EstTPS
}

// A shape on the card says the machine's memory its start holds: the set's base, the weights the
// header keeps in RAM and the expert layers moved, predicted; a CPU-only shape says its RAMMiB. A
// footprint that cannot tell (SeedFromBytes reads no header) says Unknown and 0, never a figure.
func TestShapeSaysItsHostMemory(t *testing.T) {
	g, ref := gemmaLike(), seed(refInfo(), "q4_0")
	if g.HostBaseMiB != 1536 || ref.HostBaseMiB != 1536 {
		t.Fatalf("seeded host base %v %v, want the default set's 1536", g.HostBaseMiB, ref.HostBaseMiB)
	}
	s, err := Choose(g, room(16303, 1255, 24000), game)
	if err != nil {
		t.Fatal(err)
	}
	if want := int(math.Ceil(1536 + 912)); s.HostMiB != want || s.HostBasis != Predicted {
		t.Errorf("dense card shape host %d %q, want %d predicted", s.HostMiB, s.HostBasis, want)
	}
	if h, err := g.HostNeed(s); err != nil || math.Ceil(h) != float64(s.HostMiB) {
		t.Errorf("HostNeed %v %v, shape says %d", h, err, s.HostMiB)
	}
	// MoE with experts in RAM: base + the moved layers.
	m, err := Choose(ref, room(8192, 1255, 64000), game)
	if err != nil || m.NCpuMoe == 0 {
		t.Fatalf("no split shape to read: %v %v", m, err)
	}
	if want := int(math.Ceil(1536 + ref.ExpertRAMPerLayer*float64(m.NCpuMoe))); m.HostMiB != want || m.HostBasis != Predicted {
		t.Errorf("split shape host %d %q, want %d predicted", m.HostMiB, m.HostBasis, want)
	}
	// CPU-only: the run's host memory, which is RAMMiB.
	r := room(0, 0, 64000)
	r.NoGPU = true
	c, err := Choose(g, r, game)
	if err != nil || !c.CPUOnly || c.HostMiB != c.RAMMiB || c.HostBasis != Predicted {
		t.Errorf("CPU-only shape %+v %v", c, err)
	}
	// Unknown where the footprint cannot tell.
	fb, _ := SeedFromBytes(4<<30, false, "q4_0")
	u, err := Choose(fb, room(16303, 1255, 24000), game)
	if err != nil || u.HostMiB != 0 || u.HostBasis != Unknown {
		t.Errorf("SeedFromBytes shape host %d %q %v, want 0 unknown", u.HostMiB, u.HostBasis, err)
	}
	if _, err := fb.HostNeed(u); !errors.Is(err, ErrNoHostFigure) {
		t.Errorf("HostNeed of an unknown footprint: %v", err)
	}
	// A set lacking the constant: unknown too; a set holding it without its word: refused.
	set := DefaultConsts()
	delete(set.Of, HostBaseMiB)
	if fp, _, err := SeedWith(denseInfo(), "q4_0", set); err != nil || fp.HostBaseMiB != 0 {
		t.Errorf("a set lacking host_base_mib seeded %v %v", fp.HostBaseMiB, err)
	}
	set = DefaultConsts()
	set.Of[HostBaseMiB] = Const{Value: 1000}
	var ce *ConstError
	if _, _, err := SeedWith(denseInfo(), "q4_0", set); !errors.As(err, &ce) || ce.Name != HostBaseMiB || !ce.NoFrom {
		t.Errorf("a host base with no word answered %v", err)
	}
}

// With no free host memory given, every planning entry answers exactly what it answered before;
// given, a model none of whose card starts fits it is *HostShortError, ErrTooBig to errors.Is,
// with what it needs and what stands; a footprint that cannot tell is ErrNoHostFigure.
func TestCardHostFree(t *testing.T) {
	g := gemmaLike()
	base := room(16303, 1255, 24000)
	cur, err := Choose(g, base, game)
	if err != nil {
		t.Fatal(err)
	}
	need := int(math.Ceil(g.cardHost(0)))
	for _, free := range []int{need + 2048, need + 2048 + 5000} { // fits exactly, and with room
		r := base
		r.CardHostFree = free
		s, err := Choose(g, r, game)
		if err != nil || !sameRun(s, cur) {
			t.Errorf("free %d: %v %v, want %v as with none given", free, s, err, cur)
		}
	}
	short := base
	short.CardHostFree = need + 2048 - 1
	none := base
	none.CardHostFree = -1
	for _, r := range []Room{short, none} {
		entries := map[string]func() error{
			"Choose":    func() error { _, err := Choose(g, r, game); return err },
			"MissRetry": func() error { _, err := MissRetry(g, cur, 500, r, game); return err },
			"StepDown":  func() error { _, err := StepDown(g, cur, r, game); return err },
			"Grow":      func() error { _, _, err := Grow(g, cur, r, game); return err },
		}
		for n, call := range entries {
			err := call()
			var he *HostShortError
			if !errors.Is(err, ErrTooBig) || !errors.As(err, &he) {
				t.Errorf("%s, free %d: %v, want a HostShortError", n, r.CardHostFree, err)
				continue
			}
			if he.NeedMiB != need || he.FreeMiB != r.CardHostFree-OSReserveMiB || he.Basis != Predicted {
				t.Errorf("%s: %+v, want need %d free %d predicted", n, *he, need, r.CardHostFree-OSReserveMiB)
			}
		}
	}
	// The reserve is the room's own.
	res := base
	res.Reserve, res.CardHostFree = -1, need
	if _, err := Choose(g, res, game); err != nil {
		t.Errorf("no reserve, free = need: %v", err)
	}
	// A footprint that cannot tell, held to a free figure: refused, never planned as zero; with no
	// free figure, planned as before; on a processor room the free figure is not read.
	fb, _ := SeedFromBytes(4<<30, false, "q4_0")
	r := base
	r.CardHostFree = 1 << 20
	if _, err := Choose(fb, r, game); !errors.Is(err, ErrNoHostFigure) {
		t.Errorf("unknown host figure held to a free figure: %v", err)
	}
	if _, err := Choose(fb, base, game); err != nil {
		t.Errorf("unknown host figure, no free figure: %v", err)
	}
	cpu := room(0, 0, 64000)
	cpu.NoGPU, cpu.CardHostFree = true, 1
	if s, err := Choose(fb, cpu, game); err != nil || !s.CPUOnly {
		t.Errorf("processor room with a card free figure: %v %v", s, err)
	}
	// Unread, as any field.
	ur := base
	ur.Unread = []string{"card_host_free"}
	var ue *UnreadError
	if _, err := Choose(g, ur, game); !errors.As(err, &ue) || ue.Field != "card_host_free" {
		t.Errorf("unread card_host_free: %v", err)
	}
}

// Expert layers move to RAM only as far as the free host memory holds them, in Choose and in the
// walk down, and the shape's host memory fits it.
func TestCardHostFreeBoundsExperts(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	base := room(8192, 1255, 64000)
	free, err := Choose(ref, base, game)
	if err != nil || free.NCpuMoe < 4 {
		t.Fatalf("no split shape to bound: %v %v", free, err)
	}
	r := base
	r.CardHostFree = int(math.Ceil(ref.cardHost(free.NCpuMoe-2))) + OSReserveMiB
	s, err := Choose(ref, r, game)
	t.Logf("unbounded %v; bounded at free %d: %v %v", free, r.CardHostFree, s, err)
	switch {
	case errors.Is(err, ErrTooBig):
		var he *HostShortError
		if errors.As(err, &he) {
			t.Errorf("a host that holds a start answered %v", err)
		}
	case err != nil:
		t.Fatal(err)
	case s.NCpuMoe > free.NCpuMoe-2 || s.HostMiB > r.CardHostFree-OSReserveMiB:
		t.Errorf("bounded shape %v (host %d), free %d", s, s.HostMiB, r.CardHostFree)
	}
	// StepDown at the floor never moves a layer past it.
	at := ref.shape(game.Norm(), 1, game.CtxFloor, free.NCpuMoe-2, false)
	if next, err := StepDown(ref, at, r, game); err == nil && next.NCpuMoe > free.NCpuMoe-2 {
		t.Errorf("StepDown moved a layer past the free host memory: %v", next)
	}
	if next, err := StepDown(ref, at, base, game); err != nil || next.NCpuMoe != free.NCpuMoe-1 {
		t.Errorf("StepDown with no free figure: %v %v, want one more layer moved", next, err)
	}
}

// A card reading may carry the process's resident memory: the book keeps it with the reading and
// Mark gives the shape's host memory as measured; a reading without it leaves the prediction.
func TestCalibrateCardKeepsHostMemory(t *testing.T) {
	g := gemmaLike()
	s, err := Choose(g, room(16303, 1255, 24000), game)
	if err != nil {
		t.Fatal(err)
	}
	b := &Book{}
	if c, err := b.Calibrate("k", g, s, s.VRAMMiB, stamp); !taken(c, err) {
		t.Fatalf("%v %v", c, err)
	}
	if m := b.Mark("k", s); m.VRAMBasis != Measured || m.HostBasis != Predicted || m.HostMiB != s.HostMiB {
		t.Errorf("a card reading without host memory marked %+v", m)
	}
	c, err := b.CalibrateCard("k", g, s, s.VRAMMiB, 2600, stamp)
	if !taken(c, err) {
		t.Fatalf("%v %v", c, err)
	}
	if n := len(b.Entries["k"].Readings); n != 1 {
		t.Errorf("%d readings of one shape", n)
	}
	m := b.Mark("k", s)
	if m.VRAMBasis != Measured || m.HostMiB != 2600 || m.HostBasis != Measured {
		t.Errorf("marked %+v, want host 2600 measured", m)
	}
	// Another shape is not marked by it.
	o := s
	o.CtxPerSlot -= 1024
	if mo := b.Mark("k", o); mo.HostBasis == Measured {
		t.Errorf("another shape marked by the reading: %+v", mo)
	}
	// No card reading: nothing kept, the host memory with it.
	nb := &Book{}
	if c, err := nb.CalibrateCard("k", g, s, 0, 2600, stamp); err != nil || c.Verdict != NoReading || len(nb.Entries) != 0 {
		t.Errorf("no card reading: %v %v %+v", c, err, nb.Entries)
	}
	// CPU-only: the reading is the host memory, as Calibrate's.
	cpu := room(0, 0, 64000)
	cpu.NoGPU = true
	cs, _ := Choose(g, cpu, game)
	if c, err := nb.CalibrateCard("k", g, cs, 0, 9000, stamp); err != nil || c.Verdict != CPUOnly {
		t.Fatalf("%v %v", c, err)
	}
	if m := nb.Mark("k", cs); m.RAMMiB != 9000 || m.HostMiB != 9000 || m.HostBasis != Measured || m.RAMBasis != Measured {
		t.Errorf("CPU-only marked %+v", m)
	}
}

// The book keys the fixed part on the constants it was worked out under, which the host base is
// not: a book an older build wrote (its set without host_base_mib) still overlays its fixed part,
// and the set the book keeps holds no constant an older build does not know.
func TestHostBaseLeavesTheBookAlone(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	old := DefaultConsts()
	delete(old.Of, HostBaseMiB)
	b := &Book{Entries: map[string]Calib{"k": {FixedMiB: 11900, Consts: &old}}}
	if fp := b.Footprint("k", ref); fp.FixedMiB != 11900 || fp.Basis != Measured {
		t.Errorf("an older book's fixed part not laid: %+v", fp)
	}
	s := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	vram, _, _ := ref.Need(s)
	nb := &Book{}
	if c, err := nb.Calibrate("k", ref, s, int(vram), stamp); !taken(c, err) {
		t.Fatal(c, err)
	}
	if _, ok := nb.Entries["k"].Consts.Of[HostBaseMiB]; ok {
		t.Error("the book kept host_base_mib beside its fixed part")
	}
	if fp := nb.Footprint("k", ref); fp.Basis != Measured {
		t.Errorf("the book's own fixed part not laid on the seed: %+v", fp)
	}
}

// The new fields go both ways through JSON: an older build reads a newer shape, room, reading and
// footprint with every field it knew, and a newer build reads an older one's as the zero values,
// which are the behaviour of before.
func TestHostMemJSONBothWays(t *testing.T) {
	g := gemmaLike()
	s, _ := Choose(g, room(16303, 1255, 24000), game)
	b := &Book{}
	b.CalibrateCard("k", g, s, s.VRAMMiB, 2600, stamp)
	r := room(16303, 1255, 24000)
	r.CardHostFree = 30000
	j, err := json.Marshal(struct {
		Shape Shape
		Room  Room
		Book  *Book
		FP    Footprint
	}{s, r, b, g})
	if err != nil {
		t.Fatal(err)
	}
	// The older reader, as core v0.1.3 declared them (the fields it knew).
	type oldShape struct {
		Slots      int    `json:"slots"`
		CtxPerSlot int    `json:"ctx_per_slot"`
		NCpuMoe    int    `json:"n_cpu_moe"`
		KVType     string `json:"kv_type"`
		VRAMMiB    int    `json:"vram_mib"`
		RAMMiB     int    `json:"ram_mib"`
		VRAMBasis  Basis  `json:"vram_basis,omitempty"`
		RAMBasis   Basis  `json:"ram_basis,omitempty"`
	}
	type oldRoom struct {
		VRAMTotal int `json:"vram_total"`
		RAMAvail  int `json:"ram_avail"`
	}
	type oldReading struct {
		Slots      int `json:"slots"`
		CtxPerSlot int `json:"ctx_per_slot"`
		VRAMMiB    int `json:"vram_mib"`
	}
	type oldCalib struct {
		FixedMiB float64      `json:"fixed_mib"`
		Readings []oldReading `json:"readings"`
		Consts   *Consts      `json:"consts"`
	}
	var o struct {
		Shape oldShape
		Room  oldRoom
		Book  struct {
			Entries map[string]oldCalib `json:"entries"`
		}
		FP struct {
			HostMiB float64 `json:"host_mib"`
		}
	}
	if err := json.Unmarshal(j, &o); err != nil {
		t.Fatal(err)
	}
	if o.Shape.VRAMMiB != s.VRAMMiB || o.Shape.RAMMiB != s.RAMMiB || o.Shape.VRAMBasis != s.VRAMBasis || o.Room.RAMAvail != r.RAMAvail {
		t.Errorf("the older reader saw %+v", o)
	}
	e := o.Book.Entries["k"]
	if len(e.Readings) != 1 || e.Readings[0].VRAMMiB != s.VRAMMiB || e.FixedMiB == 0 || e.Consts == nil {
		t.Errorf("the older reader saw the book as %+v", e)
	}
	if _, ok := e.Consts.Of[HostBaseMiB]; ok {
		t.Error("the book's set carries a constant the older build does not know")
	}
	// The newer reader keeps every new field.
	var n struct {
		Shape Shape
		Room  Room
		Book  *Book
		FP    Footprint
	}
	if err := json.Unmarshal(j, &n); err != nil {
		t.Fatal(err)
	}
	if n.Shape != s || n.Room.CardHostFree != 30000 || n.FP.HostBaseMiB != 1536 || n.Book.Entries["k"].Readings[0].HostMiB != 2600 {
		t.Errorf("the newer reader saw %+v", n)
	}
	// An older build's JSON is the zero values: no host figure, no free figure, no host reading.
	oldJ := []byte(`{"Shape":{"slots":3,"ctx_per_slot":8192,"vram_mib":9000,"vram_basis":"predicted"},"Room":{"vram_total":16303,"ram_avail":24000},"FP":{"fixed_mib":9000,"host_mib":912}}`)
	var z struct {
		Shape Shape
		Room  Room
		FP    Footprint
	}
	if err := json.Unmarshal(oldJ, &z); err != nil {
		t.Fatal(err)
	}
	if z.Shape.HostMiB != 0 || z.Shape.HostBasis != "" || z.Room.CardHostFree != 0 || z.FP.HostBaseMiB != 0 {
		t.Errorf("an older build's JSON read as %+v", z)
	}
}
