package fit

import (
	"errors"
	"testing"
)

// The wants and the room another program states (whole, the cache's types best first, the
// model's own most context, the room's reserve, margin, largest free block, a processor node,
// a field not read): these wants, this room, this footprint, expect this shape or this error.
func TestWantsAndRoomTable(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	ling := seed(lingInfo(), "q4_0")
	dense := seed(denseInfo(), "q4_0")
	measured := ref
	measured.KVType = "" // a footprint built from measured numbers says no type it can be re-priced from
	noWeights := ref
	noWeights.WeightsMiB = 0
	trained := func(fp Footprint, n int) Footprint { fp.CtxTrain = n; return fp }
	with := func(f func(*Wants)) Wants { w := game; f(&w); return w }
	whole := with(func(w *Wants) { w.Whole = true })
	types := with(func(w *Wants) { w.KVTypes = []string{"f16", "q8_0", "q4_0"} })
	wholeTypes := types
	wholeTypes.Whole = true
	fromModel := with(func(w *Wants) { w.CtxFromModel = true })
	reserve := func(r Room, n int) Room { r.Reserve = n; return r }
	margin := func(r Room, n int) Room { r.Margin = n; return r }
	block := func(r Room, n int) Room { r.LargestBlock = n; return r }

	type want struct {
		slots, ctx, ncpu int
		kv               string
		cpuOnly          bool
	}
	cases := []struct {
		name   string
		fp     Footprint
		room   Room
		w      Wants
		err    error  // errors.Is target
		unread string // the field an *UnreadError names
		want   want
	}{
		// 1. whole
		{name: "whole, 16GB card: all on the card, as without the want", fp: ref, room: room(16303, 1255, 24000), w: whole, want: want{3, 65536, 0, "q4_0", false}},
		{name: "whole, 12GB card: only a split shape would fit", fp: ref, room: room(12288, 800, 24000), w: whole, err: ErrTooBig},
		{name: "whole, 8GB card", fp: ref, room: room(8192, 600, 24000), w: whole, err: ErrTooBig},
		{name: "Ling on a 7.5GB card, split: 3x64k with 5 experts in RAM", fp: ling, room: room(7680, 600, 24000), w: game, want: want{3, 65536, 5, "q4_0", false}},
		{name: "whole, Ling on a 7.5GB card: ctx and slots go before any expert", fp: ling, room: room(7680, 600, 24000), w: whole, want: want{1, 19456, 0, "q4_0", false}},
		{name: "whole, dense on a 24GB card, as without the want", fp: dense, room: room(24564, 800, 64000), w: whole, want: want{3, 46080, 0, "q4_0", false}},
		// 2. the cache's types, best first
		{name: "whole, types, 20GB card: f16 holds the most", fp: ref, room: room(20480, 1255, 24000), w: wholeTypes, want: want{3, 65536, 0, "f16", false}},
		{name: "whole, types, 20GB card, 4000 others: q8_0 is the first that holds the most", fp: ref, room: room(20480, 4000, 24000), w: wholeTypes, want: want{3, 65536, 0, "q8_0", false}},
		{name: "whole, types, 20GB card, 5000 others: only q4_0 holds the most", fp: ref, room: room(20480, 5000, 24000), w: wholeTypes, want: want{3, 65536, 0, "q4_0", false}},
		{name: "whole, types, 16GB card, 2800 others: none holds the most, the last takes what fits", fp: ref, room: room(16303, 2800, 24000), w: wholeTypes, want: want{3, 35840, 0, "q4_0", false}},
		{name: "types, split allowed: f16 holds the most with 9 experts in RAM", fp: ref, room: room(16303, 1255, 24000), w: types, want: want{3, 65536, 9, "f16", false}},
		{name: "types, 4GB card: under the floor at the last type", fp: ref, room: room(4096, 500, 24000), w: types, err: ErrTooBig},
		{name: "types, a footprint with no type to re-price from", fp: measured, room: room(16303, 1255, 24000), w: types, err: ErrNoKVType},
		{name: "one type in the list is that type alone", fp: ref, room: room(16303, 2800, 24000), w: with(func(w *Wants) { w.KVTypes = []string{"q4_0"} }), want: want{3, 35840, 0, "q4_0", false}},
		// 3. the model's own most context
		{name: "most ctx from the model: 32k trained", fp: trained(ref, 32768), room: room(16303, 1255, 24000), w: fromModel, want: want{3, 32768, 0, "q4_0", false}},
		{name: "most ctx from the model: trained under the floor, the floor follows it", fp: trained(ref, 4096), room: room(16303, 1255, 24000), w: fromModel, want: want{3, 4096, 0, "q4_0", false}},
		{name: "most ctx from the model: trained under the core's least ctx (1024)", fp: trained(ref, 512), room: room(16303, 1255, 24000), w: fromModel, err: ErrCtxTrainUnder},
		{name: "most ctx from the model: the header does not say it", fp: ref, room: room(16303, 1255, 24000), w: fromModel, err: ErrNoCtxTrain},
		// 4. the room as given
		{name: "no GPU, 12000 free: today's 2048 reserve leaves too little", fp: ref, room: Room{NoGPU: true, RAMAvail: 12000}, w: game, err: ErrTooBig},
		{name: "no GPU, 12000 free, reserve 512", fp: ref, room: reserve(Room{NoGPU: true, RAMAvail: 12000}, 512), w: game, want: want{3, 39936, 0, "q4_0", true}},
		{name: "8GB card, 7000 RAM: with today's reserve the experts do not fit in RAM", fp: ref, room: room(8192, 600, 7000), w: game, err: ErrTooBig},
		{name: "8GB card, 7000 RAM, no reserve", fp: ref, room: reserve(room(8192, 600, 7000), -1), w: game, want: want{3, 17408, 22, "q4_0", false}},
		{name: "margin unset is the default margin", fp: ref, room: margin(room(16303, 2800, 24000), 0), w: game, want: want{3, 35840, 0, "q4_0", false}},
		{name: "no margin", fp: ref, room: margin(room(16303, 2800, 24000), -1), w: game, want: want{3, 57344, 0, "q4_0", false}},
		{name: "largest free block under the weights: experts leave until they fit", fp: ref, room: block(room(16303, 1255, 24000), 8000), w: game, want: want{3, 65536, 7, "q4_0", false}},
		{name: "largest free block under the weights, whole", fp: ref, room: block(room(16303, 1255, 24000), 8000), w: whole, err: ErrTooBig},
		{name: "largest free block, a footprint that cannot tell its weights", fp: noWeights, room: block(room(16303, 1255, 24000), 8000), w: game, err: ErrNoWeights},
		{name: "a processor node, no card: its free memory alone", fp: ling, room: Room{NoGPU: true, RAMAvail: 16000, VRAMTotal: 16303, Others: 1255, Margin: 652}, w: game, want: want{3, 65536, 0, "q4_0", true}},
		{name: "others not read is never zero", fp: ref, room: Room{VRAMTotal: 16303, GameReserve: 1500, RAMAvail: 24000, Unread: []string{"others"}}, w: game, unread: "others"},
	}
	for _, c := range cases {
		s, err := Choose(c.fp, c.room, c.w)
		var ue *UnreadError
		switch {
		case c.unread != "":
			if !errors.As(err, &ue) || ue.Field != c.unread {
				t.Errorf("%s: want unread %q, got %v (%v)", c.name, c.unread, err, s)
			}
			continue
		case c.err != nil:
			if !errors.Is(err, c.err) {
				t.Errorf("%s: want %v, got %v (%v)", c.name, c.err, err, s)
			}
			continue
		case err != nil:
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		got := want{s.Slots, s.CtxPerSlot, s.NCpuMoe, s.KVType, s.CPUOnly}
		if got != c.want {
			t.Errorf("%s: got %+v (%v), want %+v", c.name, got, s, c.want)
		}
		if c.room.LargestBlock > 0 && float64(c.room.LargestBlock) < c.fp.largest(s.Slots, s.CtxPerSlot, s.NCpuMoe) {
			t.Errorf("%s: largest allocation over the block: %v", c.name, s)
		}
		// a type in the list is priced as the header's seed at that type, never tried
		if len(c.w.KVTypes) > 0 {
			w1 := c.w
			w1.KVTypes, w1.KVType = nil, s.KVType
			s1, err1 := Choose(seed(refInfo(), s.KVType), c.room, w1)
			if err1 != nil || s1.Slots != s.Slots || s1.CtxPerSlot != s.CtxPerSlot || s1.NCpuMoe != s.NCpuMoe {
				t.Errorf("%s: the list's %v differs from the seed at %s: %v %v", c.name, s, s.KVType, s1, err1)
			}
		}
	}
}

// With whole set, no shape from Choose, MissRetry, StepDown or Grow has an expert off the card,
// on any card from 4 to 24 GB.
func TestWholeNeverSplits(t *testing.T) {
	w := game
	w.Whole = true
	for _, fp := range []Footprint{seed(refInfo(), "q4_0"), seed(lingInfo(), "q4_0")} {
		for vram := 4096; vram <= 24576; vram += 512 {
			r := room(vram, 600, 64000)
			split := func(op string, s Shape) {
				if s.NCpuMoe != 0 {
					t.Fatalf("%s on %d MiB: %v", op, vram, s)
				}
			}
			cur, err := Choose(fp, r, w)
			if err != nil {
				if !errors.Is(err, ErrTooBig) {
					t.Fatal(err)
				}
				continue
			}
			split("Choose", cur)
			small, err := Choose(fp, room(vram, 600+vram/3, 64000), w)
			if err == nil {
				g, _ := Grow(fp, small, r, w)
				split("Grow", g)
			}
			if s, err := MissRetry(fp, cur, 900, room(vram, 1600, 64000), w); err == nil {
				split("MissRetry", s)
			}
			for i := 0; ; i++ {
				next, err := StepDown(fp, cur, r, w)
				if err != nil {
					break
				}
				split("StepDown", next)
				cur = next
				if i > 2000 {
					t.Fatal("StepDown does not end")
				}
			}
			if cur.Slots != 1 || cur.CtxPerSlot != 8192 {
				t.Errorf("whole walk on %d MiB ended at %v", vram, cur)
			}
		}
	}
}

// Asking the model for its most context is the same as naming that number: MissRetry and
// StepDown give the same shape either way, and a trained context under the core's least is
// refused by each, never planned above what the model was trained for.
func TestCtxFromModelIsTheNamedNumber(t *testing.T) {
	fp := seed(refInfo(), "q4_0")
	fp.CtxTrain = 65536
	named, asked := game, game
	named.CtxMax = 65536
	asked.CtxFromModel = true
	r := room(16303, 1255, 24000)
	cur, err := Choose(fp, r, named)
	if err != nil {
		t.Fatal(err)
	}
	m1, e1 := MissRetry(fp, cur, 2000, r, named)
	m2, e2 := MissRetry(fp, cur, 2000, r, asked)
	if m1 != m2 || e1 != e2 {
		t.Errorf("MissRetry: named %v %v, from the model %v %v", m1, e1, m2, e2)
	}
	d1, e1 := StepDown(fp, cur, r, named)
	d2, e2 := StepDown(fp, cur, r, asked)
	if d1 != d2 || e1 != e2 {
		t.Errorf("StepDown: named %v %v, from the model %v %v", d1, e1, d2, e2)
	}
	fp.CtxTrain = 512
	small := Shape{Slots: 1, CtxPerSlot: 1024, KVType: "q4_0"}
	if s, err := MissRetry(fp, small, 2000, r, asked); !errors.Is(err, ErrCtxTrainUnder) {
		t.Errorf("MissRetry at 512 trained: %v %v", s, err)
	}
	if s, err := StepDown(fp, small, r, asked); !errors.Is(err, ErrCtxTrainUnder) {
		t.Errorf("StepDown at 512 trained: %v %v", s, err)
	}
}

// With a list of cache types, the walk down after a failed load moves on to the next type at an
// earlier type's floor, and answers too big only at the last type's: a room Choose places the
// model in is never refused by MissRetry or StepDown.
func TestStepDownMovesToTheNextType(t *testing.T) {
	fp := seed(refInfo(), "q4_0")
	w := game
	w.Whole = true
	w.KVTypes = []string{"f16", "q4_0"}
	r := room(17920, 5208, 64000)
	want, err := Choose(fp, r, w)
	if err != nil || want.KVType != "q4_0" {
		t.Fatalf("Choose: %v %v", want, err)
	}
	atFloor := Shape{Slots: 1, CtxPerSlot: 8192, KVType: "f16"}
	if s, err := StepDown(fp, atFloor, r, w); err != nil || s != want {
		t.Errorf("StepDown at the f16 floor: %v %v, want %v", s, err, want)
	}
	if s, err := MissRetry(fp, atFloor, 900, r, w); err != nil || s != want {
		t.Errorf("MissRetry at the f16 floor: %v %v, want %v", s, err, want)
	}
	last := Shape{Slots: 1, CtxPerSlot: 8192, KVType: "q4_0"}
	if s, err := StepDown(fp, last, r, w); !errors.Is(err, ErrTooBig) {
		t.Errorf("StepDown at the last type's floor: %v %v", s, err)
	}
}
