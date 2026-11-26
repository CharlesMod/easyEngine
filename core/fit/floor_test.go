package fit

import (
	"errors"
	"math"
	"testing"
)

// A floor not on the step of context is held exactly: Choose never answers a shape whose context
// per slot is under Wants.CtxFloor. Where no aligned context at or above the floor fits, the floor
// itself is the answer; otherwise the largest aligned context that fits; where the floor itself
// does not fit, ErrTooBig. The shape at the floor is priced as Need prices it and held to the
// largest block as any other. Card and CPU-only, at each cache type alone and through the list.
func TestChooseHoldsTheFloorBetweenSteps(t *testing.T) {
	dense := seed(denseInfo(), "q4_0")
	const ample = 0 // the room holds far more than CtxMax
	rows := []struct {
		name       string
		floor, max int
		train      int // the model's trained context, CtxFromModel; 0: CtxMax as given
		capTotal   int // the book's cap of total ctx; 0: none
		at         int // the ctx whose need sizes the room exactly; ample: room to spare
		less       int // MiB taken from that room
		block      bool
		want       int // 0: ErrTooBig
	}{
		{name: "just above a step, room for the floor alone", floor: 8193, max: 65536, at: 8193, want: 8193},
		{name: "just above a step, room for more", floor: 8193, max: 65536, at: 20480, want: 20480},
		{name: "the floor is the model's trained context", floor: 40000, train: 40000, at: ample, want: 40000},
		{name: "ctx max is the floor", floor: 9000, max: 9000, at: ample, want: 9000},
		{name: "only the floor fits the room", floor: 9000, max: 65536, at: 9000, want: 9000},
		{name: "an aligned ctx above the floor fits", floor: 9000, max: 65536, at: 9216, want: 9216},
		{name: "the floor itself does not fit", floor: 9000, max: 65536, at: 9000, less: 1},
		{name: "a cap under the floor", floor: 9000, max: 65536, capTotal: 8192, at: ample},
		{name: "the largest block under the floor shape's", floor: 9000, max: 65536, at: ample, block: true},
	}
	types := []string{"f16", "q8_0", "q4_0"}
	for _, cpu := range []bool{false, true} {
		for _, r := range rows {
			if r.block && cpu {
				continue // a CPU-only shape lays nothing on a card
			}
			fp := dense
			fp.CtxTrain, fp.CapCtxTotal = r.train, r.capTotal
			for i, kv := range types {
				sizeAt := r.at
				if sizeAt == ample {
					sizeAt = 2 * 65536
				}
				v, h, err := fp.Need(Shape{Slots: 1, CtxPerSlot: sizeAt, KVType: kv, CPUOnly: cpu})
				if err != nil {
					t.Fatal(err)
				}
				rm := Room{Margin: -1, Reserve: -1, RAMAvail: 64000, VRAMTotal: int(math.Ceil(v)) - r.less}
				if cpu {
					rm = Room{NoGPU: true, Reserve: -1, RAMAvail: int(math.Ceil(h)) - r.less}
				}
				if r.block {
					rm.LargestBlock = int(fp.largest(1, r.floor, 0)) - 1
				}
				w := Wants{Slots: 1, MinSlots: 1, CtxMax: r.max, CtxFloor: r.floor, KVType: kv, CtxFromModel: r.train > 0}
				name := r.name + " (" + kv + map[bool]string{false: ", card)", true: ", CPU-only)"}[cpu]
				check := func(how string, s Shape, err error, exact bool) {
					t.Helper()
					if r.want == 0 {
						if !errors.Is(err, ErrTooBig) {
							t.Errorf("%s%s: want ErrTooBig, got %v (%v)", name, how, err, s)
						}
						return
					}
					if err != nil {
						t.Errorf("%s%s: %v", name, how, err)
						return
					}
					if s.CtxPerSlot < r.floor || (exact && s.CtxPerSlot != r.want) || s.CPUOnly != cpu {
						t.Errorf("%s%s: got %v, want ctx %d", name, how, s, r.want)
					}
					nv, nh, err := fp.Need(s)
					if err != nil || s.VRAMMiB != int(math.Ceil(nv)) || s.RAMMiB != int(math.Ceil(nh)) {
						t.Errorf("%s%s: %v is not priced as Need prices it (%.1f, %.1f, %v)", name, how, s, nv, nh, err)
					}
					if rm.LargestBlock > 0 && fp.largest(s.Slots, s.CtxPerSlot, s.NCpuMoe) > float64(rm.LargestBlock) {
						t.Errorf("%s%s: %v is over the largest block", name, how, s)
					}
				}
				s, err := Choose(fp, rm, w)
				check("", s, err, true)
				// Through the list, best first, down to this type: the shape still holds the floor,
				// and at the last type it is the one this type alone gives.
				wl := w
				wl.KVTypes = types[:i+1]
				s, err = Choose(fp, rm, wl)
				check(" through the list", s, err, false)
				if err == nil && r.want > 0 && s.KVType == kv && s.CtxPerSlot != r.want {
					t.Errorf("%s through the list: got %v, want ctx %d", name, s, r.want)
				}
			}
		}
	}
}

// A shape at a floor off the step walks on to the next cache type at that floor, never to
// ErrTooBig because the floor was rounded under itself.
func TestStepDownKeepsAFloorOffTheStep(t *testing.T) {
	dense := seed(denseInfo(), "q4_0")
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 65536, CtxFloor: 9000, KVTypes: []string{"f16", "q4_0"}}
	cur := dense.shape(Wants{KVType: "f16"}, 1, 9000, 0, false)
	rm := Room{Margin: -1, Reserve: -1, RAMAvail: 64000, VRAMTotal: cur.VRAMMiB}
	s, err := StepDown(dense, cur, rm, w)
	if err != nil || s.KVType != "q4_0" || s.CtxPerSlot != 9000 || s.Slots != 1 {
		t.Errorf("from %v: got %v %v, want q4_0 at 1x9000", cur, s, err)
	}
}
