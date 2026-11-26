package fit

import (
	"math"
	"testing"

	"github.com/CharlesMod/easyEngine/core/gguf"
)

// hybridInfo is a hybrid model's header: 32 layers, every fourth full attention (4 KV heads x
// 256), the rest linear attention holding 50.3 MiB of state a slot; 2,351.8 MiB of weights and a
// 248,320-word vocabulary, so the seed's fixed part is 2,784.8 MiB.
func hybridInfo() gguf.Info {
	in := gguf.Info{Arch: "hybrid", NLayer: 32, NKVHead: 4, HeadDimK: 256, HeadDimV: 256,
		DenseBytes: 23518 * mib / 10, VocabSize: 248320, StateBytesPerSlot: 503 * mib / 10}
	for i := 0; i < 32; i++ {
		l := gguf.Layer{Recurrent: true}
		if i%4 == 3 {
			l = gguf.Layer{HasKV: true, KVHeads: 4, HeadDimK: 256, HeadDimV: 256}
		}
		in.Layers = append(in.Layers, l)
	}
	return in
}

// A processor lays no f16 expansion of a quantised cache: its compute buffer is the same at any
// cache type. So on a room with no card the footprint carries no expansion term, and every row
// read on a processor pool (one slot, resident memory at ready) is inside the calibration gate of
// 15 % of the fixed part. On a card the expansion is real, and the card's figures for the same
// shapes stay what they were, to the MiB.
func TestProcessorPricesNoExpansion(t *testing.T) {
	rows := []struct {
		kv       string
		ctx      int
		measured float64 // MiB resident at ready on a processor pool
		card     int     // MiB the card needs for the same shape
	}{
		{"f16", 8192, 2954.1, 3153},
		{"f16", 65536, 4754.2, 4970},
		{"f16", 262144, 10906.3, 11198},
		{"q8_0", 8192, 2835.3, 3065},
		{"q8_0", 65536, 3794.4, 4266},
		{"q8_0", 262144, 7067.2, 8382},
		{"q4_0", 8192, 2771.2, 3001},
		{"q4_0", 65536, 3282.9, 3754},
		{"q4_0", 262144, 5019.2, 6334},
	}
	processor := Room{NoGPU: true, RAMAvail: 64000}
	card := room(81920, 0, 64000)
	for _, seededAt := range []string{"f16", "q4_0"} {
		fp := seed(hybridInfo(), seededAt)
		for _, r := range rows {
			w := Wants{Slots: 1, MinSlots: 1, CtxMax: r.ctx, CtxFloor: r.ctx, KVType: r.kv}
			s, err := Choose(fp, processor, w)
			if err != nil || !s.CPUOnly || s.CtxPerSlot != r.ctx {
				t.Fatalf("seeded at %s, %s %d: processor chose %v %v", seededAt, r.kv, r.ctx, s, err)
			}
			_, need, err := fp.Need(Shape{Slots: 1, CtxPerSlot: r.ctx, KVType: r.kv, CPUOnly: true})
			if err != nil || math.Ceil(need) != float64(s.RAMMiB) {
				t.Errorf("seeded at %s, %s %d: Need says %.1f MiB, Choose %d (%v)", seededAt, r.kv, r.ctx, need, s.RAMMiB, err)
			}
			off := (r.measured - need) / fp.FixedMiB // measured fixed less predicted fixed, over predicted
			if math.Abs(off) > 0.15 {
				t.Errorf("seeded at %s, %s %d: predicted %.1f MiB, measured %.1f: fixed part off %+.1f %%, outside the gate",
					seededAt, r.kv, r.ctx, need, r.measured, 100*off)
			}
			v, _, err := fp.Need(Shape{Slots: 1, CtxPerSlot: r.ctx, KVType: r.kv})
			cs, cerr := Choose(fp, card, w)
			if err != nil || cerr != nil || int(math.Ceil(v)) != r.card || cs.VRAMMiB != r.card || cs.CPUOnly {
				t.Errorf("seeded at %s, %s %d: card needs %.1f MiB, chose %v (%v %v), want %d", seededAt, r.kv, r.ctx, v, cs, err, cerr, r.card)
			}
		}
	}
}

// The figures a caller holds or gets back always keep the expansion: only the pricing of a shape
// that lays nothing on a card leaves it out. So in a card room every card shape an entry answers,
// whatever shape it was walked from (a CPU-only one included), needs exactly what Need says of
// that shape, and what the core answered before a processor's pricing was mended.
func TestCardShapeFromCPUOnlyKeepsExpansion(t *testing.T) {
	dense := seed(denseInfo(), "q4_0")
	card := Room{VRAMTotal: 24576, GameReserve: 1500, Margin: DefaultMargin(24576), RAMAvail: 32000}
	w := Wants{Slots: 2, MinSlots: 1, CtxMax: 131072, CtxFloor: 4096, KVType: "q4_0"}
	listed := w
	listed.KVTypes = []string{"q8_0", "q4_0"}
	cases := []struct {
		name string
		step func() (Shape, error)
		vram int // MiB the card shape needs
	}{
		{"MissRetry from 2x8192 q4_0 CPU-only", func() (Shape, error) {
			return MissRetry(dense, Shape{Slots: 2, CtxPerSlot: 8192, KVType: "q4_0", CPUOnly: true}, 500, card, w)
		}, 14925},
		{"StepDown, q8_0 then q4_0, from 1x4096 q8_0 CPU-only", func() (Shape, error) {
			return StepDown(dense, Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q8_0", CPUOnly: true}, card, listed)
		}, 14727},
	}
	for _, c := range cases {
		s, err := c.step()
		if err != nil || s.CPUOnly {
			t.Fatalf("%s: %v %v", c.name, s, err)
		}
		v, _, err := dense.Need(s)
		if err != nil || s.VRAMMiB != c.vram || s.VRAMMiB != int(math.Ceil(v)) {
			t.Errorf("%s: answered %v needing %d MiB; Need says %.1f, and it was %d", c.name, s, s.VRAMMiB, v, c.vram)
		}
	}
}

// Swept over card rooms, footprints, wants and shapes walked from (card and CPU-only, each type):
// every card shape Choose, MissRetry, StepDown and Grow answer needs what Need says of it.
func TestEveryCardShapePricesAsNeed(t *testing.T) {
	fps := []Footprint{seed(refInfo(), "q4_0"), seed(denseInfo(), "q4_0"), seed(denseInfo(), "f16"), seed(hybridInfo(), "q8_0")}
	lists := [][]string{nil, {"f16", "q8_0", "q4_0"}, {"q8_0", "q4_0"}}
	for _, fp := range fps {
		for vram := 8192; vram <= 32768; vram += 8192 {
			rm := room(vram, 1000, 32000)
			for _, list := range lists {
				w := Wants{Slots: 2, MinSlots: 1, CtxMax: 131072, CtxFloor: 4096, KVType: "q4_0", KVTypes: list}
				check := func(op string, s Shape, err error) {
					if err != nil || s.CPUOnly {
						return
					}
					v, _, nerr := fp.Need(s)
					if nerr != nil || s.VRAMMiB != int(math.Ceil(v)) {
						t.Errorf("%s on %d, %v: %v needs %d MiB, Need says %.1f (%v)", op, vram, list, s, s.VRAMMiB, v, nerr)
					}
				}
				s, err := Choose(fp, rm, w)
				check("Choose", s, err)
				for _, kv := range []string{"f16", "q8_0", "q4_0"} {
					for _, cpu := range []bool{false, true} {
						cur := Shape{Slots: 2, CtxPerSlot: 8192, KVType: kv, CPUOnly: cpu}
						s, err = MissRetry(fp, cur, 500, rm, w)
						check("MissRetry", s, err)
						s, err = StepDown(fp, cur, rm, w)
						check("StepDown", s, err)
						if s, grew, err := Grow(fp, cur, rm, w); grew { // else it answers cur as given
							check("Grow", s, err)
						}
					}
				}
			}
		}
	}
}
