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
