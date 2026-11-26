package fit

import (
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"strings"
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
// that shape, to the MiB the figures price it at.
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

// A re-plan after a miss bounds the ctx to drop by the failed buffer over the price of a token, and
// that price is the one Need gives the shape that failed: from a CPU-only shape, the slope of its
// host memory, with no f16 expansion in it.
func TestMissRetryFromHostShapePricesAsNeed(t *testing.T) {
	fp := seed(hybridInfo(), "q4_0")
	cur := Shape{Slots: 1, CtxPerSlot: 262144, KVType: "q4_0", CPUOnly: true}
	under := cur
	under.CtxPerSlot = 131072
	_, hi, err1 := fp.Need(cur)
	_, lo, err2 := fp.Need(under)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	slope := (hi - lo) / float64(cur.CtxPerSlot-under.CtxPerSlot)
	drop := int(math.Ceil((1000 + fp.ScratchMiB) / slope))
	w := Wants{Slots: 1, MinSlots: 1, CtxMax: 262144, CtxFloor: 8192, KVType: "q4_0"}
	s, err := MissRetry(fp, cur, 1000, Room{NoGPU: true, RAMAvail: 64000}, w)
	if want := alignDown(cur.CtxPerSlot - drop); err != nil || !s.CPUOnly || s.CtxPerSlot != want {
		t.Errorf("answered %v (%v); a slope of %.5f MiB a token drops %d tokens, so 1x%d", s, err, slope, drop, want)
	}
}

// A shape that lays nothing on a card has no allocation on it: its largest block there is none.
func TestLargestOfHostShapeIsNone(t *testing.T) {
	fp := seed(hybridInfo(), "q4_0")
	for _, kv := range []string{"f16", "q8_0", "q4_0"} {
		if v, err := fp.Largest(Shape{Slots: 1, CtxPerSlot: 65536, KVType: kv, CPUOnly: true}); err != nil || v != 0 {
			t.Errorf("%s CPU-only: largest block on the card %.1f MiB (%v), want none", kv, v, err)
		}
	}
}

// The price of a token for a shape is said in one place, fig.perToken: the per-token terms that
// differ between a card and a processor are read only there, by price (which prices the figures
// at a type), by coherent (which refuses figures no footprint can hold) and by the seeds that
// write them. A second spelling would price a shape apart from
// Need. This reads the package's non-test source and fails on any other reading.
func TestOnePriceOfAToken(t *testing.T) {
	allowed := map[string]bool{"perToken": true, "price": true, "coherent": true,
		"Seed": true, "SeedWith": true, "SeedFromBytes": true, "SeedFromBytesWith": true}
	terms := map[string]bool{"ComputePerTokenMiB": true, "DequantPerTokenMiB": true}
	paths, _ := filepath.Glob("*.go")
	fset := token.NewFileSet()
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		f, err := parser.ParseFile(fset, p, src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fn, ok := d.(*ast.FuncDecl); ok && allowed[fn.Name.Name] {
				continue
			}
			if g, ok := d.(*ast.GenDecl); ok && g.Tok == token.TYPE {
				continue // the fields' own declaration
			}
			ast.Inspect(d, func(n ast.Node) bool {
				if id, ok := n.(*ast.Ident); ok && terms[id.Name] {
					t.Errorf("%s reads %s outside fig.perToken", fset.Position(id.Pos()), id.Name)
				}
				return true
			})
		}
	}
}

// The f16 expansion is part of the compute a token costs, so figures whose expansion is below zero
// or above the compute that holds it cannot be true: a price from them would fall as ctx grows.
// Every entry refuses them in settle, naming the figure, never priced.
func TestIncoherentFiguresRefused(t *testing.T) {
	fp := seed(denseInfo(), "q4_0")
	fp.PerTokenMiB, fp.ComputePerTokenMiB, fp.DequantPerTokenMiB = 0.01, 0.01, 0.05
	shapes := []Shape{
		{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0", CPUOnly: true},
		{Slots: 1, CtxPerSlot: 131072, KVType: "q4_0", CPUOnly: true},
		{Slots: 1, CtxPerSlot: 4096, KVType: "f16"},
		{Slots: 1, CtxPerSlot: 131072, KVType: "f16"},
	}
	for _, s := range shapes {
		v, r, err := fp.Need(s)
		var fe *FigureError
		if !errors.As(err, &fe) || fe.Name != "DequantPerTokenMiB" {
			t.Errorf("%v at %s: priced %.1f card, %.1f host (%v), want the expansion refused", s, s.KVType, v, r, err)
		}
	}
	f16 := seed(denseInfo(), "f16")
	f16.DequantPerTokenMiB = -0.01
	if _, _, err := f16.Need(shapes[0]); !errors.As(err, new(*FigureError)) {
		t.Errorf("figures at f16 with the expansion below zero: %v, want refused", err)
	}
}

// No producer makes figures settle refuses: every seed at every type, carried through JSON, the
// book's overlay and a calibration's corrected fixed part, prices at every type, on a card and not.
func TestProducersMakeCoherentFigures(t *testing.T) {
	types := []string{"f16", "q8_0", "q4_0"}
	var fps []Footprint
	for _, in := range []gguf.Info{refInfo(), denseInfo(), hybridInfo()} {
		for _, kv := range types {
			fps = append(fps, seed(in, kv))
		}
	}
	for _, kv := range types {
		for _, moe := range []bool{false, true} {
			fp, err := SeedFromBytes(10<<30, moe, kv)
			if err != nil {
				t.Fatal(err)
			}
			fps = append(fps, fp)
		}
	}
	for i, fp := range fps {
		b := &Book{}
		s := Shape{Slots: 1, CtxPerSlot: 8192, KVType: fp.KVType}
		if _, err := b.Calibrate("k", fp, s, int(fp.need(s))+300, stamp); err != nil {
			t.Fatalf("%d: calibrate: %v", i, err)
		}
		raw, _ := json.Marshal(b.Footprint("k", fp))
		var back Footprint
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatal(err)
		}
		for _, kv := range types {
			for _, cpu := range []bool{false, true} {
				if _, _, err := back.Need(Shape{Slots: 1, CtxPerSlot: 65536, KVType: kv, CPUOnly: cpu}); err != nil {
					t.Errorf("%d seeded at %s, at %s CPU-only %v: %v", i, fp.KVType, kv, cpu, err)
				}
			}
		}
	}
}
