package fit

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/core/gguf"
)

// refInfo is the reference model's header as read by core/gguf (see TestReferenceFromFile).
func refInfo() gguf.Info {
	in := gguf.Info{Arch: "gemma4", NLayer: 30, NKVHead: 2, HeadDimK: 512, HeadDimV: 512, SlidingWindow: 1024, SWALayers: 25,
		NExpert: 96, NExpertUsed: 8, ExpertBytes: 7734594816, DenseBytes: 2256820680, FileSize: 10001779136,
		ExpertLayers: 30, HeadDimKSWA: 256, HeadDimVSWA: 256, VocabSize: 262144}
	for i := 0; i < 30; i++ {
		l := gguf.Layer{HasKV: true, SWA: i%6 != 5, KVHeads: 8, HeadDimK: 256, HeadDimV: 256}
		if !l.SWA {
			l.KVHeads, l.HeadDimK, l.HeadDimV = 2, 512, 512
		}
		in.Layers = append(in.Layers, l)
	}
	return in
}

// denseInfo: a 14 GiB dense model, 40 layers, 8 KV heads x 128.
func denseInfo() gguf.Info {
	return gguf.Info{Arch: "llama", NLayer: 40, NKVHead: 8, HeadDimK: 128, HeadDimV: 128, DenseBytes: 14 << 30}
}

// lingInfo is Ling 3.0-tiny from its real header: 6 MLA layers (576-wide latent, no V) among
// 18 Kimi delta-attention layers that hold a fixed state; 23 MoE layers.
func lingInfo() gguf.Info {
	in := gguf.Info{Arch: "bailingmoe3", NLayer: 24, NKVHead: 1, HeadDimK: 576, NExpert: 128, NExpertUsed: 8,
		ExpertBytes: 4192468992, DenseBytes: 624909952 + 6516000, FileSize: 4823894944, ExpertLayers: 23,
		KVLoraRank: 512, StateBytesPerSlot: 20201472, VocabSize: 157184}
	for i := 0; i < 24; i++ {
		l := gguf.Layer{Recurrent: true}
		if i%4 == 3 {
			l = gguf.Layer{HasKV: true, KVHeads: 1, HeadDimK: 576}
		}
		in.Layers = append(in.Layers, l)
	}
	return in
}

// stamp is the time the tests give the book; the core never reads the clock.
var stamp = time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

var game = Wants{Slots: 3, MinSlots: 1, CtxMax: 65536, CtxFloor: 8192, KVType: "q4_0"}

// seed is Seed without its notes, for the tests that look only at the footprint.
func seed(info gguf.Info, kvType string) Footprint {
	fp, _, err := Seed(info, kvType)
	if err != nil {
		panic(err)
	}
	return fp
}

func room(vram, others, ram int) Room {
	return Room{VRAMTotal: vram, Others: others, GameReserve: 1500, Margin: DefaultMargin(vram), RAMAvail: ram}
}

func TestSeedReferencePredictsMeasuredFootprint(t *testing.T) {
	fp := seed(refInfo(), "q4_0")
	got := fp.vram(3, 65536, 0)
	t.Logf("reference seed: fixed %.0f scratch %.0f perTok %.5f swaPerTok %.5f expert/layer %.1f -> 3x64k q4_0 n-cpu-moe 0: %.0f MiB (measured process use after requests: 12134, sweep.csv)",
		fp.FixedMiB, fp.ScratchMiB, fp.PerTokenMiB, fp.SWAPerTokenMiB, fp.ExpertMiBPerLayer, got)
	if math.Abs(got-12134) > 200 {
		t.Errorf("predicted %.0f MiB, measured 12134", got)
	}
	// 5 full layers x 2 heads x 1024 elts at 4.5 bit
	if want := 5 * 2 * 1024 * 4.5 / 8 / mib; math.Abs(fp.PerTokenMiB-want) > 1e-9 {
		t.Errorf("per token %.6f want %.6f", fp.PerTokenMiB, want)
	}
	if !fp.MoE || fp.ExpertLayers != 30 || math.Abs(fp.ExpertMiBPerLayer-245.9) > 0.5 {
		t.Errorf("%+v", fp)
	}
}

func TestSeedLingMLAAndState(t *testing.T) {
	fp := seed(lingInfo(), "q4_0")
	// 6 full layers x 576 latent elements at 4.5 bit, no V cache
	if want := 6 * 576 * 4.5 / 8 / mib; math.Abs(fp.PerTokenMiB-want) > 1e-9 {
		t.Errorf("per token %.7f want %.7f", fp.PerTokenMiB, want)
	}
	if math.Abs(fp.StatePerSlotMiB-19.26) > 0.01 || fp.SWAPerTokenMiB != 0 {
		t.Errorf("state %.3f", fp.StatePerSlotMiB)
	}
	if !fp.MoE || fp.ExpertLayers != 23 {
		t.Errorf("%+v", fp)
	}
	got := fp.vram(3, 65536, 0)
	t.Logf("Ling seed: fixed %.0f perTok %.6f state/slot %.2f expert/layer %.1f -> 3x64k q4_0: %.0f MiB", fp.FixedMiB, fp.PerTokenMiB, fp.StatePerSlotMiB, fp.ExpertMiBPerLayer, got)
	if math.Abs(got-5598) > 200 { // measured process use after requests, sweep.csv
		t.Errorf("3x64k predicted %.0f MiB, measured 5598", got)
	}
	// the state is per slot and does not grow with ctx
	if d := fp.vram(3, 8192, 0) - fp.vram(1, 8192, 0); math.Abs(d-2*((fp.PerTokenMiB+fp.ComputePerTokenMiB)*8192+fp.StatePerSlotMiB)) > 1e-6 {
		t.Errorf("slot cost %.3f", d)
	}
}

func TestSeedUnknownArchitectureNeverZero(t *testing.T) {
	fp, notes, _ := Seed(gguf.Info{Arch: "mystery", NLayer: 10, DenseBytes: 4 << 30}, "q4_0")
	logged := strings.Join(notes, "\n")
	if fp.PerTokenMiB <= 0 || !fp.Estimated || logged == "" {
		t.Errorf("%+v log %q", fp, logged)
	}
	fp = seed(gguf.Info{Arch: "mystery", NLayer: 2, Estimated: true, Note: "assumed", NKVHead: 4, HeadDimK: 64, HeadDimV: 64,
		Layers: []gguf.Layer{{HasKV: true, KVHeads: 4, HeadDimK: 64, HeadDimV: 64}, {HasKV: true, KVHeads: 4, HeadDimK: 64, HeadDimV: 64}}}, "q4_0")
	if fp.PerTokenMiB <= 0 || !fp.Estimated {
		t.Errorf("%+v", fp)
	}
}

func TestSeedKVTypes(t *testing.T) {
	q4 := seed(refInfo(), "q4_0").PerTokenMiB
	q8 := seed(refInfo(), "q8_0").PerTokenMiB
	f16 := seed(refInfo(), "f16").PerTokenMiB
	if math.Abs(q8/q4-8.5/4.5) > 1e-9 || math.Abs(f16/q4-16/4.5) > 1e-9 {
		t.Error(q4, q8, f16)
	}
}

func TestSeedDenseFullAttentionEveryLayer(t *testing.T) {
	fp := seed(denseInfo(), "f16")
	if want := 40.0 * 8 * 256 * 2 / mib; math.Abs(fp.PerTokenMiB-want) > 1e-9 || fp.MoE || fp.SWAPerTokenMiB != 0 {
		t.Errorf("%+v want per token %v", fp, want)
	}
}

func TestSeedSWAWithoutLayers(t *testing.T) {
	in := gguf.Info{Arch: "x", NLayer: 10, NKVHead: 4, HeadDimK: 64, HeadDimV: 64, SlidingWindow: 512, SWALayers: 8, DenseBytes: 1 << 30}
	fp := seed(in, "f16")
	if fp.PerTokenMiB == 0 || fp.SWAPerTokenMiB == 0 || fp.SWAWindow != 512+ubatchTokens {
		t.Errorf("%+v", fp)
	}
	// the window part must not grow with ctx (the scratch that scales with ctx does)
	if a, b := fp.kvAt(1, 4096), fp.kvAt(1, 8192); math.Abs((b-a)-(fp.PerTokenMiB+fp.ComputePerTokenMiB)*4096) > 1e-6 {
		t.Errorf("SWA part grew with ctx: %v %v", a, b)
	}
}

type tcase struct {
	name       string
	fp         Footprint
	room       Room
	wantErr    bool
	slots      int
	ncpuMin    int
	ncpuMax    int
	ctxMin     int
	cpuOnly    bool
	checkExact bool
}

func TestChooseTable(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	ling := seed(lingInfo(), "q4_0")
	cases := []tcase{
		{name: "16GB card, desktop 1255 others, reference model, full wants", fp: ref, room: room(16303, 1255, 24000), slots: 3, ncpuMax: 0, ctxMin: 65536},
		{name: "16GB card, 2800 others (browser + game loaded): ctx shrinks, experts stay on the card", fp: ref, room: room(16303, 2800, 24000), slots: 3, ncpuMin: 0, ncpuMax: 0, ctxMin: 32768},
		{name: "12GB card", fp: ref, room: room(12288, 800, 24000), slots: 3, ncpuMin: 8, ncpuMax: 14, ctxMin: 65536},
		{name: "8GB card", fp: ref, room: room(8192, 600, 24000), slots: 3, ncpuMin: 22, ncpuMax: 30, ctxMin: 65536},
		{name: "Ling, 8GB card", fp: ling, room: room(8192, 600, 24000), slots: 3, ncpuMin: 0, ncpuMax: 0, ctxMin: 32768},
		{name: "Ling, 6GB card", fp: ling, room: room(6144, 600, 24000), slots: 3, ncpuMin: 14, ncpuMax: 23, ctxMin: 65536},
		{name: "Ling, 16GB card", fp: ling, room: room(16303, 1255, 24000), slots: 3, ncpuMax: 0, ctxMin: 65536},
		{name: "Ling, no GPU, 16GB RAM free", fp: ling, room: Room{NoGPU: true, RAMAvail: 16000}, slots: 3, ctxMin: 65536, cpuOnly: true},
		{name: "8GB card, only 5 GB host RAM free: experts will not fit in RAM", fp: ref, room: room(8192, 600, 5000), wantErr: true},
		{name: "4GB card: dense part alone does not fit", fp: ref, room: room(4096, 500, 24000), wantErr: true},
		{name: "no GPU, 32GB RAM (24000 free)", fp: ref, room: Room{NoGPU: true, RAMAvail: 24000}, slots: 3, ctxMin: 8192, cpuOnly: true},
		{name: "no GPU, 12GB RAM free: model alone needs ~11 GiB", fp: ref, room: Room{NoGPU: true, RAMAvail: 12000}, wantErr: true},
		{name: "Apple unified 16GB (usable 11468): no expert offload", fp: ref, room: Room{VRAMTotal: 11468, GameReserve: 1500, Margin: 512, RAMAvail: 12000, Unified: true}, wantErr: true},
		{name: "Apple unified 32GB (usable 22937)", fp: ref, room: Room{VRAMTotal: 22937, GameReserve: 1500, Margin: 917, RAMAvail: 28000, Unified: true}, slots: 3, ctxMin: 65536},
		{name: "dense 14GiB on 8GB card: never split", fp: seed(denseInfo(), "q4_0"), room: room(8192, 600, 64000), wantErr: true},
		{name: "dense 14GiB on 24GB card", fp: seed(denseInfo(), "q4_0"), room: room(24564, 800, 64000), slots: 3, ctxMin: 8192},
		{name: "dense 14GiB, no GPU, plenty of RAM: CPU only is allowed", fp: seed(denseInfo(), "q4_0"), room: Room{NoGPU: true, RAMAvail: 32000}, slots: 3, ctxMin: 8192, cpuOnly: true},
	}
	for _, c := range cases {
		s, err := Choose(c.fp, c.room, game)
		if c.wantErr {
			if !errors.Is(err, ErrTooBig) {
				t.Errorf("%s: want ErrTooBig, got %v (%v)", c.name, err, s)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		t.Logf("%-70s -> %v", c.name, s)
		if s.Slots != c.slots || s.NCpuMoe < c.ncpuMin || s.NCpuMoe > c.ncpuMax || s.CtxPerSlot < c.ctxMin || s.CPUOnly != c.cpuOnly {
			t.Errorf("%s: got %v", c.name, s)
		}
		if s.CtxPerSlot%1024 != 0 || s.CtxPerSlot > 65536 {
			t.Errorf("%s: ctx %d not aligned or over max", c.name, s.CtxPerSlot)
		}
		if !s.CPUOnly {
			if float64(s.VRAMMiB) > vramBudget(c.room)+1 {
				t.Errorf("%s: vram %d over budget %.0f", c.name, s.VRAMMiB, vramBudget(c.room))
			}
		}
		if c.fp.MoE == false && s.NCpuMoe != 0 {
			t.Errorf("%s: dense model got n-cpu-moe %d", c.name, s.NCpuMoe)
		}
	}
}

func TestChooseOrderCtxBeforeExpertsBeforeSlots(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	// Find a budget where full ctx does not fit but floor ctx does with zero CPU layers:
	// ctx must shrink first, experts stay on the GPU.
	need := func(c int) float64 { return ref.vram(3, c, 0) }
	r := room(16303, 1255, 24000)
	r.VRAMTotal = int(need(40960)) + r.Others + r.GameReserve + r.Margin + 1
	s, err := Choose(ref, r, game)
	if err != nil || s.NCpuMoe != 0 || s.Slots != 3 || s.CtxPerSlot < 40960-1024 || s.CtxPerSlot >= 65536 {
		t.Errorf("ctx should shrink first: %v %v", s, err)
	}
	// Below half the max ctx at zero layers: layers move to RAM and ctx grows back past half
	r.VRAMTotal = int(need(20480)) + r.Others + r.GameReserve + r.Margin
	s, err = Choose(ref, r, game)
	if err != nil || s.NCpuMoe < 1 || s.Slots != 3 || s.CtxPerSlot <= 32768 {
		t.Errorf("experts should move before ctx drops under half: %v %v", s, err)
	}
	// equal ctx: the fewer CPU layers wins
	if s.NCpuMoe > 1 {
		lower := ref.vram(3, s.CtxPerSlot, s.NCpuMoe-1)
		if lower <= vramBudget(r) {
			t.Errorf("a smaller n-cpu-moe also fits: %v", s)
		}
	}
	// Below the floor ctx at zero layers: layers move, ctx grows back
	r.VRAMTotal = int(need(8192)) + r.Others + r.GameReserve + r.Margin - 300
	s, err = Choose(ref, r, game)
	if err != nil || s.NCpuMoe < 1 || s.Slots != 3 || s.CtxPerSlot < 8192 {
		t.Errorf("experts should move next: %v %v", s, err)
	}
	// Slots go last: only room for one slot's floor
	oneSlot := room(8192, 600, 24000)
	oneSlot.VRAMTotal = int(ref.vram(1, 8192, 30)) + oneSlot.Others + oneSlot.GameReserve + oneSlot.Margin + 5
	s, err = Choose(ref, oneSlot, game)
	if err != nil || s.Slots != 1 || s.NCpuMoe != 30 {
		t.Errorf("fewer slots last: %v %v", s, err)
	}
	// MinSlots respected
	w := game
	w.MinSlots = 3
	if _, err = Choose(ref, oneSlot, w); !errors.Is(err, ErrTooBig) {
		t.Errorf("min slots must hold: %v", err)
	}
}

func TestChooseWantsDefaultsAndCap(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	s, err := Choose(ref, room(16303, 600, 24000), Wants{})
	if err != nil || s.Slots != 1 || s.CtxPerSlot != 65536 || s.KVType != "q4_0" {
		t.Errorf("%v %v", s, err)
	}
	capped := ref
	capped.CapCtxTotal = 100000
	s, err = Choose(capped, room(16303, 600, 24000), game)
	if err != nil || s.CtxTotal() > 100000 {
		t.Errorf("cap ignored: %v %v", s, err)
	}
	// f16 KV needs more room than q4_0
	f16 := seed(refInfo(), "f16")
	w := game
	w.KVType = "f16"
	s1, _ := Choose(f16, room(16303, 1255, 24000), w)
	s2, _ := Choose(ref, room(16303, 1255, 24000), game)
	if s1.CtxPerSlot >= s2.CtxPerSlot && s1.NCpuMoe <= s2.NCpuMoe {
		t.Errorf("f16 should cost more: %v vs %v", s1, s2)
	}
}

func TestArgs(t *testing.T) {
	got := Args(Shape{Slots: 3, CtxPerSlot: 65536, NCpuMoe: 4, KVType: "q4_0"})
	want := []string{"-ngl", "999", "-np", "3", "-c", "196608", "--n-cpu-moe", "4", "-ctk", "q4_0", "-ctv", "q4_0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%v", got)
	}
	got = Args(Shape{Slots: 1, CtxPerSlot: 8192, CPUOnly: true, NCpuMoe: 9})
	want = []string{"-ngl", "0", "-np", "1", "-c", "8192", "-ctk", "q4_0", "-ctv", "q4_0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%v", got)
	}
}

func TestStepDownAndMissRetryAndGrow(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	r := room(16303, 1255, 24000)
	cur, _ := Choose(ref, r, game)
	// StepDown walks down to ErrTooBig, strictly shrinking each time
	seen := 0
	for {
		next, err := StepDown(ref, cur, r, game)
		if err != nil {
			if !errors.Is(err, ErrTooBig) {
				t.Fatal(err)
			}
			break
		}
		if !better(cur, next) {
			t.Fatalf("StepDown did not shrink: %v -> %v", cur, next)
		}
		cur = next
		if seen++; seen > 2000 {
			t.Fatal("StepDown does not terminate")
		}
	}
	if cur.Slots != 1 || cur.CtxPerSlot != 8192 || cur.NCpuMoe != 30 {
		t.Errorf("walk ended at %v", cur)
	}

	// MissRetry: the failed buffer (the KV of the failing ctx) forces a smaller ctx, never above cur-1024
	start := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	tight := room(16303, 4000, 24000) // others grew since planning
	next, err := MissRetry(ref, start, 900, tight, game)
	if err != nil || !better(start, next) {
		t.Fatalf("miss retry: %v %v", next, err)
	}
	if next.CtxPerSlot > 65536-1024 && next.NCpuMoe == 0 {
		t.Errorf("must shrink: %v", next)
	}
	t.Logf("MissRetry 3x64k, failed buffer 900 MiB, others 4000: %v", next)
	// at the floor with nothing left to cut at this shape: moves on (experts)
	atFloor := Shape{Slots: 3, CtxPerSlot: 8192, KVType: "q4_0"}
	next, err = MissRetry(ref, atFloor, 100, room(16303, 1255, 24000), game)
	if err != nil || next.NCpuMoe != 1 || next.CtxPerSlot != 8192 {
		t.Errorf("floor miss: %v %v", next, err)
	}
	// a miss that the planner alone would not predict: failed buffer larger than the planner's view
	next, err = MissRetry(ref, start, 6000, room(16303, 1255, 24000), game)
	if err != nil || next.CtxPerSlot > 65536-6000*1024/int(ref.PerTokenMiB*1024*3)/1 && next.NCpuMoe == 0 {
		t.Logf("big-buffer miss -> %v (%v)", next, err)
	}

	// Grow: bigger room offers a bigger shape; same room does not
	small, _ := Choose(ref, room(16303, 5000, 24000), game)
	if g, ok, _ := Grow(ref, small, room(16303, 1255, 24000), game); !ok || !better(g, small) {
		t.Errorf("grow: %v %v", g, ok)
	}
	best, _ := Choose(ref, room(16303, 1255, 24000), game)
	if g, ok, _ := Grow(ref, best, room(16303, 1255, 24000), game); ok {
		t.Errorf("grow at best: %v", g)
	}
}

func TestBook(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	shape := Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}
	var b Book
	key := BookKey("sha", "RTX 5080", "cuda-linux-x64")
	// within +-15 %: accepted, and the footprint now carries the measured fixed load
	if !taken(b.Calibrate(key, ref, shape, 12500, stamp)) {
		t.Fatal("12500 should calibrate")
	}
	fp := b.Footprint(key, ref)
	if math.Abs(fp.vram(3, 65536, 0)-12500) > 1 {
		t.Errorf("calibrated model predicts %.0f, measured 12500", fp.vram(3, 65536, 0))
	}
	// a wild reading (others changed during load) is refused and changes nothing
	if taken(b.Calibrate(key, ref, shape, 16000, stamp)) || taken(b.Calibrate(key, ref, shape, 8000, stamp)) {
		t.Error("out-of-range reading accepted")
	}
	if got := b.Footprint(key, ref).FixedMiB; math.Abs(got-fp.FixedMiB) > 1e-9 {
		t.Error("book changed by refused reading")
	}
	if e := b.Entries[key]; e.Loads != 1 || !e.Updated.Equal(stamp) { // the stamp is the caller's
		t.Errorf("%+v", e)
	}
	// runtime OOM lowers the cap; the next plan honours it; a later OOM never raises it (LetGo lifts it)
	b.LowerCap(key, ref, shape, 150000, room(16303, 600, 24000), "cuda-linux-x64", stamp)
	b.LowerCap(key, ref, shape, 170000, room(16303, 600, 24000), "cuda-linux-x64", stamp)
	fp = b.Footprint(key, ref)
	if fp.CapCtxTotal != 150000 {
		t.Errorf("cap %d", fp.CapCtxTotal)
	}
	s, _ := Choose(fp, room(16303, 600, 24000), game)
	if s.CtxTotal() > 150000 {
		t.Errorf("cap ignored %v", s)
	}
	// other keys are untouched; nil book is harmless
	if got := b.Footprint("other", ref); got != ref {
		t.Error("foreign key altered footprint")
	}
	var nb *Book
	if nb.Footprint(key, ref) != ref {
		t.Error("nil book")
	}
	// CPU-only runs never calibrate a GPU fixed load
	if taken(b.Calibrate("k2", ref, Shape{Slots: 1, CtxPerSlot: 8192, CPUOnly: true}, 12000, stamp)) {
		t.Error("cpu-only calibrated")
	}
}

func TestCalibrateWithExpertsOnCPU(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	sh := Shape{Slots: 3, CtxPerSlot: 32768, NCpuMoe: 10, KVType: "q4_0"}
	truth := ref
	truth.FixedMiB += 150 // the real fixed load is 150 MiB above the seed
	measured := int(truth.vram(3, 32768, 10))
	var b Book
	if !taken(b.Calibrate("k", ref, sh, measured, stamp)) {
		t.Fatal("not accepted")
	}
	if got := b.Entries["k"].FixedMiB; math.Abs(got-truth.FixedMiB) > 1 {
		t.Errorf("fixed %.0f want %.0f", got, truth.FixedMiB)
	}
}

// With the real reference file present, the header-based seed must predict near the measured use.
func TestReferenceFromFile(t *testing.T) {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, ".hive/models/gemma4-rp-maxxed/gemma4-rp-maxxed-Q3_K_XL.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skip("reference model not on this machine")
	}
	fh, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Parse(fh, st.Size())
	if err != nil {
		t.Fatal(err)
	}
	in, err := f.Info()
	if err != nil {
		t.Fatal(err)
	}
	fp := seed(in, "q4_0")
	got := fp.vram(3, 65536, 0)
	t.Logf("from the real header: %.0f MiB at 3x64k q4_0 n-cpu-moe 0 (measured 12134)", got)
	if math.Abs(got-12134) > 200 {
		t.Errorf("predicted %.0f", got)
	}
	if !reflect.DeepEqual(in.Layers, refInfo().Layers) || in.ExpertBytes != refInfo().ExpertBytes {
		t.Error("refInfo() fixture is out of date with the real header")
	}
}

func TestJSONTags(t *testing.T) {
	for _, v := range []any{Wants{}, Room{}, Footprint{}, Shape{}, Calib{}, Book{}, CapLift{}, HeldCap{}, Reading{}, Calibration{}, Consts{}, Const{}} {
		rt := reflect.TypeOf(v)
		for i := 0; i < rt.NumField(); i++ {
			if rt.Field(i).Tag.Get("json") == "" {
				t.Errorf("%s.%s has no json tag", rt.Name(), rt.Field(i).Name)
			}
		}
	}
}
