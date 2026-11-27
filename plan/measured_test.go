package plan

import (
	"math"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"

	"github.com/CharlesMod/easyEngine/catalog"
)

func measuredFP() *catalog.Footprint {
	return &catalog.Footprint{
		FixedMiB:        3000,
		MiBPer1kCtx:     map[string]float64{"q8_0": 20, "q4_0": 10},
		StateMiBPerSlot: map[string]float64{"q8_0": 100, "q4_0": 60},
		ScratchMiB:      300, ExpertLayers: 30, ExpertMiBPerLayer: 200, ExpertRAMMiBPerLayer: 250, MoESchedMiB: 180,
		TPSByNCpuMoe: []fit.TPSPoint{{NCpuMoe: 0, TPS: 40}, {NCpuMoe: 10, TPS: 20}},
	}
}

// must is a footprint whose error the test does not expect.
func must(fp fit.Footprint, err error) fit.Footprint {
	if err != nil {
		panic(err)
	}
	return fp
}

func TestFromMeasuredUsesBlock(t *testing.T) {
	fp := must(FromMeasured(measuredFP(), "q4_0"))
	if fp.FixedMiB != 3000 || math.Abs(fp.PerTokenMiB-10.0/1024) > 1e-9 || fp.StatePerSlotMiB != 60 || !fp.MoE || fp.ExpertLayers != 30 || fp.TPS == nil {
		t.Fatalf("%+v", fp)
	}
	// q5_0 sits between q4_0 and q8_0 by bytes per element, so its slope is in between.
	mid := must(FromMeasured(measuredFP(), "q5_0"))
	if !(mid.PerTokenMiB > fp.PerTokenMiB && mid.PerTokenMiB < 20.0/1024) {
		t.Fatalf("interpolated slope %v", mid.PerTokenMiB)
	}
	if must(FromMeasured(measuredFP(), "")).PerTokenMiB != fp.PerTokenMiB {
		t.Fatal("default kv is q4_0")
	}
}

func TestSpeedScaleFor(t *testing.T) {
	m := measuredFP()
	m.MeasuredOn.GPU = "NVIDIA GeForce RTX 5080"
	if SpeedScaleFor("nvidia geforce rtx 5080 ", m) != 1 {
		t.Fatal("same card scales 1")
	}
	if SpeedScaleFor("NVIDIA GeForce RTX 3060", m) != OtherHardwareSpeed || SpeedScaleFor("", m) != OtherHardwareSpeed || SpeedScaleFor("x", nil) != OtherHardwareSpeed {
		t.Fatal("other hardware is scaled down")
	}
}

func TestSeedFromSizeUsesMeasured(t *testing.T) {
	m := catalog.Model{ID: "x", Size: 10 << 30, Footprint: measuredFP()}
	fp := must(SeedFromSize(m, "q4_0"))
	if fp.FixedMiB != 3000 || fp.Estimated {
		t.Fatalf("%+v", fp)
	}
	g := must(SeedFromSize(catalog.Model{ID: "y", Size: 10 << 30}, "q4_0"))
	if !g.Estimated {
		t.Fatal("a size guess is flagged estimated")
	}
}

func TestSpeedFloorPrefersSpeedOverCtx(t *testing.T) {
	fp := must(FromMeasured(measuredFP(), "q4_0"))
	// Card too small for the experts: n must be raised, but the floor of 20 caps n at 10.
	room := fit.Room{VRAMTotal: 3500, GameReserve: 0, Margin: 200, RAMAvail: 64000, SpeedScale: 1}
	s, err := fit.Choose(fp, room, fit.Wants{Slots: 1, CtxMax: 8192, CtxFloor: 2048, KVType: "q4_0"})
	if err != nil {
		t.Fatal(err)
	}
	if s.EstTPS < fit.DefaultMinTPS-0.01 {
		t.Fatalf("shape under the floor: %+v", s)
	}
	// On a machine half as fast the same card cannot hold the floor: it falls back to a slow shape.
	room.SpeedScale = 0.5
	s2, err := fit.Choose(fp, room, fit.Wants{Slots: 1, CtxMax: 8192, CtxFloor: 2048, KVType: "q4_0"})
	if err == nil && s2.EstTPS >= fit.DefaultMinTPS {
		t.Fatalf("expected a slow or refused shape, got %+v", s2)
	}
	// A negative floor turns the rule off.
	s3, err := fit.Choose(fp, room, fit.Wants{Slots: 1, CtxMax: 8192, CtxFloor: 2048, KVType: "q4_0", MinTPS: -1})
	if err != nil {
		t.Fatal(err)
	}
	if s3.NCpuMoe < s.NCpuMoe {
		t.Fatalf("without a floor n should not shrink: %d vs %d", s3.NCpuMoe, s.NCpuMoe)
	}
}
