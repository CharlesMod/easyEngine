package plan

import (
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"

	"hermesengine/catalog"
	"hermesengine/probe"
)

// Each "no" names what really stops the model.
func TestPlanLimits(t *testing.T) {
	c := testCatalog()
	none := func(catalog.Model) bool { return false }
	disk := hwNvidia(24576, 300, 60000, "Tesla P40")
	disk.DiskFreeMiB = 2000
	cases := []struct {
		name  string
		hw    probe.Hardware
		id    string
		limit string
		says  []string
	}{
		// the 26B MoE on an 8 GB card with little RAM free: the card holds its part, RAM cannot
		// hold the expert layers the card cannot (this said "needs ... VRAM" before)
		{"moe ram", hwNvidia(8192, 300, 3000, "NVIDIA GeForce RTX 4060"), "gemma4-rp-maxxed-q3kxl", LimitRAM,
			[]string{"MiB of system RAM for the", "expert layers the card cannot hold"}},
		{"cpu ram", probe.Hardware{OS: "linux", Arch: "amd64", RAMTotalMiB: 4000, RAMAvailMiB: 2000, DiskFreeMiB: 500000},
			"gemma4-e2b-q4km", LimitRAM, []string{"system RAM to run on the CPU", " 0 MiB available", "(no usable GPU)"}},
		{"card too small, cpu ram", hwNvidia(4096, 300, 6000, "NVIDIA GeForce RTX 3050 Laptop GPU"), "gemma4-rp-maxxed-q3kxl", LimitRAM,
			[]string{"system RAM to run on the CPU", "the card has under 2048 MiB left after the game's room"}},
		{"disk", disk, "gemma4-rp-maxxed-q3kxl", LimitDisk, []string{"not enough free disk: needs 9.8 GB: model 9.3 GB, headroom ", "2.0 GB free"}},
	}
	for _, tc := range cases {
		v := verdicts(Plan(c, tc.hw, game, 2048, none))[tc.id]
		if v.Fits != "no" || v.Limit != tc.limit {
			t.Errorf("%s: fits=%s limit=%q (%s)", tc.name, v.Fits, v.Limit, v.Why)
			continue
		}
		for _, s := range tc.says {
			if !strings.Contains(v.Why, s) {
				t.Errorf("%s: %q lacks %q", tc.name, v.Why, s)
			}
		}
	}
	// what fits names no limit
	for _, v := range Plan(c, hwNvidia(24576, 300, 60000, "Tesla P40"), game, 2048, none).Models {
		if v.Fits == "no" || v.Limit != "" {
			t.Errorf("%s: fits=%s limit=%q", v.ID, v.Fits, v.Limit)
		}
	}
}

func TestWhyNotEachLimit(t *testing.T) {
	dense := fit.Footprint{FixedMiB: 5000, PerTokenMiB: 0.05, ScratchMiB: 500, WeightsMiB: 4000}
	moe, err := SeedFromSize(testCatalog().Models[0], "q4_0")
	if err != nil {
		t.Fatal(err)
	}
	card := func(total int) fit.Room { return fit.Room{VRAMTotal: total, Margin: 256, RAMAvail: 60000} }
	cases := []struct {
		name  string
		fp    fit.Footprint
		room  fit.Room
		w     fit.Wants
		err   error
		limit string
		says  string
	}{
		{"dense", dense, card(4096), game, fit.ErrTooBig, LimitDense, "dense models are never split between card and RAM"},
		{"moe card", moe, card(2400), game, fit.ErrTooBig, LimitVRAM, "VRAM even with every expert layer in system RAM"},
		{"moe whole", moe, card(8192), func() fit.Wants { w := game; w.Whole = true; return w }(), fit.ErrTooBig, LimitVRAM, "kept whole on the card"},
		{"unified", dense, fit.Room{VRAMTotal: 4096, Unified: true, Margin: 256, RAMAvail: 60000}, game, fit.ErrTooBig, LimitRAM, "system memory the GPU shares"},
		{"block", dense, fit.Room{VRAMTotal: 24576, Margin: 256, RAMAvail: 60000, LargestBlock: 1000}, game, fit.ErrTooBig, LimitBlock, "largest free block is 1000 MiB"},
		{"kv type", dense, card(24576), game, &fit.KVTypeError{Name: "q9"}, LimitKVType, "q9"},
		{"header", dense, card(24576), game, fit.ErrNoCtxTrain, LimitModel, "trained context"},
	}
	for _, tc := range cases {
		limit, why := whyNot(tc.fp, tc.room, tc.w, tc.err)
		if limit != tc.limit || !strings.Contains(why, tc.says) {
			t.Errorf("%s: %q %q", tc.name, limit, why)
		}
		if _, err := fit.Choose(tc.fp, tc.room, tc.w); tc.err == fit.ErrTooBig && err == nil {
			t.Errorf("%s: the case fits, so it tests nothing", tc.name)
		}
	}
}

// The plan's disk verdict uses the installer's count (DiskNeed, with DiskHeadroomMiB): a disk the
// installer would accept is not refused by the plan, and one MiB less is.
func TestPlanDiskHeadroomMatchesInstaller(t *testing.T) {
	c := testCatalog()
	none := func(catalog.Model) bool { return false }
	const id = "gemma4-rp-maxxed-q3kxl"
	var size int64
	for _, m := range c.Models {
		if m.ID == id {
			size = m.Size
		}
	}
	if size == 0 {
		t.Fatalf("%s not in the test catalog", id)
	}
	hw := hwNvidia(24576, 300, 60000, "Tesla P40")
	m, _ := c.Model(id)
	need := ModelOnlyNeed(m).Total
	if need < size+DiskHeadroomMiB<<20 {
		t.Fatalf("need %d is under size+headroom", need)
	}
	hw.DiskFreeMiB = int((need + mib - 1) / mib) // the installer's free MiB at exactly the need
	if v := verdicts(Plan(c, hw, game, 2048, none))[id]; v.Limit == LimitDisk {
		t.Errorf("exactly size+headroom free: refused for disk (%s)", v.Why)
	}
	hw.DiskFreeMiB--
	if v := verdicts(Plan(c, hw, game, 2048, none))[id]; v.Limit != LimitDisk {
		t.Errorf("one MiB short: fits=%s limit=%q, want disk", v.Fits, v.Limit)
	}
}
