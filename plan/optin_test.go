package plan

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/probe"
)

func TestPlanOptInIsNeverRecommended(t *testing.T) {
	c := &catalog.Catalog{Models: []catalog.Model{
		{ID: "mid", Name: "Mid", Codename: "Flashlight", Size: 6 << 30, Quality: 60, URL: "https://x"},
		{ID: "small", Size: 2 << 30, Quality: 40, URL: "https://x"},
		{ID: "bonfire", Name: "Big", Codename: "Bonfire", Size: 7 << 30, Quality: 120, URL: "https://x", OptIn: true},
	}}
	hw := hwNvidia(12288, 600, 24000, "NVIDIA GeForce RTX 4070")
	none := func(catalog.Model) bool { return false }
	p := Plan(c, hw, game, 1500, none)
	v := verdicts(p)
	if !v["bonfire"].OptIn || v["bonfire"].Fits == "no" || v["mid"].OptIn {
		t.Fatalf("verdicts: %+v", v)
	}
	if p.Recommended != "mid" {
		t.Errorf("recommended %q, want the best non-opt-in that fits", p.Recommended)
	}
	if v["bonfire"].Codename != "Bonfire" || v["mid"].Codename != "Flashlight" || v["small"].Codename != "" {
		t.Errorf("codenames not copied: %+v", v)
	}
	// installed or not, the opt-in model is not recommended
	if p := Plan(c, hw, game, 1500, func(catalog.Model) bool { return true }); p.Recommended == "bonfire" {
		t.Errorf("installed opt-in model recommended")
	}
	// only the opt-in model fits: it is still not recommended
	only := &catalog.Catalog{Models: []catalog.Model{
		{ID: "huge", Size: 40 << 30, Quality: 100, URL: "https://x"},
		c.Models[2],
	}}
	p = Plan(only, hw, game, 1500, none)
	if verdicts(p)["bonfire"].Fits == "no" {
		t.Fatalf("opt-in model should fit: %+v", p.Models)
	}
	if p.Recommended == "bonfire" {
		t.Errorf("recommended the only fitting model though opt-in")
	}
	// the verdict JSON carries codename always and opt_in only when set
	b, _ := json.Marshal(v["bonfire"])
	s, _ := json.Marshal(v["mid"])
	if !strings.Contains(string(b), `"opt_in":true`) || strings.Contains(string(s), "opt_in") || !strings.Contains(string(s), `"codename":"Flashlight"`) {
		t.Errorf("verdict JSON: %s / %s", b, s)
	}
}

func TestPlanCPUOnlyNeverPicksOptIn(t *testing.T) {
	c := &catalog.Catalog{Models: []catalog.Model{
		{ID: "e4b", Size: 3 << 30, Quality: 60, URL: "https://x"},
		{ID: "tinyoptin", Size: 1 << 30, Quality: 120, URL: "https://x", OptIn: true},
	}}
	hw := probe.Hardware{OS: "linux", Arch: "amd64", RAMTotalMiB: 32000, RAMAvailMiB: 24000, DiskFreeMiB: 500000}
	p := Plan(c, hw, game, 1500, func(catalog.Model) bool { return false })
	if v := verdicts(p)["tinyoptin"]; v.Fits == "no" || !v.OptIn {
		t.Fatalf("opt-in model should be viable on CPU: %+v", p.Models)
	}
	if p.Recommended == "tinyoptin" {
		t.Errorf("CPU-only pick is the opt-in model")
	}
	only := &catalog.Catalog{Models: c.Models[1:]}
	if p := Plan(only, hw, game, 1500, nil); p.Recommended == "tinyoptin" {
		t.Errorf("CPU-only pick is the only (opt-in) model")
	}
}
