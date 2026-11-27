package plan

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/core/gguf"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/probe"
)

// refInfo is the reference model's header as read by core/gguf (the core's fit tests hold the
// same fixture against the real file).
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

var game = fit.Wants{Slots: 3, MinSlots: 1, CtxMax: 65536, CtxFloor: 8192, KVType: "q4_0"}

func testCatalog() *catalog.Catalog {
	c, err := catalog.Load("")
	if err != nil {
		panic(err)
	}
	return c
}

func hwNvidia(vram, used, ram int, name string) probe.Hardware {
	return probe.Hardware{OS: "linux", Arch: "amd64", RAMTotalMiB: ram + 4000, RAMAvailMiB: ram, DiskFreeMiB: 500000,
		GPUs: []probe.GPU{{Vendor: "nvidia", Name: name, VRAMTotalMiB: vram, VRAMUsedMiB: used, Source: "nvidia-smi"}}}
}

func verdicts(p PlanResult) map[string]Verdict {
	m := map[string]Verdict{}
	for _, v := range p.Models {
		m[v.ID] = v
	}
	return m
}

func TestPlanTable(t *testing.T) {
	c := testCatalog()
	allInstalled := func(catalog.Model) bool { return true }
	noneInstalled := func(catalog.Model) bool { return false }
	refInfoFn := func(m catalog.Model) (gguf.Info, bool) {
		if m.ID == "gemma4-rp-maxxed-q3kxl" {
			return refInfo(), true
		}
		return gguf.Info{}, false
	}
	type pc struct {
		name      string
		hw        probe.Hardware
		installed func(catalog.Model) bool
		info      func(catalog.Model) (gguf.Info, bool)
		rec       string
		fits      map[string]string
	}
	apple := probe.Hardware{OS: "darwin", Arch: "arm64", RAMTotalMiB: 16384, RAMAvailMiB: 11000, DiskFreeMiB: 500000,
		GPUs: []probe.GPU{{Vendor: "apple", Name: "Apple M2", VRAMTotalMiB: 11468, UnifiedMemory: true, Source: "sysctl"}}}
	cases := []pc{
		{"16GB, reference installed with header", hwNvidia(16303, 1255, 24000, "NVIDIA GeForce RTX 5080"), allInstalled, refInfoFn,
			"gemma4-rp-maxxed-q3kxl", map[string]string{"gemma4-rp-maxxed-q3kxl": "yes", "gemma4-e4b-rp-q40": "yes", "gemma4-e2b-q4km": "yes"}},
		{"16GB, 2800 others: a few expert layers in RAM keep the long ctx", func() probe.Hardware { h := hwNvidia(16303, 2800, 24000, "NVIDIA GeForce RTX 5080"); return h }(), allInstalled, refInfoFn,
			"gemma4-rp-maxxed-q3kxl", map[string]string{"gemma4-rp-maxxed-q3kxl": "yes"}},
		{"16GB, nothing installed: reference is downloadable now", hwNvidia(16303, 1255, 24000, "NVIDIA GeForce RTX 5080"), noneInstalled, nil,
			"gemma4-rp-maxxed-q3kxl", nil},
		{"12GB", hwNvidia(12288, 800, 24000, "NVIDIA GeForce RTX 4070"), noneInstalled, nil, "gemma4-rp-maxxed-q3kxl", map[string]string{"gemma4-rp-maxxed-q3kxl": "yes", "gemma4-e4b-rp-q40": "yes", "gemma4-e2b-q4km": "yes"}},
		{"8GB", hwNvidia(8192, 600, 24000, "NVIDIA GeForce RTX 3060 Ti"), noneInstalled, nil, "gemma4-e4b-rp-q40", map[string]string{"gemma4-e4b-rp-q40": "yes", "gemma4-e2b-q4km": "yes", "gemma4-rp-maxxed-q3kxl": "tight"}},
		{"6GB", hwNvidia(6144, 600, 24000, "NVIDIA GeForce RTX 2060"), noneInstalled, nil, "gemma4-e4b-rp-q40", map[string]string{"gemma4-e4b-rp-q40": "yes", "gemma4-e2b-q4km": "yes", "gemma4-rp-maxxed-q3kxl": "tight"}},
		{"4GB: too little card, CPU-only and tight", hwNvidia(4096, 500, 24000, "NVIDIA GeForce GTX 1650"), noneInstalled, nil, "", map[string]string{"gemma4-e2b-q4km": "tight"}},
		{"no GPU 32GB", probe.Hardware{OS: "linux", Arch: "amd64", RAMTotalMiB: 32000, RAMAvailMiB: 24000, DiskFreeMiB: 500000}, noneInstalled, nil, "", map[string]string{"gemma4-e4b-rp-q40": "tight"}},
		{"no GPU 8GB RAM", probe.Hardware{OS: "linux", Arch: "amd64", RAMTotalMiB: 8000, RAMAvailMiB: 5000, DiskFreeMiB: 500000}, noneInstalled, nil, "", map[string]string{"gemma4-e4b-rp-q40": "no", "gemma4-rp-maxxed-q3kxl": "no"}},
		{"Apple unified 16GB", apple, noneInstalled, nil, "", map[string]string{"gemma4-rp-maxxed-q3kxl": "no"}},
		{"16GB but 3 GB of disk", func() probe.Hardware {
			h := hwNvidia(16303, 600, 24000, "NVIDIA GeForce RTX 5080")
			h.DiskFreeMiB = 3000
			return h
		}(), noneInstalled, nil, "", map[string]string{"gemma4-e4b-rp-q40": "no", "gemma4-rp-maxxed-q3kxl": "no"}},
	}
	for _, tc := range cases {
		p := PlanWith(c, tc.hw, game, 1500, tc.installed, tc.info)
		vs := verdicts(p)
		for _, v := range p.Models {
			t.Logf("[%s] %-26s %-5s %-s | %v", tc.name, v.ID, v.Fits, v.Why, v.Shape)
		}
		t.Logf("[%s] recommended %q backend %q", tc.name, p.Recommended, p.Backend)
		for id, want := range tc.fits {
			if vs[id].Fits != want {
				t.Errorf("%s: %s fits=%q want %q (%s)", tc.name, id, vs[id].Fits, want, vs[id].Why)
			}
		}
		if tc.rec != "" && p.Recommended != tc.rec {
			t.Errorf("%s: recommended %q want %q", tc.name, p.Recommended, tc.rec)
		}
		if len(p.Models) != len(c.Models) {
			t.Errorf("%s: %d verdicts", tc.name, len(p.Models))
		}
	}
}

func TestPlanRecommendationRules(t *testing.T) {
	c := &catalog.Catalog{Models: []catalog.Model{
		{ID: "big", Size: 20 << 30, Quality: 100, URL: "https://x"},
		{ID: "mid", Size: 6 << 30, Quality: 60, URL: "https://x"},
		{ID: "importonly", Size: 1 << 30, Quality: 99, URL: ""},
		{ID: "small", Size: 2 << 30, Quality: 40, URL: "https://x"},
	}}
	hw := hwNvidia(12288, 600, 24000, "NVIDIA GeForce RTX 4070")
	p := Plan(c, hw, game, 1500, func(catalog.Model) bool { return false })
	v := verdicts(p)
	if v["big"].Fits != "no" || !v["importonly"].ImportOnly || v["mid"].ImportOnly {
		t.Errorf("%+v", v)
	}
	if p.Recommended != "mid" { // import-only and not installed cannot be recommended despite quality 99
		t.Errorf("recommended %q", p.Recommended)
	}
	p = Plan(c, hw, game, 1500, func(m catalog.Model) bool { return m.ID == "importonly" })
	if p.Recommended != "importonly" {
		t.Errorf("installed import-only model should win: %q", p.Recommended)
	}
	// only a tight option: it is recommended
	hw4 := hwNvidia(4096, 500, 24000, "GTX")
	if p := Plan(c, hw4, game, 1500, nil); p.Recommended == "" || verdicts(p)[p.Recommended].Fits != "tight" {
		t.Errorf("tight fallback: %+v", p)
	}
	if p := Plan(&catalog.Catalog{}, hw, game, 1500, nil); p.Recommended != "" || len(p.Models) != 0 {
		t.Errorf("%+v", p)
	}
}

func TestPlanBackend(t *testing.T) {
	c := testCatalog()
	p := Plan(c, hwNvidia(16303, 600, 24000, "NVIDIA GeForce RTX 5080"), game, 1500, nil)
	if p.Backend != "cuda-linux-x64" {
		t.Errorf("backend %q", p.Backend)
	}
}

func TestPlanCarriesFallback(t *testing.T) {
	c := &catalog.Catalog{Models: []catalog.Model{
		{ID: "big", Size: 6 << 30, Quality: 60, URL: "https://x"},
		{ID: "tiny", Size: 2 << 30, Quality: 40, URL: "https://x", Fallback: true, FallbackNote: "smallest"},
	}}
	p := Plan(c, hwNvidia(12288, 600, 24000, "NVIDIA GeForce RTX 4070"), game, 1500, nil)
	v := verdicts(p)
	if v["big"].Fallback || v["big"].FallbackNote != "" || !v["tiny"].Fallback || v["tiny"].FallbackNote != "smallest" {
		t.Errorf("%+v", v)
	}
	b, _ := json.Marshal(p)
	if !strings.Contains(string(b), `"fallback":true`) || !strings.Contains(string(b), `"fallback_note":"smallest"`) {
		t.Errorf("json %s", b)
	}
}

func TestJSONTags(t *testing.T) {
	for _, v := range []any{Verdict{}, PlanResult{}} {
		rt := reflect.TypeOf(v)
		for i := 0; i < rt.NumField(); i++ {
			if rt.Field(i).Tag.Get("json") == "" {
				t.Errorf("%s.%s has no json tag", rt.Name(), rt.Field(i).Name)
			}
		}
	}
}
