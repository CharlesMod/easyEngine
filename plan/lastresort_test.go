package plan

import (
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/probe"
)

const (
	e2b = "gemma4-e2b-q4km"
	e4b = "gemma4-e4b-rp-q40"
	big = "gemma4-rp-maxxed-q3kxl"
)

// sweep plans the embedded catalog on each machine hw(i) for i in [from, to) and returns the first
// plan for which want holds, or fails.
func sweep(t *testing.T, what string, from, to, step int, hw func(int) probe.Hardware, want func(map[string]Verdict) bool) PlanResult {
	t.Helper()
	c := testCatalog()
	if m, ok := c.Model(e2b); !ok || !m.Fallback {
		t.Fatalf("the embedded E2B is not the fallback: %+v", m)
	}
	for i := from; i < to; i += step {
		p := Plan(c, hw(i), game, 1500, func(catalog.Model) bool { return false })
		if want(verdicts(p)) {
			return p
		}
	}
	t.Fatalf("no machine in the sweep where %s", what)
	return PlanResult{}
}

// The fallback model is recommended only when nothing else can run: a model that fits tight beats
// it fitting with room to spare.
func TestFallbackLosesToATightModelOnACard(t *testing.T) {
	// a machine with little RAM, so the 26B cannot lay its expert layers there and does not fit
	p := sweep(t, "the E4B fits tight, the E2B fits, the 26B does not", 2048, 16384, 32,
		func(v int) probe.Hardware { return hwNvidia(v, 300, 3000, "NVIDIA GeForce RTX 3050") },
		func(v map[string]Verdict) bool {
			return v[e4b].Fits == "tight" && v[e2b].Fits == "yes" && v[big].Fits == "no"
		})
	if p.Recommended != e4b {
		t.Fatalf("recommended %q, want the E4B (tight) over the fallback E2B (yes): %+v", p.Recommended, verdicts(p))
	}
}

// Where only the fallback fits the card, it is offered.
func TestFallbackRecommendedWhenOnlyItFitsTheCard(t *testing.T) {
	p := sweep(t, "only the E2B fits the card", 2048, 16384, 64,
		func(v int) probe.Hardware { return hwNvidia(v, 300, 24000, "NVIDIA GeForce GTX 1650") },
		func(v map[string]Verdict) bool {
			return v[e2b].Fits != "no" && v[e4b].Fits == "no" && v[big].Fits == "no" && v[e2b].Shape.CPUOnly == false
		})
	if p.Recommended != e2b {
		t.Fatalf("recommended %q, want the E2B (the only one that runs): %+v", p.Recommended, verdicts(p))
	}
}

func cpuOnly(ram int) probe.Hardware {
	return probe.Hardware{OS: "linux", Arch: "amd64", RAMTotalMiB: ram + 4000, RAMAvailMiB: ram, DiskFreeMiB: 500000}
}

// With no card, the E4B is recommended whenever it fits, comfortable or not, ahead of the smaller
// fallback: on the smallest machine where both fit (the E4B there past the comfortable share of
// RAM, the E2B within it) and on a large one.
func TestFallbackLosesOnTheProcessor(t *testing.T) {
	p := sweep(t, "both fit on the processor", 1024, 65536, 64, cpuOnly,
		func(v map[string]Verdict) bool { return v[e4b].Fits != "no" && v[e2b].Fits != "no" })
	if v := verdicts(p); p.Recommended != e4b || v[e4b].EstRAMMiB <= v[e2b].EstRAMMiB {
		t.Fatalf("recommended %q, want the E4B: %+v", p.Recommended, v)
	}
	p = Plan(testCatalog(), cpuOnly(60000), game, 1500, func(catalog.Model) bool { return false })
	if p.Recommended != e4b {
		t.Fatalf("recommended %q on a large machine, want the E4B: %+v", p.Recommended, verdicts(p))
	}
}

// With no card and room for the fallback alone, the fallback is recommended.
func TestFallbackRecommendedWhenOnlyItFitsTheProcessor(t *testing.T) {
	p := sweep(t, "only the E2B fits on the processor", 1024, 65536, 128, cpuOnly,
		func(v map[string]Verdict) bool { return v[e2b].Fits != "no" && v[e4b].Fits == "no" })
	if p.Recommended != e2b {
		t.Fatalf("recommended %q, want the E2B: %+v", p.Recommended, verdicts(p))
	}
}
