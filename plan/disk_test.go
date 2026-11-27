package plan

import (
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
)

func TestNeedOfRunningMaximum(t *testing.T) {
	steps := []DiskStep{
		{What: "a", Peak: 10, Keep: 4},
		{What: "b", Peak: 3, Keep: 3},
		{What: "c", Peak: 8, Keep: 1},
	}
	const logs, host = 1000, 2000
	n := NeedOf(steps, logs, host)
	if n.Peak != 15 { // max(10, 4+3, 7+8)
		t.Errorf("peak = %d, want 15", n.Peak)
	}
	head := int64(DiskHeadroomMiB) << 20
	if n.Headroom != head || n.Logs != logs || n.Host != host {
		t.Errorf("parts = %+v", n)
	}
	if want := 15 + logs + host + head; n.Total != want {
		t.Errorf("total = %d, want %d", n.Total, want)
	}
	if !n.Downloads() || len(n.Steps) != 3 {
		t.Errorf("steps = %+v", n.Steps)
	}
}

func TestNeedOfKeepAboveEveryPeak(t *testing.T) {
	// the base after the last step can exceed every step's own peak
	n := NeedOf([]DiskStep{{Peak: 1, Keep: 5}, {Peak: 1, Keep: 5}}, 0, 0)
	if n.Peak != 10 {
		t.Errorf("peak = %d, want 10", n.Peak)
	}
}

func TestNeedOfSkipsEmptyStepsAndNegatives(t *testing.T) {
	n := NeedOf([]DiskStep{{What: "none"}, {What: "neg", Peak: -5, Keep: -5}}, -7, -9)
	if n.Downloads() || len(n.Steps) != 0 {
		t.Errorf("empty steps kept: %+v", n.Steps)
	}
	if n.Logs != 0 || n.Host != 0 || n.Peak != 0 {
		t.Errorf("negatives not clamped: %+v", n)
	}
	if n.Total != int64(DiskHeadroomMiB)<<20 {
		t.Errorf("total = %d, want the headroom alone", n.Total)
	}
	if n := NeedOf(nil, 5, 5); n.Downloads() {
		t.Error("no steps must not download")
	}
}

func TestShortBoundary(t *testing.T) {
	n := NeedOf([]DiskStep{{Peak: 100, Keep: 100}}, 10, 10)
	if n.Short(n.Total) {
		t.Error("exactly Total free must not be short")
	}
	if !n.Short(n.Total - 1) {
		t.Error("Total-1 free must be short")
	}
	if n.Short(-1) {
		t.Error("unknown free space (-1) must not be short")
	}
	if !n.Short(0) {
		t.Error("0 free is short")
	}
}

func TestBackendStepsExact(t *testing.T) {
	a := catalog.Asset{ID: "b", Size: 10000, UnpackedSize: 50000, Extra: []string{"x"}}
	cuda := catalog.Asset{ID: "x", Size: 5000, UnpackedSize: 6000}
	st := BackendSteps(a, []catalog.Asset{cuda})
	if len(st) != 2 {
		t.Fatalf("steps = %+v", st)
	}
	// 10000 rounds to 12288 (3 blocks); the archive and the tree sit side by side
	if st[0].What != "model server" || st[0].Peak != 12288+50000 || st[0].Keep != 50000 {
		t.Errorf("backend step = %+v", st[0])
	}
	// archive 5000 -> 8192; archive+tree = 14192 beats 2*tree = 12000
	if st[1].What != "graphics runtime" || st[1].Peak != 8192+6000 || st[1].Keep != 6000 {
		t.Errorf("extra step = %+v", st[1])
	}
	// the tree dwarfs the archive: two trees (the copy) is the peak
	small := catalog.Asset{Size: 1000, UnpackedSize: 9000}
	if s := BackendSteps(a, []catalog.Asset{small}); s[1].Peak != 18000 || s[1].Keep != 9000 {
		t.Errorf("2*tree rule: %+v", s[1])
	}
}

func TestBackendStepsUnpackedFallback(t *testing.T) {
	a := catalog.Asset{Size: 10000} // no UnpackedSize: 4x the archive
	st := BackendSteps(a, nil)
	if len(st) != 1 || st[0].Keep != 40000 || st[0].Peak != 12288+40000 {
		t.Errorf("fallback step = %+v", st)
	}
	if u := UVStep(catalog.Asset{Size: 1000}); u.Keep != 4000 || u.Peak != max(4096+4000, 8000) {
		t.Errorf("uv fallback = %+v", u)
	}
}

func TestUVStepExact(t *testing.T) {
	u := UVStep(catalog.Asset{Size: 5000, UnpackedSize: 20000})
	if u.What != "Python installer" || u.Keep != 20000 || u.Peak != 40000 { // 8192+20000 < 2*20000
		t.Errorf("uv step = %+v", u)
	}
}

func TestModelStepPartCredit(t *testing.T) {
	const size = 1 << 20
	full := ModelStep("m", size, 0)
	if full.Peak != size || full.Keep != size {
		t.Errorf("fresh = %+v", full)
	}
	half := ModelStep("m", size, size/2)
	if half.Peak != size/2 || half.Keep != size/2 {
		t.Errorf("half done = %+v", half)
	}
	// the remainder rounds up to a block
	odd := ModelStep("m", 100000, 30000)
	if odd.Peak != roundBlock(70000) || odd.Peak%4096 != 0 {
		t.Errorf("rounding = %+v", odd)
	}
	// a .part at or past the size is no credit (it will be refetched)
	for _, part := range []int64{size, size + 5000} {
		if s := ModelStep("m", size, part); s.Peak != size {
			t.Errorf("part %d gave credit: %+v", part, s)
		}
	}
}

func TestAgentStepHaveAbovePeak(t *testing.T) {
	rt := catalog.RuntimeSize{PeakBytes: 1000, KeepBytes: 800}
	if s := AgentStep(rt, 5000); s.Peak != 0 || s.Keep != 0 {
		t.Errorf("have > peak: %+v", s)
	}
	if s := AgentStep(rt, 300); s.Peak != 700 || s.Keep != 500 {
		t.Errorf("partial: %+v", s)
	}
	if s := AgentStep(rt, 0); s.What != "agent runtime" || s.Peak != 1000 || s.Keep != 800 {
		t.Errorf("none on disk: %+v", s)
	}
}

func TestExplainPartsAddUp(t *testing.T) {
	const g = int64(1) << 30
	steps := []DiskStep{
		{What: "model server", Peak: g + g/2, Keep: g},
		{What: "graphics runtime", Peak: 2 * g, Keep: g / 2},
		{What: "Gemma", Peak: 4 * g, Keep: 4 * g},
	}
	n := NeedOf(steps, g, g)
	txt := n.Explain()
	for _, want := range []string{
		"needs " + GB(n.Total) + ":",
		"model server " + GB(g),
		"graphics runtime " + GB(g/2),
		"Gemma " + GB(4*g),
		"logs " + GB(g),
		"the host's files " + GB(g),
		"headroom " + GB(n.Headroom),
	} {
		if !strings.Contains(txt, want) {
			t.Errorf("%q missing from %q", want, txt)
		}
	}
	// peak is max(1.5, 1+2, 1.5+4)=5.5 GiB; kept is 5.5 GiB: no temporary part
	if n.Peak != g*11/2 {
		t.Fatalf("peak = %d", n.Peak)
	}
	if strings.Contains(txt, "temporary files") {
		t.Errorf("no temporary room expected: %q", txt)
	}
}

func TestExplainTemporaryAndOptionalParts(t *testing.T) {
	n := NeedOf([]DiskStep{{What: "model server", Peak: 3 << 30, Keep: 1 << 30}}, 0, 0)
	txt := n.Explain()
	if !strings.Contains(txt, "temporary files while unpacking "+GB(2<<30)) {
		t.Errorf("temporary part: %q", txt)
	}
	if strings.Contains(txt, "logs") || strings.Contains(txt, "host") {
		t.Errorf("zero parts must be left out: %q", txt)
	}
	if !strings.Contains(txt, "headroom") {
		t.Errorf("headroom always named: %q", txt)
	}
}

func TestGBFormat(t *testing.T) {
	if got := GB(0); got != "0 MB" {
		t.Errorf("GB(0) = %q", got)
	}
	if got := GB(1); got != "1 MB" {
		t.Errorf("GB(1) = %q", got)
	}
	if got := GB(100 << 20); got != "0.1 GB" {
		t.Errorf("GB(100MiB) = %q", got)
	}
	if got := GB(3 << 30); got != "3.0 GB" {
		t.Errorf("GB(3GiB) = %q", got)
	}
}
