package run

import (
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"
)

// The projector's room: margin made explicit, then widened; the budget shrinks by exactly it.
// vram is the room's VRAM budget in MiB.
func vram(t *testing.T, r fit.Room) float64 {
	t.Helper()
	v, _, err := r.Budget()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestProjectorRoom(t *testing.T) {
	const mib = 1300
	base := fit.Room{VRAMTotal: 16384}
	w := withProjectorRoom(base, mib)
	if want := fit.DefaultMargin(16384) + mib; w.Margin != want {
		t.Fatalf("margin 0 became %d, want %d", w.Margin, want)
	}
	if got := vram(t, base) - vram(t, w); got != float64(mib) {
		t.Fatalf("budget smaller by %v, want %d", got, mib)
	}
	if back := withoutProjectorRoom(w, mib); back.Margin != w.Margin-mib || vram(t, back) != vram(t, base) {
		t.Fatalf("without: margin %d budget %v, want budget %v", back.Margin, vram(t, back), vram(t, base))
	}
	// an explicit margin just grows
	e := withProjectorRoom(fit.Room{VRAMTotal: 16384, Margin: 700}, mib)
	if e.Margin != 700+mib {
		t.Fatalf("explicit margin %d", e.Margin)
	}
	// "no margin" (negative) is zero, then the projector's room
	n := fit.Room{VRAMTotal: 16384, Margin: -1}
	nw := withProjectorRoom(n, mib)
	if nw.Margin != mib || vram(t, n)-vram(t, nw) != mib {
		t.Fatalf("negative margin: %d, budget diff %v", nw.Margin, vram(t, n)-vram(t, nw))
	}
	// no card, or no projector: untouched
	cpu := fit.Room{NoGPU: true, VRAMTotal: 16384}
	if withProjectorRoom(cpu, mib).Margin != 0 || withoutProjectorRoom(cpu, mib).Margin != 0 {
		t.Fatal("a processor-only room was changed")
	}
	if r := withProjectorRoom(base, 0); r.Margin != base.Margin {
		t.Fatalf("0 MiB changed the room: %+v", r)
	}
}

func TestLayBackMismatchVision(t *testing.T) {
	a := ctxKey{Model: "m", Build: "b", KV: "q4_0"}
	b := a
	b.Vision = "abc"
	for _, c := range [][2]ctxKey{{a, b}, {b, a}} {
		if got := layBackMismatch(c[0], c[1]); got != "the vision projector was added or removed" {
			t.Errorf("%+v -> %+v: %q", c[0], c[1], got)
		}
	}
	if !strings.HasSuffix(b.String(), "|mmproj:abc") || strings.Contains(a.String(), "mmproj") {
		t.Errorf("keys %q %q", a.String(), b.String())
	}
}
