package lookimg

import (
	"image"
	"runtime"
	"runtime/debug"
	"testing"
	"time"
)

// allocBound is the TotalAlloc ceiling of one Prepare at the pixel limit with
// the garbage collector off (so it bounds the live peak from above); the
// worst measured kind is about 93 MiB.
const allocBound = 112 << 20

// measure runs one Prepare with the collector off and returns the bytes it
// allocated and how long it took.
func measure(t *testing.T, s string, edge int) (uint64, time.Duration) {
	t.Helper()
	runtime.GC()
	old := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(old)
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	t0 := time.Now()
	r, e := Prepare(ctx0, s, edge, Default)
	d := time.Since(t0)
	runtime.ReadMemStats(&b)
	if e != nil {
		t.Fatalf("Prepare: %v", e)
	}
	if r.W != edge || r.H != edge {
		t.Fatalf("size %dx%d, want %d", r.W, r.H, edge)
	}
	return b.TotalAlloc - a.TotalAlloc, d
}

// TestResourceBoundsAtPixelLimit measures every allowed kind at 4096 x 4096
// and fails if one Prepare allocates more than allocBound. The table in the
// package comment is this test's output (go test -v -run Resource).
func TestResourceBoundsAtPixelLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates hundreds of MiB")
	}
	all := append([]kind{}, kinds...)
	all = append(all, kind{"png-rgba-flat", func(t testing.TB, n int) []byte {
		m := image.NewNRGBA(image.Rect(0, 0, n, n))
		m.Pix[3] = 100
		return pngBytes(t, m)
	}})
	for _, k := range all {
		in := k.make(t, noiseSide)
		if len(in) > Default.MaxBytes {
			t.Fatalf("%s fixture is %d bytes, over the %d limit", k.name, len(in), Default.MaxBytes)
		}
		s := b64(in)
		in = nil
		for _, edge := range []int{1024, 2048} {
			n, d := measure(t, s, edge)
			t.Logf("%-14s file %5.2f MiB  edge %4d  alloc %3d MiB  %v", k.name, float64(len(s))*3/4/(1<<20), edge, n>>20, d.Round(time.Millisecond))
			if n > allocBound {
				t.Errorf("%s edge %d: allocated %d MiB, bound %d MiB", k.name, edge, n>>20, allocBound>>20)
			}
		}
	}
}

// TestScaleAllocatesOnlyTheTarget proves the scaler's allocations do not
// grow with the source: a 3000x3000 grey source and a 300x300 one scale to
// the same 100x100 target with the same number of allocations and bytes.
func TestScaleAllocatesOnlyTheTarget(t *testing.T) {
	run := func(n int) (mallocs, bytes uint64) {
		src := image.NewGray(image.Rect(0, 0, n, n))
		var a, b runtime.MemStats
		runtime.ReadMemStats(&a)
		if _, e := scale(src, 100, 100); e != nil {
			t.Fatal(e)
		}
		runtime.ReadMemStats(&b)
		return b.Mallocs - a.Mallocs, b.TotalAlloc - a.TotalAlloc
	}
	m1, b1 := run(300)
	m2, b2 := run(3000)
	if m2 > m1+3 || b2 > b1+4096 { // a little slack for the runtime's own background allocations
		t.Fatalf("allocations grew with the source: %d/%d bytes vs %d/%d", m1, b1, m2, b2)
	}
	if m1 > 12 || b1 > 100*100*4+16<<10 {
		t.Fatalf("scale made %d allocations, %d bytes", m1, b1)
	}
}
