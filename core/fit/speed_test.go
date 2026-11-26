package fit

import (
	"math"
	"testing"
)

func TestTPSCurveAt(t *testing.T) {
	var nilc *TPSCurve
	if _, ok := nilc.At(3); ok {
		t.Fatal("nil curve has no speed")
	}
	if _, ok := (&TPSCurve{}).At(3); ok {
		t.Fatal("empty curve has no speed")
	}
	c := &TPSCurve{Points: []TPSPoint{{NCpuMoe: 10, TPS: 20}, {NCpuMoe: 0, TPS: 40}}}
	if v, _ := c.At(0); v != 40 {
		t.Fatalf("at 0: %v", v)
	}
	// 1/tps is linear in n: 25 ms at 0, 50 ms at 10, so 37.5 ms at 5 = 26.67 tok/s.
	if v, _ := c.At(5); math.Abs(v-1000/37.5) > 0.01 {
		t.Fatalf("at 5: %v", v)
	}
	if v, _ := c.At(20); v >= 20 || v <= 0 {
		t.Fatalf("past the end should keep falling: %v", v)
	}
}
