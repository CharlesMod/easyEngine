package fit

import (
	"sort"

	"hermesengine/catalog"
)

// TPSPoint is a measured per-stream decode speed with n expert layers on the CPU.
type TPSPoint = catalog.TPSPoint

// TPSCurve is decode speed (tokens/s per stream, 3 slots decoding at once, so a lower bound for
// fewer slots) against --n-cpu-moe, measured on one reference machine.
type TPSCurve struct {
	Points []TPSPoint `json:"points,omitempty"`
}

// At returns the speed with n expert layers on the CPU. Between points it interpolates the
// time per token (which is linear in n: each CPU layer costs a fixed number of milliseconds), and
// past the last point it extends the last segment's slope. ok is false for an empty curve.
func (c *TPSCurve) At(n int) (float64, bool) {
	if c == nil {
		return 0, false
	}
	p := append([]TPSPoint(nil), c.Points...)
	if len(p) == 0 {
		return 0, false
	}
	sort.Slice(p, func(i, j int) bool { return p[i].NCpuMoe < p[j].NCpuMoe })
	if len(p) == 1 {
		return p[0].TPS, true
	}
	i := sort.Search(len(p), func(i int) bool { return p[i].NCpuMoe >= n })
	switch {
	case i == 0:
		i = 1
	case i == len(p):
		i = len(p) - 1
	}
	a, b := p[i-1], p[i]
	if b.NCpuMoe == a.NCpuMoe || a.TPS <= 0 || b.TPS <= 0 {
		return b.TPS, true
	}
	ta, tb := 1/a.TPS, 1/b.TPS
	t := ta + (tb-ta)*float64(n-a.NCpuMoe)/float64(b.NCpuMoe-a.NCpuMoe)
	if t <= 0 {
		return p[0].TPS, true
	}
	return 1 / t, true
}
