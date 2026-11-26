package plan

import (
	"sort"
	"strings"

	"github.com/CharlesMod/easyEngine/core/fit"

	"hermesengine/catalog"
)

// OtherHardwareSpeed is the fraction of the measured decode speed the planner assumes on a GPU
// that is not the one the curve was measured on. Decode is memory-bandwidth bound: common 8-12 GB
// cards have 30-45 % of an RTX 5080's bandwidth and the host's RAM may be slower too, so half
// is the conservative guess (a faster card only gets a better result than planned).
const OtherHardwareSpeed = 0.5

// SpeedScaleFor is the Room.SpeedScale for a GPU: 1 on the card the footprint was measured on,
// else OtherHardwareSpeed.
func SpeedScaleFor(gpuName string, m *catalog.Footprint) float64 {
	if m != nil && gpuName != "" && strings.EqualFold(strings.TrimSpace(gpuName), strings.TrimSpace(m.MeasuredOn.GPU)) {
		return 1
	}
	return OtherHardwareSpeed
}

// interp reads a per-KV-type measured value at bytes-per-element b: exact for a measured type,
// else linear between the nearest measured types (past the ends: the nearest segment's slope,
// never below zero).
func interp(byKV map[string]float64, kvType string) float64 {
	if v, ok := byKV[kvType]; ok {
		return v
	}
	type pt struct{ b, v float64 }
	var p []pt
	for k, v := range byKV {
		p = append(p, pt{fit.KVBytes(k), v})
	}
	if len(p) == 0 {
		return 0
	}
	sort.Slice(p, func(i, j int) bool { return p[i].b < p[j].b })
	b := fit.KVBytes(kvType)
	if len(p) == 1 {
		return p[0].v
	}
	i := sort.Search(len(p), func(i int) bool { return p[i].b >= b })
	switch {
	case i == 0:
		i = 1
	case i == len(p):
		i = len(p) - 1
	}
	a, c := p[i-1], p[i]
	v := a.v + (c.v-a.v)*(b-a.b)/(c.b-a.b)
	if v < 0 {
		v = 0
	}
	return v
}

// FromMeasured turns a catalog footprint (measured with a real llama-server) into the planner's
// Footprint for one KV type. The measured per-slot cost already contains the sliding-window cache
// and recurrent state, and the per-ctx slope already contains the compute scratch, so only
// those measured numbers are used.
func FromMeasured(m *catalog.Footprint, kvType string) fit.Footprint {
	if kvType == "" {
		kvType = "q4_0"
	}
	fp := fit.Footprint{
		FixedMiB:        m.FixedMiB,
		HostMiB:         m.HostMiB,
		PerTokenMiB:     interp(m.MiBPer1kCtx, kvType) / 1024,
		StatePerSlotMiB: interp(m.StateMiBPerSlot, kvType),
		ScratchMiB:      m.ScratchMiB,
	}
	if m.ExpertLayers > 0 && m.ExpertMiBPerLayer > 0 {
		fp.MoE = true
		fp.ExpertLayers = m.ExpertLayers
		fp.ExpertMiBPerLayer = m.ExpertMiBPerLayer
		fp.ExpertRAMPerLayer = m.ExpertRAMMiBPerLayer
		fp.MoESchedMiB = m.MoESchedMiB
	}
	if len(m.TPSByNCpuMoe) > 0 {
		fp.TPS = &fit.TPSCurve{Points: append([]fit.TPSPoint(nil), m.TPSByNCpuMoe...)}
	}
	return fp
}
