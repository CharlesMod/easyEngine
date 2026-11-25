package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"hermesengine/catalog"
	"hermesengine/fit"
)

// Reference shape of the speed rows: 3 slots x 65536, q4_0.
const (
	refCtx = 65536
	refKV  = "q4_0"
)

// Fit is one model's measured terms with how well the straight lines fit.
type Fit struct {
	Model string
	FP    catalog.Footprint

	Rows     int
	ResMax   float64 // MiB, load fit residual (q4_0 rows)
	ResRMS   float64
	FitRows  int
	ScratchN [2]float64 // min, max of used - load
	ExpertN  int        // rows in the expert fits
	ExpRes   float64    // max residual of the VRAM-vs-n fit
	RSSRes   float64
	MsLayer  float64 // extra ms per token for each expert layer on the CPU (1000/tps linear fit)
	MsBase   float64
	MsRes    float64 // max |ms residual|
	RSSBase  float64
	RSSBaseN float64
}

// solve returns the least-squares x for A x ~ y by the normal equations (columns are few).
func solve(A [][]float64, y []float64) ([]float64, error) {
	n := len(A[0])
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n+1)
		for j := 0; j < n; j++ {
			for _, a := range A {
				m[i][j] += a[i] * a[j]
			}
		}
		for k, a := range A {
			m[i][n] += a[i] * y[k]
		}
	}
	for i := 0; i < n; i++ {
		p := i
		for k := i + 1; k < n; k++ {
			if math.Abs(m[k][i]) > math.Abs(m[p][i]) {
				p = k
			}
		}
		m[i], m[p] = m[p], m[i]
		if math.Abs(m[i][i]) < 1e-9 {
			return nil, fmt.Errorf("singular system")
		}
		for k := i + 1; k < n; k++ {
			f := m[k][i] / m[i][i]
			for j := i; j <= n; j++ {
				m[k][j] -= f * m[i][j]
			}
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = m[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= m[i][j] * x[j]
		}
		x[i] /= m[i][i]
	}
	return x, nil
}

func residuals(A [][]float64, y, x []float64) (maxAbs, rms float64) {
	for k, a := range A {
		var p float64
		for i := range x {
			p += a[i] * x[i]
		}
		r := y[k] - p
		maxAbs = math.Max(maxAbs, math.Abs(r))
		rms += r * r
	}
	return maxAbs, math.Sqrt(rms / float64(len(A)))
}

// FitAll fits every model in rows.
func FitAll(rows []Row) map[string]*Fit {
	out := map[string]*Fit{}
	var ids []string
	seen := map[string]bool{}
	for _, r := range rows {
		if !seen[r.Model] {
			seen[r.Model] = true
			ids = append(ids, r.Model)
		}
	}
	for _, id := range ids {
		var mr []Row
		for _, r := range rows {
			if r.Model == id {
				mr = append(mr, r)
			}
		}
		if f, err := fitModel(id, mr); err != nil {
			fmt.Fprintf(os.Stderr, "fitcheck: %s: %v\n", id, err)
		} else {
			out[id] = f
		}
	}
	return out
}

func fitModel(id string, rows []Row) (*Fit, error) {
	f := &Fit{Model: id, Rows: len(rows)}
	fp := &f.FP
	fp.MiBPer1kCtx = map[string]float64{}
	fp.StateMiBPerSlot = map[string]float64{}
	fp.DecodeTPS = map[string]float64{}
	fp.PromptTPS = map[string]float64{}
	var all, q4 []Row
	for _, r := range rows {
		if r.NCpuMoe == 0 {
			all = append(all, r)
			if r.KV == refKV {
				q4 = append(q4, r)
			}
		}
	}
	// q4_0: vram_load = fixed + a*ctx_total/1024 + s*slots over every shape.
	var A [][]float64
	var y []float64
	for _, r := range q4 {
		A = append(A, []float64{1, float64(r.CtxTotal) / 1024, float64(r.Slots)})
		y = append(y, r.VRAMLoad)
	}
	x, err := solve(A, y)
	if err != nil {
		return nil, fmt.Errorf("q4_0 fit: %w", err)
	}
	if x[2] < 0 { // a cost per slot cannot be negative: the data has none; refit without it
		var A2 [][]float64
		for _, a := range A {
			A2 = append(A2, a[:2])
		}
		x2, err := solve(A2, y)
		if err != nil {
			return nil, fmt.Errorf("q4_0 fit: %w", err)
		}
		A, x = A2, []float64{x2[0], x2[1], 0}
	}
	f.FitRows = len(A)
	f.ResMax, f.ResRMS = residuals(A, y, x[:len(A[0])])
	fp.FixedMiB, fp.MiBPer1kCtx[refKV], fp.StateMiBPerSlot[refKV] = x[0], x[1], x[2]
	// The other KV types share the fixed load. Two shapes give both slopes; one shape
	// (f16) takes the per-slot cost from the line through the other types' per-slot costs
	// against bytes per element, and fits only the per-ctx slope.
	type kvPoint struct{ b, s float64 }
	pts := []kvPoint{{fit.KVBytes(refKV), x[2]}}
	var one []string
	for _, kv := range []string{"q8_0", "f16"} {
		var Ak [][]float64
		var yk []float64
		for _, r := range all {
			if r.KV == kv {
				Ak = append(Ak, []float64{float64(r.CtxTotal) / 1024, float64(r.Slots)})
				yk = append(yk, r.VRAMLoad-x[0])
			}
		}
		switch {
		case len(Ak) >= 2 && Ak[0][0] != Ak[1][0]:
			xk, err := solve(Ak, yk)
			if err != nil {
				return nil, err
			}
			if xk[1] < 0 {
				xk, err = solve([][]float64{{Ak[0][0]}, {Ak[1][0]}}, yk)
				if err != nil {
					return nil, err
				}
				xk = []float64{xk[0], 0}
			}
			fp.MiBPer1kCtx[kv], fp.StateMiBPerSlot[kv] = xk[0], xk[1]
			pts = append(pts, kvPoint{fit.KVBytes(kv), xk[1]})
		case len(Ak) == 1:
			one = append(one, kv)
		}
	}
	for _, kv := range one {
		var r0 Row
		for _, r := range all {
			if r.KV == kv {
				r0 = r
			}
		}
		b := fit.KVBytes(kv)
		s := pts[len(pts)-1].s
		if len(pts) >= 2 {
			p0, p1 := pts[0], pts[len(pts)-1]
			s = p0.s + (p1.s-p0.s)*(b-p0.b)/(p1.b-p0.b)
		}
		s = math.Max(s, 0)
		fp.StateMiBPerSlot[kv] = s
		fp.MiBPer1kCtx[kv] = (r0.VRAMLoad - x[0] - s*float64(r0.Slots)) / (float64(r0.CtxTotal) / 1024)
	}
	// scratch: what the first requests add on top of load.
	var sum float64
	f.ScratchN = [2]float64{math.Inf(1), math.Inf(-1)}
	for _, r := range rows {
		d := r.VRAMUsed - r.VRAMLoad
		sum += d
		f.ScratchN[0], f.ScratchN[1] = math.Min(f.ScratchN[0], d), math.Max(f.ScratchN[1], d)
	}
	fp.ScratchMiB = sum / float64(len(rows))
	// speed at the reference ctx per slot, q4_0, 1/2/3 slots.
	for _, r := range all {
		if r.KV == refKV && r.CtxPerSlot == refCtx {
			k := fmt.Sprint(r.Slots)
			fp.DecodeTPS[k], fp.PromptTPS[k] = r.DecodeTPS, r.PromptTPS
		}
	}
	// experts: rows at the reference shape with n_cpu_moe varying.
	var ex []Row
	for _, r := range rows {
		if r.Slots == 3 && r.CtxPerSlot == refCtx && r.KV == refKV {
			ex = append(ex, r)
		}
	}
	sort.Slice(ex, func(i, j int) bool { return ex[i].NCpuMoe < ex[j].NCpuMoe })
	if len(ex) >= 4 {
		var Ae, Ar [][]float64
		var ye, yr, ym []float64
		for _, r := range ex {
			ind := 0.0
			if r.NCpuMoe > 0 {
				ind = 1
			}
			Ae = append(Ae, []float64{1, float64(r.NCpuMoe), ind})
			ye = append(ye, r.VRAMUsed)
			Ar = append(Ar, []float64{1, float64(r.NCpuMoe)})
			yr = append(yr, r.RSSUsed)
			ym = append(ym, 1000/r.DecodeTPS)
			fp.TPSByNCpuMoe = append(fp.TPSByNCpuMoe, catalog.TPSPoint{NCpuMoe: r.NCpuMoe, TPS: r.DecodeTPS})
		}
		if len(ex) > 1 {
			f.ExpertN = len(ex)
			xe, err := solve(Ae, ye)
			if err != nil {
				return nil, err
			}
			f.ExpRes, _ = residuals(Ae, ye, xe)
			fp.ExpertMiBPerLayer, fp.MoESchedMiB = -xe[1], xe[2]
			xr, err := solve(Ar, yr)
			if err != nil {
				return nil, err
			}
			f.RSSRes, _ = residuals(Ar, yr, xr)
			fp.ExpertRAMMiBPerLayer, f.RSSBase = xr[1], xr[0]
			Am := make([][]float64, len(Ar))
			copy(Am, Ar)
			xm, err := solve(Am, ym)
			if err != nil {
				return nil, err
			}
			f.MsBase, f.MsLayer = xm[0], xm[1]
			f.MsRes, _ = residuals(Am, ym, xm)
		}
	} else {
		fp.TPSByNCpuMoe = nil
	}
	return f, nil
}

// PrintFit writes one model's fitted terms and residuals.
func PrintFit(w io.Writer, f *Fit) {
	fp := f.FP
	fmt.Fprintf(w, "\nfitted terms: %s (%d rows)\n", f.Model, f.Rows)
	fmt.Fprintf(w, "  fixed load            %8.0f MiB\n", fp.FixedMiB)
	for _, kv := range []string{"q4_0", "q8_0", "f16"} {
		fmt.Fprintf(w, "  %-4s per 1k ctx       %8.3f MiB   per slot %7.1f MiB\n", kv, fp.MiBPer1kCtx[kv], fp.StateMiBPerSlot[kv])
	}
	fmt.Fprintf(w, "  scratch (used-load)   %8.1f MiB (range %.0f..%.0f)\n", fp.ScratchMiB, f.ScratchN[0], f.ScratchN[1])
	fmt.Fprintf(w, "  q4_0 line fit: %d rows, residual max %.1f MiB, rms %.1f MiB (q8_0 / f16 slopes are exact through their 2 / 1 rows)\n", f.FitRows, f.ResMax, f.ResRMS)
	if f.ExpertN > 0 {
		fmt.Fprintf(w, "  expert layer on CPU   frees %.1f MiB VRAM (+%.0f MiB offload buffers once n>0; residual max %.0f MiB), gains %.1f MiB RSS (residual max %.0f MiB)\n",
			fp.ExpertMiBPerLayer, fp.MoESchedMiB, f.ExpRes, fp.ExpertRAMMiBPerLayer, f.RSSRes)
		fmt.Fprintf(w, "  decode time per token %.2f ms + %.2f ms per CPU layer (residual max %.2f ms)\n", f.MsBase, f.MsLayer, f.MsRes)
		for _, p := range fp.TPSByNCpuMoe {
			fmt.Fprintf(w, "    n_cpu_moe %2d: %6.1f tok/s (line says %6.1f)\n", p.NCpuMoe, p.TPS, 1000/(f.MsBase+f.MsLayer*float64(p.NCpuMoe)))
		}
	}
	for _, k := range []string{"1", "2", "3"} {
		if v, ok := fp.DecodeTPS[k]; ok {
			fmt.Fprintf(w, "  %s slots at 3x... ref ctx: decode %.1f tok/s per stream, prompt %.0f tok/s\n", k, v, fp.PromptTPS[k])
		}
	}
}

// measuredOn describes the sweep's machine.
var measuredOn = catalog.MeasuredOn{
	GPU: "NVIDIA GeForce RTX 5080", VRAMMiB: 16303, Backend: "cuda-linux-x64", LlamaCpp: "b12680", Date: "2026-11-25",
	Flags: "-ngl 999 -fa on --no-kv-unified -ub 512 --cache-reuse 256",
}

// EmitMeasured writes the footprint block per model, rounded to what the data supports.
func EmitMeasured(path string, fits map[string]*Fit) error {
	out := map[string]catalog.Footprint{}
	r1 := func(v float64) float64 { return math.Round(v*10) / 10 }
	r3 := func(v float64) float64 { return math.Round(v*1000) / 1000 }
	for id, f := range fits {
		fp := f.FP
		fp.FixedMiB = math.Round(fp.FixedMiB)
		fp.ScratchMiB = math.Round(fp.ScratchMiB)
		fp.ExpertMiBPerLayer, fp.ExpertRAMMiBPerLayer, fp.MoESchedMiB = r1(fp.ExpertMiBPerLayer), r1(fp.ExpertRAMMiBPerLayer), math.Round(fp.MoESchedMiB)
		for k, v := range fp.MiBPer1kCtx {
			fp.MiBPer1kCtx[k] = r3(v)
		}
		for k, v := range fp.StateMiBPerSlot {
			fp.StateMiBPerSlot[k] = r1(v)
		}
		for _, m := range []map[string]float64{fp.DecodeTPS, fp.PromptTPS} {
			for k, v := range m {
				m[k] = r1(v)
			}
		}
		fp.MeasuredOn = measuredOn
		out[id] = fp
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
