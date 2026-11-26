package main

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"
)

const hdr = "model,slots,ctx_per_slot,ctx_total,kv,n_cpu_moe,status,vram_load,vram_used,rss_load,rss_used,decode_tps,prompt_tps\n"

func TestReadRowsKeepsOK(t *testing.T) {
	csv := hdr + "m,3,1024,3072,q4_0,0,ok,1000,1100,500,600,50,400\n" +
		"m,3,2048,6144,q4_0,0,oom,0,0,0,0,0,0\n"
	rows, err := ReadRows(strings.NewReader(csv))
	if err != nil || len(rows) != 1 || rows[0].VRAMUsed != 1100 || rows[0].Slots != 3 {
		t.Fatalf("rows %+v err %v", rows, err)
	}
	if _, err := ReadRows(strings.NewReader("model,slots\nm,1\n")); err == nil {
		t.Fatal("missing columns must be an error")
	}
	if _, err := ReadRows(strings.NewReader("")); err == nil {
		t.Fatal("empty input must be an error")
	}
}

func TestSolveRecoversLine(t *testing.T) {
	var A [][]float64
	var y []float64
	for _, x := range []float64{1, 2, 3, 5, 8} {
		A = append(A, []float64{1, x})
		y = append(y, 100+7*x)
	}
	c, err := solve(A, y)
	if err != nil || math.Abs(c[0]-100) > 1e-6 || math.Abs(c[1]-7) > 1e-6 {
		t.Fatalf("coef %v err %v", c, err)
	}
	if mx, rms := residuals(A, y, c); mx > 1e-6 || rms > 1e-6 {
		t.Fatalf("residuals %v %v", mx, rms)
	}
	// Collinear columns cannot be solved.
	if _, err := solve([][]float64{{1, 2}, {2, 4}, {3, 6}}, []float64{1, 2, 3}); err == nil {
		t.Fatal("singular system must be an error")
	}
}

func TestCheckSummaryAndErrors(t *testing.T) {
	rows, _ := ReadRows(strings.NewReader(hdr +
		"m,3,1024,3072,q4_0,0,ok,1000,1100,500,600,50,400\n" +
		"m,3,2048,6144,q4_0,0,ok,1100,1300,500,600,50,400\n"))
	seed := func(model, kv string) (fit.Footprint, error) {
		return fit.Footprint{FixedMiB: 1000, ScratchMiB: 0}, nil
	}
	var buf bytes.Buffer
	by, err := Check(&buf, rows, seed)
	if err != nil || len(by["m"]) != 2 {
		t.Fatalf("by %v err %v", by, err)
	}
	if e := by["m"][0]; math.Abs(e.MiB()-(-100)) > 1 || math.Abs(e.Pct()-(-100.0/1100*100)) > 0.1 {
		t.Fatalf("err %+v", e)
	}
	if !strings.Contains(buf.String(), "m") {
		t.Fatalf("no summary: %s", buf.String())
	}
}

func TestEmitMeasuredWritesJSON(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.json")
	fits := map[string]*Fit{"m": {Model: "m"}}
	fits["m"].FP.FixedMiB = 1234.56
	if err := EmitMeasured(p, fits); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil || !strings.Contains(string(b), `"m"`) || !strings.Contains(string(b), "RTX 5080") {
		t.Fatalf("%s %v", b, err)
	}
}
