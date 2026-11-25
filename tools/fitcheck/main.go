// Command fitcheck compares the planner's footprint predictions with measured llama-server
// runs (the sweep CSV): for every row it prints predicted vs measured VRAM and the RAM gained
// by moving experts to the CPU, then a per-model summary. With -fit it also fits each model's
// real terms by least squares; with -emit it writes the measured footprint block that
// gencatalog carries into the catalog.
//
//	go run ./tools/fitcheck -csv sweep.csv -models <dir with the GGUF files>
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"hermesengine/catalog"
	"hermesengine/fit"
	"hermesengine/gguf"
)

// Row is one measured run.
type Row struct {
	Model                string
	Slots, CtxPerSlot    int
	CtxTotal             int
	KV                   string
	NCpuMoe              int
	Status               string
	VRAMLoad, VRAMUsed   float64
	RSSLoad, RSSUsed     float64
	DecodeTPS, PromptTPS float64
}

// ReadRows parses the sweep CSV, keeping only status "ok" rows.
func ReadRows(r io.Reader) ([]Row, error) {
	cr := csv.NewReader(r)
	recs, err := cr.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(recs) < 2 {
		return nil, fmt.Errorf("no rows")
	}
	col := map[string]int{}
	for i, h := range recs[0] {
		col[h] = i
	}
	for _, need := range []string{"model", "slots", "ctx_per_slot", "ctx_total", "kv", "n_cpu_moe", "status", "vram_load", "vram_used", "rss_load", "rss_used", "decode_tps", "prompt_tps"} {
		if _, ok := col[need]; !ok {
			return nil, fmt.Errorf("missing column %q", need)
		}
	}
	var rows []Row
	for _, rec := range recs[1:] {
		f := func(n string) float64 { v, _ := strconv.ParseFloat(rec[col[n]], 64); return v }
		row := Row{Model: rec[col["model"]], KV: rec[col["kv"]], Status: rec[col["status"]],
			Slots: int(f("slots")), CtxPerSlot: int(f("ctx_per_slot")), CtxTotal: int(f("ctx_total")), NCpuMoe: int(f("n_cpu_moe")),
			VRAMLoad: f("vram_load"), VRAMUsed: f("vram_used"), RSSLoad: f("rss_load"), RSSUsed: f("rss_used"),
			DecodeTPS: f("decode_tps"), PromptTPS: f("prompt_tps")}
		if row.Status == "ok" {
			rows = append(rows, row)
		}
	}
	return rows, nil
}

// Err is one prediction error.
type Err struct {
	Pred, Meas float64
}

func (e Err) MiB() float64 { return e.Pred - e.Meas }
func (e Err) Pct() float64 { return 100 * (e.Pred - e.Meas) / e.Meas }

// Seeder builds the footprint to test for a model and kv type.
type Seeder func(model, kv string) (fit.Footprint, error)

func headerSeeder(modelsDir string, cat *catalog.Catalog) Seeder {
	infos := map[string]gguf.Info{}
	return func(model, kv string) (fit.Footprint, error) {
		in, ok := infos[model]
		if !ok {
			m, found := cat.Model(model)
			if !found {
				return fit.Footprint{}, fmt.Errorf("model %q is not in the catalog", model)
			}
			var err error
			in, err = gguf.Read(filepath.Join(modelsDir, m.File))
			if err != nil {
				return fit.Footprint{}, err
			}
			infos[model] = in
		}
		return fit.Seed(in, kv), nil
	}
}

// Check prints the per-row table and per-model summary and returns the errors by model.
func Check(w io.Writer, rows []Row, seed Seeder) (map[string][]Err, error) {
	byModel := map[string][]Err{}
	base := map[string]float64{} // rss_used of the n_cpu_moe=0 run at the same shape
	key := func(r Row) string { return fmt.Sprintf("%s/%d/%d/%s", r.Model, r.Slots, r.CtxPerSlot, r.KV) }
	for _, r := range rows {
		if r.NCpuMoe == 0 {
			base[key(r)] = r.RSSUsed
		}
	}
	fmt.Fprintf(w, "%-24s %2s %6s %-4s %3s | %9s %9s %7s %6s | %8s %8s %7s\n", "model", "np", "ctx/sl", "kv", "moe",
		"pred_vram", "meas_vram", "err_MiB", "err_%", "pred_ram", "meas_ram", "err_MiB")
	var order []string
	seen := map[string]bool{}
	for _, r := range rows {
		fp, err := seed(r.Model, r.KV)
		if err != nil {
			return nil, err
		}
		if !seen[r.Model] {
			seen[r.Model] = true
			order = append(order, r.Model)
		}
		e := Err{Pred: fp.VRAM(r.Slots, r.CtxPerSlot, r.NCpuMoe), Meas: r.VRAMUsed}
		byModel[r.Model] = append(byModel[r.Model], e)
		ramCols := fmt.Sprintf("%8s %8s %7s", "-", "-", "-")
		if r.NCpuMoe > 0 {
			if b, ok := base[key(r)]; ok {
				pr, me := fp.ExpertRAM(r.NCpuMoe), r.RSSUsed-b
				ramCols = fmt.Sprintf("%8.0f %8.0f %+7.0f", pr, me, pr-me)
			}
		}
		fmt.Fprintf(w, "%-24s %2d %6d %-4s %3d | %9.0f %9.0f %+7.0f %+6.1f | %s\n", r.Model, r.Slots, r.CtxPerSlot, r.KV, r.NCpuMoe,
			e.Pred, e.Meas, e.MiB(), e.Pct(), ramCols)
	}
	fmt.Fprintf(w, "\nper-model VRAM error (predicted - measured)\n%-24s %4s %9s %9s %10s %10s %9s\n", "model", "rows", "mean_MiB", "max_MiB", "signed_MiB", "mean_%", "max_%")
	for _, m := range order {
		es := byModel[m]
		var sum, sumAbs, maxAbs, sumPctAbs, maxPct float64
		for _, e := range es {
			sum += e.MiB()
			sumAbs += math.Abs(e.MiB())
			maxAbs = math.Max(maxAbs, math.Abs(e.MiB()))
			sumPctAbs += math.Abs(e.Pct())
			maxPct = math.Max(maxPct, math.Abs(e.Pct()))
		}
		n := float64(len(es))
		fmt.Fprintf(w, "%-24s %4d %9.0f %9.0f %+10.0f %10.1f %9.1f\n", m, len(es), sumAbs/n, maxAbs, sum/n, sumPctAbs/n, maxPct)
	}
	return byModel, nil
}

func main() {
	csvPath := flag.String("csv", "", "sweep CSV (required)")
	models := flag.String("models", "", "directory with the GGUF files (required; headers are read, nothing is copied)")
	doFit := flag.Bool("fit", false, "also fit each model's terms by least squares and print residuals")
	emit := flag.String("emit", "", "write the measured footprint block (JSON) to this file")
	flag.Parse()
	if *csvPath == "" || *models == "" {
		fatal(fmt.Errorf("-csv and -models are required"))
	}
	f, err := os.Open(*csvPath)
	if err != nil {
		fatal(err)
	}
	rows, err := ReadRows(f)
	f.Close()
	if err != nil {
		fatal(err)
	}
	cat, err := catalog.Load("")
	if err != nil {
		fatal(err)
	}
	if _, err := Check(os.Stdout, rows, headerSeeder(*models, cat)); err != nil {
		fatal(err)
	}
	if *doFit || *emit != "" {
		fits := FitAll(rows)
		var names []string
		for n := range fits {
			names = append(names, n)
		}
		sort.Strings(names)
		if *doFit {
			for _, n := range names {
				PrintFit(os.Stdout, fits[n])
			}
		}
		if *emit != "" {
			seed := headerSeeder(*models, cat)
			for id, f := range fits { // facts the CSV cannot give come from the header
				fp, err := seed(id, "q4_0")
				if err != nil {
					fatal(err)
				}
				f.FP.HostMiB = math.Round(fp.HostMiB)
				if fp.MoE {
					f.FP.ExpertLayers = fp.ExpertLayers
				}
			}
			if err := EmitMeasured(*emit, fits); err != nil {
				fatal(err)
			}
		}
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "fitcheck:", err)
	os.Exit(1)
}
