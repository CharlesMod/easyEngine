package fit

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// updateFlagOff rewrites testdata/flagoff.jsonl.gz (one JSON line per case, gzipped) from the code as it stands. It was written once
// from main at 4b31c98, before the dense split existed (core/v0.1.4 behaviour), and must never be
// rewritten to make a change pass: a footprint without Split plans exactly as it did then.
var updateFlagOff = flag.Bool("update-flagoff", false, "rewrite testdata/flagoff.jsonl.gz")

const flagOffFile = "testdata/flagoff.jsonl.gz"

// flagOffFootprints is every kind of footprint the core is given today: MoE, dense, MLA with
// state, hybrid, the E-series with host weights, size-only guesses and measured numbers.
func flagOffFootprints() map[string]Footprint {
	moeGuess, _ := SeedFromBytes(20<<30, true, "q4_0")
	denseGuess, _ := SeedFromBytes(4<<30, false, "q4_0")
	return map[string]Footprint{
		"ref-moe":     seed(refInfo(), "q4_0"),
		"dense-14g":   seed(denseInfo(), "q4_0"),
		"ling":        seed(lingInfo(), "q4_0"),
		"hybrid":      seed(hybridInfo(), "q4_0"),
		"gemma-like":  gemmaLike(),
		"moe-guess":   moeGuess,
		"dense-guess": denseGuess,
		// measured numbers, as plan.FromMeasured gives them for the E4B (no cache type)
		"measured-e4b": {FixedMiB: 2994, PerTokenMiB: 8.915 / 1024, ScratchMiB: 55, HostMiB: 912, Basis: Predicted},
	}
}

func flagOffRooms() map[string]Room {
	rs := map[string]Room{}
	for _, v := range []int{0, 3072, 4096, 8192, 16303, 24576} {
		for _, o := range []int{300, 3000} {
			for _, ram := range []int{6000, 64000} {
				r := Room{VRAMTotal: v, Others: o, RAMAvail: ram, NoGPU: v == 0}
				rs[fmt.Sprintf("v%d-o%d-r%d", v, o, ram)] = r
			}
		}
	}
	rs["block"] = Room{VRAMTotal: 16303, Others: 300, RAMAvail: 64000, LargestBlock: 6000}
	rs["noblock"] = Room{VRAMTotal: 16303, Others: 300, RAMAvail: 64000, LargestBlock: -1}
	rs["held"] = Room{VRAMTotal: 8192, Others: 300, RAMAvail: 64000, CardHostFree: 9000}
	rs["unified"] = Room{VRAMTotal: 24576, Others: 300, RAMAvail: 32000, Unified: true}
	rs["game"] = Room{VRAMTotal: 12288, Others: 1200, GameReserve: 1500, RAMAvail: 16000, Reserve: 4096}
	return rs
}

func flagOffWants() map[string]Wants {
	whole := game
	whole.Whole = true
	listed := game
	listed.KVTypes = []string{"f16", "q8_0", "q4_0"}
	one := Wants{Slots: 1, MinSlots: 1, CtxMax: 32768, CtxFloor: 4096, KVType: "q4_0"}
	floor := Wants{Slots: 2, MinSlots: 2, CtxMax: 16384, CtxFloor: 8192, KVType: "q8_0", MinTPS: -1}
	return map[string]Wants{"game": game, "whole": whole, "listed": listed, "one": one, "floor": floor}
}

// flagOffRecord is what every planning entry answers for one case, as text.
func flagOffRecord(fp Footprint, r Room, w Wants) map[string]string {
	out := map[string]string{}
	put := func(k string, v any, err error) {
		if err != nil {
			out[k] = "err: " + err.Error()
			return
		}
		j, _ := json.Marshal(v)
		out[k] = string(j)
	}
	s, err := Choose(fp, r, w)
	put("choose", s, err)
	if err != nil {
		return out
	}
	out["args"] = strings.Join(Args(s), " ")
	v, h, err := fp.Need(s)
	put("need", [2]float64{v, h}, err)
	hn, err := fp.HostNeed(s)
	put("hostneed", hn, err)
	l, err := fp.Largest(s)
	put("largest", l, err)
	cur := s
	for i := 0; i < 4; i++ {
		d, err := StepDown(fp, cur, r, w)
		put(fmt.Sprintf("stepdown%d", i), d, err)
		if err != nil {
			break
		}
		out[fmt.Sprintf("stepdown%d-args", i)] = strings.Join(Args(d), " ")
		cur = d
	}
	g, ok, err := Grow(fp, cur, r, w)
	put("grow", [2]any{g, ok}, err)
	m, err := MissRetry(fp, s, 512, r, w)
	put("missretry", m, err)
	return out
}

type flagOffLine struct {
	Case string            `json:"case"`
	Got  map[string]string `json:"got"`
}

func flagOffAll() []flagOffLine {
	fps, rooms, wants := flagOffFootprints(), flagOffRooms(), flagOffWants()
	var lines []flagOffLine
	for _, fn := range sortedKeys(fps) {
		for _, rn := range sortedKeys(rooms) {
			for _, wn := range sortedKeys(wants) {
				lines = append(lines, flagOffLine{Case: fn + "|" + rn + "|" + wn, Got: flagOffRecord(fps[fn], rooms[rn], wants[wn])})
			}
		}
	}
	return lines
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	for i := 1; i < len(ks); i++ { // insertion sort: no sort import needed beyond the test's own
		for j := i; j > 0 && ks[j] < ks[j-1]; j-- {
			ks[j], ks[j-1] = ks[j-1], ks[j]
		}
	}
	return ks
}

// Every planning entry answers today exactly what it answered before the dense split existed,
// for every kind of footprint the core is given (none carries Split), across rooms from no card to
// 24 GB, a held machine memory, a largest block, a unified pool and a game's reserve, and wants
// whole, listed, single-slot and fixed-floor: Choose, Args, Need, HostNeed, Largest, four steps of
// StepDown, Grow and MissRetry. The record was written from main at 4b31c98.
func TestFlagOffPlansAsBefore(t *testing.T) {
	lines := flagOffAll()
	if *updateFlagOff {
		if err := os.MkdirAll(filepath.Dir(flagOffFile), 0o755); err != nil {
			t.Fatal(err)
		}
		f, err := os.Create(flagOffFile)
		if err != nil {
			t.Fatal(err)
		}
		zw := gzip.NewWriter(f)
		bw := bufio.NewWriter(zw)
		for _, l := range lines {
			j, _ := json.Marshal(l)
			bw.Write(append(j, '\n'))
		}
		if err := bw.Flush(); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		f.Close()
		t.Logf("wrote %d cases", len(lines))
		return
	}
	f, err := os.Open(flagOffFile)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]map[string]string{}
	sc := bufio.NewScanner(zr)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var l flagOffLine
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
			t.Fatal(err)
		}
		want[l.Case] = l.Got
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(want) != len(lines) {
		t.Fatalf("record holds %d cases, the grid %d", len(want), len(lines))
	}
	bad := 0
	for _, l := range lines {
		w, ok := want[l.Case]
		if !ok {
			t.Errorf("%s: not in the record", l.Case)
			continue
		}
		for k, g := range l.Got {
			if w[k] != g && bad < 20 {
				bad++
				t.Errorf("%s %s:\n got %s\nwant %s", l.Case, k, g, w[k])
			}
		}
		for k := range w {
			if _, ok := l.Got[k]; !ok && bad < 20 {
				bad++
				t.Errorf("%s: %s no longer answered", l.Case, k)
			}
		}
	}
}
