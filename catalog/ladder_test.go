package catalog

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/probe"
)

const sha0 = "0000000000000000000000000000000000000000000000000000000000000000"

func asset(id, os, arch, accel string, extra ...string) Asset {
	return Asset{ID: id, OS: os, Arch: arch, Accel: accel, URL: "https://example.com/" + id, SHA256: sha0, Size: 1, Extra: extra}
}

// ladder is the shape the real catalog has: the fork's patched release first, stock upstream
// second holding only what the fork does not build (arm64).
func ladder() *Catalog {
	return &Catalog{Version: 10, Llama: []LlamaCpp{
		{Tag: "b12651-easy1", Repo: "CharlesMod/llama.cpp", Patched: true, Assets: []Asset{
			asset("cuda-linux-x64", "linux", "amd64", "cuda", "cudart-linux-x64"),
			asset("cudart-linux-x64", "linux", "amd64", "cudart"),
			asset("vulkan-linux-x64", "linux", "amd64", "vulkan"),
			asset("cpu-linux-x64", "linux", "amd64", "cpu"),
			asset("cuda-win-x64", "windows", "amd64", "cuda", "cudart-win-x64"),
			asset("cudart-win-x64", "windows", "amd64", "cudart"),
			asset("cuda13-win-x64", "windows", "amd64", "cuda", "cudart13-win-x64"),
			asset("cudart13-win-x64", "windows", "amd64", "cudart"),
			asset("vulkan-win-x64", "windows", "amd64", "vulkan"),
			asset("cpu-win-x64", "windows", "amd64", "cpu"),
			asset("metal-mac-arm64", "darwin", "arm64", "metal"),
			asset("cpu-mac-x64", "darwin", "amd64", "cpu"),
		}},
		{Tag: "b12651", Repo: "ggml-org/llama.cpp", Assets: []Asset{
			asset("cpu-linux-arm64", "linux", "arm64", "cpu"),
			asset("vulkan-linux-arm64", "linux", "arm64", "vulkan"),
			asset("cuda13-linux-arm64", "linux", "arm64", "cuda", "cudart13-linux-arm64"),
			asset("cudart13-linux-arm64", "linux", "arm64", "cudart"),
			asset("cpu-win-arm64", "windows", "arm64", "cpu"),
			asset("vulkan-win-arm64", "windows", "arm64", "vulkan"),
			// a stock build of a platform and accelerator the fork has: never offered
			asset("cpu-linux-x64-stock", "linux", "amd64", "cpu"),
		}},
	}}
}

// The pick over described machines: which builds, in which order, from which rung.
func TestLadderPick(t *testing.T) {
	c := ladder()
	g := func(v, n, drv string) probe.GPU { return probe.GPU{Vendor: v, Name: n, Driver: drv} }
	hw := func(os, arch string, gs ...probe.GPU) probe.Hardware { return probe.Hardware{OS: os, Arch: arch, GPUs: gs} }
	cases := []struct {
		name string
		hw   probe.Hardware
		want string
	}{
		{"linux nvidia gets the fork's CUDA", hw("linux", "amd64", g("nvidia", "NVIDIA GeForce RTX 3060", "580.95")), "cuda-linux-x64,vulkan-linux-x64,cpu-linux-x64"},
		{"linux blackwell: the fork has CUDA 12.8 only, never stock 13", hw("linux", "amd64", g("nvidia", "NVIDIA GeForce RTX 5080", "580.95")), "cuda-linux-x64,vulkan-linux-x64,cpu-linux-x64"},
		{"linux P40 on an old driver", hw("linux", "amd64", g("nvidia", "Tesla P40", "535.104")), "cuda-linux-x64,vulkan-linux-x64,cpu-linux-x64"},
		{"linux nvidia on a driver too old for CUDA 12", hw("linux", "amd64", g("nvidia", "GeForce GTX 1080", "470.82")), "vulkan-linux-x64,cpu-linux-x64"},
		{"linux nvidia beside an intel iGPU: CUDA", hw("linux", "amd64", g("intel", "Intel UHD 770", ""), g("nvidia", "NVIDIA GeForce RTX 4070", "")), "cuda-linux-x64,vulkan-linux-x64,cpu-linux-x64"},
		{"linux nvidia beside an AMD card: CUDA first", hw("linux", "amd64", g("amd", "Radeon RX 6600", ""), g("nvidia", "NVIDIA GeForce RTX 4070", "")), "cuda-linux-x64,vulkan-linux-x64,cpu-linux-x64"},
		{"linux AMD beside an intel iGPU: Vulkan", hw("linux", "amd64", g("amd", "Radeon RX 7900 XTX", ""), g("intel", "Intel UHD 770", "")), "vulkan-linux-x64,cpu-linux-x64"},
		{"linux none", hw("linux", "amd64"), "cpu-linux-x64"},
		{"win 40-series: CUDA 12.4 first, no 13 fallback while 12 exists", hw("windows", "amd64", g("nvidia", "NVIDIA GeForce RTX 4070", "581.57")), "cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win 50-series: CUDA 13.5 then 12.4", hw("windows", "amd64", g("nvidia", "NVIDIA GeForce RTX 5070 Ti", "581.57")), "cuda13-win-x64,cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win 50-series on a driver before 580: no CUDA 13", hw("windows", "amd64", g("nvidia", "NVIDIA GeForce RTX 5070", "572.16")), "cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win 50-series, driver unknown", hw("windows", "amd64", g("nvidia", "NVIDIA GeForce RTX 5090", "")), "cuda13-win-x64,cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win nvidia beside an intel iGPU", hw("windows", "amd64", g("intel", "Intel(R) Iris(R) Xe Graphics", ""), g("nvidia", "NVIDIA GeForce RTX 3050 Laptop GPU", "")), "cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win intel", hw("windows", "amd64", g("intel", "Intel Arc A770", "")), "vulkan-win-x64,cpu-win-x64"},
		{"mac arm", hw("darwin", "arm64", g("apple", "Apple M2", "")), "metal-mac-arm64"},
		{"mac intel", hw("darwin", "amd64"), "cpu-mac-x64"},
		{"linux arm64 (stock: the fork builds none)", hw("linux", "arm64"), "cpu-linux-arm64"},
		{"linux arm64 GB10", hw("linux", "arm64", g("nvidia", "NVIDIA GB10", "580.95")), "cuda13-linux-arm64,vulkan-linux-arm64,cpu-linux-arm64"},
		{"win arm64", hw("windows", "arm64", g("other", "Qualcomm Adreno X1-85", "")), "vulkan-win-arm64,cpu-win-arm64"},
		{"unsupported", hw("plan9", "mips"), ""},
	}
	for _, tc := range cases {
		if got := ids(c.Backends(tc.hw)); got != tc.want {
			t.Errorf("%s: got %q want %q", tc.name, got, tc.want)
		}
	}
}

// Each pick carries its rung: the tag, patched, and why a stock build is given.
func TestLadderRungMarks(t *testing.T) {
	c := ladder()
	x := c.Backends(probe.Hardware{OS: "linux", Arch: "amd64"})
	if len(x) != 1 || x[0].Tag != "b12651-easy1" || !x[0].Patched || x[0].Rung != 0 || x[0].RungWhy != "" {
		t.Errorf("linux x64: %+v", x)
	}
	a := c.Backends(probe.Hardware{OS: "linux", Arch: "arm64"})
	if len(a) != 1 || a[0].Tag != "b12651" || a[0].Patched || a[0].Rung != 1 || a[0].RungWhy != "no patched build for linux/arm64 (cpu)" {
		t.Errorf("linux arm64: %+v", a)
	}
	if e, ok := c.LlamaAsset("b12651", "cudart13-linux-arm64"); !ok || e.Tag != "b12651" {
		t.Errorf("extra of the stock rung: %+v %v", e, ok)
	}
	if _, ok := c.LlamaAsset("b12651-easy1", "cudart13-linux-arm64"); ok {
		t.Error("an extra is looked up in its own release only")
	}
	if a, ok := c.Asset("cpu-linux-x64"); !ok || a.Tag != "b12651-easy1" {
		t.Errorf("Asset resolves to the pin: %+v", a)
	}
	if _, ok := c.Asset("cpu-linux-x64-stock"); ok {
		t.Error("a stock build of a platform the pin covers is never offered")
	}
}

// A catalog before version 10 (one llamacpp release) is still read, as a one-rung ladder.
func TestOldCatalogOneRung(t *testing.T) {
	old := `{"version": 9, "llamacpp": {"tag": "b12651", "assets": [
	 {"id": "cpu-linux-x64", "os": "linux", "arch": "amd64", "accel": "cpu", "url": "https://example.com/a.tar.gz", "sha256": "` + sha0 + `", "size": 1}]}}`
	c, err := Parse([]byte(old))
	if err != nil {
		t.Fatal(err)
	}
	r := c.Rungs()
	if len(r) != 1 || r[0].Tag != "b12651" || r[0].Patched {
		t.Fatalf("rungs %+v", r)
	}
	b := c.Backends(probe.Hardware{OS: "linux", Arch: "amd64"})
	if len(b) != 1 || b[0].Tag != "b12651" || b[0].RungWhy != "the catalog names no patched build" {
		t.Errorf("%+v", b)
	}
	oldEasy := strings.Replace(old, `"b12651"`, `"b12651-easy1"`, 1)
	if c, err := Parse([]byte(oldEasy)); err != nil || !c.Rungs()[0].Patched {
		t.Errorf("an old catalog pinning a fork tag is patched: %v", err)
	}
}

func TestLadderValidate(t *testing.T) {
	bad := []struct {
		name string
		mod  func(c *Catalog)
		want string
	}{
		{"both shapes", func(c *Catalog) { c.LlamaCpp = LlamaCpp{Tag: "b1"} }, "both llama"},
		{"tag twice", func(c *Catalog) { c.Llama[1].Tag = c.Llama[0].Tag }, "named twice"},
		{"bad tag", func(c *Catalog) { c.Llama[0].Tag = "../x" }, "not a plain name"},
		{"extra in another release", func(c *Catalog) { c.Llama[1].Assets[2].Extra = []string{"cudart-linux-x64"} }, "does not name"},
		{"id twice", func(c *Catalog) { c.Llama[0].Assets = append(c.Llama[0].Assets, c.Llama[0].Assets[0]) }, "twice"},
	}
	for _, tc := range bad {
		c := ladder()
		tc.mod(c)
		b, _ := json.Marshal(c)
		if _, err := Parse(b); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v", tc.name, err)
		}
	}
	b, _ := json.Marshal(ladder())
	if strings.Contains(string(b), `"llamacpp"`) {
		t.Error("a v10 catalog writes no llamacpp field")
	}
	if _, err := Parse(b); err != nil {
		t.Error(err)
	}
}
