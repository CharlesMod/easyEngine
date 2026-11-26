package main

import (
	"encoding/json"
	"fmt"
	"hermesengine/catalog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func rel(tag string, pre bool, age time.Duration, now time.Time, names ...string) ghRelease {
	r := ghRelease{TagName: tag, Prerelease: pre, PublishedAt: now.Add(-age)}
	for _, n := range names {
		r.Assets = append(r.Assets, ghAsset{Name: n, Size: 10, Digest: "sha256:" + strings.Repeat("a", 64), BrowserDownloadURL: "https://x/" + n})
	}
	return r
}

func fullNames(tag string) []string {
	return []string{
		"llama-" + tag + "-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-" + tag + "-bin-ubuntu-cuda-12.8-x64.tar.gz",
		"llama-" + tag + "-bin-ubuntu-cuda-13.5-x64.tar.gz",
		"llama-" + tag + "-bin-ubuntu-vulkan-x64.tar.gz", "llama-" + tag + "-bin-ubuntu-x64.tar.gz",
		"llama-" + tag + "-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
		"llama-" + tag + "-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip",
		"llama-" + tag + "-bin-win-vulkan-x64.zip", "llama-" + tag + "-bin-win-cpu-x64.zip",
		"llama-" + tag + "-bin-macos-arm64.tar.gz", "llama-" + tag + "-bin-macos-x64.tar.gz",
	}
}

func TestSelectRelease(t *testing.T) {
	now := time.Now()
	newest := rel("b200", true, time.Hour, now, fullNames("b200")...)         // too fresh
	old := rel("b150", true, 3*24*time.Hour, now, fullNames("b150")...)       // all-prerelease world
	partial := rel("b190", true, 2*24*time.Hour, now, "llama-b190-bin-x.zip") // missing assets
	junk := rel("v1.0.0", false, 10*time.Minute, now)
	got, _, err := selectRelease([]ghRelease{junk, newest, partial, old}, now, 24*time.Hour, "")
	if err != nil || got.TagName != "b150" {
		t.Fatalf("got %v %v", got.TagName, err)
	}
	stable := rel("b100", false, 30*24*time.Hour, now, fullNames("b100")...)
	got, why, err := selectRelease([]ghRelease{junk, newest, old, stable}, now, 24*time.Hour, "")
	if err != nil || got.TagName != "b100" || !strings.Contains(why, "non-prerelease") {
		t.Fatalf("non-prerelease must win: %v %v %s", got.TagName, err, why)
	}
	got, _, err = selectRelease([]ghRelease{newest, old}, now, 24*time.Hour, "b200")
	if err != nil || got.TagName != "b200" {
		t.Fatalf("pin: %v %v", got.TagName, err)
	}
	if _, _, err := selectRelease([]ghRelease{newest}, now, 24*time.Hour, ""); err == nil {
		t.Fatal("expected failure when everything is too fresh")
	}
}

func TestLlamaAssetsAndDigest(t *testing.T) {
	now := time.Now()
	r := rel("b150", true, 3*24*time.Hour, now, fullNames("b150")...)
	c := &client{log: func(string, ...any) {}}
	as, err := c.llamaAssets(r)
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]bool{}
	for _, a := range as {
		byID[a.ID] = true
		if len(a.SHA256) != 64 || a.URL == "" {
			t.Errorf("%+v", a)
		}
	}
	for _, id := range []string{"cuda-linux-x64", "cudart-linux-x64", "cuda-win-x64", "cudart-win-x64", "cuda13-win-x64", "vulkan-linux-x64", "cpu-win-x64", "metal-mac-arm64"} {
		if !byID[id] {
			t.Errorf("missing %s", id)
		}
	}
	for _, a := range as {
		if a.ID == "cuda-win-x64" && (len(a.Extra) != 1 || a.Extra[0] != "cudart-win-x64") {
			t.Errorf("cuda-win extra %v", a.Extra)
		}
	}
}

func TestHighestToolkitWins(t *testing.T) {
	as := []ghAsset{{Name: "llama-b1-bin-ubuntu-cuda-12.4-x64.tar.gz"}, {Name: "llama-b1-bin-ubuntu-cuda-12.9-x64.tar.gz"}}
	a, ok := findAsset(as, `^llama-b1-bin-ubuntu-cuda-(12\.\d+)-x64\.tar\.gz$`)
	if !ok || !strings.Contains(a.Name, "12.9") {
		t.Fatal(a, ok)
	}
}

func TestUVAssets(t *testing.T) {
	now := time.Now()
	var names []string
	for _, sp := range uvSpecs {
		names = append(names, sp.name)
	}
	c := &client{log: func(string, ...any) {}}
	as, err := c.uvAssets(rel("0.13.10", false, time.Hour, now, names...))
	if err != nil || len(as) != 6 {
		t.Fatal(err, as)
	}
	if _, err := c.uvAssets(rel("0.1", false, time.Hour, now, names[:3]...)); err == nil {
		t.Fatal("missing asset must fail")
	}
}

func TestHFEntryParse(t *testing.T) {
	var tree []hfEntry
	js := fmt.Sprintf(`[{"path":"a.gguf","size":134,"lfs":{"oid":"%s","size":4977171584}}]`, strings.Repeat("b", 64))
	if err := json.Unmarshal([]byte(js), &tree); err != nil || tree[0].LFS.Size != 4977171584 {
		t.Fatal(err, tree)
	}
}

func TestMeasuredEmbeddedAndApplied(t *testing.T) {
	m, err := loadMeasured(measuredJSON)
	if err != nil || len(m) == 0 {
		t.Fatalf("embedded measured.json: %v (%d entries)", err, len(m))
	}
	for id, fp := range m {
		if fp.FixedMiB <= 0 || len(fp.MiBPer1kCtx) == 0 || fp.MeasuredOn.GPU == "" {
			t.Errorf("%s: incomplete footprint %+v", id, fp)
		}
	}
	models := []catalog.Model{{ID: "keep"}, {ID: "stale", Footprint: &catalog.Footprint{FixedMiB: 1}}}
	measured := map[string]catalog.Footprint{"keep": {FixedMiB: 5}, "gone": {FixedMiB: 9}}
	unused := applyMeasured(models, measured)
	if models[0].Footprint == nil || models[0].Footprint.FixedMiB != 5 {
		t.Errorf("footprint not attached: %+v", models[0])
	}
	if models[1].Footprint != nil {
		t.Errorf("a model without measurements must not keep a stale block")
	}
	if len(unused) != 1 || unused[0] != "gone" {
		t.Errorf("unused %v", unused)
	}
	if _, err := loadMeasured([]byte("{")); err == nil {
		t.Error("bad json must be an error")
	}
}

func TestCommittedCatalogCarriesMeasured(t *testing.T) {
	b, err := os.ReadFile("../../catalog/catalog.json")
	if err != nil {
		t.Fatal(err)
	}
	var c catalog.Catalog
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	if c.Version < 3 {
		t.Errorf("catalog version %d, footprints arrived in 3", c.Version)
	}
	m, _ := loadMeasured(measuredJSON)
	for id := range m {
		mod, ok := c.Model(id)
		if !ok || mod.Footprint == nil {
			t.Errorf("%s: catalog.json lacks the measured footprint (run gencatalog -measured-only)", id)
		}
	}
}

func TestAllocSize(t *testing.T) {
	root := t.TempDir()
	for name, n := range map[string]int{"a0": 0, "a1": 1, "a4096": 4096, "a4097": 4097} {
		if err := os.WriteFile(filepath.Join(root, name), make([]byte, n), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	sub := filepath.Join(root, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "b"), make([]byte, 10000), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("a1", filepath.Join(root, "link")); err != nil {
		t.Skip("no symlinks:", err)
	}
	// root 4096 + sub 4096 + link 4096 + files 0+4096+4096+8192 + b 12288
	want := int64(4096*3 + 0 + 4096 + 4096 + 8192 + 12288)
	got, err := allocSize(root)
	if err != nil || got != want {
		t.Fatalf("allocSize = %d, %v; want %d", got, err, want)
	}
}

func TestArchiveExt(t *testing.T) {
	for u, want := range map[string]string{"https://x/a.tar.gz": ".tar.gz", "https://x/a.ZIP?x=1": ".zip"} {
		if got, err := archiveExt(u); err != nil || got != want {
			t.Errorf("%s: %q %v", u, got, err)
		}
	}
	if _, err := archiveExt("https://x/a.7z"); err == nil {
		t.Error("7z must fail")
	}
}

func TestRuntimeEmbeddedAndRefreshed(t *testing.T) {
	if _, err := loadRuntime(runtimeJSON); err != nil {
		t.Fatal(err)
	}
	rt, err := loadRuntime([]byte(`[{"os":"linux","arch":"amd64","python":"3.12","peak_bytes":5,"keep_bytes":3,"measured_on":"d","measured_with":"w"}]`))
	if err != nil || len(rt) != 1 || rt[0].PeakBytes != 5 {
		t.Fatal(rt, err)
	}
	saved, savedM := runtimeJSON, measuredJSON
	defer func() { runtimeJSON, measuredJSON = saved, savedM }()
	measuredJSON = []byte(`{}`)
	runtimeJSON = []byte(`[{"os":"linux","arch":"amd64","python":"3.12","peak_bytes":5,"keep_bytes":3}]`)
	cat := catalog.Catalog{Version: 1, LlamaCpp: catalog.LlamaCpp{Tag: "b1"}}
	out := filepath.Join(t.TempDir(), "c.json")
	if err := writeCatalog(out, &cat); err != nil {
		t.Fatal(err)
	}
	if err := refreshMeasured(out, 2); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(out)
	got, err := catalog.Parse(b)
	if err != nil || len(got.AgentRuntime) != 1 || got.AgentRuntime[0].PeakBytes != 5 || got.Version != 2 {
		t.Fatal(got, err)
	}
}
