package main

import (
	"encoding/json"
	"fmt"
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
