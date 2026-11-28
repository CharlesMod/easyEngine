package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/probe"
)

// relSHA is rel with a distinct digest per asset (sha256 of its name), so a test can tell
// which asset a hash came from.
func relSHA(tag string, now time.Time, names ...string) ghRelease {
	r := rel(tag, false, time.Hour, now, names...)
	for i := range r.Assets {
		r.Assets[i].Digest = "sha256:" + nameSHA(r.Assets[i].Name)
	}
	return r
}

func nameSHA(n string) string { h := sha256.Sum256([]byte(n)); return hex.EncodeToString(h[:]) }

// forkNames is the fork's release: no arm64, Windows zips with or without the tag in the name.
func forkNames(tag string, winTag bool) []string {
	wt := ""
	if winTag {
		wt = tag + "-"
	}
	return []string{
		"llama-" + tag + "-bin-ubuntu-x64.tar.gz", "llama-" + tag + "-bin-ubuntu-vulkan-x64.tar.gz",
		"llama-" + tag + "-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-" + tag + "-bin-ubuntu-cuda-12.8-x64.tar.gz",
		"llama-" + tag + "-bin-macos-arm64.tar.gz", "llama-" + tag + "-bin-macos-x64.tar.gz",
		"llama-" + wt + "bin-win-cpu-x64.zip", "llama-" + wt + "bin-win-vulkan-x64.zip",
		"llama-" + wt + "bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip",
		"llama-" + wt + "bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip",
	}
}

func upstreamNames(tag string) []string {
	return append(fullNames(tag),
		"llama-"+tag+"-bin-ubuntu-arm64.tar.gz", "llama-"+tag+"-bin-ubuntu-vulkan-arm64.tar.gz",
		"llama-"+tag+"-bin-ubuntu-cuda-13.5-arm64.tar.gz", "cudart-llama-"+tag+"-bin-ubuntu-cuda-13.5-arm64.tar.gz",
		"llama-"+tag+"-bin-win-cpu-arm64.zip", "llama-"+tag+"-bin-win-vulkan-arm64.zip")
}

func ids(as []catalog.Asset) []string {
	var o []string
	for _, a := range as {
		o = append(o, a.ID)
	}
	return o
}

func has(as []catalog.Asset, id string) bool {
	for _, a := range as {
		if a.ID == id {
			return true
		}
	}
	return false
}

func quiet() *client { return &client{log: func(string, ...any) {}} }

func TestLlamaRungs(t *testing.T) {
	now := time.Now()
	const fork, stock = "CharlesMod/llama.cpp", "ggml-org/llama.cpp"
	for _, tc := range []struct {
		name      string
		winTag    bool
		stockTag  string
		wantStock string
	}{
		{"default stock tag from the pin", true, "", "b12651"},
		{"explicit stock tag, untagged windows names", false, "b12651", "b12651"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// two fake repos; the stock one is reached through the client's API base
			srvS, _ := fakeGitHub(t, stock, []ghRelease{relSHA("b12651", now, upstreamNames("b12651")...)})
			c := &client{http: srvS.Client(), api: srvS.URL, log: func(string, ...any) {}}
			fr := relSHA("b12651-easy1", now, forkNames("b12651-easy1", tc.winTag)...)
			srvF, _ := fakeGitHub(t, fork, []ghRelease{fr})
			var rs []ghRelease
			if err := c.getJSON(releasesURL(srvF.URL, fork), &rs); err != nil {
				t.Fatal(err)
			}
			rungs, err := c.llamaRungs(fork, stock, tc.stockTag, rs[0], true, 12651, rs, time.Hour)
			if err != nil {
				t.Fatal(err)
			}
			if len(rungs) != 2 {
				t.Fatalf("rungs %d", len(rungs))
			}
			r0, r1 := rungs[0], rungs[1]
			if r0.Tag != "b12651-easy1" || r0.Repo != fork || !r0.Patched {
				t.Errorf("rung 0 %+v", r0)
			}
			for _, id := range []string{"cuda-linux-x64", "cudart-linux-x64", "vulkan-linux-x64", "cpu-linux-x64", "cuda-win-x64", "cudart-win-x64",
				"cuda13-win-x64", "cudart13-win-x64", "vulkan-win-x64", "cpu-win-x64", "metal-mac-arm64", "cpu-mac-x64"} {
				if !has(r0.Assets, id) {
					t.Errorf("rung 0 lacks %s: %v", id, ids(r0.Assets))
				}
			}
			if r1.Tag != tc.wantStock || r1.Repo != stock || r1.Patched {
				t.Errorf("rung 1 %+v", r1)
			}
			want := map[string]bool{"cpu-linux-arm64": true, "vulkan-linux-arm64": true, "cuda13-linux-arm64": true,
				"cudart13-linux-arm64": true, "cpu-win-arm64": true, "vulkan-win-arm64": true}
			if len(r1.Assets) != len(want) {
				t.Errorf("rung 1 ids %v", ids(r1.Assets))
			}
			for _, a := range r1.Assets {
				if !want[a.ID] {
					t.Errorf("rung 1 holds %s", a.ID)
				}
				if a.ID == "cuda13-linux-arm64" && (len(a.Extra) != 1 || a.Extra[0] != "cudart13-linux-arm64") {
					t.Errorf("cuda13-linux-arm64 extra %v", a.Extra)
				}
			}
			// every sha is the digest of the asset's own file
			for _, a := range append(append([]catalog.Asset{}, r0.Assets...), r1.Assets...) {
				if base := a.URL[strings.LastIndex(a.URL, "/")+1:]; a.SHA256 != nameSHA(base) {
					t.Errorf("%s sha %s, want the digest of %s", a.ID, a.SHA256, base)
				}
			}

			// the output parses, carries no legacy field, and the ladder drives Backends
			cat := catalog.Catalog{Version: 10, Llama: rungs}
			out := filepath.Join(t.TempDir(), "c.json")
			if err := writeCatalog(out, &cat); err != nil {
				t.Fatal(err)
			}
			b, _ := os.ReadFile(out)
			var raw map[string]json.RawMessage
			if err := json.Unmarshal(b, &raw); err != nil {
				t.Fatal(err)
			}
			if _, ok := raw["llamacpp"]; ok {
				t.Error("the legacy llamacpp field was written")
			}
			got, err := catalog.Parse(b)
			if err != nil {
				t.Fatal(err)
			}
			arm := got.Backends(probe.Hardware{OS: "linux", Arch: "arm64"})
			if len(arm) == 0 || arm[0].ID != "cpu-linux-arm64" || arm[0].Patched || arm[0].Tag != "b12651" {
				t.Errorf("arm64 backends %v", ids(arm))
			}
			nv := got.Backends(probe.Hardware{OS: "linux", Arch: "amd64", GPUs: []probe.GPU{{Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3090"}}})
			if len(nv) == 0 || nv[0].ID != "cuda-linux-x64" {
				t.Errorf("nvidia backends %v", ids(nv))
			}
			for _, a := range nv {
				if !a.Patched || a.Tag != "b12651-easy1" {
					t.Errorf("amd64 nvidia got a non-patched build %s (%s)", a.ID, a.Tag)
				}
			}
		})
	}
}

func TestLlamaRungsOneRung(t *testing.T) {
	now := time.Now()
	c := quiet()
	// a stock pin writes one unpatched rung and never asks for a stock release
	r := relSHA("b12651", now, upstreamNames("b12651")...)
	rungs, err := c.llamaRungs("ggml-org/llama.cpp", "ggml-org/llama.cpp", "", r, false, 12651, nil, time.Hour)
	if err != nil || len(rungs) != 1 || rungs[0].Patched || rungs[0].Tag != "b12651" || !has(rungs[0].Assets, "cpu-linux-arm64") {
		t.Fatalf("%v %+v", err, rungs)
	}
	// a patched pin whose stock release adds nothing (the pin has every build) writes one rung
	full := relSHA("b12651-easy1", now, upstreamNames("b12651-easy1")...)
	up := []ghRelease{relSHA("b12651", now, fullNames("b12651")...)}
	rungs, err = c.llamaRungs("a/b", "a/b", "", full, true, 12651, up, time.Hour)
	if err != nil || len(rungs) != 1 || !rungs[0].Patched {
		t.Fatalf("%v %+v", err, rungs)
	}
}

func TestLlamaRungsErrors(t *testing.T) {
	now := time.Now()
	c := quiet()
	var names []string
	for _, n := range forkNames("b12651-easy1", true) {
		if !strings.Contains(n, "ubuntu-vulkan-x64") {
			names = append(names, n)
		}
	}
	_, err := c.llamaRungs("a/b", "a/b", "", relSHA("b12651-easy1", now, names...), true, 12651, nil, time.Hour)
	if err == nil || !strings.Contains(err.Error(), "vulkan-linux-x64") {
		t.Fatalf("want the missing fork asset named, got %v", err)
	}
	ok := relSHA("b12651-easy1", now, forkNames("b12651-easy1", true)...)
	if _, err := c.llamaRungs("a/b", "a/b", "b12651-easy1", ok, true, 12651, nil, time.Hour); err == nil {
		t.Error("a patched stock tag accepted")
	}
	other := []ghRelease{relSHA("b999", now, fullNames("b999")...)}
	if _, err := c.llamaRungs("a/b", "a/b", "", ok, true, 12651, other, time.Hour); err == nil || !strings.Contains(err.Error(), "stock rung") {
		t.Errorf("a missing stock release must be an error: %v", err)
	}
}

func TestStockTagDefault(t *testing.T) {
	// b12651-easy1 -> b12651: the pin's build number names the stock release
	build, patch, err := parseTag("b12651-easy1")
	if err != nil || patch != 1 {
		t.Fatal(build, patch, err)
	}
	now := time.Now()
	up := []ghRelease{relSHA("b12651", now, upstreamNames("b12651")...), relSHA("b12700", now, upstreamNames("b12700")...)}
	rungs, err := quiet().llamaRungs("a/b", "a/b", "", relSHA("b12651-easy1", now, forkNames("b12651-easy1", true)...), true, build, up, time.Hour)
	if err != nil || len(rungs) != 2 || rungs[1].Tag != "b12651" {
		t.Fatalf("%v %+v", err, rungs)
	}
}

func TestWindowsNamesWithAndWithoutTag(t *testing.T) {
	now := time.Now()
	for _, winTag := range []bool{false, true} {
		as, err := quiet().llamaAssets(relSHA("b12651-easy1", now, forkNames("b12651-easy1", winTag)...))
		if err != nil {
			t.Fatalf("winTag=%v: %v", winTag, err)
		}
		for _, id := range []string{"cpu-win-x64", "vulkan-win-x64", "cuda-win-x64", "cuda13-win-x64", "cudart-win-x64", "cudart13-win-x64"} {
			if !has(as, id) {
				t.Errorf("winTag=%v: missing %s", winTag, id)
			}
		}
	}
}

// An old one-release catalog is refreshed as it is, not converted to a ladder.
func TestRefreshMeasuredKeepsShape(t *testing.T) {
	saved, savedM := runtimeJSON, measuredJSON
	defer func() { runtimeJSON, measuredJSON = saved, savedM }()
	measuredJSON, runtimeJSON = []byte(`{}`), []byte(`[]`)
	for _, c := range []struct {
		name string
		cat  catalog.Catalog
		old  bool
	}{
		{"old", catalog.Catalog{Version: 9, LlamaCpp: catalog.LlamaCpp{Tag: "b1"}}, true},
		{"v10", catalog.Catalog{Version: 10, Llama: []catalog.LlamaCpp{{Tag: "b1-easy1", Patched: true}, {Tag: "b1"}}}, false},
	} {
		out := filepath.Join(t.TempDir(), "c.json")
		if err := writeCatalog(out, &c.cat); err != nil {
			t.Fatal(err)
		}
		if err := refreshMeasured(out, 11); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		b, _ := os.ReadFile(out)
		got, err := catalog.Parse(b)
		if err != nil || got.Version != 11 {
			t.Fatalf("%s: %v %v", c.name, got, err)
		}
		if c.old && (len(got.Llama) != 0 || got.LlamaCpp.Tag != "b1") {
			t.Errorf("old catalog was converted: %+v", got)
		}
		if !c.old && (len(got.Llama) != 2 || got.LlamaCpp.Tag != "") {
			t.Errorf("v10 catalog changed: %+v", got)
		}
	}
}
