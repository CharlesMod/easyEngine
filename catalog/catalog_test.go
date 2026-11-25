package catalog

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"hermesengine/probe"
)

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

func TestEmbedded(t *testing.T) {
	c, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if c.LlamaCpp.Tag == "" || len(c.LlamaCpp.Assets) == 0 || len(c.UV) == 0 {
		t.Fatalf("empty catalog: %+v", c.LlamaCpp)
	}
	for _, a := range append(append([]Asset{}, c.LlamaCpp.Assets...), c.UV...) {
		if !hex64.MatchString(a.SHA256) || !strings.HasPrefix(a.URL, "https://") || a.Size <= 0 || a.ID == "" || a.OS == "" || a.Arch == "" {
			t.Errorf("bad asset %+v", a)
		}
		for _, e := range a.Extra {
			if _, ok := c.Asset(e); !ok {
				t.Errorf("%s: extra %s is not an asset", a.ID, e)
			}
		}
	}
	ref, ok := c.Model("gemma4-rp-maxxed-q3kxl")
	if !ok || !ref.MoE || ref.File != "gemma4-rp-maxxed-Q3_K_XL.gguf" ||
		ref.SHA256 != "595eddca7b75d56ee4a6fa363deb27f1c70c8792b97508e3336a3a3d3f071b29" || ref.Size != 10001779136 {
		t.Errorf("reference model %+v", ref)
	}
	if ref.URL != "" && ref.URL != "https://huggingface.co/cmod/gemma4-rp-maxxed-GGUF/resolve/main/gemma4-rp-maxxed-Q3_K_XL.gguf" {
		t.Errorf("reference url %q", ref.URL)
	}
	for _, id := range []string{"gemma4-e4b-q4km", "gemma4-e2b-q4km"} {
		m, ok := c.Model(id)
		if !ok || !hex64.MatchString(m.SHA256) || !strings.HasPrefix(m.URL, "https://huggingface.co/unsloth/") || m.Size <= 0 {
			t.Errorf("%s: %+v", id, m)
		}
	}
	if _, ok := c.Model("nope"); ok {
		t.Error("unknown model found")
	}
	if a, ok := c.Asset("cuda-win-x64"); !ok || len(a.Extra) == 0 {
		t.Errorf("cuda-win-x64 %+v", a)
	}
	if _, ok := c.UVFor("linux", "amd64"); !ok {
		t.Error("uv linux amd64")
	}
}

func ids(as []Asset) string {
	var s []string
	for _, a := range as {
		s = append(s, a.ID)
	}
	return strings.Join(s, ",")
}

func TestBackends(t *testing.T) {
	c, _ := Load("")
	g := func(v, n string) probe.GPU { return probe.GPU{Vendor: v, Name: n} }
	cases := []struct {
		name string
		hw   probe.Hardware
		want string
	}{
		{"linux nvidia", probe.Hardware{OS: "linux", Arch: "amd64", GPUs: []probe.GPU{g("nvidia", "NVIDIA GeForce RTX 5080")}}, "cuda-linux-x64,vulkan-linux-x64,cpu-linux-x64"},
		{"linux amd", probe.Hardware{OS: "linux", Arch: "amd64", GPUs: []probe.GPU{g("amd", "Radeon")}}, "vulkan-linux-x64,cpu-linux-x64"},
		{"linux none", probe.Hardware{OS: "linux", Arch: "amd64"}, "cpu-linux-x64"},
		{"win 40-series", probe.Hardware{OS: "windows", Arch: "amd64", GPUs: []probe.GPU{g("nvidia", "NVIDIA GeForce RTX 4070")}}, "cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win 50-series", probe.Hardware{OS: "windows", Arch: "amd64", GPUs: []probe.GPU{g("nvidia", "NVIDIA GeForce RTX 5070 Ti")}}, "cuda13-win-x64,cuda-win-x64,vulkan-win-x64,cpu-win-x64"},
		{"win intel", probe.Hardware{OS: "windows", Arch: "amd64", GPUs: []probe.GPU{g("intel", "Intel Arc A770")}}, "vulkan-win-x64,cpu-win-x64"},
		{"mac arm", probe.Hardware{OS: "darwin", Arch: "arm64", GPUs: []probe.GPU{g("apple", "Apple M2")}}, "metal-mac-arm64"},
		{"mac intel", probe.Hardware{OS: "darwin", Arch: "amd64"}, "cpu-mac-x64"},
		{"unsupported", probe.Hardware{OS: "plan9", Arch: "mips"}, ""},
	}
	for _, tc := range cases {
		if got := ids(c.Backends(tc.hw)); got != tc.want {
			t.Errorf("%s: got %q want %q", tc.name, got, tc.want)
		}
	}
}

func TestLoadOverride(t *testing.T) {
	d := t.TempDir()
	emb, _ := Load("")
	hi := `{"version":999,"llamacpp":{"tag":"bTEST","assets":[]},"uv":[],"models":[]}`
	os.WriteFile(filepath.Join(d, "catalog.json"), []byte(hi), 0o644)
	c, err := Load(d)
	if err != nil || c.LlamaCpp.Tag != "bTEST" {
		t.Fatalf("override ignored: %v %+v", err, c)
	}
	os.WriteFile(filepath.Join(d, "catalog.json"), []byte(`{"version":0}`), 0o644)
	if c, _ := Load(d); c.LlamaCpp.Tag != emb.LlamaCpp.Tag {
		t.Error("lower version must lose")
	}
	os.WriteFile(filepath.Join(d, "catalog.json"), []byte(`{garbage`), 0o644)
	if c, err := Load(d); err != nil || c.LlamaCpp.Tag != emb.LlamaCpp.Tag {
		t.Error("garbage override must be ignored")
	}
}
