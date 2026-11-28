package run

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
)

// What the engine remembers per build carries the build's full tag: the prompt cache key, the
// cap's build (fit.CapLift.Build, let go when it differs: core/fit TestCapNamesWhatLiftsIt) and
// the book's key. b12651 and b12651-easy1 are two builds even under one backend id and with an
// archive hash that happened to match.
func TestBuildKeysCarryTheTag(t *testing.T) {
	stock, patched := buildKey("cuda-linux-x64", "b12651", "abc"), buildKey("cuda-linux-x64", "b12651-easy1", "abc")
	if stock == patched || stock != "cuda-linux-x64@b12651@abc" || patched != "cuda-linux-x64@b12651-easy1@abc" {
		t.Fatalf("build keys %q %q", stock, patched)
	}
	if legacy := buildKey("cuda-linux-x64", "", "abc"); legacy != "cuda-linux-x64@abc" {
		t.Fatalf("a build installed before tags keeps its old key (its caps still match it): %q", legacy)
	}
	lc := func(build string) loadCtx {
		return loadCtx{in: install{model: catalog.Model{SHA256: "m"}, build: build}}
	}
	if a, b := ctxKeyOf(lc(stock), fit.Shape{}), ctxKeyOf(lc(patched), fit.Shape{}); a == b || a.Build != stock {
		t.Fatalf("context keys %v %v", a, b)
	}
	if a, b := fit.BookKey("m", "Tesla P40", bookBackend("cuda-linux-x64", "b12651")), fit.BookKey("m", "Tesla P40", bookBackend("cuda-linux-x64", "b12651-easy1")); a == b {
		t.Fatalf("book keys %q %q", a, b)
	}
	if k := bookBackend("cuda-linux-x64", ""); k != "cuda-linux-x64" {
		t.Fatalf("an untagged build keeps the book's old key: %q", k)
	}
}

// Builds are installed beside each other: backends/<id>@<tag>, the old layout backends/<id>.
func TestBackendDirAndInstalled(t *testing.T) {
	d := t.TempDir()
	if got := backendDir(d, "cuda-linux-x64", "b12651-easy1"); got != filepath.Join(d, "backends", "cuda-linux-x64@b12651-easy1") {
		t.Fatal(got)
	}
	if got := backendDir(d, "cuda-linux-x64", ""); got != filepath.Join(d, "backends", "cuda-linux-x64") {
		t.Fatal(got)
	}
	dir := backendDir(d, "cpu-linux-x64", "b1")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, serverBinName()), []byte("x"), 0o755)
	if backendInstalled(d, "cpu-linux-x64", "b1") {
		t.Fatal("installed without its marker")
	}
	os.WriteFile(filepath.Join(dir, ".installed"), []byte("sha"), 0o644)
	if !backendInstalled(d, "cpu-linux-x64", "b1") || backendInstalled(d, "cpu-linux-x64", "b2") || backendInstalled(d, "cpu-linux-x64", "") {
		t.Fatal("installed is per build")
	}
}

// A Vulkan build whose only device is Mesa's software one fails its smoke test: the CPU build
// that follows is faster (a VM, a machine with no GPU driver).
func TestRealVulkan(t *testing.T) {
	for out, want := range map[string]bool{
		"Available devices:\n  Vulkan0: AMD Radeon RX 7900 XTX (RADV NAVI31) (24560 MiB, 24000 MiB free)\n":                                                                          true,
		"Available devices:\n  Vulkan0: llvmpipe (LLVM 17.0.6, 256 bits) (32000 MiB, 32000 MiB free)\n":                                                                              false,
		"Available devices:\n  Vulkan0: llvmpipe (LLVM 17.0.6, 256 bits) (32000 MiB, 32000 MiB free)\n  Vulkan1: Intel(R) Arc(tm) A770 Graphics (DG2) (16288 MiB, 16000 MiB free)\n": true,
		"ggml_vulkan: Found 1 Vulkan devices\n": true, // no device line in the usual form: the word suffices, as before
	} {
		if got := realVulkan(out); got != want {
			t.Errorf("%q: %v want %v", out, got, want)
		}
	}
}

// The patched server's line on its CUDA buffers (ee-llama: ggml-cuda prints it at device init).
func TestVMMLine(t *testing.T) {
	for l, want := range map[string]string{
		"easy: CUDA VMM buffers on (1 device)":                 "on (1 device)",
		"easy: CUDA VMM buffers off (GGML_CUDA_VMM_BUFFERS=0)": "off (GGML_CUDA_VMM_BUFFERS=0)",
		"easy: CUDA VMM buffers off (2 devices)\r":             "off (2 devices)",
		"ggml_cuda_init: found 1 CUDA devices:":                "",
		"easy: slot 0 released, idle":                          "",
	} {
		if got := vmmLine(l); got != want {
			t.Errorf("%q: %q want %q", l, got, want)
		}
	}
}
