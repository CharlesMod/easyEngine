package run_test

import (
	"os"
	"testing"

	"hermesengine/run"
)

func TestParseOOMPatterns(t *testing.T) {
	cases := []struct {
		name string
		text string
		oom  bool
		mib  int
	}{
		{"cuda alloc line", "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1024.00 MiB on device 0: cudaMalloc failed: out of memory\n", true, 1024},
		{"buffer size wins over allocating", "allocating 99.00 MiB on device 0: cudaMalloc failed: out of memory\nalloc_tensor_range: failed to allocate CUDA0 buffer of size 2097152\n", true, 2},
		{"last buffer counts", "failed to allocate CUDA0 buffer of size 1048576\nfailed to allocate CUDA0 buffer of size 10485760\n", true, 10},
		{"buffer size rounds up", "failed to allocate buffer of size 1048577\n", true, 2},
		{"fractional MiB rounds up", "allocating 10.25 MiB on device 1: cudaMalloc failed: out of memory\n", true, 11},
		{"bare CUDA error", "CUDA error: out of memory\n  current device: 0\n", true, 0},
		{"cudaMalloc failed", "cudaMalloc failed: something\n", true, 0},
		{"unable to allocate", "llama_model_load: unable to allocate CUDA0 buffer\n", true, 0},
		{"vulkan", "ggml_vulkan: Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory\n", true, 0},
		{"case-insensitive", "OUT OF MEMORY\n", true, 0},
		{"not a memory error", "llama_model_load: error loading model: tensor data is not within the file bounds\n", false, 0},
		{"segfault", "Segmentation fault (core dumped)\n", false, 0},
		{"empty", "", false, 0},
		{"memory word alone", "llama_kv_cache: memory size = 1024.00 MiB\nload_tensors: offloaded 35/35 layers to GPU\n", false, 0},
		{"pinned fallback is not a failure", "ggml_cuda_host_malloc: failed to allocate 512.00 MiB of pinned memory: out of memory\nsrv  server_run_i: exiting due to bad flag\n", false, 0},
		{"vulkan pinned fallback is not a failure", "WARNING: failed to allocate 512.00 MB of pinned memory\n", false, 0},
		{"pinned fallback beside a real one", "ggml_cuda_host_malloc: failed to allocate 512.00 MiB of pinned memory: out of memory\nalloc_tensor_range: failed to allocate CUDA0 buffer of size 3145728\n", true, 3},
	}
	for _, c := range cases {
		oom, mib := run.ParseOOM(c.text)
		if oom != c.oom || mib != c.mib {
			t.Errorf("%s: ParseOOM = %v, %d; want %v, %d", c.name, oom, mib, c.oom, c.mib)
		}
	}
}

// Real tails captured on a Tesla P40 with llama.cpp b12651: a CUDA load whose KV cache does
// not fit, and the Vulkan build's (lavapipe device) the same.
func TestParseOOMRealTails(t *testing.T) {
	for _, c := range []struct {
		file string
		mib  int
	}{
		{"testdata/oom-cuda-b12651.txt", 62500},
		{"testdata/oom-vulkan-b12651.txt", 31250},
	} {
		b, err := os.ReadFile(c.file)
		if err != nil {
			t.Fatal(err)
		}
		oom, mib := run.ParseOOM(string(b))
		if !oom || mib != c.mib {
			t.Errorf("%s: ParseOOM = %v, %d; want true, %d", c.file, oom, mib, c.mib)
		}
	}
}
