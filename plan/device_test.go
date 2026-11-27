package plan

import (
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// realList is llama-server b12651's --list-devices on a two-card CUDA machine, with the
// backend's own startup lines around it.
const realList = `ggml_cuda_init: found 2 CUDA devices:
  Device 0: Tesla P40, compute capability 6.1, VMM: yes
  Device 1: Tesla P40, compute capability 6.1, VMM: yes
load_backend: loaded CUDA backend from /x/libggml-cuda.so
Available devices:
  CUDA0: Tesla P40 (24438 MiB, 23209 MiB free)
  CUDA1: Tesla P40 (24438 MiB, 5013 MiB free)
`

func TestParseListDevices(t *testing.T) {
	got := ParseListDevices(realList)
	want := []ListedDevice{{"CUDA0", "Tesla P40", 24438, 23209}, {"CUDA1", "Tesla P40", 24438, 5013}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v", got)
	}
	got = ParseListDevices("Available devices:\r\n  Vulkan0: Intel(R) Iris(R) Xe Graphics (8000 MiB, 7000 MiB free)\r\n  Vulkan1: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB, 3900 MiB free)\r\n")
	if len(got) != 2 || got[1].Name != "Vulkan1" || got[1].Desc != "NVIDIA GeForce RTX 3050 Laptop GPU" {
		t.Fatalf("got %+v", got)
	}
	if got := ParseListDevices("  CUDA0: Fake GPU"); len(got) != 1 || got[0].Desc != "Fake GPU" || got[0].TotalMiB != 0 {
		t.Fatalf("no memory figures: %+v", got)
	}
}

func p40s(engine int, uuids bool) gpusense.GPUSummary {
	as := []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", Name: "Tesla P40", DedicatedMiB: 24576, PCI: "0000:0b:00.0", UUID: "GPU-90bd"},
		{Index: 1, Vendor: "nvidia", Name: "Tesla P40", DedicatedMiB: 24576, PCI: "0000:84:00.0", UUID: "GPU-cca6"},
	}
	if !uuids {
		as[0].UUID, as[1].UUID = "", ""
	}
	return gpusense.GPUSummary{Adapters: as, EngineAdapter: engine}
}

// Two CUDA cards: the planned one is named by its UUID, so CUDA's own order never matters.
func TestPinTwoCUDACardsByUUID(t *testing.T) {
	for _, e := range []int{0, 1} {
		p := PinDevice(p40s(e, true), "cuda", ParseListDevices(realList))
		uuid := p40s(e, true).Adapters[e].UUID
		if !p.Pinned || p.Via != "uuid" || p.Device != "CUDA0" || !reflect.DeepEqual(p.Env, []string{"CUDA_VISIBLE_DEVICES=" + uuid}) ||
			!reflect.DeepEqual(p.Args, []string{"--split-mode", "none"}) {
			t.Fatalf("engine %d: %+v", e, p)
		}
	}
	// the plan, which reads no device list, predicts the same
	if p := PinDevice(p40s(1, true), "cuda", nil); !p.Pinned || p.Env[0] != "CUDA_VISIBLE_DEVICES=GPU-cca6" {
		t.Fatalf("prediction %+v", p)
	}
}

// nvidia-smi numbers cards in PCI order; CUDA (fastest first) may number them otherwise. Without
// a UUID the card is found by its name in llama-server's own list, not by its probe index.
func TestPinCUDAOrderDiffersFromPCI(t *testing.T) {
	sum := gpusense.GPUSummary{EngineAdapter: 0, Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3060", DedicatedMiB: 12288, PCI: "0000:01:00.0"},
		{Index: 1, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3090", DedicatedMiB: 24576, PCI: "0000:02:00.0"},
	}}
	list := ParseListDevices("  CUDA0: NVIDIA GeForce RTX 3090 (24135 MiB, 23000 MiB free)\n  CUDA1: NVIDIA GeForce RTX 3060 (11910 MiB, 11000 MiB free)\n")
	p := PinDevice(sum, "cuda", list)
	if !p.Pinned || p.Via != "name" || p.Device != "CUDA1" || !reflect.DeepEqual(p.Args, []string{"--device", "CUDA1", "--split-mode", "none"}) || p.Env != nil {
		t.Fatalf("%+v", p)
	}
	sum.EngineAdapter = 1
	if p := PinDevice(sum, "cuda", list); p.Device != "CUDA0" {
		t.Fatalf("%+v", p)
	}
	// two cards of one model and no UUID: never guessed
	if p := PinDevice(p40s(1, false), "cuda", ParseListDevices(realList)); p.Pinned || !strings.Contains(p.Note, "match equally") {
		t.Fatalf("ambiguous %+v", p)
	}
}

// An Optimus laptop under Vulkan sees the iGPU too, often first.
func TestPinOptimusVulkan(t *testing.T) {
	sum := gpusense.GPUSummary{EngineAdapter: 0, Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3050 Laptop GPU", DedicatedMiB: 3965, Source: "dxgi"},
		{Index: 1, Vendor: "intel", Name: "Intel(R) Iris(R) Xe Graphics", DedicatedMiB: 128, SharedMiB: 8000, Integrated: true, Source: "dxgi"},
	}}
	list := ParseListDevices("  Vulkan0: Intel(R) Iris(R) Xe Graphics (8000 MiB, 7000 MiB free)\n  Vulkan1: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB, 3900 MiB free)\n")
	p := PinDevice(sum, "vulkan", list)
	if !p.Pinned || p.Device != "Vulkan1" || !reflect.DeepEqual(p.Args, []string{"--device", "Vulkan1", "--split-mode", "none"}) {
		t.Fatalf("%+v", p)
	}
	// the iGPU planned (no discrete card with memory): the iGPU, by name, "(R)" and all
	sum.EngineAdapter = 1
	if p := PinDevice(sum, "vulkan", list); p.Device != "Vulkan0" {
		t.Fatalf("%+v", p)
	}
	// a short sysfs name still finds its card, by vendor, when it is the only one of its vendor
	sum.Adapters[0].Name, sum.EngineAdapter = "GA107M", 0
	if p := PinDevice(sum, "vulkan", list); p.Device != "Vulkan1" {
		t.Fatalf("vendor match %+v", p)
	}
	// the plan predicts nothing for Vulkan: it is matched at launch
	if p := PinDevice(sum, "vulkan", nil); p.Pinned || !strings.Contains(p.Note, "at launch") {
		t.Fatalf("prediction %+v", p)
	}
}

// The planned card missing from llama-server's list: llama-server's default, said so.
func TestPinPlannedCardMissing(t *testing.T) {
	sum := gpusense.GPUSummary{EngineAdapter: 1, Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "amd", Name: "AMD Radeon RX 6600", DedicatedMiB: 8192},
		{Index: 1, Vendor: "amd", Name: "AMD Radeon RX 7900 XTX", DedicatedMiB: 24576},
	}}
	p := PinDevice(sum, "vulkan", ParseListDevices("  Vulkan0: AMD Radeon RX 6600 (RADV NAVI23) (8176 MiB, 8000 MiB free)\n"))
	if p.Pinned || p.Args != nil || p.Env != nil || !strings.Contains(p.Note, "not matched") || p.Adapter != 1 {
		t.Fatalf("%+v", p)
	}
	if p := PinDevice(sum, "vulkan", []ListedDevice{}); p.Pinned || !strings.Contains(p.Note, "none listed") {
		t.Fatalf("empty list %+v", p)
	}
}

func TestPinNothingToDo(t *testing.T) {
	one := gpusense.GPUSummary{EngineAdapter: 0, Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", Name: "RTX 4090", DedicatedMiB: 24576, UUID: "GPU-1"}}}
	mac := gpusense.GPUSummary{EngineAdapter: 0, Adapters: []gpusense.Adapter{{Index: 0, Vendor: "apple", Name: "Apple M2", Unified: true, Integrated: true}}}
	cpu := gpusense.GPUSummary{EngineAdapter: -1}
	for name, c := range map[string]struct {
		sum   gpusense.GPUSummary
		accel string
	}{"one card": {one, "cuda"}, "metal": {mac, "metal"}, "cpu planned": {cpu, "cuda"}, "cpu build": {p40s(0, true), "cpu"},
		"amd planned on cuda": {gpusense.GPUSummary{EngineAdapter: 1, Adapters: []gpusense.Adapter{p40s(0, true).Adapters[0], {Index: 1, Vendor: "amd", Name: "RX 7900", DedicatedMiB: 49152}}}, "cuda"}} {
		p := PinDevice(c.sum, c.accel, ParseListDevices(realList))
		if p.Pinned || p.Args != nil || p.Env != nil || p.Note == "" {
			t.Errorf("%s: %+v", name, p)
		}
	}
	if p := PinDevice(cpu, "cuda", nil); p.Adapter != -1 {
		t.Fatalf("%+v", p)
	}
}

func TestAccelOf(t *testing.T) {
	for id, want := range map[string]string{"cuda-linux-x64": "cuda", "cuda13-win-x64": "cuda", "vulkan-win-x64": "vulkan", "metal-mac-arm64": "metal", "cpu-mac-x64": "cpu", "fake": ""} {
		if got := AccelOf(id); got != want {
			t.Errorf("%s: %q", id, got)
		}
	}
}

// A name is never matched to a shorter one: a 3060 Ti planned is not the 3060 listed.
func TestPinNameNotShortened(t *testing.T) {
	sum := gpusense.GPUSummary{EngineAdapter: 1, Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3060", DedicatedMiB: 12288},
		{Index: 1, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3060 Ti", DedicatedMiB: 12288},
	}}
	if p := PinDevice(sum, "vulkan", ParseListDevices("  Vulkan0: NVIDIA GeForce RTX 3060 (12288 MiB, 12000 MiB free)\n")); p.Pinned {
		t.Fatalf("%+v", p)
	}
	list := ParseListDevices("  Vulkan0: NVIDIA GeForce RTX 3060 (12288 MiB, 12000 MiB free)\n  Vulkan1: NVIDIA GeForce RTX 3060 Ti (8192 MiB, 8000 MiB free)\n")
	if p := PinDevice(sum, "vulkan", list); p.Device != "Vulkan1" {
		t.Fatalf("%+v", p)
	}
	sum.EngineAdapter = 0
	if p := PinDevice(sum, "vulkan", list); p.Device != "Vulkan0" {
		t.Fatalf("%+v", p)
	}
}
