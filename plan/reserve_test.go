package plan

import (
	"reflect"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"

	"hermesengine/probe"
	"hermesengine/probe/gpusense"
)

var (
	rtx3050 = gpusense.Adapter{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "NVIDIA GeForce RTX 3050 Laptop GPU", DedicatedMiB: 4096, UsedMiB: 300, Source: "nvidia-smi"}
	irisXe  = gpusense.Adapter{Index: 1, Vendor: "intel", VendorID: 0x8086, Name: "Intel(R) Iris(R) Xe Graphics", DedicatedMiB: 128, SharedMiB: 7900, UsedMiB: 200, Integrated: true, Source: "dxgi"}
	hwBare  = probe.Hardware{OS: "windows", RAMTotalMiB: 16000, RAMAvailMiB: 9000} // the probe listed no GPU
	hwLap   = probe.Hardware{OS: "windows", RAMTotalMiB: 16000, RAMAvailMiB: 9000, GPUs: []probe.GPU{
		{Index: 0, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3050 Laptop GPU", VRAMTotalMiB: 4096, VRAMUsedMiB: 300, Source: "nvidia-smi"},
		{Index: 1, Vendor: "intel", Name: "Intel(R) Iris(R) Xe Graphics", VRAMTotalMiB: 128, Source: "wmi"}}}
)

func TestPickEngineEqualCardsTakesTheEmptier(t *testing.T) {
	as := []gpusense.Adapter{{Index: 0, Vendor: "nvidia", DedicatedMiB: 24576, UsedMiB: 19311}, {Index: 1, Vendor: "nvidia", DedicatedMiB: 24576, UsedMiB: 1127}}
	if e, ok := pickEngine(as); !ok || e.Index != 1 {
		t.Fatalf("picked %+v, want adapter 1", e)
	}
}

// Two equal cards: once the engine's server runs on one, another app filling that card past the
// other's use must not move the engine's reading to the other card (seen live: a 17.5 GB grab on
// GPU 0 read GPU 1's 19311 MiB as the others).
func TestMeasureReserveStaysOnOursCard(t *testing.T) {
	pic := gpusense.Picture{
		Adapters: []gpusense.Adapter{
			{Index: 0, Vendor: "nvidia", DedicatedMiB: 24576, UsedMiB: 23353},
			{Index: 1, Vendor: "nvidia", DedicatedMiB: 24576, UsedMiB: 19311},
		},
		Processes: []gpusense.Process{
			{PID: 50, Adapter: 0, DedicatedMiB: 4564, Ours: true},
			{PID: 60, Adapter: 0, DedicatedMiB: 17648},
			{PID: 70, Adapter: 0, DedicatedMiB: 1124},
			{PID: 80, Adapter: 1, DedicatedMiB: 16950},
		},
	}
	sum, _ := MeasureReserve(pic, probe.Hardware{OS: "linux"}, HostInfo{}, 1500)
	if sum.EngineAdapter != 0 || sum.Reserve.OthersMiB != 23353-4564 {
		t.Fatalf("engine adapter %d, others %d; want 0, %d", sum.EngineAdapter, sum.Reserve.OthersMiB, 23353-4564)
	}
	pic.Processes = pic.Processes[1:] // no server of ours yet: of equal cards the emptier
	if sum, _ := MeasureReserve(pic, probe.Hardware{OS: "linux"}, HostInfo{}, 1500); sum.EngineAdapter != 1 {
		t.Fatalf("engine adapter %d with no server of ours, want 1", sum.EngineAdapter)
	}
}

func TestMeasureReserve(t *testing.T) {
	const e4b = 3121 // an E4B-sized model, MiB
	ours := gpusense.Process{PID: 50, Adapter: 0, DedicatedMiB: 3000, Ours: true}
	type want struct {
		engine, host, hostAd         int
		hostMiB, othersMiB           int
		hostBasis, othersBasis, hInt string
		free                         int
		note                         bool
	}
	cases := []struct {
		name string
		pic  gpusense.Picture
		hw   probe.Hardware
		host HostInfo
		est  int
		want want
		gpu  int // engineGPU
	}{
		{name: "Optimus laptop, host on Iris Xe: no reserve, E4B fits the 3050",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{rtx3050, irisXe}, Processes: []gpusense.Process{{PID: 99, Adapter: 1, DedicatedMiB: 600, Host: true}}},
			hw:  hwLap, host: HostInfo{PID: 99}, est: 1500,
			want: want{engine: 0, hostAd: 1, host: 0, hostBasis: "other-adapter", hInt: "yes", othersMiB: 300, othersBasis: "card", free: 4096 - 300 - 512, note: true},
			gpu:  0},
		{name: "host on the dGPU before it renders: the estimate",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "RTX 3050", DedicatedMiB: 4096, UsedMiB: 0}, irisXe}},
			hw:  hwLap, host: HostInfo{Adapter: &HostAdapter{Name: "NVIDIA GeForce RTX 3050 Laptop GPU", VendorID: 0x10de}}, est: 1500,
			want: want{engine: 0, hostAd: 0, hostMiB: 1500, hostBasis: "estimate", hInt: "no", othersBasis: "card", free: 4096 - 1500 - 512},
			gpu:  0},
		{name: "host on the dGPU measured above the estimate: measured wins",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{rtx3050}, Processes: []gpusense.Process{{PID: 99, Adapter: 0, DedicatedMiB: 2200, Host: true}}},
			hw:  hwLap, host: HostInfo{PID: 99}, est: 1500,
			want: want{engine: 0, hostAd: 0, hostMiB: 2200, hostBasis: "measured", hInt: "no", othersBasis: "processes", free: 4096 - 2200 - 512},
			gpu:  0},
		{name: "host measured below the estimate: the estimate holds",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{rtx3050}, Processes: []gpusense.Process{{PID: 99, Adapter: 0, DedicatedMiB: 900}}},
			hw:  hwLap, host: HostInfo{PID: 99}, est: 1500,
			want: want{engine: 0, hostAd: 0, hostMiB: 1500, hostBasis: "estimate", hInt: "no", othersBasis: "processes", free: 4096 - 1500 - 512},
			gpu:  0},
		{name: "unknown host adapter: the estimate",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{rtx3050, irisXe}},
			hw:  hwLap, host: HostInfo{PID: 99}, est: 1500,
			want: want{engine: 0, hostAd: -1, hostMiB: 1500, hostBasis: "estimate", hInt: "unknown", othersMiB: 300, othersBasis: "card", free: 4096 - 300 - 1500 - 512, note: true},
			gpu:  0},
		{name: "other apps hold memory (per process), ours and the host excluded",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "RTX 3050", DedicatedMiB: 8192, UsedMiB: 5329}},
				Processes: []gpusense.Process{ours, {PID: 99, Adapter: 0, DedicatedMiB: 1000, Host: true}, {PID: 7, Name: "chrome.exe", Adapter: 0, DedicatedMiB: 1229}, {PID: 8, Adapter: 0, DedicatedMiB: 100}}},
			hw: hwBare, host: HostInfo{PID: 99}, est: 500,
			want: want{engine: 0, hostAd: 0, hostMiB: 1000, hostBasis: "measured", othersMiB: 1329, hInt: "no", othersBasis: "processes", free: 8192 - 1329 - 1000 - 512 - 0, note: true},
			gpu:  -1},
		{name: "other apps hold memory (per process), the card counts memory no listed process accounts for",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "RTX 3050", DedicatedMiB: 8192, UsedMiB: 6000}},
				Processes: []gpusense.Process{ours, {PID: 99, Adapter: 0, DedicatedMiB: 1000, Host: true}, {PID: 7, Name: "chrome.exe", Adapter: 0, DedicatedMiB: 1229}, {PID: 8, Adapter: 0, DedicatedMiB: 100}}},
			hw: hwBare, host: HostInfo{PID: 99}, est: 500,
			want: want{engine: 0, hostAd: 0, hostMiB: 1000, hostBasis: "measured", othersMiB: 2000, hInt: "no", othersBasis: "processes", free: 8192 - 2000 - 1000 - 512 - 0, note: true},
			gpu:  -1},
		{name: "no per-process data: the card's use less ours",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "RTX 3050", DedicatedMiB: 8192, UsedMiB: 4000}}},
			hw:  hwBare, est: 0,
			want: want{engine: 0, hostAd: -1, othersMiB: 4000, hostBasis: "none", hInt: "unknown", othersBasis: "card", free: 8192 - 4000 - 512, note: true},
			gpu:  -1},
		{name: "iGPU-only machine: a pool of system memory bounded by shared+dedicated and the RAM",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{irisXe}},
			hw:  probe.Hardware{RAMTotalMiB: 16000, RAMAvailMiB: 9000, GPUs: []probe.GPU{{Index: 0, Vendor: "intel", Name: "Intel(R) Iris(R) Xe Graphics", VRAMTotalMiB: 128}}}, est: 1500,
			want: want{engine: 1, hostAd: -1, hostMiB: 1500, hostBasis: "estimate", othersMiB: 0, hInt: "unknown", othersBasis: "card", free: 8028 - 1500 - 512},
			gpu:  0},
		{name: "Apple unified memory",
			pic: gpusense.Picture{Adapters: []gpusense.Adapter{{Index: 0, Vendor: "apple", Name: "Apple M2", DedicatedMiB: 16384, Unified: true, Integrated: true, UsedMiB: -1}}},
			hw:  probe.Hardware{OS: "darwin", RAMTotalMiB: 16384, RAMAvailMiB: 9000, GPUs: []probe.GPU{{Vendor: "apple", Name: "Apple M2", VRAMTotalMiB: 11468, UnifiedMemory: true}}}, est: 1500,
			want: want{engine: 0, hostAd: -1, hostMiB: 1500, hostBasis: "estimate", othersMiB: 11468 - 9000, hInt: "unknown", othersBasis: "card", free: 11468 - 2468 - 1500 - 512, note: true},
			gpu:  0},
		{name: "no GPU",
			pic: gpusense.Picture{}, hw: probe.Hardware{RAMTotalMiB: 16000, RAMAvailMiB: 9000}, est: 1500,
			want: want{engine: -1, hostAd: -1, hInt: "unknown", othersBasis: "none", hostBasis: "none"},
			gpu:  -1},
	}
	// the "other apps" and "no per-process" cases read hw index -1/-1 where hw does not list the adapter
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sum, gpu := MeasureReserve(c.pic, c.hw, c.host, c.est)
			w := c.want
			if sum.EngineAdapter != w.engine || sum.HostAdapter != w.hostAd || sum.HostIntegrated != w.hInt {
				t.Fatalf("engine/host/integrated = %d/%d/%s, want %d/%d/%s", sum.EngineAdapter, sum.HostAdapter, sum.HostIntegrated, w.engine, w.hostAd, w.hInt)
			}
			r := sum.Reserve
			if r.HostMiB != w.hostMiB || r.HostBasis != w.hostBasis || r.OthersMiB != w.othersMiB || r.OthersBasis != w.othersBasis {
				t.Fatalf("reserve %+v, want host %d %s others %d %s", r, w.hostMiB, w.hostBasis, w.othersMiB, w.othersBasis)
			}
			if sum.FreeForModelMiB != w.free {
				t.Fatalf("free %d, want %d (%+v)", sum.FreeForModelMiB, w.free, r)
			}
			if (sum.OthersNote != "") != w.note {
				t.Fatalf("note %q, want note=%v", sum.OthersNote, w.note)
			}
			if w.engine >= 0 && r.MarginMiB != fit.DefaultMargin(r.TotalMiB) {
				t.Fatalf("margin %d", r.MarginMiB)
			}
			if gpu < -1 || gpu >= len(c.hw.GPUs) {
				t.Fatalf("engine gpu %d out of range", gpu)
			}
		})
	}
}

func TestMeasureReserveFitsE4BOnOptimus(t *testing.T) {
	const e4b = 3121
	pic := gpusense.Picture{Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "RTX 3050", DedicatedMiB: 4096, UsedMiB: 0}, irisXe},
		Processes: []gpusense.Process{{PID: 99, Adapter: 1, DedicatedMiB: 300, Host: true}}}
	sum, gpu := MeasureReserve(pic, hwLap, HostInfo{PID: 99}, 1500)
	if sum.Reserve.HostMiB != 0 || sum.FreeForModelMiB < e4b {
		t.Fatalf("host %d free %d: E4B (%d) must fit", sum.Reserve.HostMiB, sum.FreeForModelMiB, e4b)
	}
	if gpu != 0 || hwLap.GPUs[gpu].Name == "" {
		t.Fatalf("engine gpu %d", gpu)
	}
	r := RoomForSensed(hwLap, sum, "cuda12")
	if v, _, err := r.Budget(); err != nil || int(v) < e4b || r.NoGPU || r.Unified {
		t.Fatalf("room %+v budget %v %v", r, v, err)
	}
	if r.GameReserve != 0 {
		t.Fatalf("game reserve %d, want 0 with the host on the iGPU", r.GameReserve)
	}
}

func TestMeasureReserveEmptyPictureIsTheOldNumbers(t *testing.T) {
	hw := probe.Hardware{RAMAvailMiB: 24000, GPUs: []probe.GPU{{Index: 0, Vendor: "nvidia", Name: "Fake", VRAMTotalMiB: 16384, VRAMUsedMiB: 3000, Source: "nvidia-smi"}}}
	r := RoomFor(hw, 1500)
	want := fit.Room{VRAMTotal: 16384, Others: 3000, GameReserve: 1500, Margin: fit.DefaultMargin(16384), RAMAvail: 24000, GPUName: "Fake"}
	if !reflect.DeepEqual(r, want) {
		t.Fatalf("room %+v, want %+v", r, want)
	}
}

func TestIntegratedNeverTakenAsVRAM(t *testing.T) {
	hw := probe.Hardware{RAMTotalMiB: 32000, RAMAvailMiB: 24000, GPUs: []probe.GPU{{Vendor: "intel", Name: "Intel(R) UHD Graphics", VRAMTotalMiB: 4096, Source: "wmi"}}}
	sum, _ := MeasureReserve(gpusense.Picture{}, hw, HostInfo{}, 1500)
	if sum.EngineAdapter < 0 || !sum.Adapters[0].Integrated {
		t.Fatalf("intel UHD must be integrated: %+v", sum.Adapters)
	}
	if r := RoomForSensed(hw, sum, "cpu"); !r.NoGPU {
		t.Fatalf("an integrated adapter on a non-vulkan build must be CPU: %+v", r)
	}
	r := RoomForSensed(hw, sum, "vulkan-x64")
	if r.NoGPU || !r.Unified {
		t.Fatalf("vulkan on an integrated adapter counts as unified memory: %+v", r)
	}
	if r.VRAMTotal > 32000*integratedShareOfRAM/100 || r.VRAMTotal == 4096 {
		t.Fatalf("capacity %d is not the pool", r.VRAMTotal)
	}
	// a discrete card beats the integrated one
	hw.GPUs = append(hw.GPUs, probe.GPU{Index: 0, Vendor: "nvidia", Name: "RTX", VRAMTotalMiB: 6144, Source: "nvidia-smi"})
	sum, g := MeasureReserve(gpusense.Picture{}, hw, HostInfo{}, 0)
	if g != 1 || sum.Adapters[sum.EngineAdapter].Vendor != "nvidia" {
		t.Fatalf("engine on %d (%d)", g, sum.EngineAdapter)
	}
}

// DESKTOP, live: DXGI says 3965 MiB for the 4 GB 3050 and the stand-in host on the Iris Xe holds
// shared memory only. The card keeps the probe's 4096 and the host is found on the Iris Xe.
func TestMeasureReserveDXGICardAndSharedOnlyHost(t *testing.T) {
	pic := gpusense.Picture{Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "NVIDIA GeForce RTX 3050 Laptop GPU", DedicatedMiB: 3965, SharedMiB: 16226, UsedMiB: 0, Source: "dxgi"},
		{Index: 1, Vendor: "intel", VendorID: 0x8086, Name: "Intel(R) Iris(R) Xe Graphics", DedicatedMiB: 128, SharedMiB: 16226, UsedMiB: 0, Integrated: true, Source: "dxgi"}},
		Processes: []gpusense.Process{{PID: 16288, Name: "fakehost.exe", Adapter: 1, SharedMiB: 6, Host: true}}}
	sum, gi := MeasureReserve(pic, hwLap, HostInfo{PID: 16288}, 1500)
	if gi != 0 || sum.Reserve.TotalMiB != 4096 {
		t.Fatalf("engine gpu %d total %d, want 0 and 4096", gi, sum.Reserve.TotalMiB)
	}
	if sum.HostAdapter != 1 || sum.HostIntegrated != "yes" || sum.Reserve.HostMiB != 0 || sum.Reserve.HostBasis != "other-adapter" {
		t.Fatalf("host %d %s %+v, want the Iris Xe, integrated, no host share", sum.HostAdapter, sum.HostIntegrated, sum.Reserve)
	}
	if sum.FreeForModelMiB != 4096-512 {
		t.Fatalf("free %d, want %d", sum.FreeForModelMiB, 4096-512)
	}
}
