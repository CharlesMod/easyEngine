package gpusense

import (
	"context"
	"errors"
	"testing"
)

func TestParsePDHProcInstance(t *testing.T) {
	good := []struct {
		in   string
		pid  int
		luid string
		phys int
	}{
		{"pid_1234_luid_0x00000000_0x0000D1F2_phys_0", 1234, "0x00000000_0x0000D1F2", 0},
		{"pid_4_luid_0x00000000_0x0000d1f2_phys_1", 4, "0x00000000_0x0000D1F2", 1},
		{"pid_77_luid_0x00000000_0x0000D1F2_phys_0_eng_3_engtype_3D", 77, "0x00000000_0x0000D1F2", 0},
		{"  pid_9_luid_0x0_0x1_phys_0 ", 9, "0x00000000_0x00000001", 0},
	}
	for _, g := range good {
		pid, luid, phys, ok := parsePDHProcInstance(g.in)
		if !ok || pid != g.pid || luid != g.luid || phys != g.phys {
			t.Errorf("%q = %d %s %d %v", g.in, pid, luid, phys, ok)
		}
	}
	for _, bad := range []string{"", "_Total", "pid__luid_0x0_0x1_phys_0", "pid_12_luid_0x0_phys_0", "pid_12_luid_0xZZ_0x1_phys_0",
		"pid_12_luid_0x100000000_0x1_phys_0", "luid_0x0_0x1_phys_0", "pid_x_luid_0x0_0x1_phys_0"} {
		if _, _, _, ok := parsePDHProcInstance(bad); ok {
			t.Errorf("%q should not parse", bad)
		}
	}
	if l, ph, ok := parsePDHAdapterInstance("luid_0x00000000_0x0000D1F2_phys_1"); !ok || l != "0x00000000_0x0000D1F2" || ph != 1 {
		t.Errorf("adapter instance %s %d %v", l, ph, ok)
	}
	if _, _, ok := parsePDHAdapterInstance("pid_1_luid_0x0_0x1_phys_0"); ok {
		t.Error("process instance is not an adapter instance")
	}
}

func TestIntegratedDXGI(t *testing.T) {
	cases := []struct {
		vid  int
		name string
		ded  int
		want bool
	}{
		{0x8086, "Intel(R) Iris(R) Xe Graphics", 128, true},
		{0x8086, "Intel(R) UHD Graphics 770", 0, true},
		{0x8086, "Intel(R) Arc(TM) A770 Graphics", 16100, false},
		{0x8086, "Intel(R) Arc(TM) Graphics", 128, false}, // Arc iGPU named so: treated as dedicated-class
		{0x1002, "AMD Radeon(TM) 780M", 512, true},
		{0x1002, "AMD Radeon RX 7900 XT", 20464, false},
		{0x1002, "AMD Radeon(TM) Graphics", 2048, true},
		{0x10de, "NVIDIA GeForce RTX 3050", 4096, false},
		{0x5143, "Snapdragon", 0, true},
	}
	for _, c := range cases {
		if got := integratedDXGI(c.vid, c.name, c.ded); got != c.want {
			t.Errorf("%s: %v", c.name, got)
		}
	}
}

func fakeWindows(t *testing.T) {
	t.Helper()
	od, op, on := nativeDXGI, nativePDH, nativeProcImage
	t.Cleanup(func() { nativeDXGI, nativePDH, nativeProcImage = od, op, on })
	nativeDXGI = func() ([]dxgiAdapter, error) {
		return []dxgiAdapter{
			{Name: "Intel(R) Iris(R) Xe Graphics", VendorID: 0x8086, DedicatedBytes: 128 << 20, SharedBytes: 8000 << 20, LUIDLow: 0xD1F2},
			{Name: "NVIDIA GeForce RTX 3050", VendorID: 0x10de, DedicatedBytes: 4096 << 20, SharedBytes: 8000 << 20, LUIDLow: 0xD3A1},
			{Name: "Microsoft Basic Render Driver", VendorID: 0x1414, Flags: 2, LUIDLow: 0x1},
		}, nil
	}
	nativePDH = func() ([]pdhSample, error) {
		return []pdhSample{
			{pdhProcDedicated, "pid_1234_luid_0x00000000_0x0000D3A1_phys_0", 3000 << 20},
			{pdhProcDedicated, "pid_1234_luid_0x00000000_0x0000D3A1_phys_1", 100 << 20}, // summed with phys_0
			{pdhProcShared, "pid_1234_luid_0x00000000_0x0000D3A1_phys_0", 200 << 20},
			{pdhProcDedicated, "pid_5678_luid_0x00000000_0x0000D1F2_phys_0", 90 << 20},
			{pdhProcDedicated, "pid_5678_luid_0x00000000_0x0000D3A1_phys_0", 0},    // zero everywhere: dropped
			{pdhProcDedicated, "pid_9_luid_0x00000000_0x0000FFFF_phys_0", 5 << 20}, // unknown luid
			{pdhProcDedicated, "garbage", 5 << 20},
			{pdhAdapterDedicated, "luid_0x00000000_0x0000D3A1_phys_0", 3500 << 20},
			{pdhAdapterDedicated, "luid_0x00000000_0x0000D1F2_phys_0", 100 << 20},
		}, nil
	}
	nativeProcImage = func(pid int) string {
		switch pid {
		case 1234:
			return `C:\Program Files\Game\game.exe`
		case 5678:
			return `C:\Windows\System32\dwm.exe`
		}
		return ""
	}
}

func TestSenseWindows(t *testing.T) {
	fakeWindows(t)
	f := &fake{goos: "windows", goarch: "amd64", cmds: map[string]string{"nvidia-smi -q -x": td0("nvidia-smi-two-gpus.xml")}}
	pic := Sense(context.Background(), f, []int{5678}, 1234)
	if len(pic.Adapters) != 2 {
		t.Fatalf("software adapter must be skipped: %+v", pic.Adapters)
	}
	nv, ig := pic.Adapters[0], pic.Adapters[1]
	if nv.Vendor != "nvidia" || nv.Index != 0 || nv.LUID != "0x00000000_0x0000D3A1" || nv.Integrated || nv.DedicatedMiB != 4096 ||
		nv.UsedMiB != 3500 || nv.UUID == "" || nv.Source != "dxgi" {
		t.Errorf("nvidia %+v", nv)
	}
	if ig.Vendor != "intel" || ig.Index != 1 || !ig.Integrated || ig.UsedMiB != 100 || ig.SharedMiB != 8000 {
		t.Errorf("intel %+v", ig)
	}
	if len(pic.Processes) != 2 {
		t.Fatalf("%+v", pic.Processes)
	}
	g, d := pic.Processes[0], pic.Processes[1]
	if g.PID != 1234 || g.Adapter != 0 || g.DedicatedMiB != 3100 || g.SharedMiB != 200 || g.Name != "game.exe" || !g.Host || g.Ours || g.Kind != "" {
		t.Errorf("game %+v", g)
	}
	if d.PID != 5678 || d.Adapter != 1 || d.DedicatedMiB != 90 || d.Name != "dwm.exe" || !d.Ours {
		t.Errorf("dwm %+v", d)
	}
	for _, k := range []string{"dxgi", "pdh", "nvidia-smi"} {
		if pic.Sources[k] != "ok" {
			t.Errorf("%s: %q", k, pic.Sources[k])
		}
	}
}

func TestSenseWindowsDegrades(t *testing.T) {
	fakeWindows(t)
	nativePDH = func() ([]pdhSample, error) { return nil, errors.New("no GPU counters") }
	pic := Sense(context.Background(), &fake{goos: "windows", goarch: "amd64"}, nil, 0)
	if len(pic.Adapters) != 2 || len(pic.Processes) != 0 || pic.Sources["pdh"] != "no GPU counters" || pic.Sources["nvidia-smi"] == "ok" {
		t.Errorf("%+v", pic)
	}
	if pic.Adapters[0].UsedMiB != -1 {
		t.Errorf("used unknown must be -1: %+v", pic.Adapters[0])
	}
}

func TestSenseWindowsWMIFallback(t *testing.T) {
	fakeWindows(t)
	nativeDXGI = func() ([]dxgiAdapter, error) { return nil, errors.New("dxgi.dll missing") }
	ps := "\xef\xbb\xbf" + `{"cpu":"Intel","cores":8,"ram_total":17179869184,"ram_free_kb":8000000,"gpus":[` +
		`{"name":"Intel(R) UHD Graphics","adapter_ram":1073741824,"qw":0},{"name":"NVIDIA GeForce GTX 1650","adapter_ram":4294967295,"qw":4294967296}]}`
	f := &fake{goos: "windows", goarch: "amd64", cmds: map[string]string{"powershell": ps,
		"nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits": "0, NVIDIA GeForce GTX 1650, 4096, 300\n"}}
	pic := Sense(context.Background(), f, nil, 0)
	if pic.Sources["dxgi"] != "dxgi.dll missing" || pic.Sources["wmi"] != "ok" {
		t.Errorf("%+v", pic.Sources)
	}
	if len(pic.Adapters) != 2 {
		t.Fatalf("%+v", pic.Adapters)
	}
	if a := pic.Adapters[0]; a.Vendor != "nvidia" || a.Index != 0 || a.UsedMiB != 300 || a.Integrated || a.LUID != "" {
		t.Errorf("nvidia %+v", a)
	}
	if a := pic.Adapters[1]; a.Vendor != "intel" || !a.Integrated || a.Source != "wmi" {
		t.Errorf("intel %+v", a)
	}
	if len(pic.Processes) != 0 {
		t.Error("no LUIDs, no PDH mapping")
	}
}
