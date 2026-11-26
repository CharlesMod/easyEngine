package gpusense

import (
	"context"
	"errors"
	"os"
	"path"
	"sort"
	"strings"
	"testing"
)

type fake struct {
	goos, goarch string
	cmds         map[string]string
	files        map[string]string
}

func (f *fake) Platform() (string, string)      { return f.goos, f.goarch }
func (f *fake) NumCPU() int                     { return 4 }
func (f *fake) DiskFreeMiB(string) (int, error) { return 1000, nil }
func (f *fake) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	key := strings.TrimSpace(name + " " + strings.Join(args, " "))
	if name == "powershell" {
		key = "powershell"
	}
	if v, ok := f.cmds[key]; ok {
		return []byte(v), nil
	}
	return nil, errors.New("not found: " + key)
}
func (f *fake) ReadFile(p string) ([]byte, error) {
	if v, ok := f.files[p]; ok {
		return []byte(v), nil
	}
	return nil, errors.New("no file " + p)
}
func (f *fake) Glob(pattern string) ([]string, error) {
	// Files and the directories that contain them, like a real filesystem.
	seen := map[string]bool{}
	var out []string
	for p := range f.files {
		for q := p; q != "/" && q != "."; q = path.Dir(q) {
			if ok, _ := path.Match(pattern, q); ok && !seen[q] {
				seen[q] = true
				out = append(out, q)
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

func td(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestParseNvidiaXML(t *testing.T) {
	as, ps, err := parseNvidiaXML([]byte(td(t, "nvidia-smi-two-gpus.xml")))
	if err != nil {
		t.Fatal(err)
	}
	if len(as) != 2 || as[0].Index != 0 || as[1].Index != 1 {
		t.Fatalf("adapters %+v", as)
	}
	a := as[0]
	if a.Name != "NVIDIA GeForce RTX 5080" || a.DedicatedMiB != 16303 || a.UsedMiB != 5120 ||
		a.PCI != "0000:01:00.0" || a.UUID != "GPU-aaaaaaaa-1111-2222-3333-444444444444" || a.VendorID != 0x10de || a.Source != "nvidia-smi" {
		t.Errorf("gpu0 %+v", a)
	}
	if as[1].UsedMiB != -1 || as[1].DedicatedMiB != 6144 || as[1].PCI != "0000:02:00.0" {
		t.Errorf("gpu1 (used N/A) %+v", as[1])
	}
	if len(ps) != 3 {
		t.Fatalf("procs %+v", ps)
	}
	want := []Process{
		{PID: 4242, Name: "llama-server", Adapter: 0, DedicatedMiB: 4096, Kind: "compute"},
		{PID: 5150, Name: "Voxel.exe", Adapter: 0, DedicatedMiB: 0, Kind: "graphics"},
		{PID: 977, Name: "Xorg", Adapter: 0, DedicatedMiB: 512, Kind: "graphics"},
	}
	for i, w := range want {
		if ps[i] != w {
			t.Errorf("proc %d = %+v want %+v", i, ps[i], w)
		}
	}
}

func TestParseNvidiaXMLBad(t *testing.T) {
	for _, in := range []string{"", "not xml", "<nvidia_smi_log></nvidia_smi_log>", "<nvidia_smi_log><gpu>"} {
		if _, _, err := parseNvidiaXML([]byte(in)); err == nil {
			t.Errorf("%q: want error", in)
		}
	}
}

func TestParseSizeBytes(t *testing.T) {
	cases := map[string]struct {
		n  int64
		ok bool
	}{
		"123 MiB": {123 << 20, true}, "4096 KiB": {4 << 20, true}, "1.5 GiB": {3 << 29, true}, "7": {7, true},
		"N/A": {0, false}, "[N/A]": {0, false}, "": {0, false}, "5 parsecs": {0, false}, "-3 MiB": {0, false}, "2 kB": {2048, true},
	}
	for in, w := range cases {
		n, ok := parseSizeBytes(in)
		if n != w.n || ok != w.ok {
			t.Errorf("%q = %d,%v want %d,%v", in, n, ok, w.n, w.ok)
		}
	}
}

func TestParseFdinfo(t *testing.T) {
	c, ok := parseFdinfo(td(t, "fdinfo-amdgpu.txt"))
	if !ok || c.Driver != "amdgpu" || c.PDev != "0000:03:00.0" || c.ClientID != "77" || c.DedicatedMiB != 1024 || c.SharedMiB != 64 || !c.HasMem {
		t.Errorf("amdgpu %+v", c)
	}
	c, ok = parseFdinfo(td(t, "fdinfo-xe.txt"))
	// resident beats total: vram 200; system 1 MiB + gtt 4 MiB.
	if !ok || c.Driver != "xe" || c.DedicatedMiB != 200 || c.SharedMiB != 5 {
		t.Errorf("xe %+v", c)
	}
	c, ok = parseFdinfo(td(t, "fdinfo-i915.txt"))
	if !ok || c.Driver != "i915" || c.DedicatedMiB != 0 || c.SharedMiB != 1024 || !c.HasMem {
		t.Errorf("i915 (no vram keys) %+v", c)
	}
	if _, ok := parseFdinfo(td(t, "fdinfo-not-drm.txt")); ok {
		t.Error("non-drm file parsed as drm")
	}
	// Two vram regions sum; junk and N/A values are absent; bare number is bytes.
	c, _ = parseFdinfo("drm-driver: xe\ndrm-client-id: 1\ndrm-resident-vram0: 100 MiB\ndrm-resident-vram1: 50 MiB\ndrm-resident-gtt: N/A\ndrm-memory-cpu: 1048576\n")
	if c.DedicatedMiB != 150 || c.SharedMiB != 1 {
		t.Errorf("regions/junk %+v", c)
	}
}

const drm = "/sys/class/drm/"

func sysfsFiles() map[string]string {
	return map[string]string{
		// card0: AMD discrete
		drm + "card0/device/vendor":              "0x1002\n",
		drm + "card0/device/device":              "0x744c\n",
		drm + "card0/device/uevent":              "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\n",
		drm + "card0/device/mem_info_vram_total": "17163091968\n",
		drm + "card0/device/mem_info_vram_used":  "2147483648\n",
		drm + "card0/device/mem_info_gtt_total":  "33554432000\n",
		drm + "card0/device/product_name":        "Radeon RX 7900 XT\n",
		// card1: Intel iGPU
		drm + "card1/device/vendor": "0x8086\n",
		drm + "card1/device/device": "0x46a6\n",
		drm + "card1/device/uevent": "PCI_SLOT_NAME=0000:00:02.0\n",
		// card2: an NVIDIA card, skipped
		drm + "card2/device/vendor": "0x10de\n",
		// card3: duplicate of card1's PCI slot (some systems expose two nodes)
		drm + "card3/device/vendor": "0x8086\n",
		drm + "card3/device/uevent": "PCI_SLOT_NAME=0000:00:02.0\n",
		// renderD128 must never be read as an adapter
		"/sys/class/drm/renderD128/device/vendor": "0x1002\n",
	}
}

func TestSysfsAdapters(t *testing.T) {
	as, err := sysfsAdapters(&fake{files: sysfsFiles()}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(as) != 2 {
		t.Fatalf("%+v", as)
	}
	a, b := as[0], as[1]
	if a.Index != 1 || a.Vendor != "amd" || a.Name != "Radeon RX 7900 XT" || a.DedicatedMiB != 16368 || a.UsedMiB != 2048 ||
		a.SharedMiB != 32000 || a.Integrated || a.PCI != "0000:03:00.0" || a.Source != "sysfs" || a.VendorID != 0x1002 {
		t.Errorf("amd %+v", a)
	}
	if b.Index != 2 || b.Vendor != "intel" || !b.Integrated || b.DedicatedMiB != 0 || b.UsedMiB != -1 || b.Name != "Intel GPU 0x46a6" {
		t.Errorf("intel %+v", b)
	}
}

func TestSysfsIntegratedHeuristic(t *testing.T) {
	cases := []struct {
		vendor, pci string
		ded, shared int
		boot, want  bool
		why         string
	}{
		{"intel", "0000:00:02.0", 0, 0, true, true, "intel iGPU, no vram"},
		{"intel", "", 0, 0, false, true, "intel iGPU, bus unknown"},
		{"intel", "0000:03:00.0", 0, 0, false, false, "intel off bus 0 (Arc)"},
		{"intel", "0000:00:02.0", 8192, 0, false, false, "intel reporting local vram"},
		{"amd", "0000:0c:00.0", 512, 16000, true, true, "APU, tiny carve-out"},
		{"amd", "0000:0c:00.0", 2048, 16000, true, true, "APU, 2 GiB carve-out, boot vga"},
		{"amd", "0000:0c:00.0", 2048, 16000, false, false, "2 GiB card, not boot, off bus 0: discrete"},
		{"amd", "0000:00:01.0", 2048, 16000, false, true, "2 GiB on bus 0"},
		{"amd", "0000:03:00.0", 16384, 32000, true, false, "big card even if boot vga"},
		{"amd", "0000:0c:00.0", 512, 0, true, false, "no GTT info"},
		{"other", "", 0, 0, false, false, "unknown vendor"},
	}
	for _, c := range cases {
		if got := sysfsIntegrated(c.vendor, c.pci, c.ded, c.shared, c.boot); got != c.want {
			t.Errorf("%s: got %v", c.why, got)
		}
	}
}

func fdFiles() map[string]string {
	f := sysfsFiles()
	f["/proc/100/fdinfo/12"] = td0("fdinfo-amdgpu.txt")
	f["/proc/100/fdinfo/13"] = td0("fdinfo-amdgpu.txt") // dup fd of the same client
	f["/proc/100/fdinfo/3"] = "pos:\t0\nflags:\t02\n"
	f["/proc/100/comm"] = "game\n"
	f["/proc/200/fdinfo/9"] = td0("fdinfo-i915.txt")
	f["/proc/200/comm"] = "compositor\n"
	f["/proc/300/fdinfo/9"] = "drm-driver: amdgpu\ndrm-pdev: 0000:03:00.0\ndrm-client-id: 78\ndrm-memory-vram: 100 MiB\n"
	f["/proc/300/fdinfo/10"] = "drm-driver: amdgpu\ndrm-pdev: 0000:03:00.0\ndrm-client-id: 79\ndrm-memory-vram: 28 MiB\ndrm-memory-gtt: 2 MiB\n"
	f["/proc/300/comm"] = "viewer\n"
	f["/proc/400/fdinfo/9"] = "drm-driver: amdgpu\ndrm-pdev: 0000:03:00.0\ndrm-client-id: 77\ndrm-memory-vram: 1048576 KiB\n" // inherited client 77
	f["/proc/500/fdinfo/9"] = "drm-driver: amdgpu\ndrm-pdev: 0000:03:00.0\ndrm-client-id: 80\ndrm-memory-vram: 0 KiB\n"       // zero: dropped
	f["/proc/600/fdinfo/9"] = "drm-driver: nvidia\ndrm-pdev: 0000:01:00.0\ndrm-client-id: 1\ndrm-memory-vram: 1 MiB\n"
	f["/proc/abc/fdinfo/9"] = "drm-driver: amdgpu\n"
	return f
}

func td0(name string) string { b, _ := os.ReadFile("testdata/" + name); return string(b) }

func TestFdinfoProcesses(t *testing.T) {
	f := &fake{files: fdFiles()}
	as, _ := sysfsAdapters(f, 0)
	ps, err := fdinfoProcesses(f, as)
	if err != nil {
		t.Fatal(err)
	}
	got := map[int]Process{}
	for _, p := range ps {
		got[p.PID] = p
	}
	if len(ps) != 3 {
		t.Fatalf("want 3 processes (100,200,300), got %+v", ps)
	}
	if p := got[100]; p.Adapter != 0 || p.DedicatedMiB != 1024 || p.SharedMiB != 64 || p.Name != "game" || p.Kind != "graphics" {
		t.Errorf("100: %+v (dup fd and client must count once)", p)
	}
	if p := got[200]; p.Adapter != 1 || p.DedicatedMiB != 0 || p.SharedMiB != 1024 || p.Name != "compositor" {
		t.Errorf("200 (i915 matched by drm-pdev to the iGPU): %+v", p)
	}
	if p := got[300]; p.DedicatedMiB != 128 || p.SharedMiB != 2 {
		t.Errorf("300 (two clients sum): %+v", p)
	}
	if _, ok := got[400]; ok {
		t.Error("client 77 held by pid 400 must not double count")
	}
}

func TestFdinfoSingleAdapterNoPdev(t *testing.T) {
	f := &fake{files: map[string]string{"/proc/7/fdinfo/5": "drm-driver: amdgpu\ndrm-client-id: 1\ndrm-memory-vram: 10 MiB\n"}}
	ps, _ := fdinfoProcesses(f, []Adapter{{Index: 4}})
	if len(ps) != 1 || ps[0].Adapter != 4 || ps[0].DedicatedMiB != 10 {
		t.Errorf("%+v", ps)
	}
	ps, _ = fdinfoProcesses(f, []Adapter{{Index: 4}, {Index: 5}})
	if len(ps) != 0 {
		t.Errorf("ambiguous adapter must be skipped: %+v", ps)
	}
}

func TestSenseLinuxFull(t *testing.T) {
	f := &fake{goos: "linux", goarch: "amd64", files: fdFiles(),
		cmds: map[string]string{"nvidia-smi -q -x": td0("nvidia-smi-two-gpus.xml")}}
	pic := Sense(context.Background(), f, []int{4242}, 5150)
	if len(pic.Adapters) != 4 {
		t.Fatalf("adapters %+v", pic.Adapters)
	}
	for i, a := range pic.Adapters {
		if a.Index != i {
			t.Errorf("adapter %d has Index %d", i, a.Index)
		}
	}
	if pic.Adapters[0].Vendor != "nvidia" || pic.Adapters[1].Vendor != "nvidia" || pic.Adapters[2].Vendor != "amd" || pic.Adapters[3].Vendor != "intel" {
		t.Errorf("order %+v", pic.Adapters)
	}
	for _, k := range []string{"nvidia-smi", "sysfs", "fdinfo"} {
		if pic.Sources[k] != "ok" {
			t.Errorf("source %s = %q", k, pic.Sources[k])
		}
	}
	var ours, host int
	for _, p := range pic.Processes {
		if p.Ours {
			ours = p.PID
		}
		if p.Host {
			host = p.PID
		}
	}
	if ours != 4242 || host != 5150 {
		t.Errorf("flags ours=%d host=%d", ours, host)
	}
	if n := len(pic.OnAdapter(0)); n != 3 {
		t.Errorf("OnAdapter(0) = %d", n)
	}
	if n := len(pic.OnAdapter(2)); n != 2 { // 100 and 300
		t.Errorf("OnAdapter(2) = %d", n)
	}
}

func TestSenseLinuxDegrades(t *testing.T) {
	pic := Sense(context.Background(), &fake{goos: "linux", goarch: "amd64"}, nil, 0)
	if pic.Adapters == nil || pic.Processes == nil || pic.Sources == nil {
		t.Fatal("nil slices/maps must be empty for JSON")
	}
	if len(pic.Adapters) != 0 || pic.Sources["nvidia-smi"] == "ok" || pic.Sources["sysfs"] == "ok" || pic.Sources["fdinfo"] == "" {
		t.Errorf("%+v", pic)
	}
	// nvidia-smi only, sysfs missing.
	pic = Sense(context.Background(), &fake{goos: "linux", goarch: "amd64",
		cmds: map[string]string{"nvidia-smi -q -x": td0("nvidia-smi-two-gpus.xml")}}, nil, 0)
	if len(pic.Adapters) != 2 || pic.Sources["nvidia-smi"] != "ok" || pic.Sources["sysfs"] == "ok" {
		t.Errorf("%+v", pic)
	}
	// Garbage nvidia-smi output degrades.
	pic = Sense(context.Background(), &fake{goos: "linux", goarch: "amd64",
		cmds: map[string]string{"nvidia-smi -q -x": "NVIDIA-SMI has failed"}}, nil, 0)
	if len(pic.Adapters) != 0 || !strings.Contains(pic.Sources["nvidia-smi"], "xml") {
		t.Errorf("%+v", pic.Sources)
	}
}

func TestSenseDarwin(t *testing.T) {
	f := &fake{goos: "darwin", goarch: "arm64", cmds: map[string]string{
		"sysctl -n hw.memsize machdep.cpu.brand_string": "17179869184\nApple M2\n"}}
	pic := Sense(context.Background(), f, nil, 0)
	if len(pic.Adapters) != 1 {
		t.Fatalf("%+v", pic)
	}
	a := pic.Adapters[0]
	if a.Vendor != "apple" || !a.Integrated || !a.Unified || a.DedicatedMiB != 16384*70/100 || a.Name != "Apple M2" || a.Source != "sysctl" || len(pic.Processes) != 0 {
		t.Errorf("%+v", a)
	}
	pic = Sense(context.Background(), &fake{goos: "darwin", goarch: "arm64"}, nil, 0)
	if len(pic.Adapters) != 0 || pic.Sources["sysctl"] == "ok" {
		t.Errorf("%+v", pic)
	}
	pic = Sense(context.Background(), &fake{goos: "darwin", goarch: "amd64"}, nil, 0)
	if len(pic.Adapters) != 0 {
		t.Errorf("intel mac: %+v", pic)
	}
}

func TestFindAdapterAndOnAdapter(t *testing.T) {
	as := []Adapter{
		{Index: 0, VendorID: 0x10de, Name: "NVIDIA GeForce RTX 3050"},
		{Index: 1, VendorID: 0x10de, Name: "NVIDIA GeForce RTX 5080"},
		{Index: 2, VendorID: 0x8086, Name: "Intel(R) Iris(R) Xe Graphics"},
	}
	cases := []struct {
		vid  int
		name string
		want int
	}{
		{0x8086, "", 2},
		{0x8086, "whatever", 2},
		{0x10de, "rtx 5080", 1},
		{0x10de, "", 0},
		{0x10de, "no such card", 0},
		{0, "iris", 2},
		{0, "IRIS(R) XE GRAPHICS", 2},
		{0, "Intel(R) Iris(R) Xe Graphics (extra words)", 2}, // adapter name inside the hint
		{0x1002, "Radeon", -1},
		{0x1002, "geforce rtx 3050", 0}, // wrong vendor id falls to name
		{0, "", -1},
	}
	for _, c := range cases {
		if got := FindAdapter(as, c.vid, c.name); got != c.want {
			t.Errorf("FindAdapter(%#x,%q) = %d want %d", c.vid, c.name, got, c.want)
		}
	}
	// Index, not slice position.
	if got := FindAdapter([]Adapter{{Index: 7, VendorID: 1, Name: "x"}}, 1, ""); got != 7 {
		t.Errorf("got %d", got)
	}
	p := Picture{Processes: []Process{{PID: 1, Adapter: 0}, {PID: 2, Adapter: 1}, {PID: 3, Adapter: 0}}}
	if n := len(p.OnAdapter(0)); n != 2 {
		t.Error(n)
	}
	if p.OnAdapter(9) != nil {
		t.Error("want nil")
	}
}
