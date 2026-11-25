package probe

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type fake struct {
	goos, goarch string
	cpus         int
	cmds         map[string]string // "name arg arg" -> stdout; absent = command missing
	files        map[string]string
	disk         int
}

func (f *fake) Platform() (string, string) { return f.goos, f.goarch }
func (f *fake) NumCPU() int                { return f.cpus }
func (f *fake) DiskFreeMiB(string) (int, error) {
	return f.disk, nil
}
func (f *fake) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	key := name + " " + strings.Join(args, " ")
	if name == "powershell" {
		key = "powershell"
	}
	if v, ok := f.cmds[strings.TrimSpace(key)]; ok {
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
	// only the drm pattern is used
	var out []string
	for p := range f.files {
		if strings.HasPrefix(p, "/sys/class/drm/card") && strings.HasSuffix(p, "/device/vendor") {
			out = append(out, strings.TrimSuffix(p, "/vendor"))
		}
	}
	return out, nil
}

const smiQuery = "nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits"

// Captured on the RTX 5080 box.
const meminfo = `MemTotal:       32718700 kB
MemFree:         3000000 kB
MemAvailable:   24117244 kB
Buffers:           41032 kB
`
const cpuinfo = `processor	: 0
vendor_id	: AuthenticAMD
model name	: AMD Ryzen 7 9800X3D 8-Core Processor
flags		: fpu vme de pse avx avx2 fma
processor	: 1
model name	: AMD Ryzen 7 9800X3D 8-Core Processor
`

func TestParseNvidiaSMI(t *testing.T) {
	gs := ParseNvidiaSMI("0, NVIDIA GeForce RTX 5080, 16303, 602\n1, NVIDIA, Weird, Name, 8192, [N/A]\n\n")
	if len(gs) != 2 {
		t.Fatalf("got %d gpus", len(gs))
	}
	if gs[0].Name != "NVIDIA GeForce RTX 5080" || gs[0].VRAMTotalMiB != 16303 || gs[0].VRAMUsedMiB != 602 || gs[0].Vendor != "nvidia" {
		t.Errorf("gpu0 %+v", gs[0])
	}
	if gs[1].Name != "NVIDIA, Weird, Name" || gs[1].VRAMTotalMiB != 8192 || gs[1].VRAMUsedMiB != 0 || gs[1].Index != 1 {
		t.Errorf("gpu1 %+v", gs[1])
	}
}

func TestLinuxNvidia(t *testing.T) {
	t.Setenv("HERMES_ENGINE_HOME", t.TempDir())
	f := &fake{goos: "linux", goarch: "amd64", cpus: 16, disk: 500000,
		cmds:  map[string]string{smiQuery: "0, NVIDIA GeForce RTX 5080, 16303, 602\n"},
		files: map[string]string{"/proc/meminfo": meminfo, "/proc/cpuinfo": cpuinfo}}
	hw, err := ProbeWith(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	if hw.RAMTotalMiB != 31951 || hw.RAMAvailMiB != 23551 || !hw.AVX2 || hw.CPUName != "AMD Ryzen 7 9800X3D 8-Core Processor" || hw.Cores != 16 || hw.DiskFreeMiB != 500000 {
		t.Errorf("%+v", hw)
	}
	if len(hw.GPUs) != 1 || hw.GPUs[0].VRAMTotalMiB != 16303 {
		t.Errorf("gpus %+v", hw.GPUs)
	}
}

func TestLinuxSysfsAMDAndIntegratedSkipped(t *testing.T) {
	t.Setenv("HERMES_ENGINE_HOME", t.TempDir())
	f := &fake{goos: "linux", goarch: "amd64", cpus: 8,
		files: map[string]string{
			"/proc/meminfo": meminfo, "/proc/cpuinfo": cpuinfo,
			"/sys/class/drm/card0/device/vendor":              "0x1002\n",
			"/sys/class/drm/card0/device/device":              "0x744c\n",
			"/sys/class/drm/card0/device/mem_info_vram_total": "25753026560\n", // 24560 MiB
			"/sys/class/drm/card0/device/mem_info_vram_used":  "1073741824\n",
			"/sys/class/drm/card1/device/vendor":              "0x8086\n", // iGPU, no mem_info
			"/sys/class/drm/card2/device/vendor":              "0x10de\n", // handled by nvidia-smi
		}}
	hw, err := ProbeWith(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	if len(hw.GPUs) != 1 {
		t.Fatalf("gpus %+v", hw.GPUs)
	}
	g := hw.GPUs[0]
	if g.Vendor != "amd" || g.VRAMTotalMiB != 24560 || g.VRAMUsedMiB != 1024 || g.Source != "sysfs" || g.Name != "Amd GPU 0x744c" {
		t.Errorf("%+v", g)
	}
	// Others re-reads live use from sysfs
	f.files["/sys/class/drm/card0/device/mem_info_vram_used"] = "2147483648\n"
	o, err := OthersWith(context.Background(), f, g, nil)
	if err != nil || o != 2048 {
		t.Errorf("others %d %v", o, err)
	}
}

func TestNoGPUNothingReadable(t *testing.T) {
	t.Setenv("HERMES_ENGINE_HOME", t.TempDir())
	f := &fake{goos: "linux", goarch: "amd64", cpus: 4}
	if _, err := ProbeWith(context.Background(), f); err == nil {
		t.Error("expected error when nothing is readable")
	}
}

const vmstat = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                             500000.
Pages inactive:                           200000.
Pages speculative:                         10000.
Pages purgeable:                           20000.
`

func TestDarwinAppleSilicon(t *testing.T) {
	t.Setenv("HERMES_ENGINE_HOME", t.TempDir())
	f := &fake{goos: "darwin", goarch: "arm64", cpus: 10, disk: 100000,
		cmds: map[string]string{
			"sysctl -n hw.memsize machdep.cpu.brand_string hw.logicalcpu": "17179869184\nApple M2 Pro\n10\n",
			"vm_stat": vmstat,
		}}
	hw, err := ProbeWith(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	if hw.RAMTotalMiB != 16384 || hw.CPUName != "Apple M2 Pro" || hw.Cores != 10 {
		t.Errorf("%+v", hw)
	}
	if hw.RAMAvailMiB != (100000+200000+10000)*16384/(1<<20) {
		t.Errorf("avail %d", hw.RAMAvailMiB)
	}
	if len(hw.GPUs) != 1 || !hw.GPUs[0].UnifiedMemory || hw.GPUs[0].Vendor != "apple" || hw.GPUs[0].VRAMTotalMiB != 11468 {
		t.Errorf("gpus %+v", hw.GPUs)
	}
}

func TestDarwinIntelNoGPU(t *testing.T) {
	t.Setenv("HERMES_ENGINE_HOME", t.TempDir())
	f := &fake{goos: "darwin", goarch: "amd64", cpus: 8,
		cmds: map[string]string{
			"sysctl -n hw.memsize machdep.cpu.brand_string hw.logicalcpu": "17179869184\nIntel(R) Core(TM) i9\n16\n",
			"sysctl -n machdep.cpu.leaf7_features":                        "SMEP ERMS AVX2 BMI2\n",
		}}
	hw, err := ProbeWith(context.Background(), f)
	if err != nil || len(hw.GPUs) != 0 || !hw.AVX2 || hw.Cores != 16 {
		t.Errorf("%+v %v", hw, err)
	}
}

const winJSON = `{"cpu":"AMD Ryzen 5 5600X 6-Core Processor","cores":12,"ram_total":34281795584,"ram_free_kb":20971520,"gpus":[{"name":"AMD Radeon RX 7800 XT","adapter_ram":4293918720,"qw":17163091968},{"name":"Microsoft Basic Display Adapter","adapter_ram":0,"qw":0}]}`
const winJSONSingleObject = `{"cpu":"Intel(R) Core(TM) i5","cores":8,"ram_total":17179869184,"ram_free_kb":8388608,"gpus":{"name":"Intel(R) UHD Graphics 630","adapter_ram":1073741824,"qw":0}}`

func TestWindowsAMD(t *testing.T) {
	t.Setenv("HERMES_ENGINE_HOME", t.TempDir())
	f := &fake{goos: "windows", goarch: "amd64", cpus: 12, cmds: map[string]string{"powershell": "\xef\xbb\xbf" + winJSON}}
	hw, err := ProbeWith(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	if hw.RAMTotalMiB != 32693 || hw.RAMAvailMiB != 20480 || hw.CPUName != "AMD Ryzen 5 5600X 6-Core Processor" || !hw.AVX2 {
		t.Errorf("%+v", hw)
	}
	if len(hw.GPUs) != 1 || hw.GPUs[0].Vendor != "amd" || hw.GPUs[0].VRAMTotalMiB != 16368 || hw.GPUs[0].Source != "wmi" {
		t.Errorf("gpus %+v", hw.GPUs)
	}
}

func TestWindowsSingleGPUObjectAndAdapterRAMFallback(t *testing.T) {
	_, gs, err := ParseWindowsInfo([]byte(winJSONSingleObject))
	if err != nil || len(gs) != 1 {
		t.Fatal(err, gs)
	}
	g := gpusFromWindows(gs, false)
	if len(g) != 1 || g[0].Vendor != "intel" || g[0].VRAMTotalMiB != 1024 {
		t.Errorf("%+v", g)
	}
}

func TestWindowsNvidiaSMIWinsOverWMI(t *testing.T) {
	ws := []WindowsGPU{{Name: "NVIDIA GeForce RTX 4070", QW: 12 << 30}}
	if g := gpusFromWindows(ws, true); len(g) != 0 {
		t.Errorf("duplicate nvidia entry: %+v", g)
	}
	if g := gpusFromWindows(ws, false); len(g) != 1 || g[0].Vendor != "nvidia" || g[0].VRAMTotalMiB != 12288 {
		t.Errorf("%+v", g)
	}
}

func TestOthersNvidia(t *testing.T) {
	// Captured on the 5080 box: card holds 602 MiB, of which pid 3865 (a stand-in for ours) holds 26.
	f := &fake{goos: "linux", goarch: "amd64", cmds: map[string]string{
		"nvidia-smi -i 0 --query-gpu=uuid,memory.used --format=csv,noheader,nounits":             "GPU-3eb4, 602\n",
		"nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits": "GPU-3eb4, 3865, 26\nGPU-3eb4, 100, 300\nGPU-other, 3865, 999\n",
	}}
	g := GPU{Index: 0, Vendor: "nvidia"}
	o, err := OthersWith(context.Background(), f, g, []int{3865})
	if err != nil || o != 576 {
		t.Errorf("others %d %v", o, err)
	}
	o, _ = OthersWith(context.Background(), f, g, nil)
	if o != 602 {
		t.Errorf("no ours: %d", o)
	}
	// compute-apps unavailable (WDDM): all counts as others
	delete(f.cmds, "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits")
	o, err = OthersWith(context.Background(), f, g, []int{3865})
	if err != nil || o != 602 {
		t.Errorf("fallback %d %v", o, err)
	}
}

func TestParseComputeApps(t *testing.T) {
	if got := ParseComputeApps("No running processes found\n"); len(got) != 0 {
		t.Error(got)
	}
	got := ParseComputeApps("GPU-x, 12, [N/A]\n")
	if len(got) != 1 || got[0].PID != 12 || got[0].UsedMiB != 0 {
		t.Error(got)
	}
}

func TestMeminfoFallback(t *testing.T) {
	tot, av := ParseMeminfo("MemTotal: 2048 kB\nMemFree: 1024 kB\nCached: 512 kB\n")
	if tot != 2 || av != 1 {
		t.Errorf("%d %d", tot, av)
	}
}
