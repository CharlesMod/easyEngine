// Package probe reads the machine's CPU, RAM, disk and GPUs. Every source (commands, files,
// disk) sits behind the Sources interface so tests feed real captured output through fakes.
package probe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"hermesengine/dirs"
)

type GPU struct {
	Index         int    `json:"index"`
	Vendor        string `json:"vendor"` // nvidia | amd | intel | apple | other
	Name          string `json:"name"`
	VRAMTotalMiB  int    `json:"vram_total_mib"`
	VRAMUsedMiB   int    `json:"vram_used_mib"` // all processes, ours included
	UnifiedMemory bool   `json:"unified_memory"`
	Source        string `json:"source"` // nvidia-smi | sysfs | sysctl | llama-list-devices | wmi

	sysfsPath string // unexported: lets Others re-read live use
}

type Hardware struct {
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	CPUName     string `json:"cpu_name"`
	Cores       int    `json:"cores"`
	AVX2        bool   `json:"avx2"`
	RAMTotalMiB int    `json:"ram_total_mib"`
	RAMAvailMiB int    `json:"ram_avail_mib"`
	DiskFreeMiB int    `json:"disk_free_mib"` // on the data dir's filesystem
	GPUs        []GPU  `json:"gpus"`
}

// Sources is everything the probe reads from the machine.
type Sources interface {
	Platform() (goos, goarch string)
	// Run executes a command and returns stdout. An error means the command is missing or failed.
	Run(ctx context.Context, name string, args ...string) ([]byte, error)
	ReadFile(path string) ([]byte, error)
	Glob(pattern string) ([]string, error)
	DiskFreeMiB(path string) (int, error)
	NumCPU() int
}

const mib = 1 << 20

// Probe inspects the real machine.
func Probe(ctx context.Context) (Hardware, error) { return ProbeWith(ctx, OSSources{}) }

// ProbeWith inspects whatever the given sources describe. It errors only when nothing about
// the machine could be read; partial results are returned with a nil error.
func ProbeWith(ctx context.Context, s Sources) (Hardware, error) {
	goos, goarch := s.Platform()
	hw := Hardware{OS: goos, Arch: goarch, Cores: s.NumCPU()}
	var errs []error
	switch goos {
	case "linux":
		if err := probeLinuxHost(s, &hw); err != nil {
			errs = append(errs, err)
		}
	case "darwin":
		if err := probeDarwinHost(ctx, s, &hw); err != nil {
			errs = append(errs, err)
		}
	case "windows":
		if err := probeWindowsHost(ctx, s, &hw); err != nil {
			errs = append(errs, err)
		}
	}
	if d, err := s.DiskFreeMiB(dirs.Data()); err == nil {
		hw.DiskFreeMiB = d
	}

	if out, err := s.Run(ctx, "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"); err == nil {
		hw.GPUs = append(hw.GPUs, ParseNvidiaSMI(string(out))...)
	}
	switch goos {
	case "linux":
		hw.GPUs = append(hw.GPUs, probeSysfsGPUs(s, len(hw.GPUs))...)
	case "darwin":
		if goarch == "arm64" && hw.RAMTotalMiB > 0 {
			hw.GPUs = append(hw.GPUs, GPU{Index: 0, Vendor: "apple", Name: hw.CPUName,
				VRAMTotalMiB: hw.RAMTotalMiB * 70 / 100, UnifiedMemory: true, Source: "sysctl"})
		}
	case "windows":
		hw.GPUs = append(hw.GPUs, windowsGPUs(ctx, s, hw.GPUs)...)
	}
	for i := range hw.GPUs {
		if hw.GPUs[i].Source != "nvidia-smi" { // nvidia-smi's own index is what -i takes
			hw.GPUs[i].Index = i
		}
	}
	if hw.RAMTotalMiB == 0 && hw.CPUName == "" && len(hw.GPUs) == 0 {
		return hw, fmt.Errorf("probe: nothing readable: %w", errors.Join(errs...))
	}
	return hw, nil
}

// ---- nvidia-smi ----

// ParseNvidiaSMI parses `--query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits`.
func ParseNvidiaSMI(out string) []GPU {
	var gs []GPU
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) < 4 {
			continue
		}
		idx, err := strconv.Atoi(strings.TrimSpace(f[0]))
		if err != nil {
			continue
		}
		name := strings.TrimSpace(strings.Join(f[1:len(f)-2], ","))
		gs = append(gs, GPU{
			Index: idx, Vendor: "nvidia", Name: name,
			VRAMTotalMiB: atoiMiB(f[len(f)-2]), VRAMUsedMiB: atoiMiB(f[len(f)-1]),
			Source: "nvidia-smi",
		})
	}
	return gs
}

func atoiMiB(s string) int {
	n, err := strconv.Atoi(strings.TrimSpace(s)) // "[N/A]" and friends -> 0
	if err != nil || n < 0 {
		return 0
	}
	return n
}

// Others returns the VRAM, in MiB, used on gpu by processes that are not ours: the card's
// live used memory minus what our own pids hold (nvidia-smi --query-compute-apps). Where the
// per-process split is unavailable (non-NVIDIA, Windows WDDM) it returns the card's total use,
// and the caller should subtract its own calibrated figure.
func Others(ctx context.Context, gpu GPU, ourPIDs []int) (int, error) {
	return OthersWith(ctx, OSSources{}, gpu, ourPIDs)
}

func OthersWith(ctx context.Context, s Sources, gpu GPU, ourPIDs []int) (int, error) {
	switch gpu.Vendor {
	case "nvidia":
		idx := strconv.Itoa(gpu.Index)
		out, err := s.Run(ctx, "nvidia-smi", "-i", idx, "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits")
		if err != nil {
			return 0, fmt.Errorf("nvidia-smi: %w", err)
		}
		uuid, used, ok := parseUUIDUsed(string(out))
		if !ok {
			return 0, fmt.Errorf("nvidia-smi: cannot parse %q", strings.TrimSpace(string(out)))
		}
		out, err = s.Run(ctx, "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")
		if err != nil {
			return used, nil // no per-process data: everything counts as others
		}
		ours := 0
		want := map[int]bool{}
		for _, p := range ourPIDs {
			want[p] = true
		}
		for _, a := range ParseComputeApps(string(out)) {
			if a.UUID == uuid && want[a.PID] {
				ours += a.UsedMiB
			}
		}
		if o := used - ours; o > 0 {
			return o, nil
		}
		return 0, nil
	default:
		if gpu.Source == "sysfs" && gpu.sysfsPath != "" {
			if b, err := s.ReadFile(gpu.sysfsPath + "/mem_info_vram_used"); err == nil {
				if n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
					return int(n / mib), nil
				}
			}
		}
		return gpu.VRAMUsedMiB, nil
	}
}

func parseUUIDUsed(out string) (uuid string, used int, ok bool) {
	line := strings.TrimSpace(strings.SplitN(strings.TrimSpace(out), "\n", 2)[0])
	f := strings.Split(line, ",")
	if len(f) != 2 {
		return "", 0, false
	}
	return strings.TrimSpace(f[0]), atoiMiB(f[1]), true
}

type ComputeApp struct {
	UUID    string
	PID     int
	UsedMiB int
}

// ParseComputeApps parses `--query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits`.
func ParseComputeApps(out string) []ComputeApp {
	var as []ComputeApp
	for _, line := range strings.Split(out, "\n") {
		f := strings.Split(strings.TrimSpace(line), ",")
		if len(f) != 3 {
			continue // also covers "No running processes found"
		}
		pid, err := strconv.Atoi(strings.TrimSpace(f[1]))
		if err != nil {
			continue
		}
		as = append(as, ComputeApp{UUID: strings.TrimSpace(f[0]), PID: pid, UsedMiB: atoiMiB(f[2])})
	}
	return as
}

// ---- Linux ----

func probeLinuxHost(s Sources, hw *Hardware) error {
	mem, err := s.ReadFile("/proc/meminfo")
	if err != nil {
		return err
	}
	hw.RAMTotalMiB, hw.RAMAvailMiB = ParseMeminfo(string(mem))
	if cpu, err := s.ReadFile("/proc/cpuinfo"); err == nil {
		hw.CPUName, hw.AVX2 = ParseCPUInfo(string(cpu))
	}
	return nil
}

// ParseMeminfo returns MemTotal and MemAvailable (MemFree+Cached as a fallback) in MiB.
func ParseMeminfo(s string) (total, avail int) {
	kv := map[string]int{}
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		f := strings.Fields(v)
		if len(f) == 0 {
			continue
		}
		n, err := strconv.Atoi(f[0])
		if err != nil {
			continue
		}
		kv[k] = n
	}
	total = kv["MemTotal"] / 1024
	if a, ok := kv["MemAvailable"]; ok {
		avail = a / 1024
	} else {
		avail = (kv["MemFree"] + kv["Cached"]) / 1024
	}
	return
}

// ParseCPUInfo returns the first model name and whether the flags include avx2.
func ParseCPUInfo(s string) (name string, avx2 bool) {
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch {
		case name == "" && (k == "model name" || k == "Model Name" || k == "Hardware" || k == "cpu model"):
			name = v
		case k == "flags":
			for _, f := range strings.Fields(v) {
				if f == "avx2" {
					avx2 = true
				}
			}
		}
	}
	return
}

var cardRe = regexp.MustCompile(`/card(\d+)/device$`)

func probeSysfsGPUs(s Sources, startIdx int) []GPU {
	devs, _ := s.Glob("/sys/class/drm/card*/device")
	sort.Strings(devs)
	var gs []GPU
	for _, d := range devs {
		if !cardRe.MatchString(d) {
			continue
		}
		vb, err := s.ReadFile(d + "/vendor")
		if err != nil {
			continue
		}
		vendor := sysfsVendor(strings.TrimSpace(string(vb)))
		if vendor == "nvidia" || vendor == "" {
			continue // NVIDIA is covered by nvidia-smi
		}
		tb, err := s.ReadFile(d + "/mem_info_vram_total")
		if err != nil {
			continue // integrated parts without dedicated VRAM
		}
		total, err := strconv.ParseInt(strings.TrimSpace(string(tb)), 10, 64)
		if err != nil || total <= 0 {
			continue
		}
		var used int64
		if ub, err := s.ReadFile(d + "/mem_info_vram_used"); err == nil {
			used, _ = strconv.ParseInt(strings.TrimSpace(string(ub)), 10, 64)
		}
		name := ""
		if nb, err := s.ReadFile(d + "/product_name"); err == nil {
			name = strings.TrimSpace(string(nb))
		}
		if name == "" {
			dev := ""
			if db, err := s.ReadFile(d + "/device"); err == nil {
				dev = strings.TrimSpace(string(db))
			}
			name = strings.ToUpper(vendor[:1]) + vendor[1:] + " GPU " + dev
			name = strings.TrimSpace(name)
		}
		gs = append(gs, GPU{Index: startIdx + len(gs), Vendor: vendor, Name: name,
			VRAMTotalMiB: int(total / mib), VRAMUsedMiB: int(used / mib), Source: "sysfs", sysfsPath: d})
	}
	return gs
}

func sysfsVendor(id string) string {
	switch strings.ToLower(id) {
	case "0x1002":
		return "amd"
	case "0x8086":
		return "intel"
	case "0x10de":
		return "nvidia"
	}
	return "other"
}

// ---- macOS ----

func probeDarwinHost(ctx context.Context, s Sources, hw *Hardware) error {
	out, err := s.Run(ctx, "sysctl", "-n", "hw.memsize", "machdep.cpu.brand_string", "hw.logicalcpu")
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	if len(lines) >= 1 {
		if n, err := strconv.ParseInt(strings.TrimSpace(lines[0]), 10, 64); err == nil {
			hw.RAMTotalMiB = int(n / mib)
		}
	}
	if len(lines) >= 2 {
		hw.CPUName = strings.TrimSpace(lines[1])
	}
	if len(lines) >= 3 {
		if n, err := strconv.Atoi(strings.TrimSpace(lines[2])); err == nil && n > 0 {
			hw.Cores = n
		}
	}
	if vm, err := s.Run(ctx, "vm_stat"); err == nil {
		hw.RAMAvailMiB = ParseVMStat(string(vm))
	}
	if hw.Arch == "amd64" {
		if f, err := s.Run(ctx, "sysctl", "-n", "machdep.cpu.leaf7_features"); err == nil {
			hw.AVX2 = strings.Contains(strings.ToUpper(string(f)), "AVX2")
		}
	}
	return nil
}

var vmPageSize = regexp.MustCompile(`page size of (\d+) bytes`)

// ParseVMStat returns free+inactive+speculative pages in MiB.
func ParseVMStat(s string) int {
	page := 4096
	if m := vmPageSize.FindStringSubmatch(s); m != nil {
		page, _ = strconv.Atoi(m[1])
	}
	pages := 0
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(k) {
		case "Pages free", "Pages inactive", "Pages speculative":
			n, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimSpace(v), "."))
			pages += n
		}
	}
	return int(int64(pages) * int64(page) / mib)
}

// ---- Windows ----

// windowsScript prints one compact JSON object describing the machine (see ParseWindowsInfo).
const windowsScript = `$ErrorActionPreference='SilentlyContinue';` +
	`$cs=Get-CimInstance Win32_ComputerSystem;$os=Get-CimInstance Win32_OperatingSystem;` +
	`$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1;` +
	`$gp=@(Get-CimInstance Win32_VideoController|ForEach-Object{$n=$_.Name;$q=[int64]0;` +
	`Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11cd-be10-08002be10318}\0*'|` +
	`Where-Object{$_.DriverDesc -eq $n}|ForEach-Object{$v=[int64]$_.'HardwareInformation.qwMemorySize';if($v -gt $q){$q=$v}};` +
	`[pscustomobject]@{name=$n;adapter_ram=[int64]$_.AdapterRAM;qw=$q}});` +
	`[pscustomobject]@{cpu=$cpu.Name;cores=[int]$cpu.NumberOfLogicalProcessors;ram_total=[int64]$cs.TotalPhysicalMemory;` +
	`ram_free_kb=[int64]$os.FreePhysicalMemory;gpus=$gp}|ConvertTo-Json -Depth 4 -Compress`

type WindowsInfo struct {
	CPU       string          `json:"cpu"`
	Cores     int             `json:"cores"`
	RAMTotal  int64           `json:"ram_total"`
	RAMFreeKB int64           `json:"ram_free_kb"`
	GPUs      json.RawMessage `json:"gpus"` // array, or a single object in older PowerShell
}

type WindowsGPU struct {
	Name       string `json:"name"`
	AdapterRAM int64  `json:"adapter_ram"` // 32-bit capped by WMI (max ~4 GiB)
	QW         int64  `json:"qw"`          // HardwareInformation.qwMemorySize (true size)
}

func ParseWindowsInfo(b []byte) (WindowsInfo, []WindowsGPU, error) {
	var wi WindowsInfo
	if err := json.Unmarshal([]byte(strings.TrimPrefix(string(b), "\ufeff")), &wi); err != nil {
		return wi, nil, err
	}
	var gs []WindowsGPU
	raw := strings.TrimSpace(string(wi.GPUs))
	switch {
	case strings.HasPrefix(raw, "["):
		if err := json.Unmarshal(wi.GPUs, &gs); err != nil {
			return wi, nil, err
		}
	case strings.HasPrefix(raw, "{"):
		var g WindowsGPU
		if err := json.Unmarshal(wi.GPUs, &g); err != nil {
			return wi, nil, err
		}
		gs = []WindowsGPU{g}
	}
	return wi, gs, nil
}

func probeWindowsHost(ctx context.Context, s Sources, hw *Hardware) error {
	out, err := s.Run(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", windowsScript)
	if err != nil {
		return err
	}
	wi, _, err := ParseWindowsInfo(out)
	if err != nil {
		return err
	}
	hw.CPUName = strings.TrimSpace(wi.CPU)
	if wi.Cores > 0 {
		hw.Cores = wi.Cores
	}
	hw.RAMTotalMiB = int(wi.RAMTotal / mib)
	hw.RAMAvailMiB = int(wi.RAMFreeKB / 1024)
	hw.AVX2 = hw.Arch == "amd64" // every x86-64 gaming CPU of the last decade; llama.cpp also self-detects
	return nil
}

func windowsGPUs(ctx context.Context, s Sources, have []GPU) []GPU {
	out, err := s.Run(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", windowsScript)
	if err != nil {
		return nil
	}
	_, ws, err := ParseWindowsInfo(out)
	if err != nil {
		return nil
	}
	return gpusFromWindows(ws, len(have) > 0)
}

func gpusFromWindows(ws []WindowsGPU, haveNvidia bool) []GPU {
	var gs []GPU
	for _, w := range ws {
		l := strings.ToLower(w.Name)
		var vendor string
		switch {
		case strings.Contains(l, "nvidia"):
			if haveNvidia {
				continue // nvidia-smi already described it, with live use
			}
			vendor = "nvidia"
		case strings.Contains(l, "amd") || strings.Contains(l, "radeon"):
			vendor = "amd"
		case strings.Contains(l, "intel"):
			vendor = "intel"
		case strings.Contains(l, "microsoft basic") || strings.Contains(l, "remote display") || l == "":
			continue
		default:
			vendor = "other"
		}
		b := w.QW
		if b <= 0 {
			b = w.AdapterRAM
		}
		gs = append(gs, GPU{Vendor: vendor, Name: strings.TrimSpace(w.Name), VRAMTotalMiB: int(b / mib), Source: "wmi"})
	}
	return gs
}
