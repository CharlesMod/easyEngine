// STUB: replaced by eng/fetch
package probe

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

type GPU struct {
	Index         int    `json:"index"`
	Vendor        string `json:"vendor"`
	Name          string `json:"name"`
	VRAMTotalMiB  int    `json:"vram_total_mib"`
	VRAMUsedMiB   int    `json:"vram_used_mib"`
	UnifiedMemory bool   `json:"unified_memory"`
	Source        string `json:"source"`
}

type Hardware struct {
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	CPUName     string `json:"cpu_name"`
	Cores       int    `json:"cores"`
	AVX2        bool   `json:"avx2"`
	RAMTotalMiB int    `json:"ram_total_mib"`
	RAMAvailMiB int    `json:"ram_avail_mib"`
	DiskFreeMiB int    `json:"disk_free_mib"`
	GPUs        []GPU  `json:"gpus"`
}

type Sources interface {
	Run(ctx context.Context, name string, args ...string) ([]byte, error)
}

type execSources struct{}

func (execSources) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).Output()
}

func Probe(ctx context.Context) (Hardware, error) { return ProbeWith(ctx, execSources{}) }

func ProbeWith(ctx context.Context, s Sources) (Hardware, error) {
	h := Hardware{OS: runtime.GOOS, Arch: runtime.GOARCH, Cores: runtime.NumCPU()}
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			f := strings.Fields(l)
			if len(f) >= 2 {
				kb, _ := strconv.Atoi(f[1])
				switch f[0] {
				case "MemTotal:":
					h.RAMTotalMiB = kb / 1024
				case "MemAvailable:":
					h.RAMAvailMiB = kb / 1024
				}
			}
		}
	}
	out, err := s.Run(ctx, "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits")
	if err == nil {
		for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			f := strings.Split(l, ",")
			if len(f) < 4 {
				continue
			}
			i, _ := strconv.Atoi(strings.TrimSpace(f[0]))
			t, _ := strconv.Atoi(strings.TrimSpace(f[2]))
			u, _ := strconv.Atoi(strings.TrimSpace(f[3]))
			h.GPUs = append(h.GPUs, GPU{Index: i, Vendor: "nvidia", Name: strings.TrimSpace(f[1]), VRAMTotalMiB: t, VRAMUsedMiB: u, Source: "nvidia-smi"})
		}
	}
	return h, nil
}

// Others is the VRAM used by processes that are not ours.
func Others(ctx context.Context, gpu GPU, ourPIDs []int) (int, error) {
	ours := 0
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return gpu.VRAMUsedMiB, nil
	}
	for _, l := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Split(l, ",")
		if len(f) < 2 {
			continue
		}
		p, _ := strconv.Atoi(strings.TrimSpace(f[0]))
		m, _ := strconv.Atoi(strings.TrimSpace(f[1]))
		for _, o := range ourPIDs {
			if o == p {
				ours += m
			}
		}
	}
	o := gpu.VRAMUsedMiB - ours
	if o < 0 {
		o = 0
	}
	return o, nil
}
