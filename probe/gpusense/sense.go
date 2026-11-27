package gpusense

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/probe"
)

const mib = 1 << 20

// senseBudget bounds one whole sense; each command also has its own timeout in probe.OSSources.
const senseBudget = 40 * time.Second

// Sense reads adapters and per-process use. ours/host pids only set the flags.
//
// Adapter indices: NVIDIA adapters come first and take nvidia-smi's own order (so Index equals
// probe.GPU.Index, the number `nvidia-smi -i` takes); every other adapter follows in the order
// listed (sysfs card number on Linux, DXGI enumeration on Windows).
func Sense(ctx context.Context, s probe.Sources, ours []int, host int) Picture {
	ctx, cancel := context.WithTimeout(ctx, senseBudget)
	defer cancel()
	pic := Picture{Adapters: []Adapter{}, Processes: []Process{}, Sources: map[string]string{}}
	goos, goarch := s.Platform()
	switch goos {
	case "linux":
		senseLinux(ctx, s, &pic)
	case "windows":
		senseWindows(ctx, s, &pic)
	case "darwin":
		senseDarwin(ctx, s, goarch, &pic)
	default:
		pic.Sources["platform"] = "unsupported: " + goos
	}
	markFlags(&pic, ours, host)
	return pic
}

func markFlags(p *Picture, ours []int, host int) {
	o := map[int]bool{}
	for _, pid := range ours {
		o[pid] = true
	}
	for i := range p.Processes {
		pr := &p.Processes[i]
		pr.Ours = o[pr.PID]
		pr.Host = host > 0 && pr.PID == host
	}
}

func srcStatus(err error) string {
	if err == nil {
		return "ok"
	}
	return err.Error()
}

func senseDarwin(ctx context.Context, s probe.Sources, goarch string, p *Picture) {
	if goarch != "arm64" {
		p.Sources["sysctl"] = "not Apple silicon: no adapter read"
		return
	}
	out, err := s.Run(ctx, "sysctl", "-n", "hw.memsize", "machdep.cpu.brand_string")
	if err != nil {
		p.Sources["sysctl"] = err.Error()
		return
	}
	lines := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	n, perr := strconv.ParseInt(strings.TrimSpace(lines[0]), 10, 64)
	if perr != nil || n <= 0 {
		p.Sources["sysctl"] = fmt.Sprintf("cannot parse hw.memsize %q", strings.TrimSpace(lines[0]))
		return
	}
	name := "Apple GPU"
	if len(lines) > 1 && strings.TrimSpace(lines[1]) != "" {
		name = strings.TrimSpace(lines[1])
	}
	ram := int(n / mib)
	p.Adapters = append(p.Adapters, Adapter{Index: 0, Vendor: "apple", Name: name,
		DedicatedMiB: ram * 70 / 100, UsedMiB: -1, Integrated: true, Unified: true, Source: "sysctl"})
	p.Sources["sysctl"] = "ok"
}

// OnAdapter returns the processes holding memory on the adapter with the given Adapter.Index.
func (p Picture) OnAdapter(idx int) []Process {
	var out []Process
	for _, pr := range p.Processes {
		if pr.Adapter == idx {
			out = append(out, pr)
		}
	}
	return out
}

// FindAdapter resolves a (vendor_id, name) hint, as a profile or POST gives it, to an
// Adapter.Index; -1 when nothing matches. A vendor_id match (when vendorID != 0) comes first: a
// single match wins; among several the one whose name matches wins, else the first. With no
// vendor match, the name is matched by case-insensitive substring in either direction.
func FindAdapter(as []Adapter, vendorID int, name string) int {
	name = strings.ToLower(strings.TrimSpace(name))
	nameMatch := func(a Adapter) bool {
		n := strings.ToLower(a.Name)
		return name != "" && n != "" && (strings.Contains(n, name) || strings.Contains(name, n))
	}
	if vendorID != 0 {
		var cands []Adapter
		for _, a := range as {
			if a.VendorID == vendorID {
				cands = append(cands, a)
			}
		}
		for _, a := range cands {
			if len(cands) > 1 && nameMatch(a) {
				return a.Index
			}
		}
		if len(cands) > 0 {
			return cands[0].Index
		}
	}
	for _, a := range as {
		if nameMatch(a) {
			return a.Index
		}
	}
	return -1
}

// ---- shared helpers ----

// parseSizeBytes parses "123 MiB", "4096 KiB", "1.5 GiB", "N/A", "[N/A]" or a bare number
// (bytes). ok is false for anything unreadable.
func parseSizeBytes(s string) (int64, bool) {
	f := strings.Fields(strings.TrimSpace(s))
	if len(f) == 0 {
		return 0, false
	}
	v, err := strconv.ParseFloat(f[0], 64)
	if err != nil || v < 0 {
		return 0, false
	}
	mult := 1.0
	if len(f) > 1 {
		switch strings.ToLower(f[1]) {
		case "b", "bytes":
		case "kib", "kb", "k":
			mult = 1 << 10
		case "mib", "mb", "m":
			mult = 1 << 20
		case "gib", "gb", "g":
			mult = 1 << 30
		default:
			return 0, false
		}
	}
	return int64(v * mult), true
}

func baseName(p string) string {
	p = strings.TrimSpace(p)
	if i := strings.LastIndexAny(p, `/\`); i >= 0 {
		p = p[i+1:]
	}
	return p
}

// normPCI lower-cases a PCI bus id and trims nvidia-smi's 8-digit domain to 4: "00000000:01:00.0" -> "0000:01:00.0".
func normPCI(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if i := strings.Index(s, ":"); i > 4 {
		s = s[i-4:]
	}
	return s
}

func vendorFromID(id int) string {
	switch id {
	case 0x10de:
		return "nvidia"
	case 0x1002:
		return "amd"
	case 0x8086:
		return "intel"
	case 0x106b:
		return "apple"
	}
	return "other"
}
