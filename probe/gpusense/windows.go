package gpusense

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/probe"
)

// dxgiAdapter is one DXGI_ADAPTER_DESC1, software adapters already dropped.
type dxgiAdapter struct {
	Name           string
	VendorID       int
	DedicatedBytes uint64
	SharedBytes    uint64
	LUIDHigh       uint32
	LUIDLow        uint32
	Flags          uint32
}

// pdhSample is one instance of a wildcard PDH counter (value in bytes).
type pdhSample struct {
	Counter  string // pdhProcDedicated | pdhProcShared | pdhAdapterDedicated
	Instance string
	Value    int64
}

const (
	pdhProcDedicated    = `\GPU Process Memory(*)\Dedicated Usage`
	pdhProcShared       = `\GPU Process Memory(*)\Shared Usage`
	pdhAdapterDedicated = `\GPU Adapter Memory(*)\Dedicated Usage`
)

// The native (syscall) readers; set by win_native_windows.go, errors elsewhere. Tests replace them.
var (
	nativeDXGI      = func() ([]dxgiAdapter, error) { return nil, errors.New("dxgi: not windows") }
	nativePDH       = func() ([]pdhSample, error) { return nil, errors.New("pdh: not windows") }
	nativeProcImage = func(pid int) string { return "" }
)

func luidString(high, low uint32) string { return fmt.Sprintf("0x%08X_0x%08X", high, low) }

var (
	pdhProcRe = regexp.MustCompile(`(?i)^pid_(\d+)_luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)`)
	pdhAdptRe = regexp.MustCompile(`(?i)^luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)`)
)

// parsePDHProcInstance parses a "GPU Process Memory" instance such as
// "pid_1234_luid_0x00000000_0x0000D1F2_phys_0" (anything after _phys_N, an engtype suffix for
// the engine counters, is ignored). The LUID comes back normalised as "0x%08X_0x%08X".
func parsePDHProcInstance(name string) (pid int, luid string, phys int, ok bool) {
	m := pdhProcRe.FindStringSubmatch(strings.TrimSpace(name))
	if m == nil {
		return 0, "", 0, false
	}
	hi, ok1 := parseHex32(m[2])
	lo, ok2 := parseHex32(m[3])
	if !ok1 || !ok2 {
		return 0, "", 0, false
	}
	pid, _ = strconv.Atoi(m[1])
	phys, _ = strconv.Atoi(m[4])
	return pid, luidString(hi, lo), phys, true
}

// parsePDHAdapterInstance parses a "GPU Adapter Memory" instance "luid_0x00000000_0x0000D1F2_phys_0".
func parsePDHAdapterInstance(name string) (luid string, phys int, ok bool) {
	m := pdhAdptRe.FindStringSubmatch(strings.TrimSpace(name))
	if m == nil {
		return "", 0, false
	}
	hi, ok1 := parseHex32(m[1])
	lo, ok2 := parseHex32(m[2])
	if !ok1 || !ok2 {
		return "", 0, false
	}
	phys, _ = strconv.Atoi(m[3])
	return luidString(hi, lo), phys, true
}

func parseHex32(s string) (uint32, bool) {
	n, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(s), "0x"), 16, 32)
	return uint32(n), err == nil
}

// integratedDXGI: Intel is integrated unless it is an Arc (name contains "arc") or reports more
// than 2048 MiB of dedicated memory (iGPUs report 0..~512 MiB). AMD is integrated when its
// DedicatedVideoMemory is <= 2048 MiB (an APU carve-out; discrete cards today have 4 GiB+).
// Qualcomm (0x5143) is integrated. NVIDIA and everything else are not.
func integratedDXGI(vendorID int, name string, dedicatedMiB int) bool {
	switch vendorID {
	case 0x8086:
		return !strings.Contains(strings.ToLower(name), "arc") && dedicatedMiB <= 2048
	case 0x1002:
		return dedicatedMiB <= 2048
	case 0x5143:
		return true
	}
	return false
}

// dxgiToAdapters turns DXGI descriptions into adapters. NVIDIA adapters are placed first (so
// their Index lines up with nvidia-smi's order, which DXGI enumeration follows); the rest keep
// enumeration order after them.
func dxgiToAdapters(ds []dxgiAdapter) []Adapter {
	var nv, rest []Adapter
	for _, d := range ds {
		if d.Flags&2 != 0 { // DXGI_ADAPTER_FLAG_SOFTWARE
			continue
		}
		ded := int(d.DedicatedBytes / mib)
		a := Adapter{Vendor: vendorFromID(d.VendorID), VendorID: d.VendorID, Name: strings.TrimSpace(d.Name),
			DedicatedMiB: ded, SharedMiB: int(d.SharedBytes / mib), UsedMiB: -1,
			Integrated: integratedDXGI(d.VendorID, d.Name, ded), LUID: luidString(d.LUIDHigh, d.LUIDLow), Source: "dxgi"}
		if a.Vendor == "nvidia" {
			nv = append(nv, a)
		} else {
			rest = append(rest, a)
		}
	}
	all := append(nv, rest...)
	for i := range all {
		all[i].Index = i
	}
	return all
}

// pdhToProcesses sums the process counters per (pid, luid) over physical-adapter indices and
// the adapter counter per luid. Malformed instances and instances on unknown LUIDs are skipped.
func pdhToProcesses(samples []pdhSample, as []Adapter) (procs []Process, adapterUsed map[int]int) {
	byLUID := map[string]int{}
	for _, a := range as {
		if a.LUID != "" {
			byLUID[strings.ToUpper(a.LUID)] = a.Index
		}
	}
	type pa struct{ pid, adapter int }
	sums := map[pa]*Process{}
	adapterUsed = map[int]int{}
	adapterBytes := map[int]int64{}
	for _, sm := range samples {
		switch sm.Counter {
		case pdhProcDedicated, pdhProcShared:
			pid, luid, _, ok := parsePDHProcInstance(sm.Instance)
			if !ok {
				continue
			}
			idx, ok := byLUID[strings.ToUpper(luid)]
			if !ok {
				continue
			}
			k := pa{pid, idx}
			if sums[k] == nil {
				sums[k] = &Process{PID: pid, Adapter: idx}
			}
			if sm.Counter == pdhProcDedicated {
				sums[k].DedicatedMiB += int(sm.Value / mib)
			} else {
				sums[k].SharedMiB += int(sm.Value / mib)
			}
		case pdhAdapterDedicated:
			luid, _, ok := parsePDHAdapterInstance(sm.Instance)
			if !ok {
				continue
			}
			if idx, ok := byLUID[strings.ToUpper(luid)]; ok {
				adapterBytes[idx] += sm.Value
			}
		}
	}
	for idx, b := range adapterBytes {
		adapterUsed[idx] = int(b / mib)
	}
	for _, p := range sums {
		if p.DedicatedMiB == 0 && p.SharedMiB == 0 {
			continue
		}
		procs = append(procs, *p)
	}
	sort.Slice(procs, func(i, j int) bool {
		if procs[i].Adapter != procs[j].Adapter {
			return procs[i].Adapter < procs[j].Adapter
		}
		return procs[i].PID < procs[j].PID
	})
	return procs, adapterUsed
}

func senseWindows(ctx context.Context, s probe.Sources, p *Picture) {
	ds, err := nativeDXGI()
	p.Sources["dxgi"] = srcStatus(err)
	var as []Adapter
	if err == nil {
		as = dxgiToAdapters(ds)
	}
	if len(as) == 0 {
		as = wmiFallback(ctx, s, p)
	}
	// nvidia-smi: UUID, PCI and card-wide use for NVIDIA adapters, and the process kinds.
	kinds := map[int]string{}
	smi, smiProcs, serr := []Adapter(nil), []Process(nil), error(nil)
	if out, rerr := s.Run(ctx, "nvidia-smi", "-q", "-x"); rerr != nil {
		serr = rerr
	} else {
		smi, smiProcs, serr = parseNvidiaXML(out)
	}
	p.Sources["nvidia-smi"] = srcStatus(serr)
	if serr == nil {
		n := 0
		for i := range as {
			if as[i].Vendor == "nvidia" && n < len(smi) {
				as[i].UUID, as[i].PCI = smi[n].UUID, smi[n].PCI
				if as[i].UsedMiB < 0 {
					as[i].UsedMiB = smi[n].UsedMiB
				}
				if as[i].DedicatedMiB == 0 {
					as[i].DedicatedMiB = smi[n].DedicatedMiB
				}
				n++
			}
		}
		for _, sp := range smiProcs {
			if sp.Kind != "" {
				kinds[sp.PID] = sp.Kind
			}
		}
	}

	samples, perr := nativePDH()
	p.Sources["pdh"] = srcStatus(perr)
	if perr == nil {
		procs, used := pdhToProcesses(samples, as)
		for i := range as {
			if u, ok := used[as[i].Index]; ok {
				as[i].UsedMiB = u // PDH is card-wide and live; it wins over nvidia-smi's figure
			}
		}
		names := map[int]string{}
		for i := range procs {
			pid := procs[i].PID
			if _, ok := names[pid]; !ok {
				names[pid] = baseName(nativeProcImage(pid))
			}
			procs[i].Name = names[pid]
			procs[i].Kind = kinds[pid]
		}
		p.Processes = append(p.Processes, procs...)
	}
	p.Adapters = append(p.Adapters, as...)
}

// wmiFallback is the DXGI-less path: the probe's WMI/nvidia-smi GPU list, without LUIDs
// (so no PDH mapping is possible; the per-process list stays empty).
func wmiFallback(ctx context.Context, s probe.Sources, p *Picture) []Adapter {
	hw, err := probe.ProbeWith(ctx, s)
	if err != nil {
		p.Sources["wmi"] = err.Error()
		return nil
	}
	var as []Adapter
	for _, g := range hw.GPUs {
		a := Adapter{Index: g.Index, Vendor: g.Vendor, VendorID: vendorID(g.Vendor), Name: g.Name,
			DedicatedMiB: g.VRAMTotalMiB, UsedMiB: -1, Source: g.Source}
		if g.Source == "nvidia-smi" {
			a.UsedMiB = g.VRAMUsedMiB
		}
		a.Integrated = integratedDXGI(a.VendorID, a.Name, a.DedicatedMiB) && g.Vendor != "nvidia" && g.Vendor != "other"
		as = append(as, a)
	}
	if len(as) == 0 {
		p.Sources["wmi"] = "no adapters"
		return nil
	}
	p.Sources["wmi"] = "ok"
	return as
}

func vendorID(v string) int {
	switch v {
	case "nvidia":
		return 0x10de
	case "amd":
		return 0x1002
	case "intel":
		return 0x8086
	}
	return 0
}
