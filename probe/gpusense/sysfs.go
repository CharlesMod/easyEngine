package gpusense

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/probe"
)

var cardDevRe = regexp.MustCompile(`/card(\d+)/device$`)

// sysfsAdapters lists the AMD, Intel and other non-NVIDIA adapters under /sys/class/drm,
// starting at Index startIdx. NVIDIA is skipped (nvidia-smi covers it). Only cardN nodes are
// read, never renderDN, and two cardN that share a PCI slot count once.
func sysfsAdapters(s probe.Sources, startIdx int) ([]Adapter, error) {
	devs, err := s.Glob("/sys/class/drm/card*/device")
	if err != nil {
		return nil, err
	}
	sort.Slice(devs, func(i, j int) bool { return cardNum(devs[i]) < cardNum(devs[j]) })
	var as []Adapter
	seen := map[string]bool{}
	for _, d := range devs {
		if !cardDevRe.MatchString(d) {
			continue
		}
		vb, err := s.ReadFile(d + "/vendor")
		if err != nil {
			continue
		}
		vid := hexInt(string(vb))
		if vid == 0x10de {
			continue
		}
		pci := ""
		if ub, err := s.ReadFile(d + "/uevent"); err == nil {
			for _, line := range strings.Split(string(ub), "\n") {
				if v, ok := strings.CutPrefix(line, "PCI_SLOT_NAME="); ok {
					pci = normPCI(v)
				}
			}
		}
		if pci != "" {
			if seen[pci] {
				continue
			}
			seen[pci] = true
		}
		dev := ""
		if db, err := s.ReadFile(d + "/device"); err == nil {
			dev = strings.TrimSpace(string(db))
		}
		a := Adapter{Index: startIdx + len(as), Vendor: vendorFromID(vid), VendorID: vid, UsedMiB: -1, PCI: pci, Source: "sysfs"}
		var vram, gtt int64 = -1, -1
		if b, err := s.ReadFile(d + "/mem_info_vram_total"); err == nil {
			vram = parseInt(b)
		}
		if b, err := s.ReadFile(d + "/mem_info_gtt_total"); err == nil {
			gtt = parseInt(b)
		}
		if b, err := s.ReadFile(d + "/mem_info_vram_used"); err == nil && parseInt(b) >= 0 {
			a.UsedMiB = int(parseInt(b) / mib)
		}
		if vram > 0 {
			a.DedicatedMiB = int(vram / mib)
		}
		if gtt > 0 {
			a.SharedMiB = int(gtt / mib)
		}
		boot := false
		if b, err := s.ReadFile(d + "/boot_vga"); err == nil {
			boot = strings.TrimSpace(string(b)) == "1"
		}
		a.Name = sysfsName(s, d, a.Vendor, dev)
		a.Integrated = sysfsIntegrated(a.Vendor, pci, a.DedicatedMiB, a.SharedMiB, boot)
		as = append(as, a)
	}
	if len(as) == 0 {
		return nil, fmt.Errorf("no non-NVIDIA adapters under /sys/class/drm")
	}
	return as, nil
}

func sysfsName(s probe.Sources, d, vendor, dev string) string {
	if nb, err := s.ReadFile(d + "/product_name"); err == nil && strings.TrimSpace(string(nb)) != "" {
		return strings.TrimSpace(string(nb))
	}
	n := strings.ToUpper(vendor[:1]) + vendor[1:] + " GPU"
	if vendor == "amd" {
		n = "AMD GPU"
	}
	return strings.TrimSpace(n + " " + dev)
}

// sysfsIntegrated decides whether an adapter shares the CPU's memory. There is no kernel flag, so:
//   - Intel: integrated unless it reports local VRAM (mem_info_vram_total > 0, an Arc) or sits
//     off PCI bus 00 (iGPUs are always 0000:00:02.0; Arc cards are on a discrete bus).
//   - AMD: integrated when it has a small VRAM carve-out (<= 2048 MiB) and a GTT pool, and either
//     the carve-out is tiny (<= 1024 MiB), or it is the boot VGA device, or it sits on bus 00.
//     APUs usually sit on a non-zero bus and carve 512 MiB..2 GiB. An APU with a BIOS carve-out
//     above 2 GiB reads as discrete: that errs toward "dedicated", the safe side for a reserve.
//   - Anything else: not integrated.
func sysfsIntegrated(vendor, pci string, dedicatedMiB, sharedMiB int, bootVGA bool) bool {
	bus0 := strings.HasPrefix(pci[min(len(pci), 5):], "00:")
	switch vendor {
	case "intel":
		if dedicatedMiB > 0 {
			return false
		}
		return pci == "" || bus0
	case "amd":
		if dedicatedMiB > 2048 || sharedMiB <= 0 {
			return false
		}
		return dedicatedMiB <= 1024 || bootVGA || bus0
	}
	return false
}

func cardNum(p string) int {
	if m := cardDevRe.FindStringSubmatch(p); m != nil {
		n, _ := strconv.Atoi(m[1])
		return n
	}
	return 1 << 30
}

func hexInt(s string) int {
	n, err := strconv.ParseInt(strings.TrimPrefix(strings.ToLower(strings.TrimSpace(s)), "0x"), 16, 64)
	if err != nil {
		return 0
	}
	return int(n)
}

func parseInt(b []byte) int64 {
	n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	if err != nil {
		return -1
	}
	return n
}
