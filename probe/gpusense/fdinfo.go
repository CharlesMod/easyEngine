package gpusense

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/probe"
)

// fdClient is one DRM client read from a /proc/<pid>/fdinfo/<fd> file.
type fdClient struct {
	Driver, PDev, ClientID  string
	DedicatedMiB, SharedMiB int
	HasMem                  bool
}

// Key names carry a region suffix ("vram0", "local1"); trimming digits folds regions together.
var fdKeyRe = regexp.MustCompile(`^drm-(memory|resident|total)-([a-z]+)\d*$`)

// parseFdinfo reads the drm-* lines of one fdinfo file. ok is false when it is not a DRM file.
//
// Dedicated = vram (amdgpu, xe) and local (i915 discrete) regions; shared = gtt, system, cpu.
// For each class the best available kind wins, per name: drm-resident-* (what is actually
// there) over drm-memory-* (amdgpu) over drm-total-* (everything allocated, resident or not).
// Regions with the same name (vram0, vram1) are summed. Bare numbers are bytes; KiB/MiB/GiB
// are honoured; "N/A" and junk count as absent.
func parseFdinfo(text string) (fdClient, bool) {
	var c fdClient
	type key struct{ class, name string }
	best := map[key]map[string]int64{} // (class,name) -> kind -> summed bytes
	rank := map[string]int{"resident": 0, "memory": 1, "total": 2}
	for _, line := range strings.Split(text, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch k {
		case "drm-driver":
			c.Driver = v
			continue
		case "drm-pdev":
			c.PDev = normPCI(v)
			continue
		case "drm-client-id":
			c.ClientID = v
			continue
		}
		m := fdKeyRe.FindStringSubmatch(k)
		if m == nil {
			continue
		}
		var class string
		switch m[2] {
		case "vram", "local":
			class = "d"
		case "gtt", "system", "cpu":
			class = "s"
		default:
			continue
		}
		n, ok := parseSizeBytes(v)
		if !ok {
			continue
		}
		bk := key{class, m[2]}
		if best[bk] == nil {
			best[bk] = map[string]int64{}
		}
		best[bk][m[1]] += n
		c.HasMem = true
	}
	if c.Driver == "" && c.PDev == "" && c.ClientID == "" {
		return c, false
	}
	var d, s int64
	for bk, kinds := range best {
		pick, pr := int64(0), 99
		for kind, n := range kinds {
			if rank[kind] < pr {
				pick, pr = n, rank[kind]
			}
		}
		if bk.class == "d" {
			d += pick
		} else {
			s += pick
		}
	}
	c.DedicatedMiB, c.SharedMiB = int(d/mib), int(s/mib)
	return c, true
}

var fdPathRe = regexp.MustCompile(`^/proc/(\d+)/fdinfo/[^/]+$`)

// fdinfoProcesses sums DRM client memory per (pid, adapter) over /proc/*/fdinfo/*. A client
// (pdev + client-id) is counted once however many fds or processes hold it. Unreadable entries
// are skipped silently. Adapters are matched by PCI slot; with no usable drm-pdev and exactly
// one candidate adapter, that one is assumed.
func fdinfoProcesses(s probe.Sources, as []Adapter) ([]Process, error) {
	paths, err := s.Glob("/proc/*/fdinfo/*")
	if err != nil {
		return nil, err
	}
	byPCI := map[string]int{}
	for _, a := range as {
		if a.PCI != "" {
			byPCI[a.PCI] = a.Index
		}
	}
	type pa struct{ pid, adapter int }
	sums := map[pa]*Process{}
	var order []pa
	seen := map[string]bool{}
	for _, p := range paths {
		m := fdPathRe.FindStringSubmatch(p)
		if m == nil {
			continue
		}
		b, err := s.ReadFile(p)
		if err != nil || !strings.Contains(string(b), "drm-") {
			continue
		}
		c, ok := parseFdinfo(string(b))
		if !ok || !c.HasMem {
			continue
		}
		if c.Driver == "nvidia" {
			continue
		}
		pid, _ := strconv.Atoi(m[1])
		idx, found := byPCI[c.PDev]
		if !found {
			if len(as) != 1 {
				continue
			}
			idx = as[0].Index
		}
		if c.ClientID != "" {
			k := c.PDev + "/" + c.ClientID
			if seen[k] {
				continue
			}
			seen[k] = true
		}
		if c.DedicatedMiB == 0 && c.SharedMiB == 0 {
			continue
		}
		k := pa{pid, idx}
		if sums[k] == nil {
			sums[k] = &Process{PID: pid, Adapter: idx, Kind: "graphics"}
			order = append(order, k)
		}
		sums[k].DedicatedMiB += c.DedicatedMiB
		sums[k].SharedMiB += c.SharedMiB
	}
	var out []Process
	for _, k := range order {
		p := sums[k]
		if b, err := s.ReadFile("/proc/" + strconv.Itoa(p.PID) + "/comm"); err == nil {
			p.Name = strings.TrimSpace(string(b))
		}
		out = append(out, *p)
	}
	return out, nil
}
