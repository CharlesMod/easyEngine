package gpusense

import (
	"bytes"
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/probe"
)

// Procs lists the processes holding memory on the given adapters, as cheaply as the platform allows,
// for polling about once a second. as is the adapter list from an earlier Sense (indices, PCI, LUID, VendorID).
// The map is source -> status, like Picture.Sources. Ours/Host are not marked here.
func Procs(ctx context.Context, s probe.Sources, as []Adapter) ([]Process, map[string]string) {
	src := map[string]string{}
	goos, _ := s.Platform()
	switch goos {
	case "linux":
		var nv, other []Adapter
		for _, a := range as {
			if a.VendorID == 0x10de || a.Vendor == "nvidia" {
				nv = append(nv, a)
			} else {
				other = append(other, a)
			}
		}
		var out []Process
		if len(nv) > 0 {
			b, err := s.Run(ctx, "nvidia-smi", "-q", "-d", "PIDS")
			if err == nil {
				var ps []Process
				ps, err = parseNvidiaPIDs(b, nv)
				out = append(out, ps...)
			}
			src["nvidia-smi"] = srcStatus(err)
		}
		if len(other) > 0 {
			ps, err := fdinfoProcesses(s, other)
			src["fdinfo"] = srcStatus(err)
			out = append(out, ps...)
		}
		return out, src
	case "windows":
		samples, err := nativePDH()
		src["pdh"] = srcStatus(err)
		if err != nil {
			return nil, src
		}
		procs, _ := pdhToProcesses(samples, as)
		names := map[int]string{}
		for i := range procs {
			pid := procs[i].PID
			if _, ok := names[pid]; !ok {
				names[pid] = baseName(nativeProcImage(pid))
			}
			procs[i].Name = names[pid]
		}
		return procs, src
	}
	src["procs"] = "not available on " + goos
	return nil, src
}

// parseNvidiaPIDs parses the text of `nvidia-smi -q -d PIDS`. Sections start with a
// "GPU <busid>" line; each is matched to an adapter by normalised PCI id (or, with exactly
// one adapter and no match, that adapter). Sections that match nothing are skipped.
func parseNvidiaPIDs(out []byte, as []Adapter) ([]Process, error) {
	if !bytes.Contains(out, []byte("GPU ")) {
		return nil, fmt.Errorf("nvidia-smi pids: no GPU sections")
	}
	byPCI := map[string]int{}
	for _, a := range as {
		if a.PCI != "" {
			byPCI[a.PCI] = a.Index
		}
	}
	var res []Process
	adapter, inSec := -1, false
	var cur *Process
	flush := func() {
		if cur != nil && inSec && adapter >= 0 {
			res = append(res, *cur)
		}
		cur = nil
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimRight(line, "\r")
		t := strings.TrimSpace(line)
		if strings.HasPrefix(line, "GPU ") && !strings.Contains(line, " : ") {
			flush()
			inSec = true
			adapter = -1
			if i, ok := byPCI[normPCI(strings.TrimPrefix(line, "GPU "))]; ok {
				adapter = i
			} else if len(as) == 1 {
				adapter = as[0].Index
			}
			continue
		}
		k, v, ok := strings.Cut(t, " : ")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch k {
		case "Process ID":
			flush()
			pid, err := strconv.Atoi(v)
			if err != nil {
				continue
			}
			cur = &Process{PID: pid, Adapter: adapter}
		case "Type":
			if cur != nil {
				cur.Kind = smiKind(v)
			}
		case "Name":
			if cur != nil {
				if n := baseName(v); n != "N/A" {
					cur.Name = n
				}
			}
		case "Used GPU Memory":
			if cur != nil {
				if f := strings.Fields(v); len(f) >= 1 {
					if n, err := strconv.Atoi(f[0]); err == nil {
						cur.DedicatedMiB = n
					}
				}
			}
		}
	}
	flush()
	return res, nil
}
