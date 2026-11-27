package gpusense

import (
	"encoding/xml"
	"fmt"
	"strconv"
	"strings"
)

type smiLog struct {
	GPUs []smiGPU `xml:"gpu"`
}

type smiGPU struct {
	BusID   string `xml:"id,attr"`
	Product string `xml:"product_name"`
	UUID    string `xml:"uuid"`
	FB      struct {
		Total string `xml:"total"`
		Used  string `xml:"used"`
	} `xml:"fb_memory_usage"`
	Procs []smiProc `xml:"processes>process_info"`
}

type smiProc struct {
	PID  string `xml:"pid"`
	Type string `xml:"type"`
	Name string `xml:"process_name"`
	Mem  string `xml:"used_memory"`
}

// parseNvidiaXML parses `nvidia-smi -q -x`. Adapters take Index in document order (which is
// nvidia-smi's own GPU index). Per-process memory that nvidia-smi cannot report ("N/A", as under
// Windows WDDM) is 0. Kind: "C" compute, "G" and "C+G" graphics (a process that renders is
// what a game looks like), anything else "".
func parseNvidiaXML(b []byte) ([]Adapter, []Process, error) {
	var log smiLog
	if err := xml.Unmarshal(b, &log); err != nil {
		return nil, nil, fmt.Errorf("nvidia-smi xml: %w", err)
	}
	if len(log.GPUs) == 0 {
		return nil, nil, fmt.Errorf("nvidia-smi xml: no gpu elements")
	}
	var as []Adapter
	var ps []Process
	for i, g := range log.GPUs {
		a := Adapter{Index: i, Vendor: "nvidia", VendorID: 0x10de, Name: strings.TrimSpace(g.Product),
			UsedMiB: -1, UUID: strings.TrimSpace(g.UUID), PCI: normPCI(g.BusID), Source: "nvidia-smi"}
		if n, ok := parseSizeBytes(g.FB.Total); ok {
			a.DedicatedMiB = int(n / mib)
		}
		if n, ok := parseSizeBytes(g.FB.Used); ok {
			a.UsedMiB = int(n / mib)
		}
		as = append(as, a)
		for _, p := range g.Procs {
			pid, err := strconv.Atoi(strings.TrimSpace(p.PID))
			if err != nil {
				continue
			}
			pr := Process{PID: pid, Adapter: i, Kind: smiKind(p.Type)}
			if n := baseName(p.Name); n != "" && n != "N/A" {
				pr.Name = n
			}
			if n, ok := parseSizeBytes(p.Mem); ok {
				pr.DedicatedMiB = int(n / mib)
			}
			ps = append(ps, pr)
		}
	}
	return as, ps, nil
}

func smiKind(t string) string {
	switch strings.ToUpper(strings.ReplaceAll(strings.TrimSpace(t), " ", "")) {
	case "C":
		return "compute"
	case "G", "C+G", "G+C":
		return "graphics"
	}
	return ""
}

// smiCSVQuery is the cheap read of the card: the adapters, one line each (~0.004 s against the
// XML's ~0.09 s and 125 KB). pci.bus_id and uuid ride along so that `-q -d PIDS` can be matched to
// an adapter as the XML's own gpu elements are.
var smiCSVQuery = []string{"--query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"}

// parseNvidiaCSV parses smiCSVQuery's output. Adapters take Index in line order (nvidia-smi's own
// GPU index, as the XML's document order is). The name may hold commas: the first three fields and
// the last two are fixed. A size that is not a number ("[N/A]") leaves the adapter's use -1, its
// size 0, as the XML's does.
func parseNvidiaCSV(b []byte) ([]Adapter, error) {
	var as []Adapter
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) < 6 {
			return nil, fmt.Errorf("nvidia-smi csv: short line %q", line)
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		n := len(f)
		a := Adapter{Index: len(as), Vendor: "nvidia", VendorID: 0x10de, UsedMiB: -1,
			PCI: normPCI(f[1]), UUID: f[2], Name: strings.Join(f[3:n-2], ", "), Source: "nvidia-smi"}
		if v, err := strconv.Atoi(f[n-2]); err == nil {
			a.DedicatedMiB = v
		}
		if v, err := strconv.Atoi(f[n-1]); err == nil {
			a.UsedMiB = v
		}
		as = append(as, a)
	}
	if len(as) == 0 {
		return nil, fmt.Errorf("nvidia-smi csv: no gpu lines")
	}
	return as, nil
}
