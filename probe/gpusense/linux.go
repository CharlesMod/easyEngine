package gpusense

import (
	"context"
	"sort"

	"hermesengine/probe"
)

func senseLinux(ctx context.Context, s probe.Sources, p *Picture) {
	out, err := s.Run(ctx, "nvidia-smi", "-q", "-x")
	if err == nil {
		var as []Adapter
		var ps []Process
		as, ps, err = parseNvidiaXML(out)
		if err == nil {
			p.Adapters = append(p.Adapters, as...)
			p.Processes = append(p.Processes, ps...)
		}
	}
	p.Sources["nvidia-smi"] = srcStatus(err)

	sys, err := sysfsAdapters(s, len(p.Adapters))
	p.Sources["sysfs"] = srcStatus(err)
	p.Adapters = append(p.Adapters, sys...)

	if len(sys) == 0 {
		p.Sources["fdinfo"] = "skipped: no AMD/Intel adapter"
		return
	}
	ps, err := fdinfoProcesses(s, sys)
	p.Sources["fdinfo"] = srcStatus(err)
	p.Processes = append(p.Processes, ps...)
	sort.SliceStable(p.Processes, func(i, j int) bool { return p.Processes[i].Adapter < p.Processes[j].Adapter })
}
