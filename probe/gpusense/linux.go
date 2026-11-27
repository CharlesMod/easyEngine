package gpusense

import (
	"context"
	"sort"

	"github.com/CharlesMod/easyEngine/probe"
)

func senseLinux(ctx context.Context, s probe.Sources, p *Picture) {
	// The card is read by two cheap queries (a CSV line per adapter, `-q -d PIDS` for the
	// processes), 0.02 s together; the XML (0.09 s, 125 KB) is the fallback when they do not
	// parse. Both give the same adapters and processes (TestCheapReadEqualsXML).
	nvA, nvP, err := senseCheap(ctx, s)
	if err != nil {
		nvA, nvP, err = senseXML(ctx, s)
	}
	if err == nil {
		p.Adapters = append(p.Adapters, nvA...)
		p.Processes = append(p.Processes, nvP...)
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

func senseCheap(ctx context.Context, s probe.Sources) ([]Adapter, []Process, error) {
	out, err := s.Run(ctx, "nvidia-smi", smiCSVQuery...)
	if err != nil {
		return nil, nil, err
	}
	as, err := parseNvidiaCSV(out)
	if err != nil {
		return nil, nil, err
	}
	out, err = s.Run(ctx, "nvidia-smi", "-q", "-d", "PIDS")
	if err != nil {
		return nil, nil, err
	}
	ps, err := parseNvidiaPIDs(out, as)
	if err != nil {
		return nil, nil, err
	}
	return as, ps, nil
}

func senseXML(ctx context.Context, s probe.Sources) ([]Adapter, []Process, error) {
	out, err := s.Run(ctx, "nvidia-smi", "-q", "-x")
	if err != nil {
		return nil, nil, err
	}
	return parseNvidiaXML(out)
}
