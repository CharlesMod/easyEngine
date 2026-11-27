package gpusense

import (
	"context"
	"errors"
	"testing"
)

func TestParseNvidiaPIDsReal(t *testing.T) {
	ps, err := parseNvidiaPIDs([]byte(td0("nvidia-smi-pids.txt")), []Adapter{{Index: 0, VendorID: 0x10de, PCI: "0000:01:00.0"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 10 {
		t.Fatalf("got %d: %+v", len(ps), ps)
	}
	if k := ps[0]; k.PID != 3865 || k.Name != "kwin_wayland" || k.Kind != "graphics" || k.DedicatedMiB != 74 {
		t.Errorf("kwin %+v", k)
	}
	l := ps[9]
	if l.PID != 1290082 || l.Name != "llama-server" || l.DedicatedMiB != 10332 || l.Kind != "compute" || l.Adapter != 0 {
		t.Errorf("llama %+v", l)
	}
}

func TestParseNvidiaPIDsTwoGPUs(t *testing.T) {
	as := []Adapter{{Index: 0, VendorID: 0x10de, PCI: "0000:01:00.0"}, {Index: 1, VendorID: 0x10de, PCI: "0000:65:00.0"}}
	ps, err := parseNvidiaPIDs([]byte(td0("nvidia-smi-pids-two.txt")), as)
	if err != nil {
		t.Fatal(err)
	}
	if len(ps) != 2 {
		t.Fatalf("%+v", ps)
	}
	if ps[0].PID != 777 || ps[0].Adapter != 1 || ps[0].Name != "some tool" || ps[0].DedicatedMiB != 512 {
		t.Errorf("%+v", ps[0])
	}
	if ps[1].PID != 778 || ps[1].Adapter != 1 || ps[1].DedicatedMiB != 0 || ps[1].Kind != "graphics" {
		t.Errorf("%+v", ps[1])
	}
	// An unknown section is skipped when several adapters exist.
	ps, _ = parseNvidiaPIDs([]byte(td0("nvidia-smi-pids-two.txt")), as[:1])
	if len(ps) != 2 { // single adapter, no match for 65:00.0 -> assumed
		t.Errorf("single-adapter fallback: %+v", ps)
	}
	ps, _ = parseNvidiaPIDs([]byte(td0("nvidia-smi-pids-two.txt")), []Adapter{{Index: 0, PCI: "0000:01:00.0"}, {Index: 1, PCI: "0000:99:00.0"}})
	if len(ps) != 0 {
		t.Errorf("unknown section must be skipped: %+v", ps)
	}
	if _, err := parseNvidiaPIDs([]byte("garbage"), as); err == nil {
		t.Error("garbage must error")
	}
}

func TestProcsLinux(t *testing.T) {
	f := &fake{goos: "linux", goarch: "amd64",
		cmds: map[string]string{"nvidia-smi -q -d PIDS": td0("nvidia-smi-pids.txt")}}
	ps, src := Procs(context.Background(), f, []Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, PCI: "0000:01:00.0"}})
	if len(ps) != 10 || src["nvidia-smi"] != "ok" {
		t.Errorf("%d %v", len(ps), src)
	}
	if _, ok := src["fdinfo"]; ok {
		t.Errorf("NVIDIA-only must not read fdinfo: %v", src)
	}
	// nvidia-smi missing.
	ps, src = Procs(context.Background(), &fake{goos: "linux"}, []Adapter{{Index: 0, VendorID: 0x10de}})
	if len(ps) != 0 || src["nvidia-smi"] == "ok" || src["nvidia-smi"] == "" {
		t.Errorf("%v %v", ps, src)
	}
	// Mixed: NVIDIA via smi, others via fdinfo.
	f = &fake{goos: "linux", goarch: "amd64", files: fdFiles(),
		cmds: map[string]string{"nvidia-smi -q -d PIDS": td0("nvidia-smi-pids-two.txt")}}
	as, _ := sysfsAdapters(f, 0)
	as = append(as, Adapter{Index: 9, VendorID: 0x10de, PCI: "0000:65:00.0"})
	_, src = Procs(context.Background(), f, as)
	if src["nvidia-smi"] != "ok" || src["fdinfo"] != "ok" {
		t.Errorf("%v", src)
	}
}

func TestProcsWindows(t *testing.T) {
	fakeWindows(t)
	f := &fake{goos: "windows", goarch: "amd64"}
	as := []Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, LUID: "0x00000000_0x0000D3A1"}}
	ps, src := Procs(context.Background(), f, as)
	if src["pdh"] != "ok" || len(ps) != 1 || ps[0].PID != 1234 || ps[0].Name != "game.exe" || ps[0].DedicatedMiB != 3100 {
		t.Errorf("%+v %v", ps, src)
	}
	nativePDH = func() ([]pdhSample, error) { return nil, errors.New("no counters") }
	ps, src = Procs(context.Background(), f, as)
	if len(ps) != 0 || src["pdh"] != "no counters" {
		t.Errorf("%+v %v", ps, src)
	}
}

func TestProcsOtherOS(t *testing.T) {
	ps, src := Procs(context.Background(), &fake{goos: "darwin"}, nil)
	if ps != nil || src["procs"] != "not available on darwin" {
		t.Errorf("%v %v", ps, src)
	}
}
