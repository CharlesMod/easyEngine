package gpusense

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

const smiCSVKey = "nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used --format=csv,noheader,nounits"

// The same card state, recorded from a real card in its three forms (names and pids neutralised),
// read by the cheap path and by the XML path: the same adapters and the same processes.
func TestCheapReadEqualsXML(t *testing.T) {
	cheap := &fake{goos: "linux", goarch: "amd64", cmds: map[string]string{
		smiCSVKey:               td0("real-1gpu.csv"),
		"nvidia-smi -q -d PIDS": td0("real-1gpu-pids.txt"),
	}}
	xml := &fake{goos: "linux", goarch: "amd64", cmds: map[string]string{"nvidia-smi -q -x": td0("real-1gpu.xml")}}
	a := Sense(context.Background(), cheap, []int{2007}, 0)
	b := Sense(context.Background(), xml, []int{2007}, 0)
	if len(a.Adapters) != 1 || len(a.Processes) != 10 {
		t.Fatalf("cheap read: %+v", a)
	}
	if !reflect.DeepEqual(a.Adapters, b.Adapters) {
		t.Fatalf("adapters differ:\n%+v\n%+v", a.Adapters, b.Adapters)
	}
	if !reflect.DeepEqual(a.Processes, b.Processes) {
		t.Fatalf("processes differ:\n%+v\n%+v", a.Processes, b.Processes)
	}
	if a.Sources["nvidia-smi"] != "ok" || b.Sources["nvidia-smi"] != "ok" {
		t.Fatalf("sources %v %v", a.Sources, b.Sources)
	}
}

// When the cheap queries do not parse, the XML is read; when neither does, the source says so.
func TestCheapReadFallsBackToXML(t *testing.T) {
	f := &fake{goos: "linux", goarch: "amd64", cmds: map[string]string{
		smiCSVKey:               "Failed to initialize NVML",
		"nvidia-smi -q -d PIDS": td0("real-1gpu-pids.txt"),
		"nvidia-smi -q -x":      td0("real-1gpu.xml"),
	}}
	pic := Sense(context.Background(), f, nil, 0)
	if len(pic.Adapters) != 1 || len(pic.Processes) != 10 || pic.Sources["nvidia-smi"] != "ok" {
		t.Fatalf("%+v", pic)
	}
	f.cmds["nvidia-smi -q -x"] = "garbage"
	pic = Sense(context.Background(), f, nil, 0)
	if len(pic.Adapters) != 0 || !strings.Contains(pic.Sources["nvidia-smi"], "xml") {
		t.Fatalf("%+v", pic)
	}
}

func TestParseNvidiaCSVTwoGPUsAndCommaName(t *testing.T) {
	as, err := parseNvidiaCSV([]byte("0, 00000000:01:00.0, GPU-a, NVIDIA Foo, Bar, 8192, 100\n1, 00000000:65:00.0, GPU-b, NVIDIA Baz, 24576, [N/A]\n"))
	if err != nil || len(as) != 2 {
		t.Fatalf("%v %+v", err, as)
	}
	if as[0].Name != "NVIDIA Foo, Bar" || as[0].PCI != "0000:01:00.0" || as[0].DedicatedMiB != 8192 || as[0].UsedMiB != 100 {
		t.Errorf("%+v", as[0])
	}
	if as[1].Index != 1 || as[1].PCI != "0000:65:00.0" || as[1].UsedMiB != -1 || as[1].DedicatedMiB != 24576 {
		t.Errorf("%+v", as[1])
	}
	if _, err := parseNvidiaCSV([]byte("No devices were found\n")); err == nil {
		t.Error("garbage parsed")
	}
}
