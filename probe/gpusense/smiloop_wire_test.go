package gpusense

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"syscall"
	"testing"

	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/smiloop"
)

// wireSMI prints the same values as a one-shot and as a loop block, so the two paths can be compared.
const wireSMI = `#!/bin/sh
d=$(dirname "$0")
csv() { echo "$1 0, Fake GPU, 16384, 1234"; }
pids() { printf '\n==============NVSMI LOG==============\n\nAttached GPUs : 1\nGPU 00000000:01:00.0\n    Processes\n        Process ID                        : 7\n            Type                          : G\n            Name                          : /usr/bin/game\n            Used GPU Memory               : 99 MiB\n\n'; }
case "$*" in
  *-lms*) ;;
  --query-gpu*) csv ""; exit 0 ;;
  *) pids; exit 0 ;;
esac
case "$1$2" in
  --query-gpu*) k=csv ;;
  *) k=pids ;;
esac
while read -r w < "$d/ctl.$k"; do
  case "$w" in
    csv) csv "1," ;;
    pids) pids ;;
    exit) exit 0 ;;
  esac
done
`

// A read through OSSources gives the same values from the loop's block as from the one-shot.
func TestSampledReadsGiveTheOneShotsValues(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("linux only")
	}
	dir := t.TempDir()
	for _, k := range []string{"csv", "pids"} {
		if err := syscall.Mkfifo(filepath.Join(dir, "ctl."+k), 0o600); err != nil {
			t.Skip(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte(wireSMI), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	heard := make(chan smiloop.Kind, 16)
	old := smiloop.Default
	smiloop.Default = smiloop.NewManager(smiloop.Config{Heard: func(k smiloop.Kind) { heard <- k }})
	t.Cleanup(func() { smiloop.Default.Close(); smiloop.Default = old })
	defer smiloop.Default.Hold()()

	ctx := context.Background()
	sctx := smiloop.Sampled(ctx)
	as := []Adapter{{Index: 0, Vendor: "nvidia", VendorID: 0x10de, PCI: "0000:01:00.0"}}

	oneHW, err := probe.ProbeWith(ctx, probe.OSSources{})
	if err != nil {
		t.Skip(err)
	}
	oneProcs, _ := Procs(ctx, probe.OSSources{}, as)

	say := func(k string) {
		f, err := os.OpenFile(filepath.Join(dir, "ctl."+k), os.O_WRONLY, 0)
		if err != nil {
			t.Fatal(err)
		}
		f.WriteString(k + "\n")
		f.Close()
	}
	probe.ProbeWith(sctx, probe.OSSources{}) // starts the csv child; its read is the one-shot's
	say("csv")
	<-heard
	Procs(sctx, probe.OSSources{}, as) // starts the pids child
	say("pids")
	<-heard

	loopHW, err := probe.ProbeWith(sctx, probe.OSSources{})
	if err != nil {
		t.Fatal(err)
	}
	loopProcs, _ := Procs(sctx, probe.OSSources{}, as)
	if len(oneHW.GPUs) == 0 || !reflect.DeepEqual(oneHW.GPUs[0], loopHW.GPUs[0]) {
		t.Errorf("gpus differ:\n one-shot %+v\n loop     %+v", oneHW.GPUs, loopHW.GPUs)
	}
	if len(oneProcs) != 1 || !reflect.DeepEqual(oneProcs, loopProcs) {
		t.Errorf("procs differ:\n one-shot %+v\n loop     %+v", oneProcs, loopProcs)
	}
	if n := smiloop.Default.Starts(); n != 2 {
		t.Errorf("two queries, %d children", n)
	}
}
