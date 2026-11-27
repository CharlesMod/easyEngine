package plan

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

type recSrc struct{ cmds map[string]string }

func (recSrc) Platform() (string, string)      { return "linux", "amd64" }
func (recSrc) NumCPU() int                     { return 4 }
func (recSrc) DiskFreeMiB(string) (int, error) { return 1000, nil }
func (recSrc) ReadFile(string) ([]byte, error) { return nil, errors.New("none") }
func (recSrc) Glob(string) ([]string, error)   { return nil, nil }
func (r recSrc) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	if v, ok := r.cmds[strings.TrimSpace(name+" "+strings.Join(args, " "))]; ok {
		return []byte(v), nil
	}
	return nil, errors.New("not found")
}

func rec(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("../probe/gpusense/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The reserve (others, host, margin) the room computes is the same whether the card was read by
// the cheap queries or by the XML, from one recorded state of a real card.
func TestReserveSameFromCheapReadAndXML(t *testing.T) {
	cheap := recSrc{map[string]string{
		"nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,memory.total,memory.used --format=csv,noheader,nounits": rec(t, "real-1gpu.csv"),
		"nvidia-smi -q -d PIDS": rec(t, "real-1gpu-pids.txt"),
	}}
	xml := recSrc{map[string]string{"nvidia-smi -q -x": rec(t, "real-1gpu.xml")}}
	ours := []int{2007}
	pa := gpusense.Sense(context.Background(), cheap, ours, 0)
	pb := gpusense.Sense(context.Background(), xml, ours, 0)
	hw := probe.Hardware{OS: "linux"}
	sa, ia := MeasureReserveOn(pa, hw, HostInfo{}, 1500, 0)
	sb, ib := MeasureReserveOn(pb, hw, HostInfo{}, 1500, 0)
	if ia != ib || !reflect.DeepEqual(sa, sb) {
		t.Fatalf("reserve differs:\n%+v %d\n%+v %d", sa, ia, sb, ib)
	}
	if sa.EngineAdapter != 0 || sa.Reserve.OthersMiB == 0 {
		t.Fatalf("the recorded card should show other apps on it: %+v", sa)
	}
}
