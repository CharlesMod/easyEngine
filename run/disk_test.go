package run_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/run"
)

const mib = int64(1) << 20

func rb(n int64) int64 { return (n + 4095) / 4096 * 4096 }

// diskFix is a supervisor that was never started (nothing writes its logs behind the test's
// back), over the install harness's site: two llama.cpp builds, uv, a downloadable model, and an
// agent profile with a wheel, with sizes the tests can compute from.
type diskFix struct {
	*site
	sup  *run.Supervisor
	cat  *catalog.Catalog
	hw   probe.Hardware
	free atomic.Int64 // fake free MiB; -1 makes DiskFree fail
	be   catalog.Asset
	dl   catalog.Model
}

const (
	unpBackend = 7 << 20
	unpUV      = 2 << 20
	agentPeak  = 300 << 20
	agentKeep  = 200 << 20
)

func newDiskFix(t *testing.T) *diskFix {
	t.Helper()
	d := &diskFix{site: newSite(t)}
	d.cat = d.site.cat
	for i := range d.cat.LlamaCpp.Assets {
		d.cat.LlamaCpp.Assets[i].UnpackedSize = unpBackend
	}
	d.cat.UV[0].UnpackedSize = unpUV
	d.cat.AgentRuntime = []catalog.RuntimeSize{{OS: runtime.GOOS, Arch: runtime.GOARCH, Python: "3.12", PeakBytes: agentPeak, KeepBytes: agentKeep}}
	d.free.Store(1 << 20)
	cfg := d.e.Config()
	cfg.DiskFree = func(string) (int, error) {
		f := d.free.Load()
		if f < 0 {
			return 0, errors.New("statfs failed")
		}
		return int(f), nil
	}
	sup, err := run.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sup.Shutdown)
	d.sup = sup
	hw, err := sup.Hardware(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	d.hw = hw
	d.be = d.cat.Backends(hw)[0]
	d.dl, _ = d.cat.Model("dl")
	return d
}

func steps(n plan.DiskNeed) []string {
	var out []string
	for _, s := range n.Steps {
		out = append(out, s.What)
	}
	return out
}

func wantSteps(t *testing.T, n plan.DiskNeed, want ...string) {
	t.Helper()
	got := strings.Join(steps(n), "|")
	if got != strings.Join(want, "|") {
		t.Errorf("steps = %q, want %q", got, strings.Join(want, "|"))
	}
}

func (d *diskFix) installBackend(t *testing.T) {
	t.Helper()
	dir := filepath.Join(d.e.Data, "backends", d.be.ID+"@"+d.be.Tag) // the catalog's build of it: id@tag
	os.MkdirAll(dir, 0o755)
	bin := "llama-server"
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	os.WriteFile(filepath.Join(dir, bin), []byte("x"), 0o755)
	os.WriteFile(filepath.Join(dir, ".installed"), []byte("sha"), 0o644)
}

func TestSetupNeedFresh(t *testing.T) {
	d := newDiskFix(t)
	n := d.sup.SetupNeed(d.cat, d.hw, "", d.dl)
	wantSteps(t, n, "model server", d.dl.Name, "Python installer", "agent runtime")
	st := n.Steps
	if st[0].Peak != rb(d.be.Size)+unpBackend || st[0].Keep != unpBackend {
		t.Errorf("backend step = %+v", st[0])
	}
	if st[1].Peak != rb(d.dl.Size) || st[1].Keep != rb(d.dl.Size) {
		t.Errorf("model step = %+v", st[1])
	}
	uv := d.cat.UV[0]
	if st[2].Keep != unpUV || st[2].Peak != max(rb(uv.Size)+unpUV, 2*unpUV) {
		t.Errorf("uv step = %+v", st[2])
	}
	if st[3].Peak != agentPeak || st[3].Keep != agentKeep {
		t.Errorf("agent step = %+v", st[3])
	}
	// the running maximum: the agent peak lands on top of everything kept before it
	base := st[0].Keep + st[1].Keep + st[2].Keep
	if want := max(st[0].Peak, st[0].Keep+st[1].Peak, st[0].Keep+st[1].Keep+st[2].Peak, base+st[3].Peak); n.Peak != want {
		t.Errorf("peak = %d, want %d", n.Peak, want)
	}
	if n.Host != 0 || n.Headroom != int64(plan.DiskHeadroomMiB)*mib {
		t.Errorf("host/headroom = %d/%d", n.Host, n.Headroom)
	}
	if want := n.Peak + n.Logs + n.Host + n.Headroom; n.Total != want {
		t.Errorf("total = %d, want %d", n.Total, want)
	}
}

func TestSetupNeedHostFilesAndNoAgent(t *testing.T) {
	d := newDiskFix(t)
	cfg := d.e.Config()
	cfg.Profile.HostFilesMiB = 40
	cfg.Profile.Agent = nil
	cfg.DiskFree = func(string) (int, error) { return 1 << 20, nil }
	sup, err := run.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sup.Shutdown)
	n := sup.SetupNeed(d.cat, d.hw, "", d.dl)
	wantSteps(t, n, "model server", d.dl.Name)
	if n.Host != 40*mib {
		t.Errorf("host = %d, want 40 MiB", n.Host)
	}
}

func TestSetupNeedAgentDiskBytesBeatsCatalog(t *testing.T) {
	d := newDiskFix(t)
	cfg := d.e.Config()
	ag := *cfg.Profile.Agent
	ag.DiskBytes = 50 * mib
	cfg.Profile.Agent = &ag
	cfg.DiskFree = func(string) (int, error) { return 1 << 20, nil }
	sup, _ := run.New(cfg)
	t.Cleanup(sup.Shutdown)
	n := sup.SetupNeed(d.cat, d.hw, "", d.dl)
	last := n.Steps[len(n.Steps)-1]
	if last.What != "agent runtime" || last.Peak != 50*mib || last.Keep != 50*mib {
		t.Errorf("agent step = %+v", last)
	}
}

func TestSetupNeedBackendPresent(t *testing.T) {
	d := newDiskFix(t)
	d.installBackend(t)
	n := d.sup.SetupNeed(d.cat, d.hw, "", d.dl)
	wantSteps(t, n, d.dl.Name, "Python installer", "agent runtime")
	// naming the installed build explicitly counts it as in place too
	wantSteps(t, d.sup.SetupNeed(d.cat, d.hw, d.be.ID, d.dl), d.dl.Name, "Python installer", "agent runtime")
	// an installed uv drops its step
	os.WriteFile(filepath.Join(d.e.Data, "runtime", "uv"), []byte("x"), 0o755)
	if runtime.GOOS == "windows" {
		os.WriteFile(filepath.Join(d.e.Data, "runtime", "uv.exe"), []byte("x"), 0o755)
	}
	wantSteps(t, d.sup.SetupNeed(d.cat, d.hw, "", d.dl), d.dl.Name, "agent runtime")
}

func TestSetupNeedModelInstalledAndPartCredit(t *testing.T) {
	d := newDiskFix(t)
	d.installBackend(t)
	part := filepath.Join(d.e.Data, "models", d.dl.File+".part")
	half := int64(len(d.site.model)) / 2
	os.WriteFile(part, make([]byte, half), 0o644)
	n := d.sup.SetupNeed(d.cat, d.hw, "", d.dl)
	var ms plan.DiskStep
	for _, s := range n.Steps {
		if s.What == d.dl.Name {
			ms = s
		}
	}
	if want := rb(d.dl.Size - half); ms.Peak != want || ms.Keep != want {
		t.Errorf("model step with a half .part = %+v, want %d", ms, want)
	}
	// a .part at least the size gives no credit
	os.WriteFile(part, make([]byte, d.dl.Size+100), 0o644)
	for _, s := range d.sup.SetupNeed(d.cat, d.hw, "", d.dl).Steps {
		if s.What == d.dl.Name && s.Peak != rb(d.dl.Size) {
			t.Errorf("an oversize .part gave credit: %+v", s)
		}
	}
	// installed: no model step at all
	os.Remove(part)
	os.WriteFile(filepath.Join(d.e.Data, "models", d.dl.File), d.site.model, 0o644)
	for _, s := range d.sup.SetupNeed(d.cat, d.hw, "", d.dl).Steps {
		if s.What == d.dl.Name {
			t.Errorf("an installed model still counted: %+v", s)
		}
	}
}

func TestSetupNeedStarterAndTargetTogether(t *testing.T) {
	s := newStage(t)
	cfg := s.e.Config()
	cfg.DiskFree = func(string) (int, error) { return 1 << 20, nil }
	sup, err := run.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(sup.Shutdown)
	hw, _ := sup.Hardware(context.Background(), false)
	st, _ := s.cat.Model("starter")
	tg, _ := s.cat.Model("target")
	both := sup.SetupNeed(s.cat, hw, "", st, tg)
	only := sup.SetupNeed(s.cat, hw, "", tg)
	var names []string
	for _, x := range both.Steps {
		names = append(names, x.What)
	}
	got := strings.Join(names, "|")
	if !strings.Contains(got, "Starter|Target") {
		t.Errorf("starter then target expected in order, got %q", got)
	}
	if want := only.Total + rb(st.Size); both.Total != want {
		t.Errorf("both = %d, want target-only %d + starter %d", both.Total, only.Total, rb(st.Size))
	}
}

func TestLogRoomShrinksByTheLogsOnDisk(t *testing.T) {
	d := newDiskFix(t)
	logs := filepath.Join(d.e.Data, "logs")
	for _, f := range []string{"engine.log", "llama-server.log", "agent.log", lifecycle.StdioLogName} {
		os.Remove(filepath.Join(logs, f))
	}
	if got, want := d.sup.LogRoom(), 3*run.LogCeiling()+lifecycle.StdioLogCeiling(); got != want {
		t.Fatalf("empty logs: room %d, want 3*ceiling+stdio = %d", got, want)
	}
	empty := d.sup.LogRoom()
	os.WriteFile(filepath.Join(logs, "engine.log"), make([]byte, 5<<20), 0o644)
	if got := d.sup.LogRoom(); got != empty-rb(5<<20) {
		t.Errorf("5 MiB engine.log: room %d, want %d", got, empty-rb(5<<20))
	}
	// an odd size rounds up to a block; an old copy counts against its own log
	os.WriteFile(filepath.Join(logs, "engine.log.1"), make([]byte, 5000), 0o644)
	if got, want := d.sup.LogRoom(), empty-rb(5<<20)-rb(5000); got != want {
		t.Errorf("with .1: room %d, want %d", got, want)
	}
	// the stdio log has one old copy
	os.WriteFile(filepath.Join(logs, lifecycle.StdioLogName), make([]byte, 100), 0o644)
	os.WriteFile(filepath.Join(logs, lifecycle.StdioLogName+".1"), make([]byte, 100), 0o644)
	if got, want := d.sup.LogRoom(), empty-rb(5<<20)-rb(5000)-2*4096; got != want {
		t.Errorf("with stdio: room %d, want %d", got, want)
	}
}

func TestLogRoomNeverNegative(t *testing.T) {
	d := newDiskFix(t)
	logs := filepath.Join(d.e.Data, "logs")
	for _, f := range []string{"engine.log", "llama-server.log", "agent.log", lifecycle.StdioLogName} {
		os.Remove(filepath.Join(logs, f))
	}
	empty := d.sup.LogRoom()
	// engine.log and its copies far past the ceiling: it gives 0, the others stay as they were
	big := make([]byte, run.LogCeiling()+(8<<20))
	os.WriteFile(filepath.Join(logs, "engine.log"), big, 0o644)
	got := d.sup.LogRoom()
	if got < 0 {
		t.Fatalf("room %d is negative", got)
	}
	if want := empty - run.LogCeiling(); got != want {
		t.Errorf("an over-ceiling log must give 0 for itself only: room %d, want %d", got, want)
	}
	// the stdio log too
	os.WriteFile(filepath.Join(logs, lifecycle.StdioLogName), make([]byte, lifecycle.StdioLogCeiling()+4096), 0o644)
	if want := empty - run.LogCeiling() - lifecycle.StdioLogCeiling(); d.sup.LogRoom() != want {
		t.Errorf("over-ceiling stdio: room %d, want %d", d.sup.LogRoom(), want)
	}
}

func TestCheckNeedBoundary(t *testing.T) {
	d := newDiskFix(t)
	n := d.sup.SetupNeed(d.cat, d.hw, "", d.dl)
	needMiB := (n.Total + mib - 1) / mib
	d.free.Store(needMiB)
	if err := d.sup.CheckNeed(n, d.dl.Name); err != nil {
		t.Fatalf("at exactly the need (%d MiB): %v", needMiB, err)
	}
	d.free.Store(needMiB - 1)
	err := d.sup.CheckNeed(n, d.dl.Name)
	if err == nil {
		t.Fatalf("one MiB under the need passed")
	}
	if !errors.Is(err, run.ErrNoSpace) {
		t.Errorf("error does not wrap ErrNoSpace: %v", err)
	}
	for _, w := range []string{"needs", "model", d.dl.Name, "free"} {
		if !strings.Contains(err.Error(), w) {
			t.Errorf("%q missing from %q", w, err)
		}
	}
	// nothing left to write: never refused, whatever the free space
	d.free.Store(0)
	if err := d.sup.CheckNeed(plan.NeedOf(nil, 100, 0), "x"); err != nil {
		t.Errorf("a need with no steps was refused: %v", err)
	}
}

func TestInstallRefusedUpFront(t *testing.T) {
	d := newDiskFix(t)
	n := d.sup.SetupNeed(d.cat, d.hw, "", d.dl)
	d.free.Store((n.Total+mib-1)/mib - 2) // 2 MiB under: past what a log line could shift
	err := d.sup.InstallSync(context.Background(), "dl", "")
	if !errors.Is(err, run.ErrNoSpace) {
		t.Fatalf("InstallSync = %v, want ErrNoSpace", err)
	}
	if !strings.Contains(err.Error(), "needs") || !strings.Contains(err.Error(), "model") {
		t.Errorf("refusal text: %v", err)
	}
	if ents, _ := os.ReadDir(filepath.Join(d.e.Data, "backends")); len(ents) != 0 {
		t.Errorf("a refused install wrote a backend: %v", ents)
	}
	if ents, _ := os.ReadDir(filepath.Join(d.e.Data, "models")); len(ents) != 0 {
		t.Errorf("a refused install wrote a model: %v", ents)
	}
}

func TestUnknownFreeSpaceIsSaidAndDoesNotBlock(t *testing.T) {
	d := newDiskFix(t)
	d.free.Store(-1)
	n := d.sup.SetupNeed(d.cat, d.hw, "", d.dl)
	if err := d.sup.CheckNeed(n, d.dl.Name); err != nil {
		t.Fatalf("unreadable free space blocked: %v", err)
	}
	var found bool
	for _, e := range d.sup.Status().Errors {
		if strings.Contains(e, "could not read the free disk space") {
			found = true
		}
	}
	if !found {
		t.Errorf("no error line in status: %v", d.sup.Status().Errors)
	}
}

func TestPlanDiskVerdictUsesSetupNeed(t *testing.T) {
	d := newDiskFix(t)
	d.cat.Models = append(d.cat.Models, catalog.Model{ID: "big", Name: "Big", File: "big.gguf", URL: "http://x.invalid/big", SHA256: "00", Size: 600 * mib, Quality: 90})
	big, _ := d.cat.Model("big")
	need := d.sup.SetupNeed(d.cat, d.hw, "", big)
	d.free.Store(need.Total/mib - 2)
	res, err := d.sup.Plan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var v *plan.Verdict
	for i := range res.Models {
		if res.Models[i].ID == "big" {
			v = &res.Models[i]
		}
	}
	if v == nil {
		t.Fatal("big not in the plan")
	}
	if v.Limit != plan.LimitDisk || v.Fits != "no" {
		t.Errorf("verdict = fits %q limit %q (%s)", v.Fits, v.Limit, v.Why)
	}
	if v.DiskNeed == nil || v.DiskNeed.Total != v.DiskNeedBytes || v.DiskNeedBytes != need.Total {
		t.Errorf("disk need = %+v / %d, want total %d", v.DiskNeed, v.DiskNeedBytes, need.Total)
	}
	if !strings.Contains(v.Why, "not enough free disk") || !strings.Contains(v.Why, "model server") {
		t.Errorf("why = %q", v.Why)
	}
	// with the room, the same model is not refused for disk
	d.free.Store((need.Total + mib - 1) / mib * 2)
	res, _ = d.sup.Plan(context.Background())
	for _, m := range res.Models {
		if m.ID == "big" && m.Limit == plan.LimitDisk {
			t.Errorf("refused for disk with room: %s", m.Why)
		}
	}
}
