package run_test

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

const pinTag = "b9-easy1"

// pinEnv is the fake backend installed the old way (backends/fake, no tag recorded) and a catalog
// whose pin for that backend is the patched build b9-easy1, served from a local release. While
// fail is 1 the release answers 503.
type pinEnv struct {
	e    *testkit.Env
	fail atomic.Int32
	hits atomic.Int32
}

func newPinEnv(t *testing.T) *pinEnv {
	testkit.SkipWindows(t)
	p := &pinEnv{e: testkit.New(t)}
	p.e.InstallFake()
	os.WriteFile(filepath.Join(p.e.Data, "backends", "fake", ".installed"), []byte("oldsha"), 0o644)
	bin, _ := os.ReadFile(testkit.FakeLlama(t))
	arch := tgz(t, map[string][]byte{"llama-" + pinTag + "/llama-server": bin})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.hits.Add(1)
		if p.fail.Load() == 1 {
			http.Error(w, "release unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Write(arch)
	}))
	t.Cleanup(srv.Close)
	p.e.Cat.Llama = []catalog.LlamaCpp{{Tag: pinTag, Repo: "CharlesMod/llama.cpp", Patched: true, Assets: []catalog.Asset{
		{ID: "fake", OS: runtime.GOOS, Arch: runtime.GOARCH, Accel: "cpu", URL: srv.URL + "/fake.tar.gz", SHA256: sum(arch), Size: int64(len(arch))},
	}}}
	return p
}

func (p *pinEnv) legacy() string { return filepath.Join(p.e.Data, "backends", "fake") }
func (p *pinEnv) pinned() string { return filepath.Join(p.e.Data, "backends", "fake@"+pinTag) }

func next(s *run.Supervisor) *run.BuildNext {
	if b := s.Status().Build; b != nil {
		return b.Next
	}
	return nil
}

// An engine running a build other than the pin installs the pin beside it, says so, runs it from
// the next start, and removes the old build once the new one has served ready.
func TestBuildMovesToThePinAtTheNextStart(t *testing.T) {
	p := newPinEnv(t)
	p.e.Llama["FAKELLAMA_VMM"] = "on (1 device)"
	cfg := p.e.Config()
	cfg.StartSleeping = true
	s := p.e.StartSup(cfg)
	testkit.WaitFor(t, 15*time.Second, "the pin installed beside the old build", func() bool {
		n := next(s)
		return n != nil && n.State == "ready"
	})
	b := s.Status().Build
	if b.Backend != "fake" || b.Tag != "" || b.Patched || b.Rung != -1 || b.Pin != pinTag || !strings.Contains(b.Why, "before the engine recorded") {
		t.Fatalf("build before the start: %+v", b)
	}
	if st := run.LoadState(p.e.Data); st.Build != "" || st.NextBuild != pinTag {
		t.Fatalf("state %+v", st)
	}
	if _, err := os.Stat(p.legacy()); err != nil {
		t.Fatal("the old build went before the new one served")
	}
	if _, err := os.Stat(filepath.Join(p.pinned(), ".installed")); err != nil {
		t.Fatal("the pin is not installed under backends/fake@" + pinTag)
	}
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	b = s.Status().Build
	if b.Tag != pinTag || !b.Patched || b.Rung != 0 || b.Why != "" || b.Next != nil {
		t.Fatalf("build after the start: %+v", b)
	}
	if b.VMM != "on (1 device)" {
		t.Errorf("vmm = %q: the patched server's line was not read", b.VMM)
	}
	if f, ok := s.Hub().Fact("build", ""); !ok || !strings.Contains(string(f.Body), `"tag":"`+pinTag+`"`) {
		t.Errorf("build fact %s", f.Body)
	}
	starts := p.e.Starts()
	if l := starts[len(starts)-1]; !strings.Contains(l, "fake@"+pinTag) && !strings.Contains(l, "llama-"+pinTag) {
		t.Logf("start line (the fake does not log its path): %s", l)
	}
	if _, err := os.Stat(p.legacy()); !os.IsNotExist(err) {
		t.Fatalf("the old build is still on disk after the pin served ready: %v", err)
	}
	if st := run.LoadState(p.e.Data); st.Build != pinTag || st.NextBuild != "" {
		t.Fatalf("state after %+v", st)
	}
}

// A download that fails leaves the engine serving on what it has, says why, and is tried again
// on a word (here a nudge), not on a clock.
func TestBuildMoveFailureKeepsServingAndWaitsForAWord(t *testing.T) {
	p := newPinEnv(t)
	p.fail.Store(1)
	s := p.e.StartSup(p.e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	testkit.WaitFor(t, 15*time.Second, "the failed move said", func() bool {
		n := next(s)
		return n != nil && n.State == "waiting"
	})
	n := next(s)
	if !strings.Contains(n.Error, "503") || n.Tag != pinTag {
		t.Fatalf("move %+v", n)
	}
	if s.Phase() != "ready" || s.Status().Build.Tag != "" {
		t.Fatalf("not serving on the old build: %s %+v", s.Phase(), s.Status().Build)
	}
	hits := p.hits.Load()
	p.fail.Store(0)
	s.Nudge("test: the release is back")
	testkit.WaitFor(t, 15*time.Second, "the move after the word", func() bool {
		n := next(s)
		return n != nil && n.State == "ready"
	})
	if p.hits.Load() <= hits {
		t.Fatal("the release was not asked again after the word")
	}
	if s.Status().Build.Tag != "" || s.Phase() != "ready" {
		t.Fatal("the running server was replaced without a planned start")
	}
}

// A planned restart with the pin staged saves no context: the context key carries the build, so
// the next start (another build) could lay nothing back. The start after it runs the pin.
func TestBuildStagedSavesNoContextAndRunsAtTheRestart(t *testing.T) {
	p := newPinEnv(t)
	p.fail.Store(1) // the move waits until the slot holds tokens
	p.e.WatchFile(filepath.Join(p.e.Ctl, "slots.log"))
	s := p.e.StartSup(p.e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	ask(t, s, 0, 4000)
	p.fail.Store(0)
	s.Nudge("test: the release is back")
	testkit.WaitFor(t, 15*time.Second, "the pin staged", func() bool {
		n := next(s)
		return n != nil && n.State == "ready"
	})
	from := testkit.Mark(t, s, "before the restart")
	shrinkNow(t, p.e, s)
	l := contextsSince(t, s, from)
	if len(whatsOf(l, "saved", -1)) != 0 || logCount(p.e, "save") != 0 {
		t.Fatalf("a slot was saved for a start of another build: %+v", l)
	}
	ns := whatsOf(l, "not_saved", -1)
	if len(ns) != 1 || !strings.Contains(ns[0].Why, "another llama build ("+pinTag+")") {
		t.Fatalf("not_saved: %+v", l)
	}
	if s.Status().Build.Tag != pinTag {
		t.Fatalf("the restart did not run the pin: %+v", s.Status().Build)
	}
	if c, _ := ask(t, s, 0, 4100); c != 0 {
		t.Fatalf("cache_n = %d on another build", c)
	}
}

// A patched build that fails its smoke test is never replaced by stock of the same accelerator:
// the stock CUDA build is not even fetched; the next accelerator's build is what follows.
func TestPatchedFailureNeverFallsToStockOfTheSameAccel(t *testing.T) {
	s := newSite(t)
	var stockCuda atomic.Int32
	mux := http.NewServeMux()
	mux.HandleFunc("/stock-cuda.tar.gz", func(w http.ResponseWriter, r *http.Request) { stockCuda.Add(1) })
	extra := httptest.NewServer(mux)
	t.Cleanup(extra.Close)
	cuda, cpu := s.cat.LlamaCpp.Assets[0], s.cat.LlamaCpp.Assets[1]
	s.cat.LlamaCpp = catalog.LlamaCpp{}
	stock := cuda
	stock.ID, stock.URL = "cuda-stock", extra.URL+"/stock-cuda.tar.gz"
	s.cat.Llama = []catalog.LlamaCpp{
		{Tag: "b2-easy1", Patched: true, Assets: []catalog.Asset{cuda}}, // the broken cuda build
		{Tag: "b2", Assets: []catalog.Asset{stock, cpu}},
	}
	sup := s.sup(t)
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, sup, "ready", 30*time.Second)
	st := run.LoadState(s.e.Data)
	if st.Backend != "cpu-test" || st.Build != "b2" {
		t.Fatalf("state %+v", st)
	}
	if stockCuda.Load() != 0 {
		t.Fatal("the stock CUDA build was fetched after the patched one failed")
	}
	b := sup.Status().Build
	if b.Patched || b.Rung != 1 || !strings.Contains(b.Why, "no patched build for") {
		t.Fatalf("build %+v", b)
	}
}
