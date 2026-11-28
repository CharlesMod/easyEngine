package run_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
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
// fail is 1 the release answers 503; while gate is open (not nil) a request waits for it.
type pinEnv struct {
	e     *testkit.Env
	fail  atomic.Int32
	hits  atomic.Int32
	mu    sync.Mutex
	gate  chan struct{}
	smoke string // the broken pin's smoke log (brokenPin)
}

// newPinEnv: the pin is the fake server itself (accel cpu: its smoke test is --version).
// brokenPin: the pin is a CUDA build whose --list-devices fails, logging each run to p.smoke.
func newPinEnv(t *testing.T, brokenPin bool) *pinEnv {
	testkit.SkipWindows(t)
	p := &pinEnv{e: testkit.New(t)}
	p.e.WatchFile(filepath.Join(p.e.Ctl, "release-hits")) // WaitFor looks again when the release is asked
	p.e.InstallFake()
	os.WriteFile(filepath.Join(p.e.Data, "backends", "fake", ".installed"), []byte("oldsha"), 0o644)
	bin, _ := os.ReadFile(testkit.FakeLlama(t))
	accel := "cpu"
	if brokenPin {
		accel = "cuda"
		p.smoke = filepath.Join(p.e.Ctl, "smoke.log")
		t.Setenv("FAKE_SMOKE_LOG", p.smoke)
		bin = []byte("#!/bin/sh\necho \"$*\" >> \"$FAKE_SMOKE_LOG\"\n[ \"$1\" = --version ] && { echo broken cuda; exit 0; }\necho 'no CUDA device'; exit 1\n")
	}
	arch := tgz(t, map[string][]byte{"llama-" + pinTag + "/llama-server": bin})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.hits.Add(1)
		os.WriteFile(filepath.Join(p.e.Ctl, "release-hits"), []byte(fmt.Sprint(p.hits.Load())), 0o644) // a word for WaitFor
		p.mu.Lock()
		g := p.gate
		p.mu.Unlock()
		if g != nil {
			select {
			case <-g:
			case <-r.Context().Done():
				return
			}
		}
		if p.fail.Load() == 1 {
			http.Error(w, "release unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Write(arch)
	}))
	t.Cleanup(srv.Close)
	p.e.Cat.Llama = []catalog.LlamaCpp{{Tag: pinTag, Repo: "CharlesMod/llama.cpp", Patched: true, Assets: []catalog.Asset{
		{ID: "fake", OS: runtime.GOOS, Arch: runtime.GOARCH, Accel: accel, URL: srv.URL + "/fake.tar.gz", SHA256: sum(arch), Size: int64(len(arch))},
	}}}
	return p
}

func (p *pinEnv) legacy() string { return filepath.Join(p.e.Data, "backends", "fake") }
func (p *pinEnv) pinned() string { return filepath.Join(p.e.Data, "backends", "fake@"+pinTag) }
func (p *pinEnv) smokeRuns() int { return len(lines(p.smoke)) }

func next(s *run.Supervisor) *run.BuildNext {
	if b := s.Status().Build; b != nil {
		return b.Next
	}
	return nil
}

func moveIs(s *run.Supervisor, state string) bool {
	n := next(s)
	return n != nil && n.State == state
}

func exists(p string) bool { _, err := os.Lstat(p); return err == nil }

// An engine running a build other than the pin fetches the pin beside it (state written before
// the fact says staged), runs the pin's own binary from the next load, and removes the old build
// once the new one has served ready.
func TestBuildMovesToThePinAtTheNextStart(t *testing.T) {
	p := newPinEnv(t, false)
	p.e.Llama["FAKELLAMA_VMM"] = "on (1 device)"
	cfg := p.e.Config()
	cfg.StartSleeping = true
	s := p.e.StartSup(cfg)
	testkit.WaitFor(t, 15*time.Second, "the pin staged beside the old build", func() bool {
		if !moveIs(s, "staged") {
			return false
		}
		// the fact never runs ahead of state.json
		if st := run.LoadState(p.e.Data); st.NextBuild != pinTag {
			t.Fatalf("the fact says staged while state.json holds next_build %q", st.NextBuild)
		}
		return true
	})
	b := s.Status().Build
	if b.Backend != "fake" || b.Tag != "" || b.Patched || b.Rung != -1 || b.Pin != pinTag || !strings.Contains(b.Why, "before the engine recorded") {
		t.Fatalf("build before the start: %+v", b)
	}
	if st := run.LoadState(p.e.Data); st.Build != "" || st.NextBuild != pinTag {
		t.Fatalf("state %+v", st)
	}
	if !exists(p.legacy()) || !exists(filepath.Join(p.pinned(), ".unpacked")) || exists(filepath.Join(p.pinned(), ".installed")) {
		t.Fatal("staged: the old build stays, the pin is unpacked and not yet smoke-tested")
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
	if bin := testkit.Field(starts[len(starts)-1], "bin"); !strings.Contains(bin, string(filepath.Separator)+"fake@"+pinTag+string(filepath.Separator)) {
		t.Fatalf("the server launched is %q, not the pin's binary", bin)
	}
	if !exists(filepath.Join(p.pinned(), ".installed")) {
		t.Fatal("the pin was not smoke-tested at the load")
	}
	testkit.WaitFor(t, 15*time.Second, "the old build removed after the pin served ready", func() bool { return !exists(p.legacy()) })
	if st := run.LoadState(p.e.Data); st.Build != pinTag || st.NextBuild != "" {
		t.Fatalf("state after %+v", st)
	}
}

// A fetch that fails leaves the engine serving on what it has, says why, and is tried again on a
// person's or client's word, never on a request at the door.
func TestBuildMoveFailureKeepsServingAndWaitsForAWord(t *testing.T) {
	p := newPinEnv(t, false)
	p.fail.Store(1)
	s := p.e.StartSup(p.e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	testkit.WaitFor(t, 15*time.Second, "the failed move said", func() bool { return moveIs(s, "waiting") })
	n := next(s)
	if !strings.Contains(n.Error, "503") || n.Tag != pinTag || n.Attempt != 1 {
		t.Fatalf("move %+v", n)
	}
	if s.Phase() != "ready" || s.Status().Build.Tag != "" {
		t.Fatalf("not serving on the old build: %s %+v", s.Phase(), s.Status().Build)
	}
	for i := 0; i < 3; i++ { // requests at the door are not words for the move
		end := s.Begin()
		end()
	}
	s.Nudge("test: a client's word") // still failing: one more attempt, and only one
	testkit.WaitFor(t, 15*time.Second, "the second attempt", func() bool {
		n := next(s)
		return n != nil && n.State == "waiting" && n.Attempt == 2
	})
	if n := next(s); n.Attempt != 2 || p.hits.Load() != 2 {
		t.Fatalf("attempts %d, release asked %d times: requests at the door were taken for words", n.Attempt, p.hits.Load())
	}
	p.fail.Store(0)
	s.Nudge("test: the release is back")
	testkit.WaitFor(t, 15*time.Second, "the move after the word", func() bool { return moveIs(s, "staged") })
	if s.Status().Build.Tag != "" || s.Phase() != "ready" {
		t.Fatal("the running server was replaced without a load")
	}
}

// The pin fails its smoke test: the move never ran it while staging, the load runs it once (the
// card free), fails, starts on the old build and says why; another load does not run it again,
// nothing is downloaded again; a person's word makes the next load try the smoke test again,
// still without a download.
func TestBuildSmokeFailureAtTheLoadNeverDownloadsAgain(t *testing.T) {
	p := newPinEnv(t, true)
	cfg := p.e.Config()
	cfg.StartSleeping = true
	s := p.e.StartSup(cfg)
	testkit.WaitFor(t, 15*time.Second, "the pin staged", func() bool { return moveIs(s, "staged") })
	if p.smokeRuns() != 0 {
		t.Fatalf("the move ran the build while staging it: %v", lines(p.smoke))
	}
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	b := s.Status().Build
	if b.Tag != "" || b.Next == nil || b.Next.State != "failed" || !strings.Contains(b.Next.Error, "--list-devices") {
		t.Fatalf("build after the failed smoke test: %+v %+v", b, b.Next)
	}
	runs, hits := p.smokeRuns(), p.hits.Load()
	if runs == 0 || hits != 1 {
		t.Fatalf("smoke runs %d, downloads %d", runs, hits)
	}
	for i := 0; i < 2; i++ { // loads and requests with nothing changed: no smoke test, no download
		if err := s.Stop(); err != nil {
			t.Fatal(err)
		}
		end := s.Begin()
		end()
		if err := s.Start(); err != nil {
			t.Fatal(err)
		}
		testkit.WaitPhase(t, s, "ready", 15*time.Second)
	}
	if p.smokeRuns() != runs || p.hits.Load() != hits || !moveIs(s, "failed") {
		t.Fatalf("tried again with nothing changed: smoke runs %d (was %d), downloads %d", p.smokeRuns(), runs, p.hits.Load())
	}
	s.Nudge("test: a person's word")
	if err := s.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if p.smokeRuns() <= runs || p.hits.Load() != hits {
		t.Fatalf("after a word: smoke runs %d (was %d), downloads %d (was %d)", p.smokeRuns(), runs, p.hits.Load(), hits)
	}
	if s.Status().Build.Tag != "" || !exists(p.legacy()) || !exists(filepath.Join(p.pinned(), ".unpacked")) {
		t.Fatal("the old build must run and stay; the staged one stays unpacked")
	}
}

// A planned restart with the pin staged saves no context: the context key carries the build, so
// the next start (another build) could lay nothing back. The start after it runs the pin.
func TestBuildStagedSavesNoContextAndRunsAtTheRestart(t *testing.T) {
	p := newPinEnv(t, false)
	p.fail.Store(1) // the move waits until the slot holds tokens
	p.e.WatchFile(filepath.Join(p.e.Ctl, "slots.log"))
	s := p.e.StartSup(p.e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	testkit.WaitFor(t, 15*time.Second, "the move waits", func() bool { return moveIs(s, "waiting") })
	ask(t, s, 0, 4000)
	p.fail.Store(0)
	s.Nudge("test: the release is back")
	testkit.WaitFor(t, 15*time.Second, "the pin staged", func() bool { return moveIs(s, "staged") })
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

// A Start in the middle of a remove of the downloads starts no move: nothing is left on disk and
// no next_build stands.
func TestBuildStartDuringRemoveLeavesNothing(t *testing.T) {
	p := newPinEnv(t, false)
	gate := make(chan struct{})
	p.mu.Lock()
	p.gate = gate
	p.mu.Unlock()
	cfg := p.e.Config()
	cfg.StartSleeping = true
	s := p.e.StartSup(cfg)
	testkit.WaitFor(t, 15*time.Second, "the move fetching", func() bool { return p.hits.Load() >= 1 && moveIs(s, "downloading") })
	if err := s.RemoveDownloads(); err != nil {
		t.Fatal(err)
	}
	s.Start()
	p.mu.Lock()
	p.gate = nil
	p.mu.Unlock()
	close(gate)
	testkit.WaitFor(t, 15*time.Second, "the remove done", func() bool {
		st := s.Status()
		return st.Phase != "removing" && st.Backend == "" && st.Build == nil
	})
	s.Start() // and one after it: nothing selected, nothing to move
	left, _ := os.ReadDir(filepath.Join(p.e.Data, "backends"))
	dl, _ := os.ReadDir(filepath.Join(p.e.Data, "downloads"))
	if len(left) != 0 || len(dl) != 0 {
		t.Fatalf("left after the remove: backends %v downloads %v", left, dl)
	}
	if st := run.LoadState(p.e.Data); st.NextBuild != "" || st.Build != "" || st.Backend != "" {
		t.Fatalf("state after the remove %+v", st)
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

// An install job on a data dir made by main (backends/cuda-test with its marker, state.json with
// no build) never changes the working build: the pin's CUDA build fails its smoke test, and the
// engine still runs its old CUDA build; nothing was deleted, no other accelerator fetched.
func TestInstallKeepsTheLegacyBuildWhenThePinFails(t *testing.T) {
	for _, sync := range []bool{false, true} {
		t.Run(map[bool]string{false: "Install", true: "InstallSync"}[sync], func(t *testing.T) {
			s := newSite(t)
			var cpuHits atomic.Int32
			cuda, cpu := s.cat.LlamaCpp.Assets[0], s.cat.LlamaCpp.Assets[1]
			mux := http.NewServeMux()
			mux.HandleFunc("/cpu2.tar.gz", func(w http.ResponseWriter, r *http.Request) { cpuHits.Add(1) })
			extra := httptest.NewServer(mux)
			t.Cleanup(extra.Close)
			cpu.URL = extra.URL + "/cpu2.tar.gz"
			s.cat.LlamaCpp = catalog.LlamaCpp{}
			s.cat.Llama = []catalog.LlamaCpp{{Tag: "b2-easy1", Patched: true, Assets: []catalog.Asset{cuda, cpu}}}
			// main's layout: the working fake server installed as cuda-test, no tag
			legacy := filepath.Join(s.e.Data, "backends", "cuda-test")
			testkit.CopyFile(t, testkit.FakeLlama(t), filepath.Join(legacy, "llama-server"), 0o755)
			os.WriteFile(filepath.Join(legacy, ".installed"), []byte("mainsha"), 0o644)
			os.WriteFile(filepath.Join(s.e.Data, "state.json"), []byte(`{"model":"","backend":"cuda-test"}`), 0o600)
			s.e.Prof.Agent = nil
			sup := s.sup(t)
			if sync {
				if err := sup.InstallSync(context.Background(), "dl", ""); err != nil {
					t.Fatal(err)
				}
			} else if err := sup.Install("dl", ""); err != nil {
				t.Fatal(err)
			}
			// the move stages the pin after the job; the job's own restart may already have tried it
			testkit.WaitFor(t, 30*time.Second, "served on the old build with the pin staged", func() bool {
				return sup.Phase() == "ready" && (moveIs(sup, "staged") || moveIs(sup, "failed"))
			})
			if err := sup.Stop(); err != nil { // a load: the staged pin is smoke-tested here and fails
				t.Fatal(err)
			}
			if err := sup.Start(); err != nil {
				t.Fatal(err)
			}
			testkit.WaitPhase(t, sup, "ready", 30*time.Second)
			st := run.LoadState(s.e.Data)
			if st.Backend != "cuda-test" || st.Build != "" || st.Model != "dl" {
				t.Fatalf("state %+v", st)
			}
			if !moveIs(sup, "failed") || cpuHits.Load() != 0 || !exists(filepath.Join(legacy, ".installed")) {
				t.Fatalf("move %+v, cpu fetched %d times, legacy kept %v", next(sup), cpuHits.Load(), exists(legacy))
			}
			if bin := testkit.Field(s.e.Starts()[len(s.e.Starts())-1], "bin"); !strings.Contains(bin, string(filepath.Separator)+"cuda-test"+string(filepath.Separator)) {
				t.Fatalf("launched %q, not the legacy build", bin)
			}
		})
	}
}
