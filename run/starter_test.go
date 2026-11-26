package run_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"hermesengine/catalog"
	"hermesengine/internal/testkit"
	"hermesengine/run"
)

// stage is a first-run fixture: a starter model (small) and a target model (big) on a fake host
// whose target download can be gated, corrupted or counted.
type stage struct {
	*site
	hs       *httptest.Server
	starter  []byte
	target   []byte
	mu       sync.Mutex
	reqs     []string // "starter" / "target" in arrival order
	ranges   []string // Range header of each target request
	gate     chan struct{}
	once     sync.Once
	badFirst int32 // number of target requests to answer with corrupt bytes
	free     int32 // fake free disk MiB once the starter is on disk
}

func newStage(t *testing.T) *stage {
	s := &stage{site: newSite(t), gate: make(chan struct{}), free: 1 << 20}
	s.starter = []byte(strings.Repeat("starter!", 2000))
	s.target = []byte(strings.Repeat("thetarget", 40000))
	mux := http.NewServeMux()
	mux.HandleFunc("/starter.gguf", func(w http.ResponseWriter, r *http.Request) {
		s.note("starter", "")
		w.Write(s.starter)
	})
	mux.HandleFunc("/target.gguf", func(w http.ResponseWriter, r *http.Request) {
		s.note("target", r.Header.Get("Range"))
		select {
		case <-s.gate:
		case <-r.Context().Done():
			return
		}
		if atomic.AddInt32(&s.badFirst, -1) >= 0 {
			w.Write(bytes.Repeat([]byte("x"), len(s.target)))
			return
		}
		http.ServeContent(w, r, "t", time.Time{}, bytes.NewReader(s.target))
	})
	s.hs = httptest.NewServer(mux)
	t.Cleanup(s.hs.Close)
	s.cat.Models = []catalog.Model{
		{ID: "starter", Name: "Starter", File: "starter.gguf", URL: s.hs.URL + "/starter.gguf", SHA256: sum(s.starter), Size: int64(len(s.starter)), Quality: 40},
		{ID: "target", Name: "Target", File: "target.gguf", URL: s.hs.URL + "/target.gguf", SHA256: sum(s.target), Size: int64(len(s.target)), Quality: 100},
	}
	s.e.Cat = s.cat
	return s
}

func (s *stage) note(what, rng string) {
	s.mu.Lock()
	s.reqs = append(s.reqs, what)
	if what == "target" {
		s.ranges = append(s.ranges, rng)
	}
	s.mu.Unlock()
}

func (s *stage) order() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.reqs, ",")
}

func (s *stage) release() {
	s.once.Do(func() { close(s.gate) })
}

// start builds the supervisor with a fake disk that reports s.free once the starter is on disk.
func (s *stage) start(t *testing.T) *run.Supervisor {
	cfg := s.e.Config()
	cfg.UpgradeBackoff, cfg.UpgradeBackoffMax = 60*time.Millisecond, 120*time.Millisecond
	cfg.SwapDrain = 5 * time.Second
	cfg.DiskFree = func(dir string) (int, error) {
		if _, err := os.Stat(filepath.Join(dir, "models", "starter.gguf")); err == nil {
			return int(atomic.LoadInt32(&s.free)), nil
		}
		return 1 << 20, nil
	}
	sup := s.e.StartSup(cfg)
	sup.SetPort(4242)
	return sup
}

func waitStatus(t *testing.T, sup *run.Supervisor, what string, f func(run.Status) bool) run.Status {
	t.Helper()
	var last run.Status
	testkit.WaitFor(t, 30*time.Second, what, func() bool { last = sup.Status(); return f(last) })
	return last
}

func TestStarterThenTargetSwap(t *testing.T) {
	s := newStage(t)
	sup := s.start(t)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	// the starter serves while the target is still held back by the gate
	st := waitStatus(t, sup, "starter serving with the target downloading", func(st run.Status) bool {
		return st.Phase == "ready" && st.Model == "starter" && st.ModelStage == "target_downloading"
	})
	if st.ModelTier != "starter" || !st.ModelFallback || st.ModelNote == "" {
		t.Errorf("starter tier fields: %+v", st)
	}
	if st.Upgrade == nil || st.Upgrade.Target != "target" || st.Upgrade.State != "downloading" {
		t.Fatalf("upgrade = %+v", st.Upgrade)
	}
	if o := s.order(); !strings.HasPrefix(o, "starter,target") {
		t.Errorf("download order %q: the starter must be requested first", o)
	}
	if got := len(s.e.Starts()); got != 1 {
		t.Errorf("%d server starts before the target arrived, want 1", got)
	}
	if run.LoadState(s.e.Data).Target != "target" {
		t.Errorf("state.json lost the pending target")
	}
	s.release()
	st = waitStatus(t, sup, "swap to the target", func(st run.Status) bool {
		return st.Phase == "ready" && st.Model == "target" && st.ModelStage == "target_serving"
	})
	if st.ModelTier != "full" || st.ModelFallback || st.Upgrade.State != "done" {
		t.Errorf("after swap: %+v", st)
	}
	if got := len(s.e.Starts()); got != 2 {
		t.Errorf("%d server starts, want 2 (starter then target with a fresh fit)", got)
	}
	if ss := run.LoadState(s.e.Data); ss.Model != "target" || ss.Target != "" {
		t.Errorf("state after swap: %+v", ss)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "target.gguf")); err != nil {
		t.Error(err)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "starter.gguf")); !os.IsNotExist(err) {
		t.Errorf("the starter we downloaded should be deleted after the swap (stat err %v)", err)
	}
	if ss := run.LoadState(s.e.Data); ss.StarterOwned != "" {
		t.Errorf("starter_owned not cleared: %+v", ss)
	}
}

func TestStarterAlreadyOnDiskIsKeptAfterSwap(t *testing.T) {
	s := newStage(t)
	os.MkdirAll(filepath.Join(s.e.Data, "models"), 0o755)
	if err := os.WriteFile(filepath.Join(s.e.Data, "models", "starter.gguf"), s.starter, 0o644); err != nil {
		t.Fatal(err)
	}
	sup := s.start(t)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "starter serving", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "starter" })
	s.release()
	waitStatus(t, sup, "swap to the target", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "target" })
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "starter.gguf")); err != nil {
		t.Errorf("a model that was on disk before must be kept: %v", err)
	}
}

func TestFallbackModelIsNeverTheStarter(t *testing.T) {
	s := newStage(t)
	tiny := []byte(strings.Repeat("tiny", 100))
	s.cat.Models = append(s.cat.Models, catalog.Model{ID: "tiny", Name: "Tiny", File: "tiny.gguf", URL: s.hs.URL + "/tiny.gguf",
		SHA256: sum(tiny), Size: int64(len(tiny)), Quality: 10, Fallback: true, FallbackNote: "small"})
	s.e.Cat = s.cat
	sup := s.start(t)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "the non-fallback starter serving", func(st run.Status) bool {
		return st.Phase == "ready" && st.Model == "starter" && st.ModelTier == "starter"
	})
	s.release()
}

func TestSwapWaitsForRequestsInFlight(t *testing.T) {
	s := newStage(t)
	sup := s.start(t)
	sup.Install("target", "")
	waitStatus(t, sup, "starter ready", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "starter" })
	done := sup.Begin() // a request is in flight
	s.release()
	waitStatus(t, sup, "swap to start draining", func(st run.Status) bool { return st.ModelStage == "swapping" })
	// new requests get the 503 right away, the old server is still up for the one in flight
	r := sup.Route()
	if r.Kind != "unavailable" || !strings.Contains(r.Message, "full model") {
		t.Errorf("route during swap = %+v", r)
	}
	time.Sleep(400 * time.Millisecond)
	if st := sup.Status(); st.ModelStage != "swapping" || st.Model != "starter" {
		t.Fatalf("swapped with a request in flight: %+v", st)
	}
	if got := len(s.e.Starts()); got != 1 {
		t.Fatalf("old server was stopped with a request in flight (%d starts)", got)
	}
	done()
	waitStatus(t, sup, "swap done", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "target" })
	if r := sup.Route(); r.Kind != "local" {
		t.Errorf("route after swap = %+v", r)
	}
}

func TestTargetFailureKeepsStarterAndRetries(t *testing.T) {
	s := newStage(t)
	s.release()
	atomic.StoreInt32(&s.badFirst, 2) // two corrupt downloads, then a good one
	sup := s.start(t)
	sup.Install("target", "")
	sawRetry := false
	testkit.WaitFor(t, 30*time.Second, "retry then swap", func() bool {
		st := sup.Status()
		if st.Upgrade != nil && st.Upgrade.State == "retrying" {
			sawRetry = true
			if st.Model != "starter" || st.ModelTier != "starter" || st.Upgrade.Error == "" {
				t.Errorf("while retrying: %+v", st)
			}
		}
		return st.Phase == "ready" && st.Model == "target"
	})
	if !sawRetry {
		t.Error("never saw the retrying state")
	}
	if n := strings.Count(s.order(), "target"); n != 3 {
		t.Errorf("target requested %d times, want 3 (2 corrupt + 1 good): %s", n, s.order())
	}
	if got := len(s.e.Starts()); got != 2 {
		t.Errorf("starter must keep serving through failures: %d server starts, want 2", got)
	}
}

func TestTargetResumesPartialDownload(t *testing.T) {
	s := newStage(t)
	s.release()
	part := filepath.Join(s.e.Data, "models", "target.gguf.part")
	os.MkdirAll(filepath.Dir(part), 0o755)
	half := len(s.target) / 2
	os.WriteFile(part, s.target[:half], 0o644)
	sup := s.start(t)
	sup.Install("target", "")
	waitStatus(t, sup, "swap", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "target" })
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.ranges) == 0 || s.ranges[0] != "bytes="+strconv.Itoa(half)+"-" {
		t.Errorf("target request ranges %v, want a resume from byte %d", s.ranges, half)
	}
}

func TestTargetNeverFitsStaysOnStarter(t *testing.T) {
	s := newStage(t)
	s.release()
	atomic.StoreInt32(&s.free, 100) // 100 MiB: the starter fits, the target (needs 512+) does not
	sup := s.start(t)
	sup.Install("target", "")
	st := waitStatus(t, sup, "no_space", func(st run.Status) bool {
		return st.Phase == "ready" && st.Upgrade != nil && st.Upgrade.State == "no_space"
	})
	if st.Phase != "ready" || st.Model != "starter" || st.ModelStage != "target_no_space" || !strings.Contains(st.Upgrade.Error, "disk space") {
		t.Errorf("no space: %+v upgrade %+v", st, st.Upgrade)
	}
	if strings.Contains(s.order(), "target") {
		t.Error("the target was requested despite no room")
	}
	atomic.StoreInt32(&s.free, 1<<20) // the player frees space: the next check goes ahead
	waitStatus(t, sup, "swap after space appeared", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "target" })
}

func TestTargetAlreadyOnDiskHasNoStarter(t *testing.T) {
	s := newStage(t)
	s.release()
	os.MkdirAll(filepath.Join(s.e.Data, "models"), 0o755)
	os.WriteFile(filepath.Join(s.e.Data, "models", "target.gguf"), s.target, 0o644)
	sup := s.start(t)
	sup.Install("target", "")
	st := waitStatus(t, sup, "ready", func(st run.Status) bool { return st.Phase == "ready" })
	if st.Model != "target" || st.ModelTier != "full" || st.ModelStage != "" || st.Upgrade != nil {
		t.Errorf("status = %+v", st)
	}
	if o := s.order(); o != "" {
		t.Errorf("downloads %q, want none", o)
	}
}

func TestNoStarterWhenNotUseful(t *testing.T) {
	// the target is the smallest model there is (a CPU-only machine's pick), or only a little
	// bigger than the next one down: no starter
	for name, mk := range map[string]func(s *stage){
		"only model":  func(s *stage) { s.cat.Models = s.cat.Models[1:] },
		"close sizes": func(s *stage) { s.cat.Models[0].Size = s.cat.Models[1].Size * 3 / 4 },
	} {
		t.Run(name, func(t *testing.T) {
			s := newStage(t)
			s.release()
			mk(s)
			sup := s.start(t)
			sup.Install("target", "")
			st := waitStatus(t, sup, "ready", func(st run.Status) bool { return st.Phase == "ready" })
			if st.Model != "target" || st.Upgrade != nil || st.ModelTier != "full" {
				t.Errorf("status = %+v", st)
			}
			if strings.Contains(s.order(), "starter") {
				t.Error("a starter was downloaded")
			}
		})
	}
}

func TestStarterStatusJSONFields(t *testing.T) {
	s := newStage(t)
	sup := s.start(t)
	sup.Install("target", "")
	waitStatus(t, sup, "downloading", func(st run.Status) bool { return st.ModelStage == "target_downloading" })
	b, _ := json.Marshal(sup.Status())
	var m map[string]any
	json.Unmarshal(b, &m)
	for _, k := range []string{"model_tier", "model_stage", "upgrade", "model_fallback", "model_note"} {
		if _, ok := m[k]; !ok {
			t.Errorf("status JSON lacks %q", k)
		}
	}
	s.release()
}
