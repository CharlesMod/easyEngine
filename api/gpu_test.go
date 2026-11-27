package api_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
	"github.com/CharlesMod/easyEngine/run"
)

func gpuEnv(t *testing.T, calls *atomic.Int32) (*testkit.Env, *run.Supervisor, string) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.Sense = func(ctx context.Context, ours []int, host int) gpusense.Picture {
		n := calls.Add(1) // each reading carries its number, so a report says which reading it is
		return gpusense.Picture{
			Adapters: []gpusense.Adapter{
				{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UsedMiB: 1300},
				{Index: 1, Vendor: "intel", VendorID: 0x8086, Name: "Intel(R) Iris(R) Xe Graphics", DedicatedMiB: 128, SharedMiB: 8000, UsedMiB: 100, Integrated: true},
			},
			Processes: []gpusense.Process{{PID: 4242, Name: "browser", Adapter: 0, DedicatedMiB: 1300, Kind: "graphics"}},
			Sources:   map[string]string{"fake": "ok", "call": fmt.Sprint(n)},
		}
	}
	cfg.Others = func(context.Context, probe.GPU, []int) (int, error) { return 0, nil }
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv.URL
}

func TestGPURoute(t *testing.T) {
	var calls atomic.Int32
	_, _, url := gpuEnv(t, &calls)
	c, body, _ := do(t, "GET", url+"/engine/v1/gpu?refresh=1", "", nil)
	if c != 200 {
		t.Fatalf("gpu %d %s", c, body)
	}
	var rep gpusense.GPUReport
	if err := json.Unmarshal([]byte(body), &rep); err != nil {
		t.Fatal(err)
	}
	if len(rep.Adapters) != 2 || len(rep.Processes) != 1 || rep.Sources["fake"] != "ok" || rep.EngineAdapter != 0 {
		t.Fatalf("report %s", body)
	}
	if rep.Reserve.OthersMiB != 1300 || rep.Reserve.OthersBasis != "processes" || rep.OthersNote == "" || rep.FreeForModelMiB != 16384-1300-655 {
		t.Fatalf("reserve %+v free %d note %q", rep.Reserve, rep.FreeForModelMiB, rep.OthersNote)
	}
	// Cached for 5 s: a plain GET answers the reading it already has, refresh=1 a new one. The
	// watchdog senses through the same hook on its own ticks, so the count of calls cannot tell
	// the two apart; the number the reading carries can.
	reading := func(query string) string {
		t.Helper()
		c, body, _ := do(t, "GET", url+"/engine/v1/gpu"+query, "", nil)
		var r gpusense.GPUReport
		if err := json.Unmarshal([]byte(body), &r); c != 200 || err != nil || r.Sources["call"] == "" {
			t.Fatalf("gpu%s: %d %v %s", query, c, err, body)
		}
		return r.Sources["call"]
	}
	first := rep.Sources["call"]
	if first == "" {
		t.Fatalf("report names no reading: %s", body)
	}
	if got := reading(""); got != first {
		t.Fatalf("a plain GET sensed again (reading %s -> %s)", first, got)
	}
	if got := reading("?refresh=1"); got == first {
		t.Fatal("refresh=1 did not sense")
	}
	if c, _, _ := do(t, "POST", url+"/engine/v1/gpu", "", csrf); c != 405 {
		t.Fatalf("POST gpu: %d", c)
	}
}

func TestHostRoute(t *testing.T) {
	var calls atomic.Int32
	_, _, url := gpuEnv(t, &calls)
	h := url + "/engine/v1/host"
	if c, _, _ := do(t, "POST", h, `{"reserve_mib": 900}`, nil); c != 403 {
		t.Fatalf("no CSRF header: %d", c)
	}
	if c, _, _ := do(t, "GET", h, "", nil); c != 405 {
		t.Fatalf("GET: %d", c)
	}
	for _, bad := range []string{`not json`, `{"pid": -3}`, `{"pid": 2147483000}`, `{"reserve_mib": -1}`, `{"adapter": {"memory_mib": 5}}`, `{"adapter": {"name": "x", "memory_mib": -5}}`} {
		if c, _, _ := do(t, "POST", h, bad, csrf); c != 400 {
			t.Errorf("%s: %d, want 400", bad, c)
		}
	}
	// the host renders on the integrated adapter: no reserve on the card
	body := fmt.Sprintf(`{"pid": %d, "adapter": {"name": "Intel(R) Iris(R) Xe Graphics", "vendor_id": 32902, "memory_mib": 128}, "reserve_mib": 1700}`, os.Getpid())
	if c, b, _ := do(t, "POST", h, body, csrf); c != 204 || b != "" {
		t.Fatalf("host: %d %q", c, b)
	}
	_, out, _ := do(t, "GET", url+"/engine/v1/gpu", "", nil) // the POST dropped the cached reading
	var rep gpusense.GPUReport
	json.Unmarshal([]byte(out), &rep)
	if rep.HostAdapter != 1 || rep.HostIntegrated != "yes" || rep.Reserve.HostMiB != 0 || rep.Reserve.HostBasis != "other-adapter" {
		t.Fatalf("after host: %s", out)
	}
	// and on the card: the posted estimate stands
	do(t, "POST", h, `{"adapter": {"name": "Fake GPU", "vendor_id": 4318}}`, csrf)
	_, out, _ = do(t, "GET", url+"/engine/v1/gpu", "", nil)
	json.Unmarshal([]byte(out), &rep)
	if rep.HostAdapter != 0 || rep.Reserve.HostMiB != 1700 || rep.Reserve.HostBasis != "estimate" {
		t.Fatalf("host on the card: %s", out)
	}
	// the next watchdog tick re-reads: the status carries it
	testkit.WaitFor(t, 5*time.Second, "status gpu follows the host", func() bool {
		_, b, _ := do(t, "GET", url+"/engine/v1/status", "", nil)
		var st struct{ GPU *gpusense.GPUSummary }
		json.Unmarshal([]byte(b), &st)
		return st.GPU != nil && st.GPU.Reserve.HostMiB == 1700
	})
}

func TestPlanCarriesGPU(t *testing.T) {
	var calls atomic.Int32
	_, _, url := gpuEnv(t, &calls)
	_, body, _ := do(t, "GET", url+"/engine/v1/plan", "", nil)
	var p struct {
		GPU *gpusense.GPUReport `json:"gpu"`
	}
	json.Unmarshal([]byte(body), &p)
	if p.GPU == nil || len(p.GPU.Adapters) != 2 || p.GPU.Reserve.OthersMiB != 1300 || len(p.GPU.Processes) != 1 {
		t.Fatalf("plan gpu: %s", body)
	}
}
