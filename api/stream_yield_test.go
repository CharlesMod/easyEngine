package api

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/probe/gpusense"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/yield"
)

// gameOS is the other programs on a pretend machine: a launcher's game that comes on the
// launcher's word, and nothing else.
type gameOS struct {
	mu   sync.Mutex
	runs []yield.Launch
	word chan struct{}
}

func (g *gameOS) Launches() []yield.Launch {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]yield.Launch(nil), g.runs...)
}
func (g *gameOS) Openers(func(int) bool) map[int]string { return map[int]string{} }
func (g *gameOS) Under(int, int) bool                   { return false }
func (g *gameOS) Adapter(string)                        {}
func (g *gameOS) Close()                                {}
func (g *gameOS) Watch() yield.Words {
	return yield.Words{Launches: g.word, Stop: func() {}}
}

// A reply cut by a real yield ends with the "yielded" event: the yield stands in the status
// while its drain runs, before the phase says so.
func TestStreamCutByARealYield(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_STREAM_GATE"] = "1"
	g := &gameOS{word: make(chan struct{}, 1)}
	baseline := make(chan struct{}, 1)
	cfg := e.Config()
	cfg.Sense = func(ctx context.Context, ours []int, host int) gpusense.Picture {
		return gpusense.Picture{Adapters: []gpusense.Adapter{
			{Index: 0, Vendor: "nvidia", VendorID: 0x10de, Name: "Fake GPU", DedicatedMiB: 16384, UUID: "GPU-aaaa"},
		}, Sources: map[string]string{"fake": "ok"}}
	}
	y := &run.YieldConfig{Except: []string{"294"}, Every: time.Hour, Drain: 200 * time.Millisecond,
		OS:    func() yield.OS { return g },
		Procs: func(context.Context, []gpusense.Adapter) []gpusense.Process { return nil },
		Looked: func(engine int, _ []yield.Use, _ []yield.Launch) {
			if engine == 0 {
				select {
				case baseline <- struct{}{}:
				default:
				}
			}
		}}
	y.Policy = yield.Config{Grace: 400 * time.Millisecond, Steady: 600 * time.Millisecond, Floor: 128}
	cfg.Yield = y
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	// network timeout: a broken engine fails the test instead of hanging it
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	select {
	case <-baseline: // the reactor has taken its first look; a game now is a change
	case <-ctx.Done():
		t.Fatal("the yield reactor never took its first look")
	}

	a := &API{S: s}
	a.heard = s.Heard
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()

	req, _ := http.NewRequestWithContext(ctx, "POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", "yield-req-1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	// event reads the next data line's payload (comments and blank lines skipped).
	event := func() (string, error) {
		for {
			l, err := br.ReadString('\n')
			if strings.HasPrefix(l, "data:") {
				return strings.TrimSpace(strings.TrimPrefix(l, "data:")), nil
			}
			if err != nil {
				return "", err
			}
		}
	}
	first, err := event()
	if err != nil || !strings.Contains(first, "t0 ") {
		t.Fatalf("first chunk %q, %v", first, err)
	}

	// The reply is held open by the gate; a game starts, and the engine yields for real.
	g.mu.Lock()
	g.runs = []yield.Launch{{PID: 777, Kind: "steam", AppID: "999", Exe: "x"}}
	g.mu.Unlock()
	select {
	case g.word <- struct{}{}:
	default:
	}

	term, err := event()
	if err != nil {
		t.Fatalf("no terminal event: %v", err)
	}
	var ev struct {
		Error struct {
			Code      string `json:"code"`
			Type      string `json:"type"`
			Message   string `json:"message"`
			Status    int    `json:"status"`
			RequestID string `json:"request_id"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(term), &ev); err != nil {
		t.Fatalf("terminal event %q: %v", term, err)
	}
	if x := ev.Error; x.Code != "yielded" || x.Type != "unavailable_error" || x.Status != 503 ||
		!strings.HasPrefix(x.Message, "Paused while ") || x.RequestID != "yield-req-1" {
		t.Fatalf("terminal event %q", term)
	}
	if l, err := event(); err != io.EOF {
		t.Fatalf("after the terminal event: %q, %v (want a clean end)", l, err)
	}
}
