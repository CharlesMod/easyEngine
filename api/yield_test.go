package api_test

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/run"
	"hermesengine/yield"
)

// gameOS reports one launcher game while on is set, and nothing else.
type gameOS struct{ on atomic.Bool }

func (g *gameOS) Launches() []yield.Launch {
	if g.on.Load() {
		return []yield.Launch{{PID: 1, Kind: "steam", AppID: "999", Exe: "x"}}
	}
	return nil
}
func (g *gameOS) Openers(func(int) bool) map[int]string { return nil }
func (g *gameOS) Under(int, int) bool                   { return false }
func (g *gameOS) Wait(ctx context.Context, d time.Duration) bool {
	select {
	case <-ctx.Done():
	case <-time.After(d):
	}
	return false
}
func (g *gameOS) Drain() {}
func (g *gameOS) Close() {}

func TestYieldedReply(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	g := &gameOS{}
	cfg := e.Config()
	cfg.Yield = &run.YieldConfig{OS: func() yield.OS { return g }, Every: 50 * time.Millisecond}
	cfg.Yield.Policy.Grace = 200 * time.Millisecond
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	g.on.Store(true)
	testkit.WaitPhase(t, s, "yielded", 5*time.Second)
	for _, path := range []string{"/v1/chat/completions", "/completion", "/v1/models"} {
		c, body, h := do(t, "POST", srv.URL+path, `{}`, nil)
		if c != 503 || h.Get("Retry-After") != "10" || h.Get("Content-Type") != "application/json" {
			t.Fatalf("%s: %d %q %s", path, c, h.Get("Retry-After"), body)
		}
		var j struct {
			Error struct{ Code, Type, Message string }
		}
		if err := json.Unmarshal([]byte(body), &j); err != nil || j.Error.Code != "yielded" || j.Error.Message != "Paused while AppId 999 (x) runs" {
			t.Fatalf("%s: body %s (%v)", path, body, err)
		}
	}
	c, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
	var st map[string]any
	json.Unmarshal([]byte(body), &st)
	y, _ := st["yield"].(map[string]any)
	if c != 200 || st["phase"] != "yielded" || y == nil || y["reason"] != "AppId 999 (x)" || y["kind"] != "game" || y["since"] == "" {
		t.Fatalf("status %d %s", c, body)
	}
	g.on.Store(false)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if c, _, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, nil); c != 200 {
		t.Fatalf("after resume: %d", c)
	}
}
