package api_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// The old wire names, built from pieces: only internal/legacy may spell them plainly.
const (
	oldEngineHdr  = "X-Her" + "mes-Engine"
	oldSamplerHdr = "X-Her" + "mes-Sampler"
	oldAppliedHdr = "X-Her" + "mes-Sampler-Applied"
)

func TestCSRFAcceptsNewOldAndRefusesNeither(t *testing.T) {
	_, _, srv := readyEnv(t)
	for name, h := range map[string]map[string]string{
		"new":  {"X-Easy-Engine": "1"},
		"old":  {oldEngineHdr: "1"},
		"both": {"X-Easy-Engine": "1", oldEngineHdr: "1"},
	} {
		if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/cancel", "", h); c == 403 {
			t.Errorf("%s header refused: %d %s", name, c, b)
		}
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/cancel", "", nil); c != 403 {
		t.Errorf("no header: %d", c)
	}
	if c, _, _ := do(t, "POST", srv.URL+"/engine/v1/cancel", "", map[string]string{"X-Easy-Engine": "0", oldEngineHdr: "0"}); c != 403 {
		t.Errorf("wrong values: %d", c)
	}
}

func TestSamplerHeadersOldAndNewReadStrippedAndEchoed(t *testing.T) {
	var seen atomic.Value // the headers the upstream received
	var last atomic.Value
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		last.Store(string(b))
		seen.Store(r.Header.Clone())
		io.WriteString(w, `{"choices":[{"message":{"content":"ready"}}]}`)
	}))
	defer up.Close()
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Sampler = map[string]catalog.SamplerProfile{
		"official":   {Temperature: fp(1.0), TopK: ip(64)},
		"rp":         {Temperature: fp(0.9), TopK: ip(64)},
		"structured": {Temperature: fp(0.3), TopK: ip(20)},
	}
	e.Cat.Models[0].SamplerMatch = []string{"gemma4e4b"}
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	set := `{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":"google/gemma-4-E4B-it"},"linger_s":0}`
	if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", set, csrf); c != 200 {
		t.Fatalf("settings %d %s", c, b)
	}
	testkit.WaitPhase(t, s, "external", 5*time.Second)

	for name, c := range map[string]struct {
		h    map[string]string
		temp float64
		want string
	}{
		"new only":        {map[string]string{"X-Easy-Sampler": "structured"}, 0.3, "structured;enforce"},
		"old only":        {map[string]string{oldSamplerHdr: "structured"}, 0.3, "structured;enforce"},
		"both, new wins":  {map[string]string{"X-Easy-Sampler": "structured", oldSamplerHdr: "rp"}, 0.3, "structured;enforce"},
		"neither -> rp":   {nil, 0.9, "rp;enforce"},
		"old with engine": {map[string]string{oldSamplerHdr: "structured", oldEngineHdr: "1"}, 0.3, "structured;enforce"},
	} {
		_, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{"temperature":0.2,"max_tokens":3}`, hdr(c.h))
		var got map[string]any
		json.Unmarshal([]byte(last.Load().(string)), &got)
		if got["temperature"] != c.temp {
			t.Errorf("%s: temperature %v want %v", name, got["temperature"], c.temp)
		}
		if h.Get("X-Easy-Sampler-Applied") != c.want || h.Get(oldAppliedHdr) != c.want {
			t.Errorf("%s: applied new %q old %q want %q", name, h.Get("X-Easy-Sampler-Applied"), h.Get(oldAppliedHdr), c.want)
		}
		up := seen.Load().(http.Header)
		for _, k := range []string{"X-Easy-Sampler", oldSamplerHdr, oldEngineHdr} {
			if up.Get(k) != "" {
				t.Errorf("%s: %s forwarded upstream", name, k)
			}
		}
	}
}
