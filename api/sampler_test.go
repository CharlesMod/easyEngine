package api_test

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"hermesengine/api"
	"hermesengine/catalog"
	"hermesengine/internal/testkit"
	"hermesengine/run"
)

func fp(v float64) *float64 { return &v }
func ip(v int) *int         { return &v }

// samplerEnv is a ready local engine whose model has three distinct profiles.
func samplerEnv(t *testing.T) (*testkit.Env, *run.Supervisor, *httptest.Server) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	off := catalog.SamplerProfile{Temperature: fp(1.0), TopK: ip(64), TopP: fp(0.95), MinP: fp(0), RepeatPenalty: fp(1.0), DryMultiplier: fp(0)}
	e.Cat.Models[0].Sampler = map[string]catalog.SamplerProfile{
		"official":   off,
		"rp":         {Temperature: fp(0.9), TopK: ip(64), TopP: fp(0.95), MinP: fp(0)},
		"structured": {Temperature: fp(0.3), TopK: ip(20), TopP: fp(0.9)},
	}
	e.Cat.Models[0].SamplerMatch = []string{"modelone"}
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv
}

func lastBody(t *testing.T, e *testkit.Env) map[string]any {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(e.Ctl, "lastbody.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("upstream body %q: %v", b, err)
	}
	return m
}

var jsonCT = map[string]string{"Content-Type": "application/json"}

func hdr(extra map[string]string) map[string]string {
	h := map[string]string{"Content-Type": "application/json"}
	for k, v := range extra {
		h[k] = v
	}
	return h
}

// every sampler field a client might send, with non-sampler fields next to them
const wild = `{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":77,"stream":false,
"temperature":0.1,"top_k":5,"top_p":0.5,"min_p":0.2,"typical_p":0.4,"repeat_penalty":1.3,"presence_penalty":0.5,
"frequency_penalty":0.6,"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":-1,
"dry_sequence_breakers":["\n"],"xtc_probability":0.5,"xtc_threshold":0.1,"mirostat":2,"mirostat_tau":5,"mirostat_eta":0.1,
"dynatemp_range":0.5,"dynatemp_exponent":1,"top_n_sigma":2,"samplers":["top_k","temperature"],
"grammar":"root ::= \"a\"","json_schema":{"type":"object"},"stop":["\n\n"],"seed":1234567890123456789,"cache_prompt":true,
"id_slot":0,"n_probs":3,"logit_bias":[[15043,-1]]}`

var nonSampler = []string{"model", "messages", "max_tokens", "stream", "grammar", "json_schema", "stop", "seed", "cache_prompt", "id_slot", "n_probs", "logit_bias"}

func TestEnforceRewritesEveryField(t *testing.T) {
	e, _, srv := samplerEnv(t)
	c, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", wild, jsonCT)
	if c != 200 {
		t.Fatalf("status %d", c)
	}
	if got := h.Get("X-Hermes-Sampler-Applied"); got != "rp;enforce" {
		t.Fatalf("applied header %q", got)
	}
	var sent map[string]json.RawMessage
	json.Unmarshal([]byte(wild), &sent)
	got := lastBody(t, e)
	// rp: temperature 0.9, top_k 64, top_p 0.95, min_p 0; everything else sampler-ish is gone
	want := map[string]any{"temperature": 0.9, "top_k": float64(64), "top_p": 0.95, "min_p": float64(0)}
	for _, k := range catalog.SamplerFields {
		v, present := got[k]
		w, wanted := want[k]
		switch {
		case wanted && (!present || v != w):
			t.Errorf("%s = %v, want %v", k, v, w)
		case !wanted && present:
			t.Errorf("%s = %v should have been removed", k, v)
		}
	}
	for _, k := range nonSampler {
		if _, ok := got[k]; !ok {
			t.Errorf("non-sampler field %q was dropped", k)
		}
	}
	// exact bytes of a large integer survive (no float round trip)
	raw, _ := os.ReadFile(filepath.Join(e.Ctl, "lastbody.json"))
	if !strings.Contains(string(raw), `"seed":1234567890123456789`) {
		t.Errorf("seed changed: %s", raw)
	}
	if got["max_tokens"] != float64(77) || got["id_slot"] != float64(0) || got["cache_prompt"] != true {
		t.Errorf("non-sampler values changed: %v", got)
	}
}

func TestEnforceOfficialAddsDryOff(t *testing.T) {
	e, _, srv := samplerEnv(t)
	do(t, "POST", srv.URL+"/v1/chat/completions", `{"dry_multiplier":0.8,"dry_base":2}`, hdr(map[string]string{"X-Hermes-Sampler": "official"}))
	got := lastBody(t, e)
	if got["dry_multiplier"] != float64(0) || got["repeat_penalty"] != float64(1) || got["temperature"] != float64(1) {
		t.Fatalf("official: %v", got)
	}
	if _, ok := got["dry_base"]; ok {
		t.Fatalf("dry_base survived enforce: %v", got)
	}
}

func TestFillOnlyFillsMissing(t *testing.T) {
	e, s, srv := samplerEnv(t)
	if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"sampler_policy":"fill","linger_s":0}`, csrf); c != 200 || !strings.Contains(b, `"sampler_policy":"fill"`) {
		t.Fatalf("settings %d %s", c, b)
	}
	testkit.WaitPhase(t, s, "ready", 5*time.Second)
	_, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{"temperature":0.1,"top_k":5,"mirostat":2,"max_tokens":9}`, jsonCT)
	if h.Get("X-Hermes-Sampler-Applied") != "rp;fill" {
		t.Fatalf("header %q", h.Get("X-Hermes-Sampler-Applied"))
	}
	got := lastBody(t, e)
	if got["temperature"] != 0.1 || got["top_k"] != float64(5) || got["mirostat"] != float64(2) {
		t.Fatalf("fill overwrote the client's fields: %v", got)
	}
	if got["top_p"] != 0.95 || got["min_p"] != float64(0) {
		t.Fatalf("fill did not add missing fields: %v", got)
	}
	if got["max_tokens"] != float64(9) {
		t.Fatalf("max_tokens: %v", got)
	}
}

func TestOffPassesThrough(t *testing.T) {
	e, s, srv := samplerEnv(t)
	do(t, "PUT", srv.URL+"/engine/v1/settings", `{"sampler_policy":"off","linger_s":0}`, csrf)
	testkit.WaitPhase(t, s, "ready", 5*time.Second)
	_, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", wild, jsonCT)
	if h.Get("X-Hermes-Sampler-Applied") != "none;off" {
		t.Fatalf("header %q", h.Get("X-Hermes-Sampler-Applied"))
	}
	var want, got map[string]any
	json.Unmarshal([]byte(wild), &want)
	got = lastBody(t, e)
	a, _ := json.Marshal(want)
	b, _ := json.Marshal(got)
	if string(a) != string(b) {
		t.Fatalf("off changed the body:\n%s\n%s", a, b)
	}
}

func TestHeaderSelectsProfileAndUnknownFallsBackToRP(t *testing.T) {
	e, s, srv := samplerEnv(t)
	for _, tc := range []struct {
		name, header, applied string
		temp                  float64
		topK                  float64
	}{
		{"structured", "structured", "structured;enforce", 0.3, 20},
		{"official", "OFFICIAL", "official;enforce", 1, 64},
		{"rp", "rp", "rp;enforce", 0.9, 64},
		{"missing", "", "rp;enforce", 0.9, 64},
		{"unknown", "spicy", "rp;enforce", 0.9, 64},
	} {
		h := jsonCT
		if tc.header != "" {
			h = hdr(map[string]string{"X-Hermes-Sampler": tc.header})
		}
		_, _, rh := do(t, "POST", srv.URL+"/v1/chat/completions", `{"temperature":2}`, h)
		got := lastBody(t, e)
		if rh.Get("X-Hermes-Sampler-Applied") != tc.applied || got["temperature"] != tc.temp || got["top_k"] != tc.topK {
			t.Errorf("%s: applied %q body %v", tc.name, rh.Get("X-Hermes-Sampler-Applied"), got)
		}
	}
	logs, _ := s.Logs("engine", 200)
	if !strings.Contains(logs, `unknown profile "spicy"`) {
		t.Fatalf("no log line for the unknown profile:\n%s", logs)
	}
}

func TestOtherCompletionEndpointsAndNonCompletionPaths(t *testing.T) {
	e, _, srv := samplerEnv(t)
	_, _, h := do(t, "POST", srv.URL+"/completion", `{"prompt":"a","temperature":3,"n_predict":4}`, jsonCT)
	got := lastBody(t, e)
	if got["temperature"] != 0.9 || got["n_predict"] != float64(4) || h.Get("X-Hermes-Sampler-Applied") != "rp;enforce" {
		t.Fatalf("/completion: %v %q", got, h.Get("X-Hermes-Sampler-Applied"))
	}
	do(t, "POST", srv.URL+"/v1/completions", `{"prompt":"a","top_p":0.1}`, jsonCT) // fake has no such route: 404 is fine
	_, _, h = do(t, "GET", srv.URL+"/props", "", nil)
	if h.Get("X-Hermes-Sampler-Applied") == "" {
		t.Fatalf("a proxied non-completion response lacks the header")
	}
	// a body that is not a JSON object is forwarded as it came
	do(t, "POST", srv.URL+"/completion", `not json`, jsonCT)
	if b, _ := os.ReadFile(filepath.Join(e.Ctl, "lastbody.json")); string(b) != "not json" {
		t.Fatalf("non-JSON body changed: %q", b)
	}
}

func TestStreamingStillWorksWithEnforce(t *testing.T) {
	e, _, srv := samplerEnv(t)
	t0 := time.Now()
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true,"temperature":5}`))
	req.Header.Set("X-Hermes-Sampler", "structured")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.Header.Get("X-Hermes-Sampler-Applied") != "structured;enforce" {
		t.Fatalf("header %q", resp.Header.Get("X-Hermes-Sampler-Applied"))
	}
	br := bufio.NewReader(resp.Body)
	line, _ := br.ReadString('\n')
	first := time.Since(t0)
	if !strings.HasPrefix(line, "data:") {
		t.Fatalf("first line %q", line)
	}
	io.Copy(io.Discard, br)
	if total := time.Since(t0); first > total/2 {
		t.Fatalf("stream was buffered: first chunk %v of %v", first, total)
	}
	if got := lastBody(t, e); got["temperature"] != 0.3 || got["stream"] != true {
		t.Fatalf("streamed request body: %v", got)
	}
}

func TestStartArgsCarryOfficialDefaults(t *testing.T) {
	e, _, _ := samplerEnv(t)
	starts := e.Starts()
	if len(starts) == 0 {
		t.Fatal("no start recorded")
	}
	line := " " + starts[len(starts)-1] + " "
	for _, w := range []string{" --temp 1 ", " --top-k 64 ", " --top-p 0.95 ", " --min-p 0 ", " --repeat-penalty 1 ", " --dry-multiplier 0 "} {
		if !strings.Contains(line, w) {
			t.Errorf("start args lack%s: %s", w, line)
		}
	}
}

func TestStatusAndSettingsShowPolicy(t *testing.T) {
	_, _, srv := samplerEnv(t)
	_, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
	var st struct {
		Sampler run.SamplerInfo `json:"sampler"`
	}
	if err := json.Unmarshal([]byte(body), &st); err != nil {
		t.Fatal(err)
	}
	if st.Sampler.Policy != "enforce" || !st.Sampler.Active || st.Sampler.Model != "m1" || len(st.Sampler.Profiles) != 3 {
		t.Fatalf("status sampler: %+v", st.Sampler)
	}
	if c, _, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"sampler_policy":"bogus"}`, csrf); c != 400 {
		t.Fatalf("bogus policy: %d", c)
	}
	// a PUT that does not mention the policy keeps it
	do(t, "PUT", srv.URL+"/engine/v1/settings", `{"sampler_policy":"fill"}`, csrf)
	do(t, "PUT", srv.URL+"/engine/v1/settings", `{"linger_s":0}`, csrf)
	_, ss, _ := do(t, "GET", srv.URL+"/engine/v1/settings", "", nil)
	if !strings.Contains(ss, `"sampler_policy":"fill"`) {
		t.Fatalf("settings: %s", ss)
	}
}

func TestExternalKnownAndUnknownModel(t *testing.T) {
	testkit.SkipWindows(t)
	var last atomic.Value
	var applied atomic.Value
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		last.Store(string(b))
		applied.Store(r.Header.Get("X-Hermes-Sampler"))
		io.WriteString(w, `{"choices":[{"message":{"content":"ready"}}]}`)
	}))
	defer up.Close()
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Sampler = map[string]catalog.SamplerProfile{
		"official": {Temperature: fp(1.0), TopK: ip(64)}, "rp": {Temperature: fp(1.0), TopK: ip(64)},
	}
	e.Cat.Models[0].SamplerMatch = []string{"gemma4e4b"}
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	put := func(model string) {
		set := `{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":` + jsq(model) + `},"linger_s":0}`
		if c, b, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", set, csrf); c != 200 {
			t.Fatalf("settings %d %s", c, b)
		}
		testkit.WaitPhase(t, s, "external", 5*time.Second)
	}
	put("google/gemma-4-E4B-it")
	_, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{"temperature":0.2,"max_tokens":3}`, hdr(map[string]string{"X-Hermes-Sampler": "rp"}))
	var got map[string]any
	json.Unmarshal([]byte(last.Load().(string)), &got)
	if got["temperature"] != float64(1) || got["top_k"] != float64(64) || got["max_tokens"] != float64(3) || h.Get("X-Hermes-Sampler-Applied") != "rp;enforce" {
		t.Fatalf("known external model: %v %q", got, h.Get("X-Hermes-Sampler-Applied"))
	}
	if applied.Load() != "" {
		t.Fatalf("the sampler header was forwarded to the player's endpoint")
	}
	put("some-other-model")
	_, _, h = do(t, "POST", srv.URL+"/v1/chat/completions", `{"temperature":0.2}`, jsonCT)
	if last.Load().(string) != `{"temperature":0.2}` || !strings.HasPrefix(h.Get("X-Hermes-Sampler-Applied"), "none;") {
		t.Fatalf("unknown external model was rewritten: %v %q", last.Load(), h.Get("X-Hermes-Sampler-Applied"))
	}
	_, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", nil)
	var st struct{ Sampler run.SamplerInfo }
	json.Unmarshal([]byte(body), &st)
	if st.Sampler.Active || !strings.Contains(st.Sampler.Note, "not one we know") {
		t.Fatalf("status should say so: %+v", st.Sampler)
	}
}

func jsq(s string) string { b, _ := json.Marshal(s); return string(b) }

func TestGameProfileAddsNamedProfile(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Sampler = map[string]catalog.SamplerProfile{"official": {Temperature: fp(1.0)}, "rp": {Temperature: fp(1.0)}}
	e.Prof.Samplers = map[string]catalog.SamplerProfile{"cold": {Temperature: fp(0.1), TopK: ip(10)}}
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	_, _, h := do(t, "POST", srv.URL+"/v1/chat/completions", `{}`, hdr(map[string]string{"X-Hermes-Sampler": "cold"}))
	if got := lastBody(t, e); got["temperature"] != 0.1 || got["top_k"] != float64(10) || h.Get("X-Hermes-Sampler-Applied") != "cold;enforce" {
		t.Fatalf("game profile: %v %q", got, h.Get("X-Hermes-Sampler-Applied"))
	}
}

func TestRewriteSamplerUnit(t *testing.T) {
	prof := catalog.SamplerProfile{Temperature: fp(1.0)}
	out, ch := api.RewriteSampler([]byte(`{"temperature":0.1,"top_p":0.2,"stop":["x"]}`), "enforce", prof)
	if !ch {
		t.Fatal("enforce reported no change")
	}
	var m map[string]any
	json.Unmarshal(out, &m)
	if m["temperature"] != float64(1) || m["stop"] == nil {
		t.Fatalf("%s", out)
	}
	if _, ok := m["top_p"]; ok {
		t.Fatalf("top_p kept: %s", out)
	}
	if o, ch := api.RewriteSampler([]byte(`[1]`), "enforce", prof); ch || string(o) != `[1]` {
		t.Fatal("array body touched")
	}
	_ = url.Values{}
}
