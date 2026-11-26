package api_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/plan"
	"hermesengine/run"
)

var csrfHdr = map[string]string{"Content-Type": "application/json", "X-Hermes-Engine": "1"}

func post(t *testing.T, base, route, body string) (int, string) {
	t.Helper()
	code, out, _ := do(t, "POST", base+"/engine/v1/"+route, body, csrfHdr)
	return code, out
}

func jsonPath(p string) string { b, _ := json.Marshal(map[string]string{"path": p}); return string(b) }

func TestCustomCheck(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	_, srv := serve(t, e, e.Config())
	dir := t.TempDir()
	good := testkit.WriteGGUF(t, dir, "My Model.gguf", true)
	bad := filepath.Join(dir, "bad.gguf")
	os.WriteFile(bad, []byte("this is not a model file at all"), 0o644)
	cut := filepath.Join(dir, "cut.gguf")
	b, _ := os.ReadFile(good)
	os.WriteFile(cut, b[:40], 0o644)

	check := func(p string) plan.CustomCheck {
		code, out := post(t, srv.URL, "custom/check", jsonPath(p))
		if code != 200 {
			t.Fatalf("%d %s", code, out)
		}
		var ck plan.CustomCheck
		if err := json.Unmarshal([]byte(out), &ck); err != nil {
			t.Fatal(err)
		}
		return ck
	}
	ck := check(good)
	if !ck.OK || ck.Arch != "llama" || ck.Quant != "Q4_K_M" || ck.Name != "My Model" || !ck.ChatTemplate || ck.MoE || ck.CtxTrain != 8192 || ck.Params != 256*100+2*(256*256+512*256) || ck.Fits == "" || ck.Known || !ck.Estimated {
		t.Fatalf("good: %+v", ck)
	}
	if ck := check(filepath.Join(dir, "nope.gguf")); ck.OK || ck.Error != "file not found" {
		t.Fatalf("missing: %+v", ck)
	}
	if ck := check(bad); ck.OK || ck.Error != "not a GGUF file" {
		t.Fatalf("bad: %+v", ck)
	}
	if ck := check(cut); ck.OK || !strings.HasPrefix(ck.Error, "header unreadable") {
		t.Fatalf("cut: %+v", ck)
	}
	noTpl := testkit.WriteGGUF(t, dir, "plain.gguf", false)
	if ck := check(noTpl); !ck.OK || ck.ChatTemplate {
		t.Fatalf("no template: %+v", ck)
	}
	if code, _ := post(t, srv.URL, "custom/use", jsonPath(bad)); code != 400 {
		t.Fatalf("use bad = %d", code)
	}
}

func TestCustomUseRestartMissingClear(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	dir := t.TempDir()
	p := testkit.WriteGGUF(t, dir, "mine.gguf", true)
	before, _ := os.ReadFile(p)

	if code, out := post(t, srv.URL, "custom/use", jsonPath(p)); code != 202 || !strings.Contains(out, "started") {
		t.Fatalf("use: %d %s", code, out)
	}
	servedCustom := func(name string) func() bool {
		return func() bool {
			st := s.Status()
			return st.Phase == "ready" && st.Model == "custom:"+name && strings.HasSuffix(strings.Join(e.Starts(), "\n"), "-a custom:"+name)
		}
	}
	testkit.WaitFor(t, 15*time.Second, "custom serving", servedCustom("mine"))
	st := s.Status()
	if st.Phase != "ready" || st.Model != "custom:mine" || st.ModelName != "mine" || st.Custom == nil || st.Custom.Missing || st.Custom.Path != p {
		t.Fatalf("status %+v custom %+v", st, st.Custom)
	}
	starts := strings.Join(e.Starts(), "\n")
	if !strings.Contains(starts, p) || !strings.Contains(starts, "--jinja") || !strings.Contains(starts, "custom:mine") {
		t.Fatalf("starts: %s", starts)
	}
	if !st.Sampler.Active || st.Sampler.Profiles["official"].TopK == nil || *st.Sampler.Profiles["official"].TopK != 64 {
		t.Fatalf("sampler %+v", st.Sampler)
	}
	pl, err := s.Plan(t.Context())
	if err != nil || pl.Custom == nil || !pl.Custom.OK {
		t.Fatalf("plan custom %+v %v", pl.Custom, err)
	}
	// state round trip
	if sv := run.LoadState(e.Data); sv.Custom == nil || sv.Custom.Path != p || sv.Model != "custom" || sv.CustomPrev != "m1" {
		t.Fatalf("state %+v", sv.Custom)
	}

	// file goes away, engine restarts
	os.Remove(p)
	s.Stop()
	s.Start()
	st = s.Status()
	if st.Phase != "error" || st.Message != "Model file not found: "+p || st.Custom == nil || !st.Custom.Missing {
		t.Fatalf("missing: %s %q %+v", st.Phase, st.Message, st.Custom)
	}
	n := len(e.Starts())
	time.Sleep(400 * time.Millisecond)
	if st := s.Status(); st.Phase != "error" || len(e.Starts()) != n {
		t.Fatalf("retry storm: %s %d->%d", st.Phase, n, len(e.Starts()))
	}
	if code, _ := post(t, srv.URL, "custom/clear", "{}"); code != 200 {
		t.Fatalf("clear = %d", code)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if st := s.Status(); st.Model != "m1" || st.Custom != nil {
		t.Fatalf("after clear: %s %+v", st.Model, st.Custom)
	}
	// the user's file is untouched by use and clear (rewrite it to compare)
	p2 := testkit.WriteGGUF(t, dir, "mine.gguf", true)
	after, _ := os.ReadFile(p2)
	if string(before) != string(after) {
		t.Fatal("fixture differs")
	}
}

func TestCustomUseMissingRecoversWithNewPath(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s, srv := serve(t, e, e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	dir := t.TempDir()
	p := testkit.WriteGGUF(t, dir, "a.gguf", true)
	post(t, srv.URL, "custom/use", jsonPath(p))
	testkit.WaitFor(t, 15*time.Second, "custom serving", func() bool {
		st := s.Status()
		return st.Phase == "ready" && strings.HasSuffix(strings.Join(e.Starts(), "\n"), "-a custom:a")
	})
	os.Remove(p)
	s.Stop()
	s.Start()
	if s.Phase() != "error" {
		t.Fatalf("phase %s", s.Phase())
	}
	p2 := testkit.WriteGGUF(t, dir, "b.gguf", true)
	if code, _ := post(t, srv.URL, "custom/use", jsonPath(p2)); code != 202 {
		t.Fatal(code)
	}
	testkit.WaitFor(t, 15*time.Second, "custom serving", func() bool {
		st := s.Status()
		return st.Phase == "ready" && strings.HasSuffix(strings.Join(e.Starts(), "\n"), "-a custom:b")
	})
}

func TestStateWithoutCustomLoads(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "state.json"), []byte(`{"mode":"local","model":"m1","backend":"fake","kv_type":"q4_0","book":{"Entries":{}}}`), 0o644)
	st := run.LoadState(dir)
	if st.Custom != nil || st.Model != "m1" || st.CustomPrev != "" {
		t.Fatalf("%+v", st)
	}
}
