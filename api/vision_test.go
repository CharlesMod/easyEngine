package api_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

const visionProjFile, visionProjSize = "mm1.gguf", 2048

// visionEnv is a ready engine whose catalog model carries a projector; with file the projector
// is in the models folder at its catalog size.
func visionEnv(t *testing.T, file bool, mut func(*run.Config)) (*testkit.Env, *run.Supervisor, *httptest.Server) {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Projector = &catalog.Projector{File: visionProjFile, URL: "http://invalid.example/mm1", SHA256: strings.Repeat("ab", 32), Size: visionProjSize, LoadMiB: 100}
	if file {
		if err := os.WriteFile(filepath.Join(e.Data, "models", visionProjFile), bytes.Repeat([]byte{1}, visionProjSize), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cfg := e.Config()
	if mut != nil {
		mut(&cfg)
	}
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv
}

func getJSON(t *testing.T, url string, hdr map[string]string) map[string]json.RawMessage {
	t.Helper()
	c, b, _ := do(t, "GET", url, "", hdr)
	m := map[string]json.RawMessage{}
	if c != 200 || json.Unmarshal([]byte(b), &m) != nil {
		t.Fatalf("GET %s: %d %s", url, c, b)
	}
	return m
}

func statusVision(t *testing.T, base string, hdr map[string]string) (bool, string) {
	t.Helper()
	m := getJSON(t, base+"/engine/v1/status", hdr)
	var v bool
	var n string
	if err := json.Unmarshal(m["vision"], &v); err != nil {
		t.Fatalf("status vision: %s %v", m["vision"], err)
	}
	if err := json.Unmarshal(m["vision_note"], &n); err != nil {
		t.Fatalf("status vision_note: %s %v", m["vision_note"], err)
	}
	return v, n
}

func diskVision(t *testing.T, e *testkit.Env) bool {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(e.Data, "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Vision bool `json:"vision"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f.Vision
}

// 1. vision is in the settings API: PUT applies at once and is written to settings.json, a PUT
// without it leaves it, a non-bool is refused.
func TestVisionInTheSettingsAPI(t *testing.T) {
	e, _, srv := readyEnv(t)
	url := srv.URL + "/engine/v1/settings"
	get := func() any { return getJSON2(t, url)["vision"] }
	if v := get(); v != false {
		t.Fatalf("default vision %v", v)
	}
	if c, b, _ := do(t, "PUT", url, `{"vision":true}`, csrf); c != 200 {
		t.Fatalf("PUT vision true: %d %s", c, b)
	}
	if v := get(); v != true || !diskVision(t, e) {
		t.Fatalf("after true: GET %v disk %v", v, diskVision(t, e))
	}
	if b, _ := os.ReadFile(filepath.Join(e.Data, "settings.json")); !strings.Contains(string(b), `"vision": true`) {
		t.Fatalf("settings.json: %s", b)
	}
	if c, b, _ := do(t, "PUT", url, `{"linger_s":9}`, csrf); c != 200 {
		t.Fatalf("PUT linger: %d %s", c, b)
	}
	if v := get(); v != true || !diskVision(t, e) {
		t.Fatalf("a PUT without vision moved it: %v", v)
	}
	for _, bad := range []string{`{"vision":"yes"}`, `{"vision":1}`, `{"vision":null,"x":`, `{"vision":[true]}`} {
		if c, _, _ := do(t, "PUT", url, bad, csrf); c != 400 {
			t.Fatalf("PUT %s: %d", bad, c)
		}
	}
	if v := get(); v != true {
		t.Fatalf("a refused PUT moved vision: %v", v)
	}
	if c, b, _ := do(t, "PUT", url, `{"vision":false}`, csrf); c != 200 {
		t.Fatalf("PUT vision false: %d %s", c, b)
	}
	if v := get(); v != false || diskVision(t, e) {
		t.Fatalf("after false: GET %v disk %v", v, diskVision(t, e))
	}
}

func getJSON2(t *testing.T, url string) map[string]any {
	t.Helper()
	c, body, _ := do(t, "GET", url, "", nil)
	var m map[string]any
	if c != 200 || json.Unmarshal([]byte(body), &m) != nil {
		t.Fatalf("GET %s %d %s", url, c, body)
	}
	return m
}

// 2. a paired token may not change settings; the master key from the LAN may.
func TestVisionSettingNeedsTheMasterKeyOffLoopback(t *testing.T) {
	pe, paired := scopeEnv(t)
	url := pe.srv.URL + "/engine/v1/settings"
	c, b, _ := do(t, "PUT", url, `{"vision":true}`, with(csrf, bearer(paired)))
	if !notAllowed(c, b) {
		t.Fatalf("paired PUT: %d %s", c, b)
	}
	if v, _ := statusVision(t, pe.srv.URL, bearer(pe.master)); v {
		t.Fatal("refused PUT changed vision")
	}
	if c, b, _ := do(t, "PUT", url, `{"vision":true}`, with(csrf, bearer(pe.master))); c != 200 {
		t.Fatalf("master PUT: %d %s", c, b)
	}
	c, b, _ = do(t, "GET", url, "", bearer(pe.master))
	if c != 200 || !strings.Contains(b, `"vision":true`) && !strings.Contains(b, `"vision": true`) {
		t.Fatalf("master GET settings: %d %s", c, b)
	}
}

// 3. an edit of settings.json's vision applies through ReloadSettings.
func TestVisionEditedInSettingsFileApplies(t *testing.T) {
	e, s, srv := readyEnv(t)
	p := settings.Path(e.Data)
	f, err := settings.Read(p)
	if err != nil {
		t.Fatal(err)
	}
	f.Vision = true
	if err := os.WriteFile(p, settings.Encode(f), 0o600); err != nil {
		t.Fatal(err)
	}
	s.ReloadSettings()
	if v := getJSON2(t, srv.URL+"/engine/v1/settings")["vision"]; v != true {
		t.Fatalf("settings after edit: %v", v)
	}
	f.Vision = false
	if err := os.WriteFile(p, settings.Encode(f), 0o600); err != nil {
		t.Fatal(err)
	}
	s.ReloadSettings()
	if v := getJSON2(t, srv.URL+"/engine/v1/settings")["vision"]; v != false {
		t.Fatalf("settings after edit back: %v", v)
	}
}

// 4. status and hello carry vision; the paired status has vision and vision_note, not
// vision_progress; the model without a projector says so; nothing carries "look".
func TestVisionInStatusAndHello(t *testing.T) {
	_, _, srv := readyEnv(t)
	if v, n := statusVision(t, srv.URL, nil); v || n != "" {
		t.Fatalf("default status vision %v note %q", v, n)
	}
	m := getJSON(t, srv.URL+"/engine/v1/status", nil)
	if _, ok := m["vision_progress"]; !ok {
		t.Fatal("full status lacks vision_progress")
	}
	// wanted from the start (serve --vision), the model has no projector
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.Vision = true
	s, srv3 := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	v, n := statusVision(t, srv3.URL, nil)
	if v || !strings.Contains(n, "has no vision projector") {
		t.Fatalf("no projector: vision %v note %q", v, n)
	}

	pe, paired := scopeEnv(t)
	hm := getJSON(t, pe.srv.URL+"/engine/v1/hello", nil)
	if string(hm["vision"]) != "false" {
		t.Fatalf("hello vision %s", hm["vision"])
	}
	pm := getJSON(t, pe.srv.URL+"/engine/v1/status", bearer(paired))
	for _, k := range []string{"vision", "vision_note"} {
		if _, ok := pm[k]; !ok {
			t.Errorf("paired status lacks %q", k)
		}
	}
	if _, ok := pm["vision_progress"]; ok {
		t.Error("paired status has vision_progress")
	}
	for name, mm := range map[string]map[string]json.RawMessage{"hello": hm, "paired status": pm, "status": m} {
		if _, ok := mm["look"]; ok {
			t.Errorf("%s carries look", name)
		}
	}
}

// 5. an attach whose profile says vision: true is accepted, local and remote, and the want shows.
func TestVisionAttachProfileLocal(t *testing.T) {
	_, _, srv := visionEnv(t, true, nil)
	body := fmt.Sprintf(`{"pid":%d,"name":"v1","profile":{"vision":true}}`, os.Getpid())
	if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", body, csrf); c != 200 {
		t.Fatalf("attach: %d %s", c, b)
	}
	testkit.WaitFor(t, 20*time.Second, "vision loaded", func() bool {
		v, _ := statusVision(t, srv.URL, nil)
		return v
	})
	if _, n := statusVision(t, srv.URL, nil); n != "" {
		t.Fatalf("note while loaded: %q", n)
	}
}

func TestVisionAttachProfileLocalWithoutProjectorFile(t *testing.T) {
	_, _, srv := visionEnv(t, false, nil)
	body := fmt.Sprintf(`{"pid":%d,"name":"v2","profile":{"vision":true}}`, os.Getpid())
	if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", body, csrf); c != 200 {
		t.Fatalf("attach: %d %s", c, b)
	}
	v, n := statusVision(t, srv.URL, nil)
	if v || n == "" {
		t.Fatalf("vision %v note %q: the want should show with a note", v, n)
	}
}

func TestVisionAttachProfileRemote(t *testing.T) {
	_, s, _ := visionEnv(t, true, nil)
	_, rsrv, _ := remoteSrv(t, s)
	h := openHold(t, rsrv, `{"hold":true,"name":"r1","profile":{"vision":true}}`, as(1))
	if h.status != 200 {
		t.Fatalf("remote attach: %d %v", h.status, h.first)
	}
	testkit.WaitFor(t, 20*time.Second, "vision loaded", func() bool {
		v, _ := statusVision(t, rsrv.URL, tok(1))
		return v
	})
}
