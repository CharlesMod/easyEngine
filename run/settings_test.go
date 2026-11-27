//go:build !windows

package run_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

func newSup(t *testing.T, e *testkit.Env) *run.Supervisor {
	t.Helper()
	s, err := run.New(e.Config())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Shutdown)
	return s
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func hasErr(s *run.Supervisor, sub string) bool {
	for _, e := range s.Status().Errors {
		if strings.Contains(e, sub) {
			return true
		}
	}
	return false
}

func TestFirstStartWritesSettingsWhole(t *testing.T) {
	e := testkit.New(t)
	newSup(t, e)
	p := filepath.Join(e.Data, "settings.json")
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("settings.json mode %v, want 0600", fi.Mode().Perm())
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(readFile(t, p)), &m); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"mode", "external", "linger_s", "kv_type", "sampler_policy", "listen", "yield"} {
		if _, ok := m[k]; !ok {
			t.Errorf("settings.json lacks %q", k)
		}
	}
	f, err := settings.Read(p)
	if err != nil || !f.Equal(settings.Defaults()) {
		t.Fatalf("not the defaults: %+v %v", f, err)
	}
}

func TestSettingsMoveOutOfOldStateOnce(t *testing.T) {
	e := testkit.New(t)
	old := `{"mode":"external","external":{"base_url":"http://x/v1","model":"m","api_key":"sekret"},"linger_s":9,
"kv_type":"q8_0","sampler_policy":"fill","model":"m1","backend":"fake","previous":"old","book":{"Entries":{}}}`
	if err := os.WriteFile(filepath.Join(e.Data, "state.json"), []byte(old), 0o600); err != nil {
		t.Fatal(err)
	}
	s := newSup(t, e)
	f, err := settings.Read(filepath.Join(e.Data, "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	if f.Mode != "external" || f.External.APIKey != "sekret" || f.External.BaseURL != "http://x/v1" || f.LingerS != 9 ||
		f.KVType != "q8_0" || f.SamplerPolicy != "fill" {
		t.Fatalf("settings.json after the split: %+v", f)
	}
	if f.Listen.Port != 47480 || f.Yield.FloorMiB != 128 {
		t.Fatalf("new keys not defaulted: %+v", f)
	}
	var st map[string]json.RawMessage
	json.Unmarshal([]byte(readFile(t, filepath.Join(e.Data, "state.json"))), &st)
	for _, k := range []string{"mode", "external", "linger_s", "kv_type", "sampler_policy"} {
		if _, ok := st[k]; ok {
			t.Errorf("state.json still holds %q", k)
		}
	}
	for _, k := range []string{"model", "backend", "previous", "book"} {
		if _, ok := st[k]; !ok {
			t.Errorf("state.json lost %q", k)
		}
	}
	if got := s.Settings(); got.Mode != "external" || got.LingerS != 9 || got.KVType != "q8_0" || got.SamplerPolicy != "fill" {
		t.Fatalf("engine settings: %+v", got)
	}
	if run.LoadState(e.Data).Model != "m1" {
		t.Fatal("the engine's own state was lost")
	}
}

func TestBadSettingsFileIsKeptReportedAndFixed(t *testing.T) {
	e := testkit.New(t)
	p := filepath.Join(e.Data, "settings.json")
	bad := "{\n  \"mode\": \"local\",\n  \"kv_type\": \"nonsense\"\n}\n"
	os.WriteFile(p, []byte(bad), 0o600)
	s := newSup(t, e)
	if readFile(t, p) != bad {
		t.Fatal("the engine overwrote a file it could not read")
	}
	if !hasErr(s, "settings.json line 3: unknown kv_type") {
		t.Fatalf("errors: %v", s.Status().Errors)
	}
	if s.Settings().KVType != "q4_0" {
		t.Fatalf("not on the defaults: %+v", s.Settings())
	}
	// Broken JSON is reported with its line, and nothing else changes.
	os.WriteFile(p, []byte("{\n  \"mode\": \n"), 0o600)
	s.ReloadSettings()
	if !hasErr(s, "using the previous settings") || !hasErr(s, "line ") {
		t.Fatalf("errors: %v", s.Status().Errors)
	}
	// Fixing it clears the error and takes effect.
	good := settings.Defaults()
	good.LingerS, good.KVType = 5, "q8_0"
	os.WriteFile(p, settings.Encode(good), 0o600)
	s.ReloadSettings()
	if hasErr(s, "settings.json") {
		t.Fatalf("error not cleared: %v", s.Status().Errors)
	}
	if got := s.Settings(); got.LingerS != 5 || got.KVType != "q8_0" {
		t.Fatalf("edit did not take effect: %+v", got)
	}
}

func TestSettingsEditsApplyLiveAndListenWaitsForRestart(t *testing.T) {
	e := testkit.New(t)
	s := newSup(t, e)
	p := filepath.Join(e.Data, "settings.json")
	f := settings.Defaults()
	f.SamplerPolicy, f.LingerS = "off", 3
	f.Listen.Port = 47999
	f.Yield.ToGames = true
	os.WriteFile(p, settings.Encode(f), 0o600)
	s.ReloadSettings()
	if got := s.Settings(); got.SamplerPolicy != "off" || got.LingerS != 3 {
		t.Fatalf("not applied: %+v", got)
	}
	if !hasErr(s, "restart easy-engine to apply the new listen address and yield options") {
		t.Fatalf("errors: %v", s.Status().Errors)
	}
	// Putting them back clears the note.
	f.Listen.Port, f.Yield.ToGames = 47480, false
	os.WriteFile(p, settings.Encode(f), 0o600)
	s.ReloadSettings()
	if hasErr(s, "restart easy-engine") {
		t.Fatalf("errors: %v", s.Status().Errors)
	}
}

func TestPutSettingsWritesTheFileAndKeepsTheRest(t *testing.T) {
	e := testkit.New(t)
	s := newSup(t, e)
	p := filepath.Join(e.Data, "settings.json")
	f := settings.Defaults()
	f.Listen.Port = 47999 // an edit waiting for a restart must survive the engine's own write
	os.WriteFile(p, settings.Encode(f), 0o600)
	s.ReloadSettings()
	in := s.Settings()
	in.LingerS = 12
	in.KVType = "q8_0"
	if err := s.SetSettings(in); err != nil {
		t.Fatal(err)
	}
	got, err := settings.Read(p)
	if err != nil || got.LingerS != 12 || got.KVType != "q8_0" || got.Listen.Port != 47999 {
		t.Fatalf("file after PUT: %+v %v", got, err)
	}
	// The engine's own write comes back as a change that is no change.
	before := readFile(t, p)
	s.ReloadSettings()
	if readFile(t, p) != before || s.Settings().LingerS != 12 {
		t.Fatal("reloading the engine's own write changed something")
	}
	// State.json holds none of it.
	if b, _ := os.ReadFile(filepath.Join(e.Data, "state.json")); strings.Contains(string(b), "linger_s") {
		t.Fatal("state.json holds a setting")
	}
}

func TestPutSettingsSetsAsideABadFile(t *testing.T) {
	e := testkit.New(t)
	p := filepath.Join(e.Data, "settings.json")
	os.WriteFile(p, []byte("not json"), 0o600)
	s := newSup(t, e)
	in := s.Settings()
	in.LingerS = 4
	if err := s.SetSettings(in); err != nil {
		t.Fatal(err)
	}
	m, _ := filepath.Glob(p + ".bad-*")
	if len(m) != 1 || readFile(t, m[0]) != "not json" {
		t.Fatalf("the unreadable file was not kept: %v", m)
	}
	if f, err := settings.Read(p); err != nil || f.LingerS != 4 {
		t.Fatalf("%+v %v", f, err)
	}
	if hasErr(s, "settings.json") {
		t.Fatalf("errors: %v", s.Status().Errors)
	}
}
