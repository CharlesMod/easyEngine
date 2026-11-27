package api_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// keep_context is in the settings API: GET shows it, PUT of it applies at once and is written to
// settings.json, a PUT of another field keeps it, a bad value is refused.
func TestKeepContextInTheSettingsAPI(t *testing.T) {
	e, s, srv := readyEnv(t)
	get := func() map[string]any {
		c, body, _ := do(t, "GET", srv.URL+"/engine/v1/settings", "", nil)
		var m map[string]any
		if c != 200 || json.Unmarshal([]byte(body), &m) != nil {
			t.Fatalf("GET settings %d %s", c, body)
		}
		return m
	}
	if v := get()["keep_context"]; v != "on" || s.KeepContext() != "on" {
		t.Fatalf("default keep_context %v", v)
	}
	if c, body, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"keep_context":"always"}`, csrf); c != 200 {
		t.Fatalf("PUT keep_context always: %d %s", c, body)
	}
	if v := get()["keep_context"]; v != "always" || s.KeepContext() != "always" {
		t.Fatalf("after PUT always: %v live %v", v, s.KeepContext())
	}
	if c, body, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"linger_s":7}`, csrf); c != 200 {
		t.Fatalf("PUT linger: %d %s", c, body)
	}
	if m := get(); m["keep_context"] != "always" || m["linger_s"] != float64(7) {
		t.Fatalf("another field's PUT moved keep_context: %v", m)
	}
	b, err := os.ReadFile(filepath.Join(e.Data, "settings.json"))
	if err != nil || !strings.Contains(string(b), `"keep_context": "always"`) {
		t.Fatalf("settings.json: %s %v", b, err)
	}
	for _, bad := range []string{`{"keep_context":"maybe"}`, `{"keep_context":""}`} {
		if c, _, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", bad, csrf); c != 400 {
			t.Fatalf("PUT %s: %d", bad, c)
		}
	}
	if c, _, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"keep_context":"off"}`, csrf); c != 200 || s.KeepContext() != "off" {
		t.Fatalf("PUT keep_context off: %d live %v", c, s.KeepContext())
	}
	if b, _ := os.ReadFile(filepath.Join(e.Data, "settings.json")); !strings.Contains(string(b), `"keep_context": "off"`) {
		t.Fatalf("settings.json after off: %s", b)
	}
}
