package api_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The streamer is in the settings API: GET shows it, PUT of it applies at once and is written to
// settings.json, a PUT of another field keeps it, a bad value is refused.
func TestStreamerInTheSettingsAPI(t *testing.T) {
	e, s, srv := readyEnv(t)
	get := func() map[string]any {
		c, body, _ := do(t, "GET", srv.URL+"/engine/v1/settings", "", nil)
		var m map[string]any
		if c != 200 || json.Unmarshal([]byte(body), &m) != nil {
			t.Fatalf("GET settings %d %s", c, body)
		}
		return m
	}
	if v := get()["streamer"]; v != "on" || !s.Streamer() {
		t.Fatalf("default streamer %v", v)
	}
	if c, body, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"streamer":"off"}`, csrf); c != 200 {
		t.Fatalf("PUT streamer off: %d %s", c, body)
	}
	if v := get()["streamer"]; v != "off" || s.Streamer() {
		t.Fatalf("after PUT off: %v live %v", v, s.Streamer())
	}
	if c, body, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"linger_s":7}`, csrf); c != 200 {
		t.Fatalf("PUT linger: %d %s", c, body)
	}
	if m := get(); m["streamer"] != "off" || m["linger_s"] != float64(7) {
		t.Fatalf("another field's PUT moved the streamer: %v", m)
	}
	b, err := os.ReadFile(filepath.Join(e.Data, "settings.json"))
	if err != nil || !strings.Contains(string(b), `"streamer": "off"`) {
		t.Fatalf("settings.json: %s %v", b, err)
	}
	for _, bad := range []string{`{"streamer":"maybe"}`, `{"streamer":""}`} {
		if c, _, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", bad, csrf); c != 400 {
			t.Fatalf("PUT %s: %d", bad, c)
		}
	}
	if c, _, _ := do(t, "PUT", srv.URL+"/engine/v1/settings", `{"streamer":"on"}`, csrf); c != 200 || !s.Streamer() {
		t.Fatalf("PUT streamer on: %d live %v", c, s.Streamer())
	}
}
