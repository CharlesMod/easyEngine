package api_test

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// Each client word at the API (attach, settings, start, wake, host) is said to the supervisor as
// a nudge, with its why: whatever waits for a word may try again.
func TestClientWordsNudgeTheSupervisor(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := e.StartSup(e.Config())
	a := &api.API{S: s, Attach: func(int) error { return nil }}
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	for _, c := range []struct{ method, path, body, why string }{
		{"POST", "attach", `{"pid":1}`, "attach"},
		{"PUT", "settings", `{"linger_s":120}`, "settings"},
		{"POST", "start", "", "start"},
		{"POST", "wake", "", "wake"},
		{"POST", "host", `{"reserve_mib": 900}`, "host"},
	} {
		mark := testkit.Mark(t, s, "before "+c.why)
		if code, b, _ := do(t, c.method, srv.URL+"/engine/v1/"+c.path, c.body, csrf); code/100 != 2 {
			t.Fatalf("%s: %d %s", c.path, code, b)
		}
		// every handler says its nudge before it answers, so it is on the hub by now
		found := false
		var got []string
		for _, ev := range testkit.Said(t, s, mark, 0, "nudge") {
			var b struct{ Why string }
			json.Unmarshal(ev.Body, &b)
			if ev.Name == "nudge" {
				got = append(got, b.Why)
				found = found || b.Why == c.why
			}
		}
		if !found {
			t.Errorf("%s: nudges after the call: %v, want %q among them", c.path, got, c.why)
		}
	}
}
