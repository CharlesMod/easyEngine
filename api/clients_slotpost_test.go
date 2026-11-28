package api_test

import (
	"os"
	"path/filepath"
	"testing"
)

// F4: the door never passes a client's POST /slots/N as sent. The action is read from the URL
// alone and must be exactly one action=erase; a form-encoded body (which cpp-httplib folds into
// the parameters, the last value winning, so it could turn an erase into a restore) is refused,
// as is a POST with no action (llama-server answers it 400 without clearing, and the old door had
// labelled the slot the caller's). The request upstream is the engine's own: no body. A JSON body
// the door reads may not come form-encoded either.
func TestSlotPostsAreTheEnginesOwn(t *testing.T) {
	for _, claimed := range []bool{false, true} {
		name := "nothing claimed"
		if claimed {
			name = "a client holds a range"
		}
		t.Run(name, func(t *testing.T) {
			e, s, srv := clientsEnv(t, true, nil)
			if claimed {
				_, rs, _ := remoteSrv(t, s)
				growRemote(t, e, s, rs, 1, "a", 1)
			}
			acts := func() string {
				b, _ := os.ReadFile(filepath.Join(e.Ctl, "slotacts.log"))
				return string(b)
			}
			form := map[string]string{"Content-Type": "application/x-www-form-urlencoded"}
			for _, c := range []struct {
				what, path, body string
				hdr              map[string]string
			}{
				{"a form body turning the erase into a restore", "/slots/1?action=erase", "action=restore&filename=x.bin", form},
				{"a form body naming save", "/slots/1?action=erase", "action=save&filename=x.bin", form},
				{"no action", "/slots/1", "", nil},
				{"two actions", "/slots/1?action=erase&action=erase", "", nil},
			} {
				before := acts()
				code, b, _ := do(t, "POST", srv.URL+c.path, c.body, c.hdr)
				if code != 400 || codeOf(b) != "bad_request" || acts() != before {
					t.Errorf("%s: %d %s; the fake saw %q", c.what, code, b, acts()[len(before):])
				}
			}
			// a plain erase goes upstream as the engine builds it
			before := acts()
			if code, b, _ := do(t, "POST", srv.URL+"/slots/1?action=erase", `{"filename":"x.bin"}`, jsonCT); code != 200 || acts()[len(before):] != "/slots/1?action=erase\n" {
				t.Errorf("erase: %d %s; the fake saw %q", code, b, acts()[len(before):])
			}
			// a completion's body is JSON, never a form
			if code, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"messages":[]}`, form); code != 400 {
				t.Errorf("a form-encoded completion: %d %s", code, b)
			}
		})
	}
}
