package api_test

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// sent is one request sent in the background: its status and body once answered.
type sent struct {
	code int
	body string
}

// sendAs posts body as remote client n in the background.
func sendAs(url, body string, n int) <-chan sent {
	out := make(chan sent, 1)
	go func() {
		req, _ := http.NewRequest("POST", url+"/v1/chat/completions", strings.NewReader(body))
		for k, v := range as(n) {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			out <- sent{0, err.Error()}
			return
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		out <- sent{resp.StatusCode, string(b)}
	}()
	return out
}

// waitHeld waits for the door fact to say n requests held.
func waitHeld(t *testing.T, s interface{ Held() int }, n int) {
	t.Helper()
	testkit.WaitFor(t, 15*time.Second, strconv.Itoa(n)+" requests held at the door", func() bool { return s.Held() == n })
}

// S3: remote requests held at the door are bounded by count: 8 of one client (its 9th is 429
// busy) and 32 of all (a fifth client's first is busy), however long the hold. Each client's grow
// waits for idle behind a local reply in flight; once the reply ends, every held one is served.
func TestRemoteHeldRequestsBounded(t *testing.T) {
	e, s, srv := gatedEnv(t, time.Hour) // named moment: no DrainMax in this test's life
	_, rs, _ := remoteSrv(t, s)
	_, br := streamGated(t, srv.URL)
	for n := 1; n <= 5; n++ {
		h := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(n))
		if sl, _ := h.first["slots"].(map[string]any); h.status != 200 || sl["why"] != "waiting_for_idle" {
			t.Fatalf("attach %d: %d %v", n, h.status, h.first)
		}
	}
	const q = `{"messages":[{"role":"user","content":"x"}]}`
	var held []<-chan sent
	for n := 1; n <= 4; n++ {
		for i := 0; i < 8; i++ {
			held = append(held, sendAs(rs.URL, q, n))
		}
		waitHeld(t, s, 8*n)
		if n == 1 {
			if r := <-sendAs(rs.URL, q, 1); r.code != 429 || codeOf(r.body) != "busy" {
				t.Fatalf("one client's 9th held: %d %s", r.code, r.body)
			}
		}
	}
	if r := <-sendAs(rs.URL, q, 5); r.code != 429 || codeOf(r.body) != "busy" {
		t.Fatalf("the 33rd held: %d %s", r.code, r.body)
	}
	for i := 1; i <= 4; i++ { // the local reply ends: the door's empty moment, the grow
		os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		br.ReadString('\n')
	}
	for _, c := range held {
		if r := <-c; r.code != 200 {
			t.Fatalf("a held request after the grow: %d %s", r.code, r.body)
		}
	}
}

// S3: the bytes of remote bodies kept (held at the door or in flight) are bounded: four held
// bodies of 15 MiB are kept, a fifth would pass 64 MiB and is 429 busy.
func TestRemoteKeptBytesBounded(t *testing.T) {
	e, s, srv := gatedEnv(t, time.Hour) // named moment: no DrainMax in this test's life
	_, rs, _ := remoteSrv(t, s)
	_, br := streamGated(t, srv.URL)
	h := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(1))
	if h.status != 200 {
		t.Fatalf("attach: %d %v", h.status, h.first)
	}
	big := `{"messages":[{"role":"user","content":"x"}],"pad":"` + strings.Repeat("a", 15<<20) + `"}`
	var held []<-chan sent
	for i := 1; i <= 4; i++ {
		held = append(held, sendAs(rs.URL, big, 1))
		waitHeld(t, s, i)
	}
	if r := <-sendAs(rs.URL, big, 1); r.code != 429 || codeOf(r.body) != "busy" || !strings.Contains(r.body, "64 MiB") {
		t.Fatalf("a fifth 15 MiB body: %d %s", r.code, r.body)
	}
	for i := 1; i <= 4; i++ {
		os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		br.ReadString('\n')
	}
	for _, c := range held {
		if r := <-c; r.code != 200 {
			t.Fatalf("a held request after the grow: %d %.200s", r.code, r.body)
		}
	}
}

// The prefix oracle: a vacant range still carries the prompts of the client that held it (the
// slot's cache, and the RAM prompt cache any slot's). A remote caller with no attachment is never
// sent where its cache hits could read them, and a remote attach is never served from another
// client's vacant range.
func TestDepartedPromptsNeverReachAnotherRemote(t *testing.T) {
	t.Run("lone owner, cache on", func(t *testing.T) {
		e, s, srv := clientsEnv(t, false, func(e *testkit.Env) { e.Llama["FAKELLAMA_CACHE_RAM"] = "1" })
		_, rs, _ := remoteSrv(t, s)
		A := growRemote(t, e, s, rs, 1, "a", 2)
		if f, _ := s.Hub().Fact("wants", ""); !strings.Contains(string(f.Body), `"ram_cache":true`) || s.Status().Shape.Slots != 2 {
			t.Fatalf("A is not the one owner with the cache on: %s, %d slots", f.Body, s.Status().Shape.Slots)
		}
		if c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"a secret prefix"}]}`, as(1)); c != 200 {
			t.Fatalf("A: %d %s", c, b)
		}
		A.cancel()
		attachmentGone(t, s, A.id)
		before := upstreamBytes(t, e)
		c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"a secret prefix"}]}`, as(2))
		if c != 409 || codeOf(b) != "attach_required" {
			t.Fatalf("an anonymous remote caller after A left: %d %s", c, b)
		}
		if upstreamBytes(t, e) != before {
			t.Fatal("it reached the model server")
		}
		// a caller of this machine is trusted: served
		if c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"messages":[]}`, jsonCT); c != 200 {
			t.Fatalf("local: %d %s", c, b)
		}
	})
	t.Run("a clean slot only", func(t *testing.T) {
		e, s, _ := clientsEnv(t, false, nil)
		_, rs, _ := remoteSrv(t, s)
		A := growRemote(t, e, s, rs, 1, "a", 2)
		sl, _ := A.first["slots"].(map[string]any)
		first, _ := sl["first"].(float64)
		A.cancel()
		attachmentGone(t, s, A.id)
		for i := 0; i < 4; i++ {
			c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(2))
			var rep struct {
				IDSlot *int `json:"id_slot"`
			}
			json.Unmarshal([]byte(b), &rep)
			switch {
			case c == 409 && codeOf(b) == "attach_required":
			case c == 200 && rep.IDSlot != nil && (*rep.IDSlot < int(first) || *rep.IDSlot >= int(first)+2):
			default:
				t.Fatalf("an anonymous remote caller after A left (A's range %v..%v): %d %s", first, first+1, c, b)
			}
		}
	})
	t.Run("two owners, cache off", func(t *testing.T) {
		e, s, _ := clientsEnv(t, false, nil)
		_, rs, _ := remoteSrv(t, s)
		A := growRemote(t, e, s, rs, 1, "a", 2)
		growRemote(t, e, s, rs, 3, "c", 1)
		A.cancel()
		attachmentGone(t, s, A.id)
		n := len(e.Starts())
		// D wants what A's vacant range holds: never given it; a grow (its own, charged) instead
		D := openHold(t, rs, `{"hold":true,"name":"d","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(4))
		if sl, _ := D.first["slots"].(map[string]any); D.status != 200 || sl["given"] != float64(0) || sl["pending"] != true {
			t.Fatalf("D: %d %v", D.status, D.first)
		}
		m := D.untilGiven(2)
		testkit.WaitPhase(t, s, "ready", 15*time.Second)
		if len(e.Starts()) != n+1 || restartsLeft(m) != 3 {
			t.Fatalf("D's grow: starts %d (from %d), restarts left %d", len(e.Starts()), n, restartsLeft(m))
		}
	})
}
