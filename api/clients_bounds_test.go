package api_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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

// answered is a background request's answer, or the test fails with what it waited for: a request
// the engine should have refused at once but held instead never hangs the test.
func answered(t *testing.T, c <-chan sent, what string) sent {
	t.Helper()
	select {
	case r := <-c:
		return r
	case <-time.After(20 * time.Second): // watchdog for silence: an answer due at once, or after the grow
		t.Fatalf("no answer to %s", what)
	}
	return sent{}
}

// attachWaiting attaches remote clients 1..n, each one slot, each waiting for idle behind the
// local reply in flight.
func attachWaiting(t *testing.T, rs *httptest.Server, n int) {
	t.Helper()
	for i := 1; i <= n; i++ {
		h := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(i))
		if sl, _ := h.first["slots"].(map[string]any); h.status != 200 || sl["why"] != "waiting_for_idle" {
			t.Fatalf("attach %d: %d %v", i, h.status, h.first)
		}
	}
}

// S3 / D4 / D8: remote requests held at the door are bounded by count: 8 of one client (its 9th is
// 429 busy) and 32 of all (a fifth client's first is busy), however long the hold, and one client
// at its share never makes another busy. Each client's grow waits for idle behind a local reply in
// flight; once the reply ends, every held one is served. Every wait is bounded (answered).
func TestRemoteHeldRequestsBounded(t *testing.T) {
	e, s, srv := gatedEnv(t, time.Hour) // named moment: no DrainMax in this test's life
	_, rs, _ := remoteSrv(t, s)
	_, br := streamGated(t, srv.URL)
	attachWaiting(t, rs, 5)
	const q = `{"messages":[{"role":"user","content":"x"}]}`
	var held []<-chan sent
	for n := 1; n <= 4; n++ {
		for i := 0; i < 8; i++ { // one at a time: each body read alone (a client's share of reads is 2)
			held = append(held, sendAs(rs.URL, q, n))
			waitHeld(t, s, 8*(n-1)+i+1)
		}
		if n == 1 {
			if r := answered(t, sendAs(rs.URL, q, 1), "one client's 9th"); r.code != 429 || codeOf(r.body) != "busy" {
				t.Fatalf("one client's 9th held: %d %s", r.code, r.body)
			}
		}
	}
	if r := answered(t, sendAs(rs.URL, q, 5), "the 33rd"); r.code != 429 || codeOf(r.body) != "busy" {
		t.Fatalf("the 33rd held: %d %s", r.code, r.body)
	}
	for i := 1; i <= 4; i++ { // the local reply ends: the door's empty moment, the grow
		os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		br.ReadString('\n')
	}
	for i, c := range held {
		if r := answered(t, c, "a held request after the grow"); r.code != 200 {
			t.Fatalf("held request %d after the grow: %d %s", i, r.code, r.body)
		}
	}
}

// S3 / D4 / D8: the bytes of remote bodies kept (held at the door or in flight) are bounded per
// client and in all: two held bodies of ~16 MiB are one client's 32 MiB (its third is busy, and a
// second client is still taken: one client never makes another busy); four clients fill 128 MiB
// and a fifth client's first is busy.
func TestRemoteKeptBytesBounded(t *testing.T) {
	e, s, srv := gatedEnv(t, time.Hour) // named moment: no DrainMax in this test's life
	_, rs, _ := remoteSrv(t, s)
	_, br := streamGated(t, srv.URL)
	attachWaiting(t, rs, 5)
	big := `{"messages":[{"role":"user","content":"x"}],"pad":"` + strings.Repeat("a", 16<<20-1<<10) + `"}`
	var held []<-chan sent
	for n := 1; n <= 4; n++ {
		for i := 0; i < 2; i++ {
			held = append(held, sendAs(rs.URL, big, n))
			waitHeld(t, s, 2*(n-1)+i+1)
		}
		if n == 1 {
			if r := answered(t, sendAs(rs.URL, big, 1), "one client's third body"); r.code != 429 || codeOf(r.body) != "busy" || !strings.Contains(r.body, "32 MiB of one client") {
				t.Fatalf("one client's third ~16 MiB body: %d %.300s", r.code, r.body)
			}
		}
	}
	if r := answered(t, sendAs(rs.URL, big, 5), "a fifth client's body"); r.code != 429 || codeOf(r.body) != "busy" || !strings.Contains(r.body, "128 MiB in all") {
		t.Fatalf("a fifth client's ~16 MiB body: %d %.300s", r.code, r.body)
	}
	for i := 1; i <= 4; i++ {
		os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		br.ReadString('\n')
	}
	for i, c := range held {
		if r := answered(t, c, "a held body after the grow"); r.code != 200 {
			t.Fatalf("held body %d after the grow: %d %.200s", i, r.code, r.body)
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

// D1: a confined /v1/messages body may not carry extra_params, which llama-server unpacks over
// the converted body (server-chat.cpp:680-685) and so past the engine's id_slot and n checks.
func TestMessagesExtraParamsNotOffered(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 2) // the local caller is confined to the commons
	before := upstreamBytes(t, e)
	for _, ep := range []string{`{"id_slot":99,"n_cmpl":4,"n":4}`, `{"seed":1}`, `null`} {
		body := `{"model":"x","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"extra_params":` + ep + `}`
		c, b, _ := do(t, "POST", srv.URL+"/v1/messages", body, jsonCT)
		if c != 400 || codeOf(b) != "extra_params_not_offered" {
			t.Fatalf("extra_params %s: %d %s", ep, c, b)
		}
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("a refused body reached the model server")
	}
	c, b, _ := do(t, "POST", srv.URL+"/v1/messages", `{"model":"x","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, jsonCT)
	if codeOf(b) == "extra_params_not_offered" || c == 400 {
		t.Fatalf("a messages body without extra_params: %d %s", c, b)
	}
	// on chat, extra_params is one opaque key the server never unpacks: passed
	if c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"messages":[],"extra_params":{"id_slot":99}}`, jsonCT); c != 200 {
		t.Fatalf("chat with extra_params: %d %s", c, b)
	}
}
