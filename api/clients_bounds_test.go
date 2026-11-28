package api_test

import (
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
