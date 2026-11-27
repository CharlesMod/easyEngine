//go:build !windows

package api_test

import (
	"bufio"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

func TestClientsSmoke(t *testing.T) {
	e, s, srv := readyEnv(t)
	n0 := len(e.Starts())
	// A: a child process (cat on a pipe we hold): its end is the attachment's leave
	cat := exec.Command("cat")
	in, _ := cat.StdinPipe()
	if err := cat.Start(); err != nil {
		t.Fatal(err)
	}
	defer cat.Wait()
	defer in.Close()
	code, body, _ := do(t, "POST", srv.URL+"/engine/v1/attach", `{"pid":`+pidStr(cat.Process.Pid)+`,"name":"a","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, csrf)
	if code != 200 {
		t.Fatalf("attach a: %d %s", code, body)
	}
	t.Logf("a: %s", body)
	if len(e.Starts()) != n0 {
		t.Fatalf("a was served from the running mind's free slots; starts %v", e.Starts())
	}
	t.Logf("starts: %v", e.Starts())
	// B: held
	req, _ := http.NewRequest("POST", srv.URL+"/engine/v1/attach", strings.NewReader(`{"hold":true,"name":"b","profile":{"slots":3,"ctx_per_slot":{"max":8192,"floor":4096}}}`))
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	br := bufio.NewReader(resp.Body)
	line, _ := br.ReadString('\n')
	t.Logf("b line1: %s", line)
	var v struct{ Attachment string }
	json.Unmarshal([]byte(line), &v)
	n1 := len(e.Starts())
	testkit.WaitFor(t, 15*time.Second, "reshape for b", func() bool { return len(e.Starts()) > n1 && s.Phase() == "ready" })
	line, _ = br.ReadString('\n')
	t.Logf("b line2: %s", line)
	t.Logf("starts: %v", e.Starts())
	// B asks its slot 1: engine slot 3
	code, body, _ = do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[{"role":"user","content":"hi"}]}`, map[string]string{ClientHeaderName: v.Attachment})
	lb, _ := os.ReadFile(e.Ctl + "/lastbody.json")
	t.Logf("chat %d %s; upstream body %s", code, body, lb)
	code, body, _ = do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":3,"messages":[]}`, map[string]string{ClientHeaderName: v.Attachment})
	t.Logf("out of range: %d %s", code, body)
	// A leaves
	in.Close()
	testkit.WaitFor(t, 15*time.Second, "a gone", func() bool { _, ok := s.Hub().Fact("attachment", ""); _ = ok; return s.Attached() == 1 })
	st := s.Status()
	b, _ := json.Marshal(st.Wants)
	t.Logf("wants after a left: %s", b)
	resp.Body.Close()
	testkit.WaitFor(t, 15*time.Second, "b gone", func() bool { return s.Attached() == 0 })
}

const ClientHeaderName = "X-Easy-Client"

func pidStr(n int) string { b, _ := json.Marshal(n); return string(b) }
