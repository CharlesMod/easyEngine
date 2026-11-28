package api_test

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// F2: the engine orders a slot's handover, never the model server. llama-server's erase and
// completion are separate tasks: an erase on a busy slot is deferred, a completion naming a busy
// slot is deferred with its id_slot, and a release takes back first the deferred task naming that
// slot (server-queue.cpp:90-112); the erase's id_slot is -1. The fake does the same. So a slot is
// labelled the newcomer's only after its erase answered, never while a request the door sent there
// runs, and every request for the slot waits behind a handover in arrival order.

type xres struct {
	code int
	hit  int
	body string
}

func probeAsync(t *testing.T, url, prompt string, slot int, hdr map[string]string) chan xres {
	ch := make(chan xres, 1)
	go func() {
		body := fmt.Sprintf(`{"id_slot":%d,"messages":[{"role":"user","content":%q}]}`, slot, prompt)
		req, _ := http.NewRequest("POST", url+"/v1/chat/completions", strings.NewReader(body))
		for k, v := range hdr {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			ch <- xres{code: -1, body: err.Error()}
			return
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var rep struct {
			Timings struct {
				CacheN int `json:"cache_n"`
			} `json:"timings"`
		}
		json.Unmarshal(b, &rep)
		ch <- xres{code: resp.StatusCode, hit: rep.Timings.CacheN, body: string(b)}
	}()
	return ch
}

func got(t *testing.T, ch chan xres, what string) xres {
	t.Helper()
	select {
	case p := <-ch:
		return p
	case <-time.After(20 * time.Second): // watchdog for silence: the request was never answered
		t.Fatalf("no answer to %s", what)
	}
	return xres{}
}

// watchNew makes a control file the fake appends to (so its changes can be followed from the
// start) and has WaitFor look again at each change.
func watchNew(t *testing.T, e *testkit.Env, name string) {
	p := filepath.Join(e.Ctl, name)
	if f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
		f.Close()
	}
	e.WatchFile(p)
}

func lines(e *testkit.Env, name string) string {
	b, _ := os.ReadFile(filepath.Join(e.Ctl, name))
	return string(b)
}

// Y runs in commons slot 1 (held by the fake after its first chunk). Untrusted X sends X1, then
// X2 (a probe of Y's prompt) on the same slot. With the old order (label at the take, erase sent
// at once) X1's erase and X2's completion are both deferred behind Y, and Y's release runs X2
// first, on Y's prompt.
func TestHandoverOnABusySlot(t *testing.T) {
	e, s, srv := clientsEnv(t, true, func(e *testkit.Env) { e.Llama["FAKELLAMA_STREAM_GATE"] = "1" })
	_, rs, _ := remoteSrv(t, s)
	watchNew(t, e, "deferred.log")
	release := func() { // Y ends (also when the test fails: never leave the fake's stream held)
		for i := 1; i <= 4; i++ {
			os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		}
	}
	t.Cleanup(release)
	ysecret := secret("Y")
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(fmt.Sprintf(`{"stream":true,"id_slot":1,"messages":[{"role":"user","content":%q}]}`, ysecret)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	if _, err := br.ReadString('\n'); err != nil {
		t.Fatalf("Y's first chunk: %v", err)
	}
	x1 := probeAsync(t, rs.URL, "zzz unrelated", 1, as(2))
	testkit.WaitFor(t, 15*time.Second, "X1 waiting (in the engine, or its erase deferred)", func() bool {
		return s.SlotsWaiting() >= 1 || strings.Contains(lines(e, "deferred.log"), "erase 1")
	})
	x2 := probeAsync(t, rs.URL, ysecret+" and then?", 1, as(2))
	testkit.WaitFor(t, 15*time.Second, "X2 waiting (in the engine, or deferred)", func() bool {
		return s.SlotsWaiting() >= 2 || strings.Contains(lines(e, "deferred.log"), "task 1 1")
	})
	release()
	io.ReadAll(br)
	p1, p2 := got(t, x1, "X1"), got(t, x2, "X2")
	if p1.code != 200 || p2.code != 200 {
		t.Fatalf("X1 %d %s; X2 %d %s", p1.code, p1.body, p2.code, p2.body)
	}
	if p2.hit != 0 {
		t.Fatalf("X2 read Y's prompt: cache_n %d (deferred: %q)", p2.hit, lines(e, "deferred.log"))
	}
}

// Y's prompt sits in idle commons slot 1. X1's erase is still on its way to the model server
// (the fake's erase gate) when X2 arrives on the same slot: with the old order X2, labelled
// X's already, runs at once on Y's prompt.
func TestHandoverOnAnIdleSlot(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	watchNew(t, e, "slotacts.log")
	ysecret := secret("Y")
	chat(t, srv.URL, ysecret, 1, jsonCT)
	gate := filepath.Join(e.Ctl, "erasegate")
	if err := syscall.Mkfifo(gate, 0o600); err != nil {
		t.Fatal(err)
	}
	opened := false
	t.Cleanup(func() { // never leave an erase held at the gate
		if !opened {
			if f, err := os.OpenFile(gate, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
				f.Close()
			}
		}
		os.Remove(gate)
	})
	x1 := probeAsync(t, rs.URL, "zzz unrelated", 1, as(2))
	testkit.WaitFor(t, 15*time.Second, "X1's erase on its way", func() bool {
		return strings.Contains(lines(e, "slotacts.log"), "/slots/1?action=erase")
	})
	x2 := probeAsync(t, rs.URL, ysecret+" and then?", 1, as(2))
	var early *xres
	testkit.WaitFor(t, 15*time.Second, "X2 waiting in the engine, or answered", func() bool {
		select {
		case p := <-x2:
			early = &p
		default:
		}
		return early != nil || s.SlotsWaiting() >= 2
	})
	if err := os.WriteFile(gate, []byte("go"), 0o600); err != nil { // X1's erase arrives
		t.Fatal(err)
	}
	opened = true
	os.Remove(gate)
	p1 := got(t, x1, "X1")
	p2 := early
	if p2 == nil {
		q := got(t, x2, "X2")
		p2 = &q
	}
	if p1.code != 200 || p2.code != 200 {
		t.Fatalf("X1 %d %s; X2 %d %s", p1.code, p1.body, p2.code, p2.body)
	}
	if p2.hit != 0 {
		t.Fatalf("X2 read Y's prompt: cache_n %d", p2.hit)
	}
}
