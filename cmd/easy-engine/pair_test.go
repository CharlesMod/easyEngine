package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/discover"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/run"
)

// lanEngine runs a real engine in this process on 127.0.0.2 (a non-loopback name for the gate,
// a loopback address for the network: it announces nothing) and returns its data dir and port.
func lanEngine(t *testing.T) (string, int) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("needs 127.0.0.2 as a second local address")
	}
	dir := t.TempDir()
	p := run.Profile{Game: "TestGame", Slots: 1, MinSlots: 1}
	p.CtxPerSlot.Max, p.CtxPerSlot.Floor = 8192, 4096
	ctx, cancel := context.WithCancel(context.Background())
	ported, done := make(chan int, 1), make(chan struct{})
	go func() {
		defer close(done)
		if _, err := app.Serve(ctx, app.Options{DataDir: dir, Profile: &p, Host: "127.0.0.2", Port: 47670, Version: "t",
			Resident: true, Started: func(p int) { ported <- p }}); err != nil {
			t.Errorf("serve: %v", err)
		}
	}()
	var port int
	select {
	case port = <-ported:
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("engine did not start")
	}
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(20 * time.Second): // watchdog for silence
		}
	})
	return dir, port
}

func TestPairCommandsAgainstARealEngine(t *testing.T) {
	dir, port := lanEngine(t)
	tg, ok := app.Find(dir)
	if !ok {
		t.Fatal("the engine is not found through its lock")
	}
	base := "http://127.0.0.2:" + strconv.Itoa(port) + "/engine/v1/"
	var out, errw bytes.Buffer
	if c := pairWith(tg, "", "", &out, &errw); c != 0 || !strings.Contains(out.String(), "No pairing request is waiting.") {
		t.Fatalf("empty list: %d %q %q", c, out.String(), errw.String())
	}
	askPair := func(name string) (string, string) {
		req, _ := http.NewRequest("POST", base+"pair", strings.NewReader(`{"name":"`+name+`"}`))
		req.Header.Set("X-Easy-Engine", "1")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var r struct{ ID, Code string }
		json.NewDecoder(resp.Body).Decode(&r)
		return r.ID, r.Code
	}
	id, code := askPair("Den PC")
	out.Reset()
	if c := pairWith(tg, "", "", &out, &errw); c != 0 || !strings.Contains(out.String(), code[:3]+" "+code[3:]) || !strings.Contains(out.String(), "Den PC") {
		t.Fatalf("list: %d %q", c, out.String())
	}
	errw.Reset()
	if c := pairWith(tg, "000000x", "", &out, &errw); c != 1 || !strings.Contains(errw.String(), "no waiting request") {
		t.Fatalf("unknown code: %d %q", c, errw.String())
	}
	out.Reset()
	if c := pairWith(tg, code[:3]+" "+code[3:], "", &out, &errw); c != 0 || !strings.Contains(out.String(), "Allowed Den PC") {
		t.Fatalf("approve: %d %q %q", c, out.String(), errw.String())
	}
	resp, err := http.Get(base + "pair/" + id)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Token    string `json:"token"`
		ClientID string `json:"client_id"`
	}
	json.NewDecoder(resp.Body).Decode(&got)
	resp.Body.Close()
	if resp.StatusCode != 200 || got.Token == "" {
		t.Fatalf("collect: %d", resp.StatusCode)
	}

	// deny
	_, code2 := askPair("Stranger")
	out.Reset()
	if c := pairWith(tg, "", code2, &out, &errw); c != 0 || !strings.Contains(out.String(), "Denied Stranger") {
		t.Fatalf("deny: %d %q", c, out.String())
	}

	// a paired client's token cannot answer through the same door
	_, code3 := askPair("Third")
	errw.Reset()
	if c := pairWith(app.Target{Host: tg.Host, Port: tg.Port, Token: got.Token}, code3, "", &out, &errw); c != 1 || !strings.Contains(errw.String(), "403") {
		t.Fatalf("a paired token approving: %d %q", c, errw.String())
	}

	// token --list shows the client without its token; --revoke ends it
	st := tokens.Open(dir)
	out.Reset()
	if c := tokenList(st, &out); c != 0 || !strings.Contains(out.String(), got.ClientID) || !strings.Contains(out.String(), "Den PC") || strings.Contains(out.String(), got.Token) {
		t.Fatalf("token --list: %d %q", c, out.String())
	}
	out.Reset()
	if c := tokenRevoke(st, got.ClientID, &out); c != 0 || !strings.Contains(out.String(), "Revoked "+got.ClientID) {
		t.Fatalf("token --revoke: %d %q", c, out.String())
	}
	req, _ := http.NewRequest("GET", base+"status", nil)
	req.Header.Set("Authorization", "Bearer "+got.Token)
	r2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var body struct{ Code string }
	json.NewDecoder(r2.Body).Decode(&body)
	r2.Body.Close()
	if r2.StatusCode != 401 || body.Code != "token_revoked" {
		t.Fatalf("revoked token: %d %q", r2.StatusCode, body.Code)
	}
}

func TestFindPrintsWhatAnswered(t *testing.T) {
	var out bytes.Buffer
	printFound(&out, nil, false)
	if !strings.Contains(out.String(), "No engine answered") {
		t.Fatalf("empty: %q", out.String())
	}
	out.Reset()
	f := discover.Found{ID: "0123456789abcdef", Name: "den", API: 1, Port: 47480, Host: "easy-engine-01234567.local.",
		Addrs: []net.IP{net.ParseIP("fd00::5"), net.IPv4(192, 168, 1, 20)}}
	printFound(&out, []discover.Found{f}, false)
	if !strings.Contains(out.String(), "192.168.1.20:47480") || !strings.Contains(out.String(), "den") {
		t.Fatalf("text: %q", out.String())
	}
	out.Reset()
	printFound(&out, []discover.Found{f}, true)
	if !strings.Contains(out.String(), `"address": "192.168.1.20:47480"`) || !strings.Contains(out.String(), `"engine_id": "0123456789abcdef"`) {
		t.Fatalf("json: %q", out.String())
	}
}
