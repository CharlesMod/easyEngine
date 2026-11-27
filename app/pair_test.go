package app_test

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/autostart"
	"github.com/CharlesMod/easyEngine/discover"
	"github.com/CharlesMod/easyEngine/tray"
)

// countingNet is a discover.Transport that only counts: an engine that announced anything
// through it would show here. No packet reaches a real network.
type countingNet struct {
	mu    sync.Mutex
	sends int
	done  chan struct{}
	once  sync.Once
}

func newCountingNet() *countingNet { return &countingNet{done: make(chan struct{})} }

func (c *countingNet) Read() (discover.Packet, error) { <-c.done; return discover.Packet{}, io.EOF }
func (c *countingNet) Multicast([]byte, int) error {
	c.mu.Lock()
	c.sends++
	c.mu.Unlock()
	return nil
}
func (c *countingNet) Unicast([]byte, *net.UDPAddr) error { return c.Multicast(nil, 0) }
func (c *countingNet) Addrs(int) []net.IP                 { return nil }
func (c *countingNet) Reach(net.IP) []net.IP              { return nil }
func (c *countingNet) Refresh() error                     { return nil }
func (c *countingNet) Close() error                       { c.once.Do(func() { close(c.done) }); return nil }
func (c *countingNet) count() int                         { c.mu.Lock(); defer c.mu.Unlock(); return c.sends }

func TestDiscoverable(t *testing.T) {
	for host, want := range map[string]bool{
		"127.0.0.1": false, "127.0.0.2": false, "localhost": false, "::1": false, "[::1]": false, "": false,
		"0.0.0.0": true, "192.168.1.20": true, "::": true, "pc.lan": true,
	} {
		if got := app.Discoverable(host); got != want {
			t.Errorf("Discoverable(%q) = %v, want %v", host, got, want)
		}
	}
}

func postJSON(t *testing.T, url, body string) (int, map[string]string) {
	t.Helper()
	req, _ := http.NewRequest("POST", url, strings.NewReader(body))
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out := map[string]string{}
	json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

// The tray shows a waiting pairing request as Allow and Deny items with one notification, the
// standing items stay, Allow from the menu hands the client its token, and an engine on a
// loopback address announces nothing.
func TestTrayAnswersAPairingRequest(t *testing.T) {
	lanOK(t)
	ft := newFakeTray()
	nw := newCountingNet()
	r := startResident(t, app.Options{Host: "127.0.0.2", Port: 47650, Tray: ft, Discovery: discover.Options{Transport: nw},
		Menu: app.MenuOptions{Autostart: autostart.Root(t.TempDir()), Exe: "/bin/true", Open: func(string) error { return nil }}})
	base := "http://127.0.0.2:" + itoa(r.port) + "/engine/v1/"

	c, res := postJSON(t, base+"pair", `{"name":"Living room"}`)
	if c != 200 || len(res["code"]) != 6 {
		t.Fatalf("pair: %d %v", c, res)
	}
	label := "Living room (" + res["code"][:3] + " " + res["code"][3:] + ")"
	var allowID string
	ft.until(t, "the pairing items", func(items []tray.Item, _ string) bool {
		if len(items) != 7 || items[0].Title != "Allow "+label || items[1].Title != "Deny "+label || !items[2].Separator ||
			items[3].ID != "chat" || items[6].ID != "quit" {
			return false
		}
		allowID = items[0].ID
		return true
	})
	ft.mu.Lock()
	notes := append([]string(nil), ft.notes...)
	ft.mu.Unlock()
	if len(notes) != 1 || notes[0] != "Allow "+label+"?" {
		t.Fatalf("notifications: %q", notes)
	}

	got := make(chan *http.Response, 1)
	go func() {
		resp, err := http.Get(base + "pair/" + res["id"])
		if err != nil {
			t.Error(err)
		}
		got <- resp
	}()
	ft.clicks <- allowID
	var resp *http.Response
	select {
	case resp = <-got:
	case <-time.After(20 * time.Second): // watchdog for silence: the held GET was never answered
		t.Fatal("no answer to the held GET")
	}
	if resp == nil {
		t.FailNow()
	}
	var tok struct {
		Token string `json:"token"`
	}
	json.NewDecoder(resp.Body).Decode(&tok)
	resp.Body.Close()
	if resp.StatusCode != 200 || tok.Token == "" {
		t.Fatalf("collect: %d", resp.StatusCode)
	}
	if c := get(t, base+"status", tok.Token); c != 200 {
		t.Fatalf("the paired token: %d", c)
	}
	ft.until(t, "the standing items alone", func(items []tray.Item, _ string) bool { return len(items) == 4 })
	r.stop()
	r.res <- <-r.res // Serve has returned; put the result back for the cleanup
	if n := nw.count(); n != 0 {
		t.Fatalf("an engine on a loopback address sent %d mDNS packets", n)
	}
}
