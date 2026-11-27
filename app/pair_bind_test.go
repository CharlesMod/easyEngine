package app_test

import (
	"encoding/json"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/autostart"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/tray"
)

func askIDFrom(t *testing.T, url, ip, name string) (string, string) {
	t.Helper()
	cl := &http.Client{Transport: &http.Transport{DialContext: (&net.Dialer{LocalAddr: &net.TCPAddr{IP: net.ParseIP(ip)}}).DialContext}}
	req, _ := http.NewRequest("POST", url, strings.NewReader(`{"name":"`+name+`"}`))
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := cl.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var r struct{ ID, Code string }
	json.NewDecoder(resp.Body).Decode(&r)
	return r.ID, r.Code
}

func heldGet(t *testing.T, url string) <-chan int {
	ch := make(chan int, 1)
	go func() {
		resp, err := http.Get(url)
		if err != nil {
			ch <- 0
			return
		}
		resp.Body.Close()
		ch <- resp.StatusCode
	}()
	return ch
}

// A press on the tray item answers the request whose code the item showed, never a replacement
// that took its key since the menu was drawn.
func TestTrayPressBindsToTheShownCode(t *testing.T) {
	lanOK(t)
	ft := newFakeTray()
	r := startResident(t, app.Options{Host: "127.0.0.2", Port: 47675, Tray: ft,
		Menu: app.MenuOptions{Autostart: autostart.Root(t.TempDir()), Exe: "/bin/true", Open: func(string) error { return nil }}})
	base := "http://127.0.0.2:" + itoa(r.port) + "/engine/v1/"

	oldID, oldCode := askIDFrom(t, base+"pair", "127.0.0.51", "device A")
	var shown string
	ft.until(t, "device A's item", func(items []tray.Item, _ string) bool {
		a := allowItems(items)
		if len(a) == 1 && strings.HasSuffix(a[0].ID, ":"+oldCode) {
			shown = a[0].ID
			return true
		}
		return false
	})
	oldGot := heldGet(t, base+"pair/"+oldID)
	newID, newCode := askIDFrom(t, base+"pair", "127.0.0.51", "someone else")
	var current string
	ft.until(t, "the replacement's item", func(items []tray.Item, _ string) bool {
		a := allowItems(items)
		if len(a) == 1 && strings.HasSuffix(a[0].ID, ":"+newCode) {
			current = a[0].ID
			return true
		}
		return false
	})
	newGot := heldGet(t, base+"pair/"+newID)
	ft.clicks <- shown // the stale press: the person compared device A's code
	select {
	case c := <-oldGot:
		if c != 410 {
			t.Fatalf("the replaced client: %d", c)
		}
	case <-time.After(20 * time.Second): // watchdog for silence
		t.Fatal("the replaced client heard nothing")
	}
	ft.clicks <- "autostart" // a fence: the stale click above has been handled when this one shows
	ft.until(t, "the fence click", func(_ []tray.Item, _ string) bool { return ft.checked() })
	if list, _ := tokens.Open(r.dir).List(); len(list) != 0 {
		t.Fatalf("a stale press issued a token: %+v", list)
	}
	select {
	case c := <-newGot:
		t.Fatalf("the new request was answered by the stale press: %d", c)
	default:
	}
	ft.clicks <- current
	select {
	case c := <-newGot:
		if c != 200 {
			t.Fatalf("the new client after the right press: %d", c)
		}
	case <-time.After(20 * time.Second): // watchdog for silence
		t.Fatal("the new client heard nothing")
	}
}
