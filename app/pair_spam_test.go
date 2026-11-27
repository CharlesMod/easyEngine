package app_test

import (
	"net"
	"net/http"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/autostart"
	"github.com/CharlesMod/easyEngine/tray"
)

func askFrom(t *testing.T, url, ip, name string) {
	t.Helper()
	cl := &http.Client{Transport: &http.Transport{DialContext: (&net.Dialer{LocalAddr: &net.TCPAddr{IP: net.ParseIP(ip)}}).DialContext}}
	req, _ := http.NewRequest("POST", url, strings.NewReader(`{"name":"`+name+`"}`))
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := cl.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("pair from %s: %d", ip, resp.StatusCode)
	}
}

func (f *fakeTray) noteCount() int { f.mu.Lock(); defer f.mu.Unlock(); return len(f.notes) }

func allowItems(items []tray.Item) (allow []tray.Item) {
	for _, it := range items {
		if strings.HasPrefix(it.ID, "allow:") {
			allow = append(allow, it)
		}
	}
	return allow
}

// A host on the network cannot spam the person: one waiting request per source, one
// notification per request not yet answered, none after the person denied that source.
func TestPairingNotificationsAreBoundedPerSource(t *testing.T) {
	lanOK(t)
	ft := newFakeTray()
	r := startResident(t, app.Options{Host: "127.0.0.2", Port: 47655, Tray: ft,
		Menu: app.MenuOptions{Autostart: autostart.Root(t.TempDir()), Exe: "/bin/true", Open: func(string) error { return nil }}})
	url := "http://127.0.0.2:" + itoa(r.port) + "/engine/v1/pair"

	for i := 0; i < 100; i++ {
		askFrom(t, url, "127.0.0.31", "spammer")
	}
	ft.until(t, "one request from the first source", func(items []tray.Item, _ string) bool {
		return len(allowItems(items)) == 1 && ft.noteCount() == 1
	})
	askFrom(t, url, "127.0.0.32", "neighbour")
	var first string
	ft.until(t, "a second source's own request", func(items []tray.Item, _ string) bool {
		a := allowItems(items)
		if len(a) != 2 || ft.noteCount() != 2 || !strings.Contains(a[0].Title, "spammer") {
			return false
		}
		first = a[0].ID // oldest first: the spammer's
		return true
	})
	ft.clicks <- "deny:" + strings.TrimPrefix(first, "allow:")
	ft.until(t, "the denied request gone", func(items []tray.Item, _ string) bool { return len(allowItems(items)) == 1 })
	for i := 0; i < 10; i++ {
		askFrom(t, url, "127.0.0.31", "spammer again")
	}
	ft.until(t, "the denied source listed again", func(items []tray.Item, _ string) bool { return len(allowItems(items)) == 2 })
	// A third source's request is a fence: its notification comes after any the denied source
	// would have raised, so the count then shows the denied source raised none.
	askFrom(t, url, "127.0.0.33", "fence")
	ft.until(t, "the fence's notification", func(items []tray.Item, _ string) bool { return ft.noteCount() >= 3 })
	ft.mu.Lock()
	notes := append([]string(nil), ft.notes...)
	ft.mu.Unlock()
	if len(notes) != 3 || !strings.Contains(notes[2], "fence") {
		t.Fatalf("notifications: %q", notes)
	}
}
