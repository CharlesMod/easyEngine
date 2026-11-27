package tray

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/godbus/dbus/v5"
)

// TestSessionTray runs the real backend on this desktop's session bus: the watcher lists the
// item, its properties and menu read back over D-Bus, a menu click sent over D-Bus arrives on
// Clicks, a SetChecked shows in the next layout, and Close takes the item away. It skips when
// there is no session bus or no StatusNotifierWatcher. The icon is on screen for a moment.
func TestSessionTray(t *testing.T) {
	peer, err := dbus.SessionBusPrivate()
	if err != nil {
		t.Skipf("no session bus: %v", err)
	}
	defer peer.Close()
	if err := peer.Auth(nil); err != nil {
		t.Skipf("session bus auth: %v", err)
	}
	if err := peer.Hello(); err != nil {
		t.Skipf("session bus hello: %v", err)
	}
	var has bool
	if err := peer.BusObject().Call("org.freedesktop.DBus.NameHasOwner", 0, watcherName).Store(&has); err != nil || !has {
		t.Skip("no StatusNotifierWatcher on the session bus")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tr, err := Run(ctx, Spec{
		Title: "easy-engine tray test",
		Icon:  DefaultIcon(),
		Items: []Item{
			{ID: "start", Title: "Launch at Startup", Checkable: true},
			{ID: "settings", Title: "Settings"},
			{Separator: true},
			{ID: "quit", Title: "Quit"},
		},
	})
	if errors.Is(err, ErrNoTray) {
		t.Skipf("no tray: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	name := tr.b.(*sniBackend).name

	// The watcher lists the item (Run returns after RegisterStatusNotifierItem has).
	var listed []string
	if err := peer.Object(watcherName, watcherPath).StoreProperty(watcherName+".RegisteredStatusNotifierItems", &listed); err != nil {
		t.Fatal(err)
	}
	if !containsPrefix(listed, name) {
		t.Fatalf("watcher does not list %s: %v", name, listed)
	}

	item := peer.Object(name, sniPath)
	var title string
	if err := item.StoreProperty(sniIface+".Title", &title); err != nil || title != "easy-engine tray test" {
		t.Fatalf("Title = %q, %v", title, err)
	}
	var menu dbus.ObjectPath
	if err := item.StoreProperty(sniIface+".Menu", &menu); err != nil || menu != menuPath {
		t.Fatalf("Menu = %q, %v", menu, err)
	}
	var icons []pixmap
	if err := item.StoreProperty(sniIface+".IconPixmap", &icons); err != nil || len(icons) != 1 ||
		len(icons[0].Data) != int(4*icons[0].W*icons[0].H) || icons[0].W < 16 {
		t.Fatalf("IconPixmap: %d pixmaps, %v", len(icons), err)
	}

	m := peer.Object(name, menuPath)
	labels, checked := readMenu(t, m)
	if got := strings.Join(labels, "|"); got != "Launch at Startup|Settings|-|Quit" {
		t.Fatalf("menu = %s", got)
	}
	if checked["Launch at Startup"] != 0 {
		t.Fatal("Launch at Startup is checked at start")
	}

	// A click sent the way a panel sends it arrives on Clicks.
	if err := m.Call(menuIface+".Event", 0, int32(2), "clicked", dbus.MakeVariant(""), uint32(0)).Err; err != nil {
		t.Fatal(err)
	}
	select {
	case id := <-tr.Clicks():
		if id != "settings" {
			t.Fatalf("click = %q, want settings", id)
		}
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("no click arrived")
	}

	// SetChecked announces a new layout, and the layout has the mark.
	if err := peer.AddMatchSignal(dbus.WithMatchSender(name), dbus.WithMatchInterface(menuIface), dbus.WithMatchMember("LayoutUpdated")); err != nil {
		t.Fatal(err)
	}
	if err := peer.AddMatchSignal(dbus.WithMatchInterface(watcherName), dbus.WithMatchMember("StatusNotifierItemUnregistered")); err != nil {
		t.Fatal(err)
	}
	sigs := make(chan *dbus.Signal, 16)
	peer.Signal(sigs)
	tr.SetChecked("start", true)
	waitSignal(t, sigs, menuIface+".LayoutUpdated", "")
	if _, checked = readMenu(t, m); checked["Launch at Startup"] != 1 {
		t.Fatal("Launch at Startup is not checked after SetChecked")
	}

	// SetItems replaces the menu.
	tr.SetItems([]Item{{ID: "quit", Title: "Quit"}})
	waitSignal(t, sigs, menuIface+".LayoutUpdated", "")
	if labels, _ = readMenu(t, m); strings.Join(labels, "|") != "Quit" {
		t.Fatalf("menu after SetItems = %v", labels)
	}

	// Close drops the bus name; the watcher unregisters the item.
	tr.Close()
	waitSignal(t, sigs, watcherName+".StatusNotifierItemUnregistered", name)
	if _, ok := <-tr.Clicks(); ok {
		t.Fatal("Clicks still open after Close")
	}
}

func containsPrefix(list []string, name string) bool {
	for _, s := range list {
		if s == name || strings.HasPrefix(s, name+"/") {
			return true
		}
	}
	return false
}

// readMenu returns the labels ("-" for a separator) and toggle states of the root's children.
func readMenu(t *testing.T, m dbus.BusObject) ([]string, map[string]int32) {
	t.Helper()
	var rev uint32
	var root layout
	if err := m.Call(menuIface+".GetLayout", 0, int32(0), int32(-1), []string{}).Store(&rev, &root); err != nil {
		t.Fatal(err)
	}
	var labels []string
	checked := map[string]int32{}
	for _, c := range root.Children {
		var l layout
		if err := dbus.Store([]any{c.Value()}, &l); err != nil {
			t.Fatal(err)
		}
		if typ, _ := l.Props["type"].Value().(string); typ == "separator" {
			labels = append(labels, "-")
			continue
		}
		label, _ := l.Props["label"].Value().(string)
		labels = append(labels, label)
		if v, ok := l.Props["toggle-state"]; ok {
			checked[label], _ = v.Value().(int32)
		}
	}
	return labels, checked
}

// waitSignal waits for the named signal (whose first argument is arg0, when arg0 is not empty).
func waitSignal(t *testing.T, sigs <-chan *dbus.Signal, name, arg0 string) {
	t.Helper()
	watchdog := time.After(10 * time.Second) // watchdog for silence
	for {
		select {
		case s := <-sigs:
			if s.Name != name {
				continue
			}
			if arg0 != "" {
				if len(s.Body) == 0 {
					continue
				}
				if a, _ := s.Body[0].(string); a != arg0 && !strings.HasPrefix(a, arg0+"/") {
					continue
				}
			}
			return
		case <-watchdog:
			t.Fatalf("no %s signal", name)
		}
	}
}
