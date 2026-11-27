package app

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/tray"
	"github.com/CharlesMod/easyEngine/web"
)

func TestMenuChatSitsAboveSettingsAndFollowsReadiness(t *testing.T) {
	for _, ready := range []bool{false, true} {
		items := menu(view{chat: ready})
		var ids []string
		for _, it := range items {
			ids = append(ids, it.ID)
			if it.ID == "chat" && (it.Title != "Chat" || it.Disabled == ready) {
				t.Errorf("ready=%v: %+v", ready, it)
			}
		}
		want := []string{"chat", "autostart", "settings", "quit"}
		for i := range want {
			if i >= len(ids) || ids[i] != want[i] {
				t.Fatalf("order %v, want %v", ids, want)
			}
		}
	}
}

func TestChatURL(t *testing.T) {
	for _, c := range []struct {
		host string
		port int
		want string
	}{
		{"", 47480, "http://127.0.0.1:47480/chat/"},
		{"127.0.0.1", 47481, "http://127.0.0.1:47481/chat/"},
		{"::1", 47480, "http://[::1]:47480/chat/"},
		{"0.0.0.0", 47480, ""}, // a network listener needs a token a page cannot send
		{"192.168.1.5", 47480, ""},
	} {
		if got := ChatURL(c.host, c.port); got != c.want {
			t.Errorf("ChatURL(%q,%d) = %q, want %q", c.host, c.port, got, c.want)
		}
	}
}

type quietTray struct {
	mu    sync.Mutex
	items []tray.Item
	drawn chan struct{} // a signal for each SetItems
}

// until waits for cond, looking again at each redraw; d bounds it.
func (q *quietTray) until(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	for !cond() {
		select {
		case <-q.drawn:
		case <-time.After(d): // watchdog for silence: a menu that never draws fails the test
			t.Fatalf("timed out after %s waiting for: %s", d, what)
		}
	}
}

func (q *quietTray) Clicks() <-chan string { return nil }
func (q *quietTray) SetItems(i []tray.Item) {
	q.mu.Lock()
	q.items = append([]tray.Item(nil), i...)
	q.mu.Unlock()
	select {
	case q.drawn <- struct{}{}:
	default:
	}
}
func (q *quietTray) SetTooltip(string)        {}
func (q *quietTray) Notify(a, b string) error { return nil }
func (q *quietTray) Close()                   {}
func (q *quietTray) chatDisabled() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, it := range q.items {
		if it.ID == "chat" {
			return it.Disabled
		}
	}
	return true
}

// A click on Chat shows the chat page in a new browser window only once a model is ready, and the menu
// item turns on by itself when the phase gets there.
func TestChatClickAndEnableFollowThePhase(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_HEALTH_DELAY"] = "1500ms"
	s := e.StartSup(e.Config())

	var mu sync.Mutex
	var shown []string
	popped := make(chan struct{}, 4)
	qt := &quietTray{drawn: make(chan struct{}, 1)}
	c := newController(qt, MenuOptions{Chat: func(u string) error {
		mu.Lock()
		shown = append(shown, u)
		mu.Unlock()
		popped <- struct{}{}
		return nil
	}},
		s, t.TempDir(), "http://127.0.0.1:1/chat/", func() {})
	count := func() int { mu.Lock(); defer mu.Unlock(); return len(shown) }

	testkit.WaitFor(t, 10*time.Second, "loading phase", func() bool { return s.Phase() == "loading" })
	c.refresh()
	if !qt.chatDisabled() {
		t.Fatal("Chat must be disabled while loading")
	}
	c.click("chat") // not ready: nothing is started, so there is nothing to wait for
	if count() != 0 {
		t.Fatal("a click while loading opened the chat")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go c.watch(ctx)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	qt.until(t, 6*time.Second, "Chat enabled", func() bool { return !qt.chatDisabled() })
	c.click("chat")
	select {
	case <-popped:
	case <-time.After(3 * time.Second): // watchdog for silence
		t.Fatal("the chat window was not shown")
	}
	if shown[0] != "http://127.0.0.1:1/chat/" {
		t.Fatalf("shown %q", shown[0])
	}

	// A controller with no reachable page (network listener) never has Chat on.
	c2 := newController(&quietTray{}, MenuOptions{Chat: func(string) error { t.Error("opened"); return nil }}, s, t.TempDir(), "", func() {})
	if c2.chatReady() {
		t.Fatal("Chat on with no page")
	}
	c2.click("chat") // a Chat call here fails the test through the t.Error above
}

func TestListenAddr(t *testing.T) {
	if got := listenAddr("", 47480); got != "127.0.0.1:47480" {
		t.Fatal(got)
	}
	if got := listenAddr("::1", 47481); got != "[::1]:47481" {
		t.Fatal(got)
	}
}

// The chat page's tab icon is the tray's icon: web/icon.png is a copy, kept equal here.
func TestChatIconIsTheTrayIcon(t *testing.T) {
	if !bytes.Equal(web.Icon, tray.DefaultIcon().PNG) {
		t.Fatal("web/icon.png differs from tray/icons/tray.png; copy it again")
	}
}
