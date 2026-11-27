package tray

import (
	"context"
	"sync"
	"testing"
	"time"
)

// fakeBackend records what the Tray asks of it and lets the test click.
type fakeBackend struct {
	mu     sync.Mutex
	menus  [][]Item
	tips   []string
	closed int
	click  func(string)
}

func (f *fakeBackend) setMenu(items []Item) { f.mu.Lock(); f.menus = append(f.menus, items); f.mu.Unlock() }
func (f *fakeBackend) setTooltip(s string)  { f.mu.Lock(); f.tips = append(f.tips, s); f.mu.Unlock() }
func (f *fakeBackend) notify(_, _ string) error {
	return nil
}
func (f *fakeBackend) close() { f.mu.Lock(); f.closed++; f.mu.Unlock() }

func startFake(t *testing.T, ctx context.Context, spec Spec) (*Tray, *fakeBackend) {
	t.Helper()
	f := &fakeBackend{}
	tr, err := start(ctx, spec, func(_ Spec, click func(string)) (backend, error) {
		f.click = click
		return f, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return tr, f
}

func recv(t *testing.T, c <-chan string) string {
	t.Helper()
	select {
	case id := <-c:
		return id
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("no click arrived")
		return ""
	}
}

func TestClicksInOrderWithoutAReader(t *testing.T) {
	tr, f := startFake(t, context.Background(), Spec{Title: "t"})
	defer tr.Close()
	// The backend's clicks never block, even with nobody reading yet.
	for _, id := range []string{"a", "b", "", "c"} {
		f.click(id)
	}
	for _, want := range []string{"a", "b", "c"} {
		if got := recv(t, tr.Clicks()); got != want {
			t.Fatalf("click = %q, want %q", got, want)
		}
	}
}

func TestMenuChanges(t *testing.T) {
	tr, f := startFake(t, context.Background(), Spec{Title: "t", Items: []Item{
		{ID: "x", Title: "X", Checkable: true}, {Separator: true}, {ID: "q", Title: "Quit"},
	}})
	defer tr.Close()
	tr.SetChecked("x", true)
	tr.SetTitle("q", "Leave")
	tr.SetTooltip("busy")
	got := tr.Items()
	if !got[0].Checked || got[2].Title != "Leave" {
		t.Fatalf("items = %+v", got)
	}
	f.mu.Lock()
	n, last, tips := len(f.menus), f.menus[len(f.menus)-1], f.tips
	f.mu.Unlock()
	if n != 2 || !last[0].Checked || last[2].Title != "Leave" || len(tips) != 1 || tips[0] != "busy" {
		t.Fatalf("backend saw %d menus, last %+v, tips %v", n, last, tips)
	}
	// The caller's slice is not shared with the tray.
	items := []Item{{ID: "only", Title: "Only"}}
	tr.SetItems(items)
	items[0].Title = "changed"
	if tr.Items()[0].Title != "Only" {
		t.Fatal("SetItems kept the caller's slice")
	}
}

func TestContextEndCloses(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	tr, f := startFake(t, ctx, Spec{Title: "t"})
	cancel()
	for range tr.Clicks() { // closed by the context's end
	}
	tr.Close() // a second close is a no-op
	tr.SetChecked("x", true)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed != 1 || len(f.menus) != 0 {
		t.Fatalf("closed %d times, %d menus after close", f.closed, len(f.menus))
	}
}
