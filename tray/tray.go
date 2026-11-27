// Package tray shows a system tray (notification area, menu bar) icon with a menu on Linux,
// Windows and macOS, in pure Go (no cgo).
//
// Use:
//
//	func main() { tray.Main(realMain) } // required on macOS, harmless elsewhere
//
//	func realMain() {
//		t, err := tray.Run(ctx, tray.Spec{Title: "easy-engine", Icon: tray.DefaultIcon(), Items: items})
//		if errors.Is(err, tray.ErrNoTray) { /* keep running without an icon */ }
//		for id := range t.Clicks() { ... } // closed when the tray closes
//	}
//
// Main: macOS runs its UI loop (NSApplication) on the process's main thread only. The program's
// main() therefore calls tray.Main(realMain): on macOS Main keeps the main thread, starts realMain
// in a goroutine and runs the UI loop until realMain returns; on Linux and Windows it just calls
// realMain. Main returns when realMain returns.
//
// Backends: Linux speaks StatusNotifierItem and com.canonical.dbusmenu over the session D-Bus
// (KDE Plasma, most other desktops; GNOME only with the AppIndicator extension). Windows uses
// Shell_NotifyIconW with a hidden window and a popup menu. macOS uses NSStatusItem and NSMenu
// through the Objective-C runtime, loaded with purego.
//
// Clicks are delivered on a channel, in order, as the user makes them; nothing polls. A
// checkable item is not toggled by the package: the program decides and calls SetChecked.
package tray

import (
	"context"
	"errors"
	"sync"
)

// ErrNoTray means this machine or session has nothing to show a tray icon in: on Linux no
// session D-Bus or no StatusNotifierWatcher (for example GNOME without the AppIndicator
// extension, or a headless session); on Windows a service session or no taskbar; on macOS
// tray.Main is not running the UI loop. The program keeps running without an icon.
var ErrNoTray = errors.New("tray: no system tray in this session")

// ErrNoNotify means desktop notifications are not available here.
var ErrNoNotify = errors.New("tray: desktop notifications are not available")

// Item is one menu entry. ID is what Clicks reports; an item with an empty ID is a label that
// never reports a click. Separator items ignore every other field.
type Item struct {
	ID        string
	Title     string
	Checkable bool
	Checked   bool
	Disabled  bool
	Separator bool
}

// Icon carries the image each system wants. Any field may be empty; a backend uses what it
// can and falls back to PNG.
type Icon struct {
	PNG      []byte // full colour, square, 64 px or more (Linux, and a fallback elsewhere)
	ICO      []byte // Windows icon file with several sizes (16-48 px)
	Template []byte // macOS menu bar: a black-on-transparent PNG, tinted by the system
}

// Spec is the tray at start.
type Spec struct {
	Title   string // the program's name, as the desktop lists it
	Tooltip string // hover text; Title when empty
	Icon    Icon
	Items   []Item
}

// backend is one system's tray. Tray serialises every call to it (never two at once) and
// calls close exactly once, last. A backend reports clicks through the click function it
// was made with, from any goroutine; click never blocks.
type backend interface {
	setMenu(items []Item)
	setTooltip(s string)
	notify(title, body string) error
	close()
}

// Tray is a running tray icon. Its methods are safe from any goroutine.
type Tray struct {
	mu      sync.Mutex
	b       backend
	items   []Item
	closed  bool
	clicks  chan string
	q       clickQueue
	done    chan struct{}
	pumpEnd chan struct{}
}

// Run shows the tray icon. It stays until ctx ends or Close is called. When the session has no
// tray it returns an error wrapping ErrNoTray.
func Run(ctx context.Context, spec Spec) (*Tray, error) { return start(ctx, spec, newBackend) }

func start(ctx context.Context, spec Spec, newB func(Spec, func(string)) (backend, error)) (*Tray, error) {
	if spec.Tooltip == "" {
		spec.Tooltip = spec.Title
	}
	spec.Items = cloneItems(spec.Items)
	t := &Tray{
		items:   spec.Items,
		clicks:  make(chan string),
		done:    make(chan struct{}),
		pumpEnd: make(chan struct{}),
	}
	t.q.init()
	b, err := newB(spec, t.q.push)
	if err != nil {
		return nil, err
	}
	t.b = b
	go t.pump()
	go func() {
		select {
		case <-ctx.Done():
			t.Close()
		case <-t.done:
		}
	}()
	return t, nil
}

// Clicks reports the ID of each menu item the user clicks. It is closed after Close.
func (t *Tray) Clicks() <-chan string { return t.clicks }

// SetChecked sets the checkmark of the item with this ID.
func (t *Tray) SetChecked(id string, on bool) {
	t.change(func(items []Item) {
		for i := range items {
			if items[i].ID == id && !items[i].Separator {
				items[i].Checked = on
			}
		}
	})
}

// SetTitle changes the text of the item with this ID.
func (t *Tray) SetTitle(id, title string) {
	t.change(func(items []Item) {
		for i := range items {
			if items[i].ID == id && !items[i].Separator {
				items[i].Title = title
			}
		}
	})
}

// SetItems replaces the whole menu.
func (t *Tray) SetItems(items []Item) {
	items = cloneItems(items)
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return
	}
	t.items = items
	t.b.setMenu(cloneItems(items))
}

// Items returns the menu as it is now.
func (t *Tray) Items() []Item {
	t.mu.Lock()
	defer t.mu.Unlock()
	return cloneItems(t.items)
}

// SetTooltip changes the hover text.
func (t *Tray) SetTooltip(s string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return
	}
	t.b.setTooltip(s)
}

// Notify shows a desktop notification, best effort. It returns ErrNoNotify where the system
// offers none.
func (t *Tray) Notify(title, body string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return ErrNoNotify
	}
	return t.b.notify(title, body)
}

// Close removes the icon and closes Clicks. It is safe to call more than once.
func (t *Tray) Close() {
	t.mu.Lock()
	if t.closed {
		t.mu.Unlock()
		return
	}
	t.closed = true
	t.b.close()
	close(t.done)
	t.mu.Unlock()
	t.q.stop()
	<-t.pumpEnd
}

func (t *Tray) change(f func([]Item)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return
	}
	f(t.items)
	t.b.setMenu(cloneItems(t.items))
}

// pump moves clicks from the queue (which never blocks the UI) to the Clicks channel.
func (t *Tray) pump() {
	defer close(t.pumpEnd)
	defer close(t.clicks)
	for {
		id, ok := t.q.pop()
		if !ok {
			return
		}
		select {
		case t.clicks <- id:
		case <-t.done:
			return
		}
	}
}

func cloneItems(items []Item) []Item { return append([]Item(nil), items...) }

// clickQueue is an unbounded queue: push never blocks, pop waits for an item or stop.
type clickQueue struct {
	mu      sync.Mutex
	cond    *sync.Cond
	ids     []string
	stopped bool
}

func (q *clickQueue) init() { q.cond = sync.NewCond(&q.mu) }

func (q *clickQueue) push(id string) {
	if id == "" {
		return
	}
	q.mu.Lock()
	if !q.stopped {
		q.ids = append(q.ids, id)
		q.cond.Signal()
	}
	q.mu.Unlock()
}

func (q *clickQueue) pop() (string, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for len(q.ids) == 0 && !q.stopped {
		q.cond.Wait()
	}
	if q.stopped {
		return "", false
	}
	id := q.ids[0]
	q.ids = q.ids[1:]
	return id, true
}

func (q *clickQueue) stop() {
	q.mu.Lock()
	q.stopped = true
	q.cond.Broadcast()
	q.mu.Unlock()
}
