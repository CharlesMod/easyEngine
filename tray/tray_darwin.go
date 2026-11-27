//go:build darwin

package tray

import (
	"fmt"
	"os/exec"
	"sync"

	"github.com/ebitengine/purego/objc"
)

// macOS backend: NSStatusItem + NSMenu through the Objective-C runtime (purego, no cgo).
// Every AppKit call runs on the main thread (see onMain).

var (
	targetOnce  sync.Once
	targetClass objc.Class
	targetErr   error

	// backends maps a menu target object to its backend, for menuClick:.
	backendsMu sync.Mutex
	backends   = map[objc.ID]*darwinBackend{}
)

// registerTarget makes the NSObject subclass whose menuClick: receives menu clicks.
func registerTarget() (objc.Class, error) {
	targetOnce.Do(func() {
		targetClass, targetErr = objc.RegisterClass(
			"EasyEngineTrayTarget",
			objc.GetClass("NSObject"),
			nil, nil,
			[]objc.MethodDef{{
				Cmd: sel("menuClick:"),
				Fn: func(self objc.ID, _ objc.SEL, sender objc.ID) {
					menuClicked(self, sender)
				},
			}},
		)
	})
	return targetClass, targetErr
}

func menuClicked(self, sender objc.ID) {
	tag := objc.Send[int](sender, sel("tag"))
	backendsMu.Lock()
	b := backends[self]
	backendsMu.Unlock()
	if b == nil {
		return
	}
	b.mu.Lock()
	id := ""
	if tag >= 1 && tag <= len(b.items) {
		id = b.items[tag-1].ID
	}
	click := b.click
	b.mu.Unlock()
	if id != "" {
		click(id)
	}
}

type darwinBackend struct {
	click func(string)

	mu     sync.Mutex // guards items and closed
	items  []Item
	closed bool

	// touched only on the main thread
	statusItem objc.ID
	button     objc.ID
	target     objc.ID
}

func newBackend(spec Spec, click func(string)) (backend, error) {
	mainMu.Lock()
	running := loopRunning
	mainMu.Unlock()
	if !running {
		return nil, fmt.Errorf("tray: call tray.Main from main() on macOS: %w", ErrNoTray)
	}
	cls, err := registerTarget()
	if err != nil {
		return nil, fmt.Errorf("tray: %v: %w", err, ErrNoTray)
	}
	b := &darwinBackend{click: click, items: cloneItems(spec.Items)}
	var failed bool
	ok := onMain(func() {
		bar := class("NSStatusBar").Send(sel("systemStatusBar"))
		// NSVariableStatusItemLength = -1
		item := bar.Send(sel("statusItemWithLength:"), float64(-1))
		if item == 0 {
			failed = true
			return
		}
		item.Send(sel("retain"))
		b.statusItem = item
		b.button = item.Send(sel("button"))
		b.target = objc.ID(cls).Send(sel("alloc")).Send(sel("init"))
		backendsMu.Lock()
		backends[b.target] = b
		backendsMu.Unlock()

		var img objc.ID
		if len(spec.Icon.Template) > 0 {
			img = nsImage(spec.Icon.Template, true)
		} else {
			img = nsImage(spec.Icon.PNG, false)
		}
		if img != 0 && b.button != 0 {
			b.button.Send(sel("setImage:"), img)
			img.Send(sel("release"))
		}
		if b.button != 0 {
			if img == 0 {
				// No image: show the title so the item is not invisible.
				b.button.Send(sel("setTitle:"), nsString(spec.Title))
			}
			b.button.Send(sel("setToolTip:"), nsString(spec.Tooltip))
		}
		b.applyMenu(b.items)
	})
	if !ok || failed {
		return nil, fmt.Errorf("tray: no status item: %w", ErrNoTray)
	}
	return b, nil
}

// applyMenu builds a fresh NSMenu and installs it (main thread only).
func (b *darwinBackend) applyMenu(items []Item) {
	menu := class("NSMenu").Send(sel("alloc")).Send(sel("initWithTitle:"), nsString(""))
	menu.Send(sel("setAutoenablesItems:"), false)
	for i, it := range items {
		if it.Separator {
			menu.Send(sel("addItem:"), class("NSMenuItem").Send(sel("separatorItem")))
			continue
		}
		var action objc.SEL
		if it.ID != "" {
			action = sel("menuClick:")
		}
		mi := class("NSMenuItem").Send(sel("alloc")).Send(
			sel("initWithTitle:action:keyEquivalent:"), nsString(it.Title), action, nsString(""))
		mi.Send(sel("setTag:"), i+1)
		if it.ID != "" {
			mi.Send(sel("setTarget:"), b.target)
		}
		mi.Send(sel("setEnabled:"), !it.Disabled)
		if it.Checkable {
			state := 0
			if it.Checked {
				state = 1
			}
			mi.Send(sel("setState:"), state)
		}
		menu.Send(sel("addItem:"), mi)
		mi.Send(sel("release"))
	}
	b.statusItem.Send(sel("setMenu:"), menu)
	menu.Send(sel("release")) // the status item holds it now
}

func (b *darwinBackend) setMenu(items []Item) {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.items = items
	b.mu.Unlock()
	onMain(func() {
		b.mu.Lock()
		closed := b.closed
		cur := b.items
		b.mu.Unlock()
		if !closed {
			b.applyMenu(cur)
		}
	})
}

func (b *darwinBackend) setTooltip(s string) {
	onMain(func() {
		b.mu.Lock()
		closed := b.closed
		b.mu.Unlock()
		if !closed && b.button != 0 {
			b.button.Send(sel("setToolTip:"), nsString(s))
		}
	})
}

func (b *darwinBackend) notify(title, body string) error {
	err := exec.Command("/usr/bin/osascript",
		"-e", "on run argv",
		"-e", "display notification (item 2 of argv) with title (item 1 of argv)",
		"-e", "end run",
		"--", title, body).Run()
	if err != nil {
		return fmt.Errorf("%w: %v", ErrNoNotify, err)
	}
	return nil
}

func (b *darwinBackend) close() {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	b.mu.Unlock()
	onMain(func() {
		if b.statusItem != 0 {
			class("NSStatusBar").Send(sel("systemStatusBar")).Send(sel("removeStatusItem:"), b.statusItem)
			b.statusItem.Send(sel("release"))
			b.statusItem = 0
		}
		backendsMu.Lock()
		delete(backends, b.target)
		backendsMu.Unlock()
		if b.target != 0 {
			b.target.Send(sel("release"))
			b.target = 0
		}
	})
}
