package app

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/CharlesMod/easyEngine/autostart"
	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/openfile"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
	"github.com/CharlesMod/easyEngine/tray"
)

// AutostartName is the engine's name in the system's startup entries.
const AutostartName = "easy-engine"

// Tray is the part of *tray.Tray the app uses; tests give a fake.
type Tray interface {
	Clicks() <-chan string
	SetItems(items []tray.Item)
	SetTooltip(s string)
	Notify(title, body string) error
	Close()
}

// Autostarter is "Launch at Startup" (*autostart.Place, whose Real() is this user's own).
type Autostarter interface {
	Enabled(name string) (bool, error)
	Enable(name, exe string, args []string) error
	Disable(name string) error
}

// MenuOptions are what the tray menu acts on. The zero value is the real thing: this user's
// own startup entry, the system's own editor, this program.
type MenuOptions struct {
	Autostart Autostarter                                    // default autostart.Real()
	Open      func(path string) error                        // default openfile.Open
	Chat      func(url string) error                         // shows the chat page in a small window; default openfile.Popup
	Exe       string                                         // the program to start at login; default this one (ExePath)
	Args      []string                                       // its arguments; default ["app"]
	Run       func(context.Context, tray.Spec) (Tray, error) // default tray.Run
}

func (m *MenuOptions) defaults() {
	if m.Autostart == nil {
		m.Autostart = autostart.Real()
	}
	if m.Open == nil {
		m.Open = func(p string) error { _, err := openfile.Open(p); return err }
	}
	if m.Chat == nil {
		m.Chat = openfile.Popup
	}
	if m.Exe == "" {
		m.Exe = ExePath()
	}
	if m.Args == nil {
		m.Args = []string{"app"}
	}
	if m.Run == nil {
		m.Run = func(ctx context.Context, s tray.Spec) (Tray, error) {
			t, err := tray.Run(ctx, s)
			if err != nil {
				return nil, err // not a typed nil inside the interface
			}
			return t, nil
		}
	}
}

// ExePath is the program a login start should run: this one with symlinks resolved, and on
// Windows the windowless easy-enginew.exe beside it when there is one.
func ExePath() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	if r, err := filepath.EvalSymlinks(exe); err == nil {
		exe = r
	}
	if runtime.GOOS == "windows" {
		w := filepath.Join(filepath.Dir(exe), "easy-enginew.exe")
		if _, err := os.Stat(w); err == nil {
			return w
		}
	}
	return exe
}

// view is what the menu is drawn from.
type view struct {
	autostart bool
	chat      bool // a model is ready to talk to, and this engine's chat page can be reached
}

// menu is the whole menu as a function of state: four items, nothing else. A later slice
// adds an item (a pairing request that waits) by adding a field to view and a line here.
func menu(v view) []tray.Item {
	return []tray.Item{
		{ID: "chat", Title: "Chat", Disabled: !v.chat},
		{ID: "autostart", Title: "Launch at Startup", Checkable: true, Checked: v.autostart},
		{ID: "settings", Title: "Settings"},
		{ID: "quit", Title: "Quit"},
	}
}

// StartTray shows the tray icon. It returns an error wrapping tray.ErrNoTray when this session
// has nothing to show it in; the caller then runs without one.
func StartTray(ctx context.Context, m MenuOptions) (Tray, error) {
	m.defaults()
	on, _ := m.Autostart.Enabled(AutostartName)
	return m.Run(ctx, tray.Spec{
		Title: "easy-engine", Tooltip: "easy-engine: starting", Icon: tray.DefaultIcon(),
		Items: menu(view{autostart: on}),
	})
}

// controller serves a tray: its clicks and its tooltip.
type controller struct {
	t    Tray
	m    MenuOptions
	sup  *run.Supervisor
	file string // settings.json
	quit func()
	chat string // the chat page in popup form ("" when this engine's page cannot be reached from here)

	mu    sync.Mutex
	v     view
	names map[string]string
	last  string
	drawn []tray.Item // the menu as last drawn
}

func newController(t Tray, m MenuOptions, sup *run.Supervisor, dir, chat string, quit func()) *controller {
	m.defaults()
	c := &controller{t: t, m: m, sup: sup, file: settings.Path(dir), quit: quit, chat: chat, names: map[string]string{}}
	if cat, err := catalog.Load(dir); err == nil && cat != nil {
		for _, md := range cat.Models {
			c.names[md.ID] = md.Name
		}
	}
	c.v.autostart, _ = m.Autostart.Enabled(AutostartName)
	c.refresh()
	return c
}

// serve handles clicks until the tray closes or ctx ends (watch follows the hub).
func (c *controller) serve(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case id, ok := <-c.t.Clicks():
			if !ok {
				return
			}
			c.click(id)
		}
	}
}

func (c *controller) click(id string) {
	switch id {
	case "autostart":
		c.toggleAutostart()
	case "chat":
		if c.chatReady() {
			go func() { // the launcher waits for the browser to start
				if err := c.m.Chat(c.chat); err != nil {
					log.Printf("easy-engine: open chat %s: %v", c.chat, err)
				}
			}()
		}
	case "settings":
		go func() { // the opener waits for the editor to start
			if err := c.m.Open(c.file); err != nil {
				log.Printf("easy-engine: open %s: %v", c.file, err)
			}
		}()
	case "quit":
		c.quit()
	}
	c.refresh()
}

// toggleAutostart flips Launch at Startup, then shows what the system now says.
func (c *controller) toggleAutostart() {
	on, _ := c.m.Autostart.Enabled(AutostartName)
	var err error
	if on {
		err = c.m.Autostart.Disable(AutostartName)
	} else {
		err = c.m.Autostart.Enable(AutostartName, c.m.Exe, c.m.Args)
	}
	if err != nil {
		log.Printf("easy-engine: launch at startup: %v", err)
		c.t.Notify("easy-engine", "Could not change Launch at Startup: "+err.Error())
	}
	c.mu.Lock()
	c.v.autostart, _ = c.m.Autostart.Enabled(AutostartName)
	c.mu.Unlock()
}

// chatReady: Chat is on only while a model is ready and the page is reachable.
func (c *controller) chatReady() bool {
	return c.chat != "" && c.sup.Status().Phase == "ready"
}

// ChatURL is the chat page in its popup form (?mini=1: the bar with one button) on this engine,
// or "" when the engine listens beyond loopback, where the page cannot send a token.
func ChatURL(host string, port int) string {
	if !IsLoopbackHost(host) {
		return ""
	}
	return "http://" + listenAddr(host, port) + "/chat/?mini=1"
}

// listenAddr is host:port as a client on this machine reaches the engine (an empty host is loopback).
func listenAddr(host string, port int) string {
	if host == "" {
		host = "127.0.0.1"
	}
	return net.JoinHostPort(strings.Trim(host, "[]"), strconv.Itoa(port))
}

// watch redraws the menu and tooltip when the supervisor's "phase", "model" or "door" fact
// changes (Chat turns on when a model is ready; the tooltip says the model and the requests the
// door holds), until ctx ends. It is driven by the engine's own event hub (Wake, then read the
// state), not a clock.
func (c *controller) watch(ctx context.Context) {
	w, err := c.sup.Hub().Wake("phase", "model", "door") // made before the first read, so no change slips between
	if err != nil {
		log.Printf("easy-engine: tray cannot follow the phase: %v", err)
		return
	}
	defer w.Close()
	for {
		c.mu.Lock()
		c.v.chat = c.chatReady()
		stale := c.tooltip() != c.last || !reflect.DeepEqual(menu(c.v), c.drawn)
		c.mu.Unlock()
		if stale {
			c.refresh()
		}
		select {
		case <-ctx.Done():
			return
		case <-c.sup.Hub().Done():
			return
		case <-w.C():
		}
	}
}

// refresh draws the menu and the tooltip from what stands now.
func (c *controller) refresh() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.v.chat = c.chatReady()
	items := menu(c.v)
	c.drawn = items
	tip := c.tooltip()
	same := tip == c.last
	c.last = tip
	c.mu.Unlock()
	c.t.SetItems(items)
	if !same {
		c.t.SetTooltip(tip)
	}
}

// tooltip is one line: the phase and the model, and the requests the door holds while the model
// comes up.
func (c *controller) tooltip() string {
	st := c.sup.Status()
	tip := c.phaseTip(st)
	if st.Held > 0 {
		tip += fmt.Sprintf(" (%d waiting)", st.Held)
	}
	return tip
}

func (c *controller) phaseTip(st run.Status) string {
	name := st.Model
	if n := c.names[st.Model]; n != "" {
		name = n
	}
	switch st.Phase {
	case "ready":
		return fmt.Sprintf("easy-engine: ready, %s", name)
	case "resizing":
		return fmt.Sprintf("easy-engine: ready, %s (resizing)", name)
	case "starting", "loading":
		return fmt.Sprintf("easy-engine: loading %s", name)
	case "sleeping":
		return "easy-engine: no model loaded"
	case "not_installed":
		return "easy-engine: no model installed"
	case "external":
		return "easy-engine: using your own endpoint"
	case "downloading", "installing":
		return "easy-engine: " + st.Phase + " " + name
	}
	return strings.TrimSpace("easy-engine: " + st.Phase)
}
