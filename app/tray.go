package app

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
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
}

// menu is the whole menu as a function of state: three items, nothing else. A later slice
// adds an item (a pairing request that waits) by adding a field to view and a line here.
func menu(v view) []tray.Item {
	return []tray.Item{
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

	mu    sync.Mutex
	v     view
	names map[string]string
	last  string
}

func newController(t Tray, m MenuOptions, sup *run.Supervisor, dir string, quit func()) *controller {
	m.defaults()
	c := &controller{t: t, m: m, sup: sup, file: settings.Path(dir), quit: quit, names: map[string]string{}}
	if cat, err := catalog.Load(dir); err == nil && cat != nil {
		for _, md := range cat.Models {
			c.names[md.ID] = md.Name
		}
	}
	c.v.autostart, _ = m.Autostart.Enabled(AutostartName)
	c.refresh()
	return c
}

// serve handles clicks until the tray closes or ctx ends, and redraws the tooltip when what it
// says changes on the hub (the phase, the model, the door), not only at a click.
func (c *controller) serve(ctx context.Context) {
	var said <-chan struct{}
	if wk, err := c.sup.Hub().Wake("phase", "model", "door"); err == nil {
		defer wk.Close()
		said = wk.C()
	}
	c.refresh() // listen first, then look: nothing said before the watch is missed
	for {
		select {
		case <-ctx.Done():
			return
		case <-said:
			c.refresh()
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

// refresh draws the menu and the tooltip from what stands now.
func (c *controller) refresh() {
	if c == nil {
		return
	}
	c.mu.Lock()
	items := menu(c.v)
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
