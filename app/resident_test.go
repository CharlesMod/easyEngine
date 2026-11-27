//go:build !windows

package app_test

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/autostart"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
	"github.com/CharlesMod/easyEngine/tray"
)

func TestMain(m *testing.M) {
	code := m.Run()
	testkit.Cleanup()
	os.Exit(code)
}

// fakeTray stands in for the system tray: it records what the app draws and lets a test click.
type fakeTray struct {
	clicks chan string

	mu      sync.Mutex
	items   []tray.Item
	tooltip string
	closed  bool
	drawn   chan struct{} // a signal for each SetItems
}

func newFakeTray() *fakeTray {
	return &fakeTray{clicks: make(chan string, 8), drawn: make(chan struct{}, 64)}
}

func (f *fakeTray) Clicks() <-chan string { return f.clicks }
func (f *fakeTray) SetItems(items []tray.Item) {
	f.mu.Lock()
	f.items = append([]tray.Item(nil), items...)
	f.mu.Unlock()
	select {
	case f.drawn <- struct{}{}:
	default:
	}
}
func (f *fakeTray) SetTooltip(s string)             { f.mu.Lock(); f.tooltip = s; f.mu.Unlock() }
func (f *fakeTray) Notify(title, body string) error { return nil }
func (f *fakeTray) Close()                          { f.mu.Lock(); f.closed = true; f.mu.Unlock() }
func (f *fakeTray) snap() ([]tray.Item, string, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]tray.Item(nil), f.items...), f.tooltip, f.closed
}

func (f *fakeTray) checked() bool {
	items, _, _ := f.snap()
	for _, it := range items {
		if it.ID == "autostart" {
			return it.Checked
		}
	}
	return false
}

type residentRun struct {
	dir  string
	port int
	res  chan app.Result
	stop context.CancelFunc
}

// startResident runs the app engine in a goroutine and waits until it listens.
func startResident(t *testing.T, o app.Options) *residentRun {
	t.Helper()
	if o.DataDir == "" {
		o.DataDir = t.TempDir()
	}
	o.Resident = true
	o.Version = "t"
	if o.Port == 0 {
		o.Port = 47620
	}
	if o.Profile == nil {
		o.Profile = prof()
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &residentRun{dir: o.DataDir, res: make(chan app.Result, 1), stop: cancel}
	ported := make(chan int, 1)
	o.Started = func(p int) { ported <- p }
	go func() {
		res, err := app.Serve(ctx, o)
		if err != nil {
			t.Errorf("serve: %v", err)
		}
		r.res <- res
	}()
	select {
	case r.port = <-ported:
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("engine did not start")
	}
	t.Cleanup(func() {
		cancel()
		select {
		case <-r.res:
		case <-time.After(20 * time.Second): // watchdog for silence
		}
	})
	return r
}

func (r *residentRun) get(t *testing.T, path string, v any) {
	t.Helper()
	resp, err := http.Get("http://127.0.0.1:" + strconv.Itoa(r.port) + path)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatal(err)
	}
}

type statusView struct {
	Phase  string   `json:"phase"`
	Errors []string `json:"errors"`
}

func (r *residentRun) status(t *testing.T) statusView {
	var s statusView
	r.get(t, "/engine/v1/status", &s)
	return s
}

func (r *residentRun) errHas(t *testing.T, sub string) bool {
	for _, e := range r.status(t).Errors {
		if strings.Contains(e, sub) {
			return true
		}
	}
	return false
}

func TestTrayMenuClicksAndQuit(t *testing.T) {
	ft := newFakeTray()
	place := autostart.Root(t.TempDir())
	var mu sync.Mutex
	var opened []string
	exe := filepath.Join(t.TempDir(), "easy-engine")
	r := startResident(t, app.Options{Tray: ft, Menu: app.MenuOptions{
		Autostart: place, Exe: exe,
		Open: func(p string) error { mu.Lock(); opened = append(opened, p); mu.Unlock(); return nil },
	}})
	// Exactly three items, in this order, and a tooltip that says what stands.
	waitFor(t, func() bool {
		items, tip, _ := ft.snap()
		return len(items) == 3 && strings.HasPrefix(tip, "easy-engine: ")
	}, "the menu")
	items, tip, _ := ft.snap()
	want := []string{"Launch at Startup", "Settings", "Quit"}
	for i, it := range items {
		if it.Title != want[i] {
			t.Fatalf("item %d is %q, want %q", i, it.Title, want[i])
		}
	}
	if !items[0].Checkable || items[0].Checked {
		t.Fatalf("Launch at Startup: %+v", items[0])
	}
	if tip != "easy-engine: no model installed" {
		t.Fatalf("tooltip %q", tip)
	}
	// Launch at Startup: a click turns it on in the (fake) place, the check follows what the OS says.
	ft.clicks <- "autostart"
	waitFor(t, ft.checked, "the checkmark")
	if on, _ := place.Enabled(app.AutostartName); !on {
		t.Fatal("autostart was not enabled")
	}
	b, err := os.ReadFile(place.Path(app.AutostartName))
	if err != nil || !strings.Contains(string(b), exe) || !strings.Contains(string(b), "app") {
		t.Fatalf("entry: %q %v", b, err)
	}
	ft.clicks <- "autostart"
	waitFor(t, func() bool { return !ft.checked() }, "the checkmark off")
	if on, _ := place.Enabled(app.AutostartName); on {
		t.Fatal("autostart was not disabled")
	}
	// Settings: the opener gets settings.json, which exists.
	ft.clicks <- "settings"
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(opened) == 1 }, "the opener")
	if opened[0] != filepath.Join(r.dir, "settings.json") {
		t.Fatalf("opened %q", opened[0])
	}
	if _, err := os.Stat(opened[0]); err != nil {
		t.Fatal(err)
	}
	// Quit ends the app and removes the lock.
	ft.clicks <- "quit"
	select {
	case res := <-r.res:
		if res.Attached {
			t.Fatal("attached")
		}
	case <-time.After(20 * time.Second): // watchdog for silence
		t.Fatal("Quit did not end the app")
	}
	if _, err := os.Stat(filepath.Join(r.dir, "engine.lock")); err == nil {
		t.Fatal("lock left behind")
	}
	if _, _, closed := ft.snap(); !closed {
		t.Fatal("the tray was not closed")
	}
}

func TestAppSecondInstanceAttachesAndAddsNoParent(t *testing.T) {
	r := startResident(t, app.Options{})
	res, err := app.Serve(context.Background(), app.Options{DataDir: r.dir, Resident: true, Profile: prof(), Port: 47620})
	if err != nil || !res.Attached || res.Port != r.port {
		t.Fatalf("second app: %+v %v", res, err)
	}
	if got := r.status(t).Phase; got == "stopping" {
		t.Fatal("first engine is stopping")
	}
}

func TestSettingsFileIsFollowedLive(t *testing.T) {
	r := startResident(t, app.Options{})
	p := filepath.Join(r.dir, "settings.json")
	if f, err := settings.Read(p); err != nil || !f.Equal(settings.Defaults()) {
		t.Fatalf("first start did not write the defaults: %+v %v", f, err)
	}
	type sv struct {
		LingerS int    `json:"linger_s"`
		KVType  string `json:"kv_type"`
	}
	read := func() sv { var s sv; r.get(t, "/engine/v1/settings", &s); return s }

	f := settings.Defaults()
	f.LingerS, f.KVType = 11, "q8_0"
	os.WriteFile(p, settings.Encode(f), 0o600)
	waitFor(t, func() bool { return read().LingerS == 11 }, "the edit to take effect")
	if read().KVType != "q8_0" {
		t.Fatalf("%+v", read())
	}
	// A broken file is reported, the last good settings stand, the file is not touched.
	broken := "{\n  \"linger_s\": 99,\n  \"kv_type\": \"nonsense\"\n}\n"
	os.WriteFile(p, []byte(broken), 0o600)
	waitFor(t, func() bool { return r.errHas(t, "settings.json line 3") }, "the error")
	if !r.errHas(t, "using the previous settings") || read().LingerS != 11 {
		t.Fatalf("%+v %v", read(), r.status(t).Errors)
	}
	if b, _ := os.ReadFile(p); string(b) != broken {
		t.Fatal("the engine overwrote a file it could not read")
	}
	// Fixing it clears the error.
	f.LingerS = 12
	f.Listen.Port = 47999
	os.WriteFile(p, settings.Encode(f), 0o600)
	waitFor(t, func() bool { return read().LingerS == 12 && !r.errHas(t, "line 3") }, "the fix")
	waitFor(t, func() bool { return r.errHas(t, "restart easy-engine to apply the new listen address") }, "the restart note")
}

func TestPutSettingsWritesTheFile(t *testing.T) {
	r := startResident(t, app.Options{})
	req, _ := http.NewRequest("PUT", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/engine/v1/settings", strings.NewReader(`{"linger_s": 21}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("%v %v", resp, err)
	}
	resp.Body.Close()
	f, err := settings.Read(filepath.Join(r.dir, "settings.json"))
	if err != nil || f.LingerS != 21 || f.Listen.Port != 47480 {
		t.Fatalf("%+v %v", f, err)
	}
}

// startFakeModel runs the resident app on an installed fake model.
func startFakeModel(t *testing.T, linger time.Duration, parents []int) *residentRun {
	e := testkit.New(t)
	e.InstallFake()
	return startResident(t, app.Options{
		DataDir: e.Data, Linger: linger, LingerSet: true, ParentPIDs: parents,
		Configure: func(c *run.Config) {
			ec := e.Config()
			ec.Parents, ec.Yield, ec.Group = c.Parents, c.Yield, c.Group
			*c = ec
		},
	})
}

func waitPhase(t *testing.T, r *residentRun, phase string) {
	t.Helper()
	waitFor(t, func() bool { return r.status(t).Phase == phase }, "phase "+phase)
}

func TestLastParentGoneThenLingerPutsTheModelToSleepNotTheEngine(t *testing.T) {
	par := newSleeper(t)
	r := startFakeModel(t, 200*time.Millisecond, []int{par.pid})
	waitPhase(t, r, "ready")
	par.kill()
	waitPhase(t, r, "sleeping")
	// The engine is still there, and still has its lock.
	if _, err := os.Stat(filepath.Join(r.dir, "engine.lock")); err != nil {
		t.Fatal("no lock:", err)
	}
	select {
	case <-r.res:
		t.Fatal("the engine ended with its parent")
	case <-time.After(500 * time.Millisecond): // watchdog for silence
	}
	// A new client's attach wakes the model again.
	par2 := newSleeper(t)
	if err := app.PostAttach(app.EngineTarget(r.dir, "127.0.0.1", r.port), par2.pid); err != nil {
		t.Fatal(err)
	}
	waitPhase(t, r, "ready")
	par2.kill()
	waitPhase(t, r, "sleeping")
}

func TestLingerHoldsWhileAParentReturns(t *testing.T) {
	par := newSleeper(t)
	r := startFakeModel(t, 1500*time.Millisecond, []int{par.pid})
	waitPhase(t, r, "ready")
	par.kill()
	par2 := newSleeper(t)
	time.Sleep(1200 * time.Millisecond) // watchdog for silence: inside the linger (the watcher notices at 1 s)
	if err := app.PostAttach(app.EngineTarget(r.dir, "127.0.0.1", r.port), par2.pid); err != nil {
		t.Fatal(err)
	}
	select {
	case <-time.After(2500 * time.Millisecond): // watchdog for silence: well past the first linger
	}
	if got := r.status(t).Phase; got != "ready" {
		t.Fatalf("phase %q: the model slept with a client attached", got)
	}
}
