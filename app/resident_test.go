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
	"github.com/CharlesMod/easyEngine/event"
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
	changed chan struct{} // a signal for each SetItems and SetTooltip (one pending is enough)
}

func newFakeTray() *fakeTray {
	return &fakeTray{clicks: make(chan string, 8), drawn: make(chan struct{}, 64), changed: make(chan struct{}, 1)}
}

func (f *fakeTray) signal() {
	select {
	case f.changed <- struct{}{}:
	default:
	}
}

// until waits for ok to hold of what the tray shows, re-checking at every redraw. The one timer
// is a watchdog for silence: the app drew nothing that made ok true.
func (f *fakeTray) until(t *testing.T, what string, ok func(items []tray.Item, tip string) bool) {
	t.Helper()
	dog := time.NewTimer(20 * time.Second) // watchdog for silence: the tray was not redrawn into what the wait wants
	defer dog.Stop()
	for {
		items, tip, _ := f.snap()
		if ok(items, tip) {
			return
		}
		select {
		case <-f.changed:
		case <-dog.C: // watchdog for silence
			_, tip, _ := f.snap()
			t.Fatalf("the tray never showed %s; the tooltip is %q", what, tip)
		}
	}
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
	f.signal()
}
func (f *fakeTray) SetTooltip(s string)             { f.mu.Lock(); f.tooltip = s; f.mu.Unlock(); f.signal() }
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
	opened := make(chan string, 4) // the opener signals each call
	exe := filepath.Join(t.TempDir(), "easy-engine")
	r := startResident(t, app.Options{Tray: ft, Menu: app.MenuOptions{
		Autostart: place, Exe: exe,
		Open: func(p string) error { opened <- p; return nil },
	}})
	// Exactly three items, in this order, and a tooltip that says what stands.
	ft.until(t, "the menu", func(items []tray.Item, tip string) bool {
		return len(items) == 3 && strings.HasPrefix(tip, "easy-engine: ")
	})
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
	ft.until(t, "the checkmark", func([]tray.Item, string) bool { return ft.checked() })
	if on, _ := place.Enabled(app.AutostartName); !on {
		t.Fatal("autostart was not enabled")
	}
	b, err := os.ReadFile(place.Path(app.AutostartName))
	if err != nil || !strings.Contains(string(b), exe) || !strings.Contains(string(b), "app") {
		t.Fatalf("entry: %q %v", b, err)
	}
	ft.clicks <- "autostart"
	ft.until(t, "the checkmark off", func([]tray.Item, string) bool { return !ft.checked() })
	if on, _ := place.Enabled(app.AutostartName); on {
		t.Fatal("autostart was not disabled")
	}
	// Settings: the opener gets settings.json, which exists.
	ft.clicks <- "settings"
	var path string
	select {
	case path = <-opened:
	case <-time.After(20 * time.Second): // watchdog for silence: the click never reached the opener
		t.Fatal("the opener was not called")
	}
	if path != filepath.Join(r.dir, "settings.json") {
		t.Fatalf("opened %q", path)
	}
	if _, err := os.Stat(path); err != nil {
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
	r.await(t, "the edit to take effect", func() bool { return read().LingerS == 11 })
	if read().KVType != "q8_0" {
		t.Fatalf("%+v", read())
	}
	// A broken file is reported, the last good settings stand, the file is not touched.
	broken := "{\n  \"linger_s\": 99,\n  \"kv_type\": \"nonsense\"\n}\n"
	os.WriteFile(p, []byte(broken), 0o600)
	r.await(t, "the error", func() bool { return r.errHas(t, "settings.json line 3") })
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
	r.await(t, "the fix", func() bool { return read().LingerS == 12 && !r.errHas(t, "line 3") })
	r.await(t, "the restart note", func() bool { return r.errHas(t, "restart easy-engine to apply the new listen address") })
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
	return startFakeModelWith(t, linger, parents, nil, nil)
}

// startFakeModelWith is startFakeModel with extra FAKELLAMA_* environment and a tweak of the
// app's options.
func startFakeModelWith(t *testing.T, linger time.Duration, parents []int, llama map[string]string, tweak func(*app.Options)) *residentRun {
	e := testkit.New(t)
	e.InstallFake()
	for k, v := range llama {
		e.Llama[k] = v
	}
	o := app.Options{
		DataDir: e.Data, Linger: linger, LingerSet: true, ParentPIDs: parents,
		Configure: func(c *run.Config) {
			ec := e.Config()
			ec.Parents, ec.Yield, ec.Group, ec.StartSleeping = c.Parents, c.Yield, c.Group, c.StartSleeping
			*c = ec
		},
	}
	if tweak != nil {
		tweak(&o)
	}
	return startResident(t, o)
}

func waitPhase(t *testing.T, r *residentRun, phase string) {
	t.Helper()
	r.await(t, "phase "+phase, func() bool { return r.status(t).Phase == phase })
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
	// A new client's attach wakes the model again: the engine that answers it is the one that
	// slept the model, so it did not end with its parent.
	par2 := newSleeper(t)
	if err := app.PostAttach(app.EngineTarget(r.dir, "127.0.0.1", r.port), par2.pid); err != nil {
		t.Fatal(err)
	}
	waitPhase(t, r, "ready")
	select {
	case <-r.res:
		t.Fatal("the engine ended with its parent")
	default: // the engine answered the attach after the model slept: it is still serving
	}
	par2.kill()
	waitPhase(t, r, "sleeping")
}

// A parent that returns while the linger runs keeps the model up: the model sleeps only after
// the last client left, and no sooner than the linger after it (the events' own timestamps).
func TestLingerHoldsWhileAParentReturns(t *testing.T) {
	const linger = 1500 * time.Millisecond
	par := newSleeper(t)
	r := startFakeModel(t, linger, []int{par.pid})
	waitPhase(t, r, "ready")
	s := r.follow(t)
	par.kill()
	par2 := newSleeper(t)
	if err := app.PostAttach(app.EngineTarget(r.dir, "127.0.0.1", r.port), par2.pid); err != nil {
		t.Fatal(err)
	}
	left := time.Now() // the last client leaves now
	par2.kill()
	evs := s.until(t, "the model asleep", func(evs []event.Event) bool {
		return indexOf(evs, 0, func(e event.Event) bool { return phaseIs(e, "sleeping") }) >= 0
	})
	slept := evs[indexOf(evs, 0, func(e event.Event) bool { return phaseIs(e, "sleeping") })]
	if d := slept.At.Sub(left); d < linger {
		t.Fatalf("the model slept %v after the last client left, linger is %v", d, linger)
	}
}
