// Package app wires lifecycle, run and api into the `serve` command.
package app

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/dirs"
	osevents "github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

// DefaultPort is the control API's first port.
const DefaultPort = 47480

// Options for Serve.
type Options struct {
	DataDir     string
	ProfilePath string
	Profile     *run.Profile // overrides ProfilePath (tests)
	ParentPIDs  []int
	Host        string // default 127.0.0.1
	// Open is serve --open: a non-loopback Host with no token. Without it a non-loopback
	// listener requires the bearer token in <data dir>/tokens.json.
	Open        bool
	Port        int // default 47480
	Linger      time.Duration
	LingerSet   bool // Linger given on the command line; else the saved setting applies
	Version     string
	DetachStdio bool // serve mode: stdio goes to logs/engine-stdio.log (capped), never to the inherited pipes
	Configure   func(*run.Config)
	// Yield, when set, makes the engine yield the card to other programs (serve --yield-to-games);
	// the profile's yield_except ids are added to it.
	Yield *run.YieldConfig
	// Started is called once the engine listens (not when it attached to another).
	Started func(port int)

	// Resident is the app (easy-engine app): the engine never ends with its clients. The model
	// waits unloaded for a connection (an attach or a request), and when the last attached
	// parent is gone and nothing is in flight, the linger runs and the model is put back to
	// sleep; the engine stays until Quit, a signal or a shutdown request.
	Resident bool
	// Tray, when set, is a tray icon already shown (StartTray); Serve answers its clicks, keeps
	// its tooltip, and closes it at the end. Menu says what the three items act on.
	Tray Tray
	Menu MenuOptions
}

// Result of Serve.
type Result struct {
	Attached bool
	Port     int
	Host     string // the listen host: of this engine, or of the one attached to (from its lock)
}

func dataDir(o Options) string {
	if o.DataDir != "" {
		return o.DataDir
	}
	return dirs.Data()
}

var statusClient = &http.Client{Timeout: 2 * time.Second}

func statusOK(t Target) bool {
	req, err := t.NewRequest("GET", "/engine/v1/status", nil)
	if err != nil {
		return false
	}
	resp, err := statusClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var st struct{ Version string }
	return resp.StatusCode == 200 && json.NewDecoder(resp.Body).Decode(&st) == nil && st.Version != ""
}

// PostAttach asks a running engine to watch another parent. It returns ErrStopping (wrapped)
// when the engine has begun shutting down and refuses.
func PostAttach(t Target, pid int) error {
	b, _ := json.Marshal(map[string]int{"pid": pid})
	req, err := t.NewRequest("POST", "/engine/v1/attach", bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := statusClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return statusErr(resp)
	}
	return nil
}

func alive(d lifecycle.LockData) bool {
	return d.Alive()
}

// Running reports the port of a healthy engine in dir, if any.
func Running(dir string) (int, bool) {
	t, ok := Find(dir)
	return t.Port, ok
}

// Serve runs the engine until the parents are gone, the context ends or shutdown
// is requested. If another engine is healthy it attaches the parents to it and
// returns at once with Attached set.
func Serve(ctx context.Context, o Options) (Result, error) {
	dir := dataDir(o)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Result{}, err
	}
	lockPath := filepath.Join(dir, "engine.lock")
	host := o.Host
	if host == "" {
		host = "127.0.0.1"
	}
	base := o.Port
	if base == 0 {
		base = DefaultPort
	}
	var lock *lifecycle.Lock
	var ln net.Listener
	var port int
	for try := 0; lock == nil; try++ {
		if try > 4 {
			return Result{}, errors.New("could not take engine.lock")
		}
		if d, err := lifecycle.ReadLock(lockPath); err == nil {
			if alive(d) {
				// A live engine holds the lock: attach to it.
				tg := targetFor(dir, d)
				// wait polls cond, one sleep line for every wait in this loop.
				wait := func(tries int, cond func() bool) bool {
					for i := 0; i < tries; i++ {
						if cond() {
							return true
						}
						time.Sleep(200 * time.Millisecond)
					}
					return false
				}
				// released: the lock is gone or now names another engine (it is released last).
				released := func() bool {
					cur, err := lifecycle.ReadLock(lockPath)
					return err != nil || cur.Pid != d.Pid || cur.Started != d.Started
				}
				if !wait(15, func() bool { return statusOK(tg) || released() }) {
					return Result{}, fmt.Errorf("engine pid %d holds engine.lock but does not answer on port %d", d.Pid, d.Port)
				}
				if released() {
					continue
				}
				stopping := false
				for _, p := range o.ParentPIDs {
					if err := PostAttach(tg, p); errors.Is(err, ErrStopping) {
						stopping = true
						break
					} else if err != nil {
						return Result{}, err
					}
				}
				if stopping {
					// A dying engine: wait for it to end (the lock is released last), then go
					// round again and take the lock ourselves.
					if !wait(50, func() bool { return !alive(d) || released() }) {
						return Result{}, fmt.Errorf("engine pid %d is stopping and has not ended", d.Pid)
					}
					continue
				}
				return Result{Attached: true, Port: d.Port, Host: d.Host}, nil
			}
			lifecycle.KillStale(d) // the engine crashed: its children may still run
			lifecycle.RemoveStale(lockPath, d)
		} else if _, serr := os.Stat(lockPath); serr == nil && !lifecycle.LockInWrite(lockPath) {
			// An unreadable lock is stale, unless it is empty and brand new: then its creator is
			// between create and write (the O_EXCL fallback), so it is held. CreateLock then
			// reports ErrLockHeld and the retry below goes round again.
			os.Remove(lockPath)
		}
		l, p, err := api.Listen(host, base)
		if err != nil {
			return Result{}, err
		}
		start, _ := lifecycle.PidStart(os.Getpid())
		lk, err := lifecycle.CreateLock(lockPath, lifecycle.LockData{Pid: os.Getpid(), Start: start, Port: p, Host: host, Started: time.Now().UTC().Format(time.RFC3339)})
		if err == lifecycle.ErrLockHeld {
			l.Close()
			time.Sleep(100 * time.Millisecond)
			continue
		}
		if err != nil {
			l.Close()
			return Result{}, err
		}
		lock, ln, port = lk, l, p
	}
	defer lock.Remove()
	loopback := IsLoopbackHost(host)
	var tk *tokens.Store
	switch {
	case loopback:
	case o.Open:
		log.Printf("easy-engine: WARNING: listening on %s:%d with no token (--open); anyone who can reach this port can use the engine", host, port)
	default:
		tk = tokens.Open(dir)
		if _, err := tk.Master(); err != nil {
			ln.Close()
			return Result{}, fmt.Errorf("token file: %w", err)
		}
	}

	if o.DetachStdio {
		// The game reads our stdio through pipes and closes them when it exits;
		// a write to a closed pipe would kill us with SIGPIPE before we clean up.
		os.MkdirAll(filepath.Join(dir, "logs"), 0o755)
		// Only a crash trace or a last exit line bypasses engine.log's sink; it goes to its own
		// capped file (lifecycle.StdioLogName), counted in the setup's disk need.
		stdio := filepath.Join(dir, "logs", lifecycle.StdioLogName)
		lifecycle.RotateStdioLog(stdio)
		detachStdio(stdio)
		log.SetOutput(io.Discard)
	}

	prof := run.Profile{}
	if o.Profile != nil {
		prof = *o.Profile
	} else if p, err := run.LoadProfile(o.ProfilePath); err != nil {
		ln.Close()
		return Result{}, err
	} else {
		prof = p
	}
	w := lifecycle.NewWatcher()
	cfg := run.Config{DataDir: dir, Profile: prof, Version: o.Version, Parents: w.PIDs, StartSleeping: o.Resident} // the app waits for a connection
	if o.Yield != nil {
		y := *o.Yield
		y.Except = append(append([]string{}, y.Except...), prof.YieldExcept...)
		cfg.Yield = &y
	}
	if o.Configure != nil {
		o.Configure(&cfg)
	}
	if cfg.Group == nil {
		cfg.Group = lifecycle.NewGroup()
	}
	cfg.Group.OnChange = lock.SetChildren
	for _, p := range o.ParentPIDs {
		if err := w.Add(p); err != nil {
			ln.Close()
			return Result{}, fmt.Errorf("parent pid %d: %w", p, err)
		}
	}
	sup, err := run.New(cfg)
	if err != nil {
		ln.Close()
		return Result{}, err
	}
	sup.SetPort(port)
	// The keeper's file is followed as it changes (an event, not a poll); the watch is made
	// before the engine answers anyone, so no edit slips between.
	ch, stopWatch, werr := osevents.Changes(settings.Path(dir))
	watching := werr == nil
	if watching {
		defer stopWatch()
	}
	shutdown := make(chan struct{}, 1)
	if cfg.Catalog == nil {
		if _, note, _ := catalog.LoadNote(dir); note != "" {
			sup.Note(note)
		}
	}
	attached := make(chan struct{}, 1) // a parent was attached (resident: the connection)
	var ctl *controller
	if o.Tray != nil {
		ctl = newController(o.Tray, o.Menu, sup, dir, func() {
			select {
			case shutdown <- struct{}{}:
			default:
			}
		})
	}
	attach := w.Add
	if o.Resident {
		attach = func(pid int) error {
			if err := w.Add(pid); err != nil {
				return err
			}
			select {
			case attached <- struct{}{}:
			default:
			}
			wakeForClient(sup)
			ctl.refresh()
			return nil
		}
	}
	a := &api.API{S: sup, Attach: attach, AnyHost: !loopback, Open: o.Open,
		Shutdown: func() {
			select {
			case shutdown <- struct{}{}:
			default:
			}
		}}
	if tk != nil {
		a.Tokens = tk
	}
	srv := &http.Server{Handler: a.Handler(), ReadHeaderTimeout: 10 * time.Second} // network timeout
	go srv.Serve(ln)
	sup.Run()
	if o.Started != nil {
		o.Started(port)
	}

	// Wait for the parents to be gone (plus linger), a signal or a shutdown request.
	wctx, wcancel := context.WithCancel(ctx)
	defer wcancel()
	gone := make(chan struct{})
	if o.Resident {
		if len(o.ParentPIDs) > 0 { // parents given on the command line count as the first connection
			attached <- struct{}{}
			wakeForClient(sup) // and load the model for them, as an attach does
		}
		go residentLoop(wctx, w, sup, attached, func() time.Duration {
			if o.LingerSet {
				return o.Linger
			}
			return time.Duration(sup.LingerS()) * time.Second
		}, ctl.refresh)
	} else {
		go func() {
			defer close(gone)
			for {
				if !w.Run(wctx) {
					return
				}
				linger := o.Linger
				if !o.LingerSet {
					linger = time.Duration(sup.LingerS()) * time.Second
				}
				if linger > 0 {
					select {
					case <-time.After(linger):
					case <-wctx.Done():
						return
					}
				}
				if len(w.PIDs()) == 0 {
					return
				}
			}
		}()
	}
	if watching {
		go func() {
			sup.ReloadSettings() // an edit made before the watch began
			for {
				select {
				case <-wctx.Done():
					return
				case <-ch:
					sup.ReloadSettings()
					ctl.refresh()
				}
			}
		}()
	}
	if ctl != nil {
		go ctl.serve(wctx)
	}
	select {
	case <-ctx.Done():
	case <-gone:
	case <-shutdown:
	}
	a.SetStopping() // status says so and attach is refused from here; the lock is released last
	if o.Tray != nil {
		o.Tray.SetTooltip("easy-engine: stopping")
		defer o.Tray.Close()
	}
	wcancel()
	sup.Shutdown()
	sctx, scancel := context.WithTimeout(context.Background(), 2*time.Second)
	if srv.Shutdown(sctx) != nil {
		srv.Close()
	}
	scancel()
	return Result{Port: port, Host: host}, nil
}
