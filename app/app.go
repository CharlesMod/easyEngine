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

	"hermesengine/api"
	"hermesengine/dirs"
	"hermesengine/lifecycle"
	"hermesengine/run"
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
	Port        int    // default 47480
	Linger      time.Duration
	LingerSet   bool // Linger given on the command line; else the saved setting applies
	Version     string
	DetachStdio bool // serve mode: stdio goes to engine.log, never to the inherited pipes
	Configure   func(*run.Config)
	// Started is called once the engine listens (not when it attached to another).
	Started func(port int)
}

// Result of Serve.
type Result struct {
	Attached bool
	Port     int
}

func dataDir(o Options) string {
	if o.DataDir != "" {
		return o.DataDir
	}
	return dirs.Data()
}

var statusClient = &http.Client{Timeout: 2 * time.Second}

func statusOK(port int) bool {
	resp, err := statusClient.Get(fmt.Sprintf("http://127.0.0.1:%d/engine/v1/status", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var st struct{ Version string }
	return resp.StatusCode == 200 && json.NewDecoder(resp.Body).Decode(&st) == nil && st.Version != ""
}

// PostAttach asks a running engine to watch another parent.
func PostAttach(port, pid int) error {
	b, _ := json.Marshal(map[string]int{"pid": pid})
	req, _ := http.NewRequest("POST", fmt.Sprintf("http://127.0.0.1:%d/engine/v1/attach", port), bytes.NewReader(b))
	req.Header.Set("X-Hermes-Engine", "1")
	req.Header.Set("Content-Type", "application/json")
	resp, err := statusClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		m, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("attach: %s %s", resp.Status, m)
	}
	return nil
}

func alive(d lifecycle.LockData) bool {
	if !lifecycle.PidAlive(d.Pid) {
		return false
	}
	if st, err := lifecycle.PidStart(d.Pid); err == nil && d.Start != "" && st != d.Start {
		return false
	}
	return true
}

// Running reports the port of a healthy engine in dir, if any.
func Running(dir string) (int, bool) {
	d, err := lifecycle.ReadLock(filepath.Join(dir, "engine.lock"))
	if err != nil || !alive(d) || !statusOK(d.Port) {
		return 0, false
	}
	return d.Port, true
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
				ok := false
				for i := 0; i < 15 && !ok; i++ {
					if ok = statusOK(d.Port); !ok {
						time.Sleep(200 * time.Millisecond)
					}
				}
				if !ok {
					return Result{}, fmt.Errorf("engine pid %d holds engine.lock but does not answer on port %d", d.Pid, d.Port)
				}
				for _, p := range o.ParentPIDs {
					if err := PostAttach(d.Port, p); err != nil {
						return Result{}, err
					}
				}
				return Result{Attached: true, Port: d.Port}, nil
			}
			lifecycle.KillStale(d) // the engine crashed: its children may still run
			os.Remove(lockPath)
		} else if _, serr := os.Stat(lockPath); serr == nil {
			os.Remove(lockPath) // unreadable lock
		}
		l, p, err := api.Listen(host, base)
		if err != nil {
			return Result{}, err
		}
		start, _ := lifecycle.PidStart(os.Getpid())
		lk, err := lifecycle.CreateLock(lockPath, lifecycle.LockData{Pid: os.Getpid(), Start: start, Port: p, Started: time.Now().UTC().Format(time.RFC3339)})
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

	if o.DetachStdio {
		// The game reads our stdio through pipes and closes them when it exits;
		// a write to a closed pipe would kill us with SIGPIPE before we clean up.
		os.MkdirAll(filepath.Join(dir, "logs"), 0o755)
		detachStdio(filepath.Join(dir, "logs", "engine.log"))
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
	cfg := run.Config{DataDir: dir, Profile: prof, Version: o.Version, Parents: w.PIDs}
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
	shutdown := make(chan struct{}, 1)
	a := &api.API{S: sup, Attach: w.Add, AnyHost: host != "127.0.0.1" && host != "localhost",
		Shutdown: func() {
			select {
			case shutdown <- struct{}{}:
			default:
			}
		}}
	srv := &http.Server{Handler: a.Handler()}
	go srv.Serve(ln)
	sup.Run()
	if o.Started != nil {
		o.Started(port)
	}

	// Wait for the parents to be gone (plus linger), a signal or a shutdown request.
	wctx, wcancel := context.WithCancel(ctx)
	defer wcancel()
	gone := make(chan struct{})
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
	select {
	case <-ctx.Done():
	case <-gone:
	case <-shutdown:
	}
	wcancel()
	sup.Shutdown()
	sctx, scancel := context.WithTimeout(context.Background(), 2*time.Second)
	if srv.Shutdown(sctx) != nil {
		srv.Close()
	}
	scancel()
	return Result{Port: port}, nil
}
