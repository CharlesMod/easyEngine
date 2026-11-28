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
	"github.com/CharlesMod/easyEngine/discover"
	osevents "github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/pair"
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
	Open      bool
	Port      int // default 47480
	Linger    time.Duration
	LingerSet bool // Linger given on the command line; else the saved setting applies
	// IdleSleep (with IdleSleepSet) is --idle-sleep: the idle rule's quiet for this run, over
	// the setting idle_sleep_s; NoIdleSleep is --no-idle-sleep (docs/ENGINE.md "Idle sleep").
	IdleSleep    time.Duration
	IdleSleepSet bool
	NoIdleSleep  bool
	Version      string
	DetachStdio  bool // serve mode: stdio goes to logs/engine-stdio.log (capped), never to the inherited pipes
	Configure    func(*run.Config)
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
	// Discovery is passed to the mDNS responder (tests give it an in-memory transport). The
	// responder runs only on a listener that is not loopback (Discoverable) with discovery on.
	Discovery discover.Options
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

// statusPatience is how long a lock whose holder lives, or that cannot be read, may stay silent
// (no change of the file, no end of the holder) before Serve gives up on it: a watchdog for silence.
const statusPatience = 10 * time.Second

// lockWritePatience is how long a lock that is held but not yet written may stay silent: a
// watchdog for silence (its creator is between create and write, which takes microseconds).
const lockWritePatience = 2 * time.Second

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
	// The lock's changes are watched from before the first look until Serve has the lock, so a
	// change made after any look is signalled (one coalesced signal waits in the channel): arm
	// first, then look, then wait.
	changed, stopChanged, err := osevents.Changes(lockPath)
	if err != nil {
		// No watch here (a system that gives none, or one that failed): Serve still starts. A lock
		// nobody contends for needs no wait at all; a contended one is then ended by the holder's
		// exit or by the watchdogs for silence below.
		log.Printf("easy-engine: engine.lock is not watched (%v)", err)
		changed, stopChanged = nil, func() {}
	}
	defer stopChanged()
	// await returns when the lock file changes, ended closes (nil: it never does), or ctx ends. If
	// patience is not zero and nothing has come by then, quiet is true: a watchdog for silence.
	await := func(ended <-chan struct{}, patience time.Duration) (quiet bool, err error) {
		var fired chan struct{}
		if patience == 0 && changed == nil && ended == nil {
			patience = statusPatience // nothing can say a word: the wait must not be endless
		}
		if patience > 0 {
			fired = make(chan struct{})
			sil := osevents.NewSilence(patience, func() { close(fired) })
			defer sil.Stop()
		}
		select {
		case <-changed:
		case <-ended:
		case <-fired:
			return true, nil
		case <-ctx.Done():
			return false, ctx.Err()
		}
		return false, nil
	}
	for held := 0; lock == nil; {
		if d, raw, err := readLockRaw(lockPath); errors.Is(err, os.ErrNotExist) {
			// No lock: go and take it. (Not the corrupt case below: a lock another serve
			// creates after this read must never be taken for a corrupt one.)
		} else if err == nil {
			// The holder's end is watched before it is asked whether it lives, so an end after the
			// question is not missed. (host.Exits, not ExitsOf: the lock's start token is the one
			// lifecycle.PidStart gives, and on darwin that is not host.StartOf's; alive(d) checks it.)
			ended, stopEnded, err := osevents.Exits(d.Pid)
			if err != nil {
				// The pid cannot be watched (a stale lock whose pid now belongs to a process this
				// user may not open, say): that is not a reason to refuse to start. alive(d) below
				// still says whether it is the engine, and a wait has the lock's change to end it.
				ended, stopEnded = nil, func() {}
			}
			if alive(d) {
				// A live engine holds the lock: attach to it.
				tg := targetFor(dir, d)
				// The lock is written after the holder's listener is bound, so one status call (it has
				// its own network timeout) is an answer. If the holder lives and does not answer, the
				// word to wait for is the lock's change or the holder's end.
				if !statusOK(tg) {
					quiet, err := await(ended, statusPatience)
					stopEnded()
					if err != nil {
						return Result{}, err
					}
					if quiet {
						return Result{}, fmt.Errorf("engine pid %d holds engine.lock but does not answer on port %d", d.Pid, d.Port)
					}
					continue
				}
				// released: the lock is gone or now names another engine (it is released last).
				if cur, err := lifecycle.ReadLock(lockPath); err != nil || cur.Pid != d.Pid || cur.Started != d.Started {
					stopEnded()
					continue
				}
				stopping := false
				for _, p := range o.ParentPIDs {
					if err := PostAttach(tg, p); errors.Is(err, ErrStopping) {
						stopping = true
						break
					} else if err != nil {
						stopEnded()
						return Result{}, err
					}
				}
				if stopping {
					// A dying engine: wait for it to end (the lock is released last), then go
					// round again and take the lock ourselves.
					_, err := await(ended, 0)
					stopEnded()
					if err != nil {
						return Result{}, err
					}
					continue
				}
				stopEnded()
				return Result{Attached: true, Port: d.Port, Host: d.Host}, nil
			}
			stopEnded()
			lifecycle.KillStale(d) // the engine crashed: its children may still run
			lifecycle.RemoveStale(lockPath, d)
		} else if isParseErr(err) {
			// The file is corrupt: stale, unless it is empty and brand new (its creator is between
			// create and write, the O_EXCL fallback), so then it is held; CreateLock reports
			// ErrLockHeld and the retry below goes round again. A corrupt file goes only if its
			// bytes are still the bytes read here, so a lock another serve wrote since is safe.
			if !lifecycle.LockInWrite(lockPath) {
				removeCorruptLock(lockPath, raw)
			}
		} else {
			// Unreadable for another reason (permission, a sharing violation while the live engine
			// rewrites its lock): not proof of anything, so remove nothing; wait for the lock to
			// change and look again. A lock that stays unreadable and silent is an error.
			quiet, aerr := await(nil, statusPatience)
			if aerr != nil {
				return Result{}, aerr
			}
			if quiet {
				return Result{}, fmt.Errorf("engine.lock stays unreadable: %w", err)
			}
			continue
		}
		l, p, err := api.Listen(host, base)
		if err != nil {
			return Result{}, err
		}
		start, _ := lifecycle.PidStart(os.Getpid())
		lk, err := lifecycle.CreateLock(lockPath, lifecycle.LockData{Pid: os.Getpid(), Start: start, Port: p, Host: host, Started: time.Now().UTC().Format(time.RFC3339)})
		if err == lifecycle.ErrLockHeld {
			l.Close()
			// Another serve took the lock between the look and the create (or its creator is
			// between create and write): wait for the lock's change. A creator that says nothing
			// (a watchdog for silence) is not robbed of its lock: Serve gives up.
			if held++; held > 8 {
				return Result{}, errors.New("could not take engine.lock")
			}
			quiet, err := await(nil, lockWritePatience)
			if err != nil {
				return Result{}, err
			}
			if quiet {
				return Result{}, errors.New("could not take engine.lock: its holder is writing it and says nothing")
			}
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
	cfg := run.Config{DataDir: dir, Profile: prof, Version: o.Version, Parents: w.PIDs, StartSleeping: o.Resident, // the app waits for a connection
		ProfileGiven: o.Profile != nil || o.ProfilePath != ""} // a profile given is the base client
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
		ctl = newController(o.Tray, o.Menu, sup, dir, ChatURL(host, port), func() {
			select {
			case shutdown <- struct{}{}:
			default:
			}
		})
	}
	attach := w.Add
	// serve: a client's attach changes nothing of the engine's life, and wakes a model the idle
	// rule (or a sleep asked) put to sleep, as the app's attach does
	present := func() { wakeForClient(sup) }

	if o.Resident {
		// a client that attached (by pid or a held connection) is the connection the app waits for
		present = func() {
			select {
			case attached <- struct{}{}:
			default:
			}
			wakeForClient(sup)
			ctl.refresh()
		}
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
		Present: present,
		Shutdown: func() {
			select {
			case shutdown <- struct{}{}:
			default:
			}
		}}
	if tk != nil {
		a.Tokens = tk
	}
	// Who this engine is to a client on the network (hello, mDNS), and pairing where tokens are
	// in force on a non-loopback listener (with --open there is nothing to pair for).
	eid, err := pair.EngineID(dir)
	if err != nil {
		log.Printf("easy-engine: engine id: %v", err)
	}
	a.EngineID = eid
	a.Name = func() string { return engineName(sup) }
	var pr *pair.Pairer
	if tk != nil {
		pr = pair.New(sup.Hub(), tk, pair.DefaultMax)
		a.Pair = pr
		if ctl != nil {
			ctl.pair = pr
		}
		// tokens.json is followed as settings.json is (the kernel's word): a revoke, by
		// `easy-engine token --revoke` or anything else that rewrites the file, takes the
		// client's fact down at once, as the gate already refuses its token.
		if stopTokens, err := followFile(tk.Path(), a.ForgetRevoked); err == nil {
			defer stopTokens()
		} else {
			log.Printf("easy-engine: cannot follow %s (%v): a revoked client's fact stands until the next start", tk.Path(), err)
		}
	}
	dopts := o.Discovery
	if dopts.Listen == nil {
		dopts.Listen = ListenSet(host) // announce only what the listener listens on
	}
	an := &announcer{eligible: Discoverable(host) && eid != "", opts: dopts}
	info := func() discover.Info {
		return discover.Info{ID: eid, Name: engineName(sup), API: run.APIVersion, Port: port}
	}
	srv := &http.Server{Handler: a.Handler(), ReadHeaderTimeout: 10 * time.Second} // network timeout
	go srv.Serve(ln)
	an.apply(sup.Discovery(), info())
	defer an.close()
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
	}
	// The idle rule, in both modes. The app's clients are its attached parents (the command
	// line's too: the app does not end with them) and the clients' attachments; serve's clients
	// are the attachments alone (an attach by pid is one), its --parent-pid parents keep the
	// engine's process, not its model. --linger is serve's engine life, the app's model linger.
	rule := func() idleRule {
		on, quiet, linger := sup.IdleSettings()
		if o.Resident && o.LingerSet {
			linger = o.Linger
		}
		if o.IdleSleepSet {
			quiet = o.IdleSleep
		}
		return idleRule{on: on && !o.NoIdleSleep, quiet: quiet, linger: linger}
	}
	if o.Resident {
		go idleLoop(wctx, w, sup, attached, func() int { return len(w.PIDs()) + sup.Attached() }, rule, ctl.refresh)
	} else {
		go idleLoop(wctx, nil, sup, nil, sup.Attached, rule, ctl.refresh)

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
					lingered := make(chan struct{})
					// named moment: the keeper's linger, the time the engine stays after its model's end
					cancel := osevents.Moment(linger, func() { close(lingered) })
					select {
					case <-lingered:
					case <-wctx.Done():
						cancel()
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
					an.apply(sup.Discovery(), info()) // the name or discovery may have changed
					ctl.refresh()
				}
			}
		}()
	}
	if ctl != nil {
		go ctl.watch(wctx)
		go ctl.serve(wctx)
	}
	select {
	case <-ctx.Done():
	case <-gone:
	case <-shutdown:
	}
	a.SetStopping() // status says so and attach is refused from here; the lock is released last
	sup.EndHolds()  // a held attachment hears {"end":"engine_ending"} before the listener closes
	if o.Tray != nil {
		o.Tray.SetTooltip("easy-engine: stopping")
		defer o.Tray.Close()
	}
	wcancel()
	// Waiting pairing requests end: each held GET pair/{id} is answered 410 and no token is
	// issued after this; the network hears the goodbye before the listener closes.
	if pr != nil {
		pr.Close()
	}
	an.close()
	sup.Shutdown()
	sctx, scancel := context.WithTimeout(context.Background(), 2*time.Second)
	if srv.Shutdown(sctx) != nil {
		srv.Close()
	}
	scancel()
	return Result{Port: port, Host: host}, nil
}

// readLockRaw reads the lock file once and parses what it read: the raw bytes go back with the
// parse so a corrupt file can be removed only while it is still these bytes.
func readLockRaw(path string) (lifecycle.LockData, []byte, error) {
	var d lifecycle.LockData
	raw, err := os.ReadFile(path)
	if err != nil {
		return d, nil, err
	}
	return d, raw, json.Unmarshal(raw, &d)
}

// isParseErr is true for an error from parsing the lock's bytes (the file was read, and is corrupt),
// false for an error from reading it.
func isParseErr(err error) bool {
	var se *json.SyntaxError
	var te *json.UnmarshalTypeError
	return errors.As(err, &se) || errors.As(err, &te)
}

// removeCorruptLock removes the lock file only if it still holds exactly raw, the corrupt bytes
// the caller read. Anything else (gone, changed, unreadable) is left alone.
func removeCorruptLock(path string, raw []byte) bool {
	cur, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(cur, raw) {
		return false
	}
	return os.Remove(path) == nil
}
