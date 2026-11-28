package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/dirs"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/settings"
)

// launch is how the engine is started: serve (a service or a shell: no tray, ends with its
// parents), setup (a setup page) or app (a tray icon, stays open and waits for a connection).
type launch int

const (
	launchServe launch = iota
	launchSetup
	launchApp
)

func cmdServe(args []string, setup bool) int {
	if setup {
		return runEngine(args, launchSetup, false)
	}
	return runEngine(args, launchServe, false)
}

// cmdApp is `easy-engine app`, and `easy-engine` with no command (implicit).
func cmdApp(args []string, implicit bool) int { return runEngine(args, launchApp, implicit) }

// isTerminal says whether f is a terminal (a character device that is not /dev/null).
func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	if err != nil || fi.Mode()&os.ModeCharDevice == 0 {
		return false
	}
	if null, err := os.Stat(os.DevNull); err == nil && os.SameFile(fi, null) {
		return false
	}
	return true
}

func ifApp[T any](isApp bool, fromFile, otherwise T) T {
	if isApp {
		return fromFile
	}
	return otherwise
}

func runEngine(args []string, kind launch, implicit bool) int {
	name := map[launch]string{launchServe: "serve", launchSetup: "setup", launchApp: "app"}[kind]
	setup, appMode := kind == launchSetup, kind == launchApp
	// The app's defaults are the keeper's settings.json; its flags only override them for one
	// run and are never written back. serve and setup keep their own defaults and never read
	// the file for these.
	conf := settings.Defaults()
	if appMode {
		if note := dirs.Migrate(); note != "" { // once, before anything reads the data dir
			fmt.Fprintln(os.Stderr, "easy-engine:", note)
		}
		// An engine already runs: say so and end (a second icon would be a second app). Asked
		// before the settings are loaded, because a load may rewrite state.json and that file
		// is the running engine's.
		if p, ok := app.Running(dirs.Data()); ok {
			fmt.Printf("easy-engine is already running (port %d)\n", p)
			return 0
		}
		conf, _ = settings.Load(dirs.Data()) // a file that cannot be read: the engine says so in its status
	}
	fs := newFlags(name)
	profile := fs.String("profile", "", "game profile file")
	port := fs.Int("port", ifApp(appMode, conf.Listen.Port, 0), "control API port (default 47480)")
	host := fs.String("host", ifApp(appMode, conf.Listen.Host, "127.0.0.1"), "listen address (beyond loopback every request needs the bearer token: easy-engine token)")
	open := fs.Bool("open", ifApp(appMode, conf.Listen.Open, false), "with a non-loopback --host: no token at all (anyone who can reach the port can use the engine)")
	linger := fs.Duration("linger", 0, "keep running this long after the last parent exits")
	idleSleep := fs.Duration("idle-sleep", 0, "unload the model after this long with nothing attached and nothing asked (default: settings idle_sleep_s, 10m)")
	noIdleSleep := fs.Bool("no-idle-sleep", false, "never unload the model for being idle (settings idle_sleep: false)")
	var parents pidList
	fs.Var(&parents, "parent-pid", "game pid to watch (repeatable)")
	y := ifApp(appMode, conf.Yield, settings.Defaults().Yield)
	yieldOn := fs.Bool("yield-to-games", ifApp(appMode, y.ToGames, false), "unload the model while another program wants the card (a launcher game, a new GPU user), load again after")
	yieldExcept := fs.String("yield-except", strings.Join(y.Except, ","), "comma-separated launcher app ids that never make the engine yield (the supported host's)")
	yieldIgnore := fs.String("yield-ignore", strings.Join(y.Ignore, ","), "comma-separated process names that never make the engine yield")
	yieldFloor := fs.Int("yield-floor-mib", y.FloorMiB, "MiB an already-running program may grow on the card before the engine yields")
	yieldMin := fs.Int("yield-min-mib", y.MinMiB, "MiB a new program must hold to make the engine yield by memory alone (0: any)")
	yieldGrace := fs.Duration("yield-grace", y.Grace(), "after a game exits, wait this long before loading again")
	yieldSteady := fs.Duration("yield-steady", y.Steady(), "load again beside a non-game program once its use has held this long")
	yieldStrict := fs.Bool("yield-strict", y.Strict, "stay unloaded until every program that made the engine yield has exited")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	lingerSet, idleSleepSet := false, false
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "linger":
			lingerSet = true
		case "idle-sleep":
			idleSleepSet = true
		}
	})
	if *idleSleep < 0 || *idleSleep > settings.IdleSleepMax*time.Second {
		fmt.Fprintf(os.Stderr, "--idle-sleep must be 0 to %s\n", settings.IdleSleepMax*time.Second)
		return 2
	}
	if kind == launchServe && len(parents) == 0 {
		fmt.Fprintln(os.Stderr, "serve needs --parent-pid (use `setup` for a run without a game)")
		return 2
	}
	if !appMode {
		if note := dirs.Migrate(); note != "" { // once, before anything reads the data dir
			fmt.Fprintln(os.Stderr, "easy-engine:", note)
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var tr app.Tray
	if appMode {
		t, err := app.StartTray(ctx, app.MenuOptions{})
		switch {
		case err == nil:
			tr = t
		case implicit && isTerminal(os.Stdout):
			// Typed in a shell with no desktop to show an icon in: not a server to start by surprise.
			usage()
			return 2
		default:
			fmt.Fprintf(os.Stderr, "easy-engine: no tray icon (%v); running without one\n", err)
		}
	}
	var yc *run.YieldConfig
	if *yieldOn {
		yc = &run.YieldConfig{Except: splitList(*yieldExcept), Ignore: splitList(*yieldIgnore), MinNewMiB: *yieldMin}
		yc.Policy.Floor, yc.Policy.Grace, yc.Policy.Steady, yc.Policy.Strict = *yieldFloor, *yieldGrace, *yieldSteady, *yieldStrict
	}
	o := app.Options{
		Yield:       yc,
		ProfilePath: *profile, ParentPIDs: parents, Host: *host, Port: *port,
		Linger: *linger, LingerSet: lingerSet, Version: version, Open: *open,
		IdleSleep: *idleSleep, IdleSleepSet: idleSleepSet, NoIdleSleep: *noIdleSleep,

		DetachStdio: !setup,
		Resident:    appMode, Tray: tr,
		Started: func(p int) {
			if setup {
				// The token (if the engine needs one) rides in the opened URL's fragment only; it is not printed.
				tg := app.EngineTarget(dirs.Data(), *host, p)
				fmt.Printf("Setup page: %s/\n", tg.Base())
				openBrowser(app.SetupURL(dirs.Data(), tg))
			}
		},
	}
	res, err := app.Serve(ctx, o)
	if err != nil {
		if tr != nil {
			tr.Close()
		}
		fmt.Fprintln(os.Stderr, "easy-engine:", err)
		return 1
	}
	if res.Attached {
		if tr != nil {
			tr.Close()
		}
		if appMode {
			fmt.Printf("easy-engine is already running (port %d)\n", res.Port)
		}
		if setup {
			tg := app.EngineTarget(dirs.Data(), res.Host, res.Port)
			fmt.Println("Engine already running:", tg.Base()+"/")
			openBrowser(app.SetupURL(dirs.Data(), tg))
		}
		return 0
	}
	return 0
}
