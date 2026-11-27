// Command traydemo shows the engine's tray icon with its three items, to try the tray by hand:
//
//   - Launch at Startup: a checkmark that turns autostart on and off. It uses a demo name in a
//     fake place under a temporary folder, so it never touches this user's real startup entries,
//     unless -real is given.
//   - Settings: opens a file with the desktop's default program (its folder when that fails).
//   - Quit: exits.
//
// Each click is printed to stdout. It also exits on Ctrl-C.
//
//	go run ./tools/traydemo [-settings file] [-real] [-name id] [-notify]
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/CharlesMod/easyEngine/autostart"
	"github.com/CharlesMod/easyEngine/openfile"
	"github.com/CharlesMod/easyEngine/tray"
)

func main() { tray.Main(func() { os.Exit(run()) }) }

func run() int {
	settings := flag.String("settings", "", "file the Settings item opens (default: a demo settings.json in a temporary folder)")
	real := flag.Bool("real", false, "turn this user's real Launch at Startup on and off (default: a fake place in a temporary folder)")
	name := flag.String("name", "easy-engine-traydemo", "the startup entry's name")
	notify := flag.Bool("notify", false, "show a desktop notification at start")
	flag.Parse()

	tmp, err := os.MkdirTemp("", "easy-engine-traydemo-")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer os.RemoveAll(tmp)

	place := autostart.Root(filepath.Join(tmp, "startup"))
	if *real {
		place = autostart.Real()
	}
	if *settings == "" {
		*settings = filepath.Join(tmp, "settings.json")
		if err := os.WriteFile(*settings, []byte("{\n  \"demo\": true\n}\n"), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
	}
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	on, err := place.Enabled(*name)
	if err != nil {
		fmt.Fprintln(os.Stderr, "autostart:", err)
	}
	fmt.Printf("startup entry: %s (on: %v)\nsettings: %s\n", place.Path(*name), on, *settings)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	t, err := tray.Run(ctx, tray.Spec{
		Title:   "easy-engine tray test",
		Tooltip: "easy-engine tray test",
		Icon:    tray.DefaultIcon(),
		Items: []tray.Item{
			{ID: "startup", Title: "Launch at Startup", Checkable: true, Checked: on},
			{ID: "settings", Title: "Settings"},
			{Separator: true},
			{ID: "quit", Title: "Quit"},
		},
	})
	if errors.Is(err, tray.ErrNoTray) {
		fmt.Fprintln(os.Stderr, "no tray here:", err)
		return 2
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer t.Close()
	fmt.Printf("tray up (pid %d)\n", os.Getpid())
	if *notify {
		if err := t.Notify("easy-engine tray test", "The tray demo is running."); err != nil {
			fmt.Fprintln(os.Stderr, "notify:", err)
		}
	}

	for id := range t.Clicks() {
		fmt.Println("click:", id)
		switch id {
		case "startup":
			on, err := place.Enabled(*name)
			if err == nil {
				if on {
					err = place.Disable(*name)
				} else {
					err = place.Enable(*name, exe, nil)
				}
			}
			if err != nil {
				fmt.Fprintln(os.Stderr, "autostart:", err)
			}
			now, _ := place.Enabled(*name)
			t.SetChecked("startup", now)
			fmt.Printf("launch at startup: %v (%s)\n", now, place.Path(*name))
		case "settings":
			go func() {
				opened, err := openfile.Open(*settings)
				if err != nil {
					fmt.Fprintln(os.Stderr, "settings:", err)
					return
				}
				fmt.Println("opened:", opened)
			}()
		case "quit":
			return 0
		}
	}
	return 0 // the context ended
}
