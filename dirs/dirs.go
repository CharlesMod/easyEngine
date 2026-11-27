// Package dirs resolves the engine's data directory and the places inside it.
package dirs

import (
	"os"
	"path/filepath"
	"runtime"

	"github.com/CharlesMod/easyEngine/internal/legacy"
)

// HomeEnv names the variable that overrides the data dir.
const HomeEnv = "EASY_ENGINE_HOME"

// Data returns the engine data dir: EASY_ENGINE_HOME when set (the old variable's value when
// only that is), else the per-OS default; of the defaults the new one if it exists, else the
// old one (internal/legacy) if it exists, else the new one. It has no side effects: the move
// of an old dir to the new one is Migrate, called once when the engine starts.
func Data() string {
	return dataFor(runtime.GOOS, os.Getenv, homeDir(), legacy.OSFS{}.Exists)
}

// Migrate moves a data dir left under the old default name to the new default, when that is
// safe (internal/legacy decides and explains). It is called once, from the serve and setup start
// path, and returns a note for the log ("" when there is nothing to say).
func Migrate() string {
	getenv, home := os.Getenv, homeDir()
	nd := newDefault(runtime.GOOS, getenv, home)
	od := legacy.OldDefaultDir(runtime.GOOS, getenv, home)
	_, note := legacy.Migrate(runtime.GOOS, getenv, HomeEnv, nd, od, legacy.OSFS{})
	return note
}

func homeDir() string {
	h, err := os.UserHomeDir()
	if err != nil || h == "" {
		return "."
	}
	return h
}

func newDefault(goos string, getenv func(string) string, home string) string {
	switch goos {
	case "windows":
		base := getenv("LOCALAPPDATA")
		if base == "" {
			base = filepath.Join(home, "AppData", "Local")
		}
		return filepath.Join(base, "EasyEngine")
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "EasyEngine")
	default:
		base := getenv("XDG_DATA_HOME")
		if base == "" {
			base = filepath.Join(home, ".local", "share")
		}
		return filepath.Join(base, "easy-engine")
	}
}

func dataFor(goos string, getenv func(string) string, home string, exists func(string) bool) string {
	if v := legacy.HomeOverride(getenv, HomeEnv); v != "" {
		return v
	}
	nd := newDefault(goos, getenv, home)
	if exists(nd) {
		return nd
	}
	if od := legacy.OldDefaultDir(goos, getenv, home); exists(od) {
		return od
	}
	return nd
}

func Backends() string  { return filepath.Join(Data(), "backends") }
func Models() string    { return filepath.Join(Data(), "models") }
func Runtime() string   { return filepath.Join(Data(), "runtime") }
func Downloads() string { return filepath.Join(Data(), "downloads") }
func Logs() string      { return filepath.Join(Data(), "logs") }
func State() string     { return filepath.Join(Data(), "state.json") }
func Lock() string      { return filepath.Join(Data(), "engine.lock") }
