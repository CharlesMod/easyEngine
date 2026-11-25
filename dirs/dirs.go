// Package dirs resolves the engine's data directory and the places inside it.
package dirs

import (
	"os"
	"path/filepath"
	"runtime"
)

// Data returns the engine data dir: HERMES_ENGINE_HOME when set, else the per-OS default.
func Data() string {
	return dataFor(runtime.GOOS, os.Getenv, homeDir())
}

func homeDir() string {
	h, err := os.UserHomeDir()
	if err != nil || h == "" {
		return "."
	}
	return h
}

func dataFor(goos string, getenv func(string) string, home string) string {
	if v := getenv("HERMES_ENGINE_HOME"); v != "" {
		return v
	}
	switch goos {
	case "windows":
		base := getenv("LOCALAPPDATA")
		if base == "" {
			base = filepath.Join(home, "AppData", "Local")
		}
		return filepath.Join(base, "HermesEngine")
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "HermesEngine")
	default:
		base := getenv("XDG_DATA_HOME")
		if base == "" {
			base = filepath.Join(home, ".local", "share")
		}
		return filepath.Join(base, "hermes-engine")
	}
}

func Backends() string  { return filepath.Join(Data(), "backends") }
func Models() string    { return filepath.Join(Data(), "models") }
func Runtime() string   { return filepath.Join(Data(), "runtime") }
func Downloads() string { return filepath.Join(Data(), "downloads") }
func Logs() string      { return filepath.Join(Data(), "logs") }
func State() string     { return filepath.Join(Data(), "state.json") }
func Lock() string      { return filepath.Join(Data(), "engine.lock") }
