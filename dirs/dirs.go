// STUB: replaced by eng/fetch
package dirs

import (
	"os"
	"path/filepath"
	"runtime"
)

// Data returns the engine data dir (HERMES_ENGINE_HOME overrides).
func Data() string {
	if v := os.Getenv("HERMES_ENGINE_HOME"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(os.Getenv("LOCALAPPDATA"), "HermesEngine")
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "HermesEngine")
	}
	if x := os.Getenv("XDG_DATA_HOME"); x != "" {
		return filepath.Join(x, "hermes-engine")
	}
	return filepath.Join(home, ".local", "share", "hermes-engine")
}
