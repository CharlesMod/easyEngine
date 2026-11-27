//go:build !linux && !windows

package yield

import (
	"os"
	"path/filepath"

	"github.com/CharlesMod/easyEngine/host"
)

type otherOS struct{ vdf string }

// NewOS watches the real machine. On macOS the launcher signal is Steam's RunningAppID in
// ~/Library/Application Support/Steam/registry.vdf; there is no device-open signal and no
// per-process GPU memory (unified memory).
func NewOS() OS {
	home, _ := os.UserHomeDir()
	return &otherOS{vdf: filepath.Join(home, "Library", "Application Support", "Steam", "registry.vdf")}
}

func (o *otherOS) Launches() []Launch {
	b, err := os.ReadFile(o.vdf)
	if err != nil {
		return nil
	}
	if id := ParseRegistryVDF(b); id != "" {
		return []Launch{{Kind: "steam-registry", AppID: id}}
	}
	return nil
}

func (o *otherOS) Openers(func(int) bool) map[int]string { return nil }
func (o *otherOS) Under(int, int) bool                   { return false }
func (o *otherOS) Adapter(string)                        {}
func (o *otherOS) Close()                                {}

// Watch is the change of registry.vdf, where Steam keeps RunningAppID.
func (o *otherOS) Watch() Words {
	ch, stop, err := host.Changes(o.vdf)
	if err != nil {
		return Words{}
	}
	return Words{Launches: ch, Stop: stop}
}
