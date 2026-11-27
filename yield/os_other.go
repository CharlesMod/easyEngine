//go:build !linux && !windows

package yield

import (
	"context"
	"os"
	"path/filepath"
	"time"
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

func (o *otherOS) Openers(func(int) bool) map[int]string          { return nil }
func (o *otherOS) Under(int, int) bool                            { return false }
func (o *otherOS) Wait(ctx context.Context, d time.Duration) bool { return sleepCtx(ctx, d) }
func (o *otherOS) Drain()                                         {}
func (o *otherOS) Close()                                         {}
