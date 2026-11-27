package yield

import (
	"context"
	"strings"
	"time"
)

// OS is the platform's view of other programs; tests replace it.
type OS interface {
	// Launches lists the games launchers are running now.
	Launches() []Launch
	// Openers lists processes holding a GPU device open (pid -> name), skipping those skip
	// says; nil where the platform cannot tell.
	Openers(skip func(pid int) bool) map[int]string
	// Adapter narrows Openers to the device nodes of the adapter at this PCI address (Linux:
	// /dev/nvidia<minor>, its DRM card and render nodes); "" or an unknown address: every GPU node.
	Adapter(pci string)
	// Under says whether pid descends from anc (false where the platform cannot tell).
	Under(pid, anc int) bool
	// Wait returns after d, or early (true) when some process opened a GPU device.
	Wait(ctx context.Context, d time.Duration) bool
	// Drain forgets device opens seen so far (the engine's own probes open the device too).
	Drain()
	Close()
}

// waitOnly is an OS piece for platforms with no open signal.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	}
	return false
}

// ParseRegistryVDF finds Steam's RunningAppID in a registry.vdf (macOS and Linux Steam keep
// it there): "0" or absent means no game.
func ParseRegistryVDF(b []byte) string {
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && strings.EqualFold(strings.Trim(f[0], `"`), "RunningAppID") {
			v := strings.Trim(f[1], `"`)
			if digits(v) && strings.Trim(v, "0") != "" {
				return v
			}
			return ""
		}
	}
	return ""
}
