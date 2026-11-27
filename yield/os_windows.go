package yield

import (
	"context"
	"strconv"
	"time"

	"golang.org/x/sys/windows/registry"
)

// SteamRegistry reads Steam's running app id; tests replace it.
type SteamRegistry interface {
	RunningAppID() (uint32, error)
}

type hkcuSteam struct{}

// RunningAppID reads HKCU\Software\Valve\Steam\RunningAppID (0: no game).
func (hkcuSteam) RunningAppID() (uint32, error) {
	k, err := registry.OpenKey(registry.CURRENT_USER, `Software\Valve\Steam`, registry.QUERY_VALUE)
	if err != nil {
		return 0, err
	}
	defer k.Close()
	v, _, err := k.GetIntegerValue("RunningAppID")
	return uint32(v), err
}

type windowsOS struct{ reg SteamRegistry }

// NewOS watches the real machine. On Windows the launcher signal is Steam's RunningAppID;
// there is no device-open signal (memory comes from the GPU Process Memory counters).
func NewOS() OS { return &windowsOS{reg: hkcuSteam{}} }

func newWindowsOS(r SteamRegistry) *windowsOS { return &windowsOS{reg: r} }

func (o *windowsOS) Launches() []Launch { return registryLaunches(o.reg) }

func registryLaunches(r SteamRegistry) []Launch {
	id, err := r.RunningAppID()
	if err != nil || id == 0 {
		return nil
	}
	return []Launch{{Kind: "steam-registry", AppID: strconv.FormatUint(uint64(id), 10)}}
}

func (o *windowsOS) Openers(func(int) bool) map[int]string          { return nil }
func (o *windowsOS) Under(int, int) bool                            { return false }
func (o *windowsOS) Wait(ctx context.Context, d time.Duration) bool { return sleepCtx(ctx, d) }
func (o *windowsOS) Drain()                                         {}
func (o *windowsOS) Close()                                         {}
