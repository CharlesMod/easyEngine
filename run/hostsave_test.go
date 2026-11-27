//go:build !windows

package run_test

import (
	"context"
	"runtime"
	"testing"
	"time"

	"hermesengine/internal/testkit"
	"hermesengine/probe"
	"hermesengine/run"
)

// A processor-only start's host reading is saved with the book at once, as a card reading is:
// it waits for no later save.
func TestHostReadingSaved(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.Probe = func(ctx context.Context) (probe.Hardware, error) {
		return probe.Hardware{OS: runtime.GOOS, Arch: runtime.GOARCH, Cores: 8, RAMTotalMiB: 65536, RAMAvailMiB: 60000}, nil
	}
	cfg.Measure = func(ctx context.Context, pid int) (int, error) { return 9000, nil }
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	testkit.WaitFor(t, 3*time.Second, "host reading in state.json", func() bool {
		for _, c := range run.LoadState(e.Data).Book.Entries {
			for _, r := range c.HostReadings {
				if r.RAMMiB == 9000 {
					return true
				}
			}
		}
		return false
	})
}
