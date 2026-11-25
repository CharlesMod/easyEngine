//go:build !linux && !windows

package lifecycle

import (
	"os/exec"
	"syscall"
)

// macOS: own process group; the engine kills the group on every exit path and
// the next start kills stale pids from engine.lock.
func prepare(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func startChild(cmd *exec.Cmd) error { return cmd.Start() }
