//go:build !windows

package lifecycle

import (
	"os/exec"
	"syscall"
)

func terminate(p *Proc, force bool) {
	sig := syscall.SIGTERM
	if force {
		sig = syscall.SIGKILL
	}
	// Negative pid: the whole process group (each child leads its own).
	if err := syscall.Kill(-p.Pid, sig); err != nil {
		syscall.Kill(p.Pid, sig)
	}
}

// reapGroup kills whatever is left in the leader's process group. The group id is the
// leader's pid and stays valid while any member lives, so this reaches survivors after
// the leader has exited; with none left the kernel answers ESRCH, which is ignored.
func reapGroup(p *Proc) {
	syscall.Kill(-p.Pid, syscall.SIGKILL)
}

func adopt(p *Proc) {}

// KillPid kills a pid and its group (used for stale children).
func KillPid(pid int) {
	if pid <= 1 {
		return
	}
	if syscall.Kill(-pid, syscall.SIGKILL) != nil {
		syscall.Kill(pid, syscall.SIGKILL)
	}
}

// PidAlive reports whether the pid exists.
func PidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}

var _ = exec.Command
