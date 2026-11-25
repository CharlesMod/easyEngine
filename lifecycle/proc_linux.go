//go:build linux

package lifecycle

import (
	"os/exec"
	"runtime"
	"sync"
	"syscall"
)

// Pdeathsig is tied to the OS thread that forked the child, so every start goes
// through one goroutine that locks its thread for the life of the process.
type startReq struct {
	cmd *exec.Cmd
	res chan error
}

var (
	launchOnce sync.Once
	launchCh   chan startReq
)

func prepare(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
}

func startChild(cmd *exec.Cmd) error {
	launchOnce.Do(func() {
		launchCh = make(chan startReq)
		go func() {
			runtime.LockOSThread() // never unlocked: the thread lives as long as the engine
			for r := range launchCh {
				r.res <- r.cmd.Start()
			}
		}()
	})
	r := startReq{cmd, make(chan error, 1)}
	launchCh <- r
	return <-r.res
}
