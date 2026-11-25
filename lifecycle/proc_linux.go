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
	spawnOnce sync.Once
	spawnCh   chan startReq
)

func prepare(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
}

func startChild(cmd *exec.Cmd) error {
	spawnOnce.Do(func() {
		spawnCh = make(chan startReq)
		go func() {
			runtime.LockOSThread() // never unlocked: the thread lives as long as the engine
			for r := range spawnCh {
				r.res <- r.cmd.Start()
			}
		}()
	})
	r := startReq{cmd, make(chan error, 1)}
	spawnCh <- r
	return <-r.res
}
