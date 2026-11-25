//go:build !linux && !windows

package app

import (
	"os"
	"os/signal"
	"syscall"
)

func detachStdio(logPath string) {
	signal.Ignore(syscall.SIGPIPE)
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		f, err = os.OpenFile(os.DevNull, os.O_RDWR, 0)
		if err != nil {
			return
		}
	}
	in, _ := os.Open(os.DevNull)
	if in != nil {
		syscall.Dup2(int(in.Fd()), 0)
	}
	syscall.Dup2(int(f.Fd()), 1)
	syscall.Dup2(int(f.Fd()), 2)
}
