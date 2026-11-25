//go:build linux

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
		syscall.Dup3(int(in.Fd()), 0, 0)
	}
	syscall.Dup3(int(f.Fd()), 1, 0)
	syscall.Dup3(int(f.Fd()), 2, 0)
}
