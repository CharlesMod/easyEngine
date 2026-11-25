//go:build windows

package app

import (
	"os"

	"golang.org/x/sys/windows"
)

func detachStdio(logPath string) {
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		f, err = os.OpenFile(os.DevNull, os.O_RDWR, 0)
		if err != nil {
			return
		}
	}
	h := windows.Handle(f.Fd())
	windows.SetStdHandle(windows.STD_OUTPUT_HANDLE, h)
	windows.SetStdHandle(windows.STD_ERROR_HANDLE, h)
	os.Stdout, os.Stderr = f, f
}
