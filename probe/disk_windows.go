//go:build windows

package probe

import (
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

func (OSSources) DiskFreeMiB(path string) (int, error) {
	for {
		if _, err := os.Stat(path); err == nil {
			break
		}
		parent := filepath.Dir(path)
		if parent == path {
			break
		}
		path = parent
	}
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	k32 := syscall.NewLazyDLL("kernel32.dll")
	proc := k32.NewProc("GetDiskFreeSpaceExW")
	var avail, total, free uint64
	r, _, e := proc.Call(uintptr(unsafe.Pointer(p)), uintptr(unsafe.Pointer(&avail)), uintptr(unsafe.Pointer(&total)), uintptr(unsafe.Pointer(&free)))
	if r == 0 {
		return 0, e
	}
	return int(avail / mib), nil
}
