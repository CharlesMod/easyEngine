//go:build windows

package run

import "golang.org/x/sys/windows"

func diskFree(dir string) (int, error) {
	p, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return 0, err
	}
	var free, total, totalFree uint64
	if err := windows.GetDiskFreeSpaceEx(p, &free, &total, &totalFree); err != nil {
		return 0, err
	}
	return int(free / (1 << 20)), nil
}
