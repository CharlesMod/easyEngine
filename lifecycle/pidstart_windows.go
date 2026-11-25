//go:build windows

package lifecycle

import (
	"fmt"

	"golang.org/x/sys/windows"
)

func PidStart(pid int) (string, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return "", err
	}
	defer windows.CloseHandle(h)
	var c, e, k, u windows.Filetime
	if err := windows.GetProcessTimes(h, &c, &e, &k, &u); err != nil {
		return "", err
	}
	return fmt.Sprintf("%d", uint64(c.HighDateTime)<<32|uint64(c.LowDateTime)), nil
}
