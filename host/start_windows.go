package host

import (
	"strconv"

	"golang.org/x/sys/windows"
)

// StartOf is an opaque token of when the process pid started: the same every
// time for the same process, and different for a later process given the
// same number. A pid with no process is an error. On windows it is the
// creation time of the process.
func StartOf(pid int) (string, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return "", err
	}
	defer windows.CloseHandle(h)
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil {
		return "", err
	}
	t := uint64(creation.HighDateTime)<<32 | uint64(creation.LowDateTime)
	return strconv.FormatUint(t, 10), nil
}
