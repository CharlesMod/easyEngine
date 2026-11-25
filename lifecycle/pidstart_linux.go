//go:build linux

package lifecycle

import (
	"fmt"
	"os"
	"strings"
)

// PidStart returns a token that is stable for one process and differs for a
// reused pid: field 22 of /proc/<pid>/stat (start time in clock ticks).
func PidStart(pid int) (string, error) {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return "", err
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	if i < 0 {
		return "", fmt.Errorf("bad stat")
	}
	f := strings.Fields(s[i+1:])
	if len(f) < 20 {
		return "", fmt.Errorf("short stat")
	}
	return f[19], nil // fields after ")" start at field 3, so field 22 is index 19
}
