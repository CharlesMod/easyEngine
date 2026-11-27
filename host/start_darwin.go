package host

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// StartOf is an opaque token of when the process pid started: the same every
// time for the same process, and different for a later process given the
// same number. A pid with no process is an error. On darwin it is the start
// time the kernel keeps for the process.
func StartOf(pid int) (string, error) {
	p, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil {
		return "", err
	}
	if int(p.Proc.P_pid) != pid {
		return "", fmt.Errorf("host: no process %d", pid)
	}
	t := p.Proc.P_starttime
	return fmt.Sprintf("%d.%06d", t.Sec, t.Usec), nil
}
