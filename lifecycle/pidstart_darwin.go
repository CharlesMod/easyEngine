//go:build darwin

package lifecycle

import (
	"fmt"
	"os/exec"
	"strings"
)

func PidStart(pid int) (string, error) {
	out, err := exec.Command("ps", "-o", "lstart=", "-p", fmt.Sprint(pid)).Output()
	s := strings.TrimSpace(string(out))
	if err != nil || s == "" {
		return "", fmt.Errorf("no such process")
	}
	return s, nil
}
