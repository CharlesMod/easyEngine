//go:build !linux && !darwin && !windows

package lifecycle

import "fmt"

func PidStart(pid int) (string, error) { return "", fmt.Errorf("unsupported") }
