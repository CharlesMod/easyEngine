//go:build !windows

package openfile

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// opener is the command that hands a path to the desktop: open on macOS, xdg-open elsewhere.
// A test replaces it.
var opener = func() string {
	if runtime.GOOS == "darwin" {
		return "open"
	}
	return "xdg-open"
}()

func open(path string) error {
	out, err := exec.Command(opener, path).CombinedOutput()
	if err != nil {
		return fmt.Errorf("openfile: %s %s: %v: %s", opener, path, err, strings.TrimSpace(string(out)))
	}
	return nil
}
