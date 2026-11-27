//go:build !linux && !darwin && !windows

package autostart

import "os"

// Other unix systems use the Linux XDG rules.
const nativeKind = kindDesktop

func realPlace() *Place {
	home, err := os.UserHomeDir()
	if err != nil {
		return &Place{kind: kindDesktop, err: err}
	}
	return &Place{kind: kindDesktop, base: configHome(os.Getenv("XDG_CONFIG_HOME"), home)}
}
