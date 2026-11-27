// Package autostart turns "Launch at Startup" on and off for the current user,
// with no admin rights and no cgo.
//
//   - Windows: a REG_SZ value named after the entry under
//     HKCU\Software\Microsoft\Windows\CurrentVersion\Run holding the command line.
//   - macOS: a launch agent plist, ~/Library/LaunchAgents/<label>.plist, where
//     label is "com.easyengine."+name unless name already contains a dot. It is
//     never loaded with launchctl: it takes effect at the next login, and the
//     program that asks for it is already running, so nothing more is needed.
//   - Linux and other unix: an XDG autostart entry,
//     $XDG_CONFIG_HOME/autostart/<name>.desktop (default ~/.config/autostart).
//
// The decision logic is shared and testable everywhere; the OS files only find
// the real root. Root gives a fake place under a directory for tests and demos.
package autostart

import (
	"errors"
	"fmt"
	"strings"
)

// validName reports whether name is a usable entry id: non-empty and made only
// of ASCII letters, digits, '.', '-' and '_'.
func validName(name string) error {
	if name == "" {
		return errors.New("autostart: empty name")
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9',
			r == '.', r == '-', r == '_':
		default:
			return fmt.Errorf("autostart: invalid name %q (letters, digits, . - _ only)", name)
		}
	}
	return nil
}

func validCommand(exe string, args []string) error {
	if exe == "" {
		return errors.New("autostart: empty executable path")
	}
	if strings.ContainsRune(exe, 0) {
		return errors.New("autostart: NUL in executable path")
	}
	for _, a := range args {
		if strings.ContainsRune(a, 0) {
			return errors.New("autostart: NUL in argument")
		}
	}
	return nil
}

// Enabled reports whether name is set to launch at startup, in this user's real place.
func Enabled(name string) (bool, error) { return Real().Enabled(name) }

// Enable makes exe (with args) launch at this user's next login.
func Enable(name, exe string, args []string) error { return Real().Enable(name, exe, args) }

// Disable removes the entry; a missing entry is not an error.
func Disable(name string) error { return Real().Disable(name) }
