package autostart

import "os"

const nativeKind = kindPlist

func realPlace() *Place {
	home, err := os.UserHomeDir()
	if err != nil {
		return &Place{kind: kindPlist, err: err}
	}
	return &Place{kind: kindPlist, base: home}
}
