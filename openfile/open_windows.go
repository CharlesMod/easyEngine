package openfile

import (
	"fmt"

	"golang.org/x/sys/windows"
)

// open asks the shell to open path with its default verb; it fails when no program is
// associated with the file's type, and Open then opens the folder.
func open(path string) error {
	verb, err := windows.UTF16PtrFromString("open")
	if err != nil {
		return err
	}
	file, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	if err := windows.ShellExecute(0, verb, file, nil, nil, windows.SW_SHOWNORMAL); err != nil {
		return fmt.Errorf("openfile: open %s: %w", path, err)
	}
	return nil
}
