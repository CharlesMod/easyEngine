package autostart

import (
	"errors"
	"os"
	"path/filepath"
)

type kind int

const (
	kindDesktop kind = iota // XDG .desktop file
	kindPlist               // macOS launch agent
	kindRun                 // Windows Run key (or its file stand-in)
)

// runKey is the Windows Run key, or a stand-in for it.
type runKey interface {
	get(name string) (value string, found bool, err error)
	set(name, value string) error
	del(name string) error
	where(name string) string
}

// Place is where startup entries live for one user: the real one, or a fake
// root for tests and demos.
type Place struct {
	kind kind
	base string // config home (desktop) or home (plist)
	key  runKey // kindRun only
	err  error  // set when the real place could not be found
}

// Real is this user's real place for this OS.
func Real() *Place { return realPlace() }

// Root is a fake place under dir that never touches the real one. On
// Linux/other unix files go to dir/autostart/, on macOS to
// dir/Library/LaunchAgents/, on Windows to dir/Run/<name>.cmd-line text files
// holding the command line.
func Root(dir string) *Place { return newPlace(nativeKind, dir) }

func newPlace(k kind, dir string) *Place {
	p := &Place{kind: k, base: dir}
	if k == kindRun {
		p.key = fileRunKey{dir: filepath.Join(dir, "Run")}
	}
	return p
}

// Path is where the entry lives: a file path, or HKCU\...\Run\<name> for the
// real Windows key. It is empty for an invalid name.
func (p *Place) Path(name string) string {
	if validName(name) != nil {
		return ""
	}
	switch p.kind {
	case kindPlist:
		return plistPath(p.base, plistLabel(name))
	case kindRun:
		return p.key.where(name)
	}
	return desktopPath(p.base, name)
}

// Enabled reports whether the entry exists (and, on Linux, is not hidden or disabled).
func (p *Place) Enabled(name string) (bool, error) {
	if err := p.check(name); err != nil {
		return false, err
	}
	switch p.kind {
	case kindRun:
		_, found, err := p.key.get(name)
		return found, err
	case kindPlist:
		_, found, err := fileStore{p.Path(name)}.read()
		return found, err
	}
	s, found, err := fileStore{p.Path(name)}.read()
	if err != nil || !found {
		return false, err
	}
	return desktopEnabled(s), nil
}

// Enable writes the entry so exe (with args) launches at the next login.
func (p *Place) Enable(name, exe string, args []string) error {
	if err := p.check(name); err != nil {
		return err
	}
	if err := validCommand(exe, args); err != nil {
		return err
	}
	switch p.kind {
	case kindRun:
		return p.key.set(name, windowsCommandLine(exe, args))
	case kindPlist:
		return fileStore{p.Path(name)}.write(launchAgentPlist(plistLabel(name), exe, args))
	}
	return fileStore{p.Path(name)}.write(desktopEntry(name, exe, args))
}

// Disable removes the entry; a missing entry is not an error.
func (p *Place) Disable(name string) error {
	if err := p.check(name); err != nil {
		return err
	}
	if p.kind == kindRun {
		return p.key.del(name)
	}
	return fileStore{p.Path(name)}.remove()
}

func (p *Place) check(name string) error {
	if p.err != nil {
		return p.err
	}
	return validName(name)
}

// fileStore is one file written atomically.
type fileStore struct{ path string }

func (f fileStore) read() (string, bool, error) {
	b, err := os.ReadFile(f.path)
	if errors.Is(err, os.ErrNotExist) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return string(b), true, nil
}

// write replaces the file atomically: temp file in the same dir, fsync, rename.
func (f fileStore) write(contents string) (err error) {
	dir := filepath.Dir(f.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(f.path)+".tmp-*")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmp.Name())
		}
	}()
	if _, err = tmp.WriteString(contents); err != nil {
		return err
	}
	if err = tmp.Chmod(0o644); err != nil {
		return err
	}
	if err = tmp.Sync(); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), f.path)
}

func (f fileStore) remove() error {
	if err := os.Remove(f.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// fileRunKey stands in for the Windows Run key: one text file per entry.
type fileRunKey struct{ dir string }

func (k fileRunKey) where(name string) string { return filepath.Join(k.dir, name+".cmd-line") }

func (k fileRunKey) get(name string) (string, bool, error) { return fileStore{k.where(name)}.read() }

func (k fileRunKey) set(name, value string) error { return fileStore{k.where(name)}.write(value) }

func (k fileRunKey) del(name string) error { return fileStore{k.where(name)}.remove() }
