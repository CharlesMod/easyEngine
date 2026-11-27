// Package legacy is the one place that names the engine's old product name (it was
// hermes-engine). The old names are part of a wire or disk contract with an installed game
// mod and a running service, so they keep working: this package holds them as constants and
// small helpers, and the rest of the code calls it. Nothing outside this package, docs/ and
// README.md may spell the old name (purity_test.go checks it).
package legacy

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/CharlesMod/easyEngine/lifecycle"
)

// Old wire names (HTTP headers).
const (
	EngineHeader  = "X-Hermes-Engine"
	SamplerHeader = "X-Hermes-Sampler"
	AppliedHeader = "X-Hermes-Sampler-Applied"
)

// Old environment names.
const (
	HomeEnv      = "HERMES_ENGINE_HOME"
	EngineURLEnv = "HERMES_ENGINE_URL"
	LLMURLEnv    = "HERMESVOICE_LLM_URL"
)

// Old data dir names.
const (
	dirLinux = "hermes-engine"
	dirOther = "HermesEngine" // Windows and macOS
)

// HeaderOr returns the value of the current header, else the old one's.
func HeaderOr(h http.Header, current, old string) string {
	if v := h.Get(current); v != "" {
		return v
	}
	return h.Get(old)
}

// CSRFOK reports whether the request carries the anti-CSRF header, under either name.
func CSRFOK(h http.Header, current string) bool {
	return h.Get(current) == "1" || h.Get(EngineHeader) == "1"
}

// MarkRequest sets the anti-CSRF header under both names, so that a request to an engine of
// either age (a new client attaching to a running old engine) is accepted.
func MarkRequest(h http.Header, current string) {
	h.Set(current, "1")
	h.Set(EngineHeader, "1")
}

// StripRequest removes the old request headers before a request is forwarded upstream.
func StripRequest(h http.Header) {
	h.Del(SamplerHeader)
	h.Del(EngineHeader)
}

// SetApplied sets the Applied response header under the current name and, with the same
// value, under the old one.
func SetApplied(h http.Header, current, v string) {
	h.Set(current, v)
	h.Set(AppliedHeader, v)
}

// Getenv reads current, and when it is empty the old variable.
func Getenv(getenv func(string) string, current, old string) string {
	if v := getenv(current); v != "" {
		return v
	}
	return getenv(old)
}

// HomeOverride is the data dir override: the current variable wins, else the old one.
func HomeOverride(getenv func(string) string, current string) string {
	return Getenv(getenv, current, HomeEnv)
}

// EngineURLVars returns KEY=value entries giving the engine's base URL to the agent, under the
// current name and the old one.
func EngineURLVars(current, base string) []string {
	return []string{current + "=" + base, EngineURLEnv + "=" + base}
}

// LLMURLNames returns the names of the variables that carry the LLM URL to the agent: the
// profile's own name when it has one, else the current default and the old default.
func LLMURLNames(configured, currentDefault string) []string {
	if configured != "" {
		return []string{configured}
	}
	return []string{currentDefault, LLMURLEnv}
}

// LLMPrefix is the stem of an LLM URL variable name: the _MODE and _MODEL companions hang on it.
func LLMPrefix(urlEnv string) string { return strings.TrimSuffix(urlEnv, "_URL") }

// OldDefaultDir is the data dir the engine used before the rename.
func OldDefaultDir(goos string, getenv func(string) string, home string) string {
	switch goos {
	case "windows":
		base := getenv("LOCALAPPDATA")
		if base == "" {
			base = filepath.Join(home, "AppData", "Local")
		}
		return filepath.Join(base, dirOther)
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", dirOther)
	default:
		base := getenv("XDG_DATA_HOME")
		if base == "" {
			base = filepath.Join(home, ".local", "share")
		}
		return filepath.Join(base, dirLinux)
	}
}

// FS is the little of the file system the data dir migration needs.
type FS interface {
	Exists(path string) bool // follows symlinks
	LockLive(dir string) bool
	MkdirAll(path string) error
	Rename(oldPath, newPath string) error
	Symlink(target, link string) error
}

// Action is what to do about the data dir.
type Action int

const (
	UseNew  Action = iota // use the new default dir (it exists, or neither does)
	UseOld                // use the old dir in place
	MoveOld               // move the old dir to the new path
)

// Decide is the migration decision, a pure function of its inputs. The env override (either
// name) means neither dir is touched. Otherwise: the new dir exists -> UseNew; only the old one
// exists -> UseOld while its engine.lock names a live engine, else MoveOld; neither -> UseNew.
func Decide(getenv func(string) string, newHomeEnv, newDir, oldDir string, fsys FS) Action {
	if HomeOverride(getenv, newHomeEnv) != "" {
		return UseNew
	}
	if fsys.Exists(newDir) || !fsys.Exists(oldDir) {
		return UseNew
	}
	if fsys.LockLive(oldDir) {
		return UseOld
	}
	return MoveOld
}

// Migrate carries out Decide for the start path of serve and setup. It deletes nothing: the
// move is one rename of the whole dir, after which (not on Windows) a symlink at the old path
// points to the new one so an older client still finds engine.lock. When the rename fails the
// old dir is used in place. It returns the dir in use and a note for the log ("" when there
// is nothing to say).
func Migrate(goos string, getenv func(string) string, newHomeEnv, newDir, oldDir string, fsys FS) (dir, note string) {
	switch Decide(getenv, newHomeEnv, newDir, oldDir, fsys) {
	case UseOld:
		return oldDir, "data dir: a running engine holds " + oldDir + ", using it in place"
	case MoveOld:
		if err := fsys.MkdirAll(filepath.Dir(newDir)); err != nil {
			return oldDir, "data dir: cannot move " + oldDir + " to " + newDir + " (" + err.Error() + "), using it in place"
		}
		if err := fsys.Rename(oldDir, newDir); err != nil {
			return oldDir, "data dir: cannot move " + oldDir + " to " + newDir + " (" + err.Error() + "), using it in place"
		}
		note = "data dir: moved " + oldDir + " to " + newDir
		if goos != "windows" {
			if err := fsys.Symlink(newDir, oldDir); err != nil {
				note += " (no symlink left at the old path: " + err.Error() + ")"
			}
		}
		return newDir, note
	}
	return newDir, ""
}

// OSFS is the real file system.
type OSFS struct{}

func (OSFS) Exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// LockLive reports whether dir/engine.lock names a live engine: its pid is alive and still
// carries the start time the lock recorded (the check app.Serve makes before attaching).
func (OSFS) LockLive(dir string) bool {
	d, err := lifecycle.ReadLock(filepath.Join(dir, "engine.lock"))
	return err == nil && d.Alive()
}

func (OSFS) MkdirAll(p string) error        { return os.MkdirAll(p, 0o755) }
func (OSFS) Rename(a, b string) error       { return os.Rename(a, b) }
func (OSFS) Symlink(target, l string) error { return os.Symlink(target, l) }
