package lifecycle

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// LockData is the content of engine.lock.
type LockData struct {
	Pid     int    `json:"pid"`
	Start   string `json:"start"`
	Port    int    `json:"port"`
	Started string `json:"started"`
	// Host is the address the control API listens on. Empty (an old lock) means loopback.
	Host     string     `json:"host,omitempty"`
	Children []ChildRef `json:"children,omitempty"`
}

// Alive reports whether the engine the lock names is still running: its pid exists and still
// carries the recorded start time (a reused pid does not count).
func (d LockData) Alive() bool {
	if !PidAlive(d.Pid) {
		return false
	}
	if st, err := PidStart(d.Pid); err == nil && d.Start != "" && st != d.Start {
		return false
	}
	return true
}

// Lock is the single-instance lock file.
type Lock struct {
	mu      sync.Mutex
	path    string
	data    LockData
	removed bool
}

// ReadLock reads an existing lock file.
func ReadLock(path string) (LockData, error) {
	var d LockData
	b, err := os.ReadFile(path)
	if err != nil {
		return d, err
	}
	err = json.Unmarshal(b, &d)
	return d, err
}

// KillStale kills the children recorded in a stale lock, but only those whose
// pid still carries the recorded start time (a reused pid is left alone).
func KillStale(d LockData) []int {
	var killed []int
	for _, c := range d.Children {
		if !PidAlive(c.Pid) {
			continue
		}
		if st, err := PidStart(c.Pid); err == nil && c.Start != "" && st != c.Start {
			continue
		}
		KillPid(c.Pid)
		killed = append(killed, c.Pid)
	}
	return killed
}

// ErrLockHeld means the lock file already exists.
var ErrLockHeld = errors.New("engine.lock exists")

// CreateLock creates the lock exclusively and with its content: a reader never sees it empty.
//
// The content goes to a uniquely named temp file in the same dir, is fsynced, and is then
// hard-linked to the lock path. Link fails if the lock exists, so exactly one creator wins, and
// the lock appears whole. On a filesystem with no hard links we fall back to an O_EXCL create,
// write and fsync; in that window the lock is briefly empty, so a reader must treat an empty
// lock that is younger than LockWriteWindow as held, not stale (see LockInWrite).
func CreateLock(path string, d LockData) (*Lock, error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	b, err := json.Marshal(d)
	if err != nil {
		return nil, err
	}
	l := &Lock{path: path, data: d}
	tmp, err := os.CreateTemp(dir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	_, werr := tmp.Write(b)
	if werr == nil {
		werr = tmp.Sync()
	}
	if cerr := tmp.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		return nil, werr
	}
	os.Chmod(tmpPath, 0o644)
	lerr := os.Link(tmpPath, path)
	if lerr == nil {
		syncDir(dir)
		return l, nil
	}
	if errors.Is(lerr, os.ErrExist) {
		return nil, ErrLockHeld
	}
	// No hard links here: create exclusively, then write and fsync the content.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return nil, ErrLockHeld
		}
		return nil, err
	}
	_, werr = f.Write(b)
	if werr == nil {
		werr = f.Sync()
	}
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		os.Remove(path)
		return nil, werr
	}
	return l, nil
}

// RemoveStale removes the lock file if it still holds the stale lock d that the caller judged
// dead. Two starters can both see the same stale lock; the one that loses the race must not
// remove the lock the winner has just created, so the content is read again right before the
// remove. (A window of a few instructions remains; the O_EXCL/link create is what guarantees
// a single winner.)
func RemoveStale(path string, d LockData) bool {
	cur, err := ReadLock(path)
	if err != nil || cur.Pid != d.Pid || cur.Start != d.Start || cur.Port != d.Port || cur.Started != d.Started {
		return false
	}
	return os.Remove(path) == nil
}

// LockWriteWindow is how long a freshly created, still empty lock is taken to be mid-write
// (the O_EXCL fallback of CreateLock) rather than stale. It is a comparison of file times, not a timer.
const LockWriteWindow = 5 * time.Second

// LockInWrite reports whether the lock file exists, is empty and was modified less than
// LockWriteWindow ago: its creator is between create and write, so it is held, not stale.
func LockInWrite(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.Size() == 0 && time.Since(fi.ModTime()) < LockWriteWindow
}

func (l *Lock) write() error {
	b, err := json.Marshal(l.data)
	if err != nil {
		return err
	}
	return WriteFileAtomic(l.path, b, 0o644)
}

// SetChildren records the live children.
func (l *Lock) SetChildren(c []ChildRef) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.removed {
		return
	}
	l.data.Children = c
	l.write()
}

// Remove deletes the lock file.
func (l *Lock) Remove() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.removed = true
	os.Remove(l.path)
}
