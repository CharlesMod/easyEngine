package lifecycle

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

// LockData is the content of engine.lock.
type LockData struct {
	Pid      int        `json:"pid"`
	Start    string     `json:"start"`
	Port     int        `json:"port"`
	Started  string     `json:"started"`
	Children []ChildRef `json:"children,omitempty"`
}

// Lock is the single-instance lock file.
type Lock struct {
	mu   sync.Mutex
	path string
	data LockData
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

// CreateLock creates the lock exclusively.
func CreateLock(path string, d LockData) (*Lock, error) {
	os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return nil, ErrLockHeld
		}
		return nil, err
	}
	f.Close()
	l := &Lock{path: path, data: d}
	return l, l.write()
}

func (l *Lock) write() error {
	b, _ := json.Marshal(l.data)
	tmp := l.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, l.path)
}

// SetChildren records the live children.
func (l *Lock) SetChildren(c []ChildRef) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.data.Children = c
	l.write()
}

// Remove deletes the lock file.
func (l *Lock) Remove() { os.Remove(l.path) }
