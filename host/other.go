//go:build !linux && !darwin && !windows

package host

// other.go — a system this package has no words for: each answer is
// ErrNoEvent, so the tree builds everywhere and a caller knows.

// Exits answers ErrNoEvent on this system.
func Exits(pid int) (<-chan struct{}, func(), error) { return nil, nil, ErrNoEvent }

// StartOf answers ErrNoEvent on this system.
func StartOf(pid int) (string, error) { return "", ErrNoEvent }

// Changes answers ErrNoEvent on this system.
func Changes(path string) (<-chan struct{}, func(), error) { return nil, nil, ErrNoEvent }

// NetworkChanges answers ErrNoEvent on this system.
func NetworkChanges() (<-chan struct{}, func(), error) { return nil, nil, ErrNoEvent }
