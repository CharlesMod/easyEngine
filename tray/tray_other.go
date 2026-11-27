//go:build !linux && !windows && !darwin

package tray

func newBackend(Spec, func(string)) (backend, error) { return nil, ErrNoTray }
