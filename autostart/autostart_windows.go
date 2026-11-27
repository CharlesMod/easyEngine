package autostart

import (
	"errors"

	"golang.org/x/sys/windows/registry"
)

const nativeKind = kindRun

func realPlace() *Place { return &Place{kind: kindRun, key: registryRunKey{}} }

// registryRunKey is HKCU\Software\Microsoft\Windows\CurrentVersion\Run.
type registryRunKey struct{}

func (registryRunKey) where(name string) string { return `HKCU\` + runSubkey + `\` + name }

func (registryRunKey) get(name string) (string, bool, error) {
	k, err := registry.OpenKey(registry.CURRENT_USER, runSubkey, registry.QUERY_VALUE)
	if errors.Is(err, registry.ErrNotExist) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	defer k.Close()
	v, _, err := k.GetStringValue(name)
	if errors.Is(err, registry.ErrNotExist) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return v, true, nil
}

func (registryRunKey) set(name, value string) error {
	k, _, err := registry.CreateKey(registry.CURRENT_USER, runSubkey, registry.SET_VALUE)
	if err != nil {
		return err
	}
	defer k.Close()
	return k.SetStringValue(name, value)
}

func (registryRunKey) del(name string) error {
	k, err := registry.OpenKey(registry.CURRENT_USER, runSubkey, registry.SET_VALUE)
	if errors.Is(err, registry.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer k.Close()
	if err := k.DeleteValue(name); err != nil && !errors.Is(err, registry.ErrNotExist) {
		return err
	}
	return nil
}
