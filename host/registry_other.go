//go:build !windows

package host

// RegistryChanges is ErrNoEvent here: only windows has a registry.
func RegistryChanges(root, path string) (<-chan struct{}, func(), error) {
	return nil, nil, ErrNoEvent
}
