//go:build !linux

package host

// Opens answers ErrNoEvent: only linux says when a device node is opened
// (inotify IN_OPEN).
func Opens(paths []string) (<-chan struct{}, func(), error) { return nil, nil, ErrNoEvent }

// Own runs f: there is no device-open watch here to keep from hearing it.
func Own(f func()) { f() }
