package host

import (
	"fmt"
	"runtime"
	"sync"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

// RegistryChanges is a channel of one coalesced signal for each change of
// the registry key path under root ("HKCU", "HKLM", "HKCR" or "HKU"), its
// values or its subkeys' names, and a func that lets the watch go. On
// windows it is RegNotifyChangeKeyValue with an event.
func RegistryChanges(root, path string) (<-chan struct{}, func(), error) {
	var rk registry.Key
	switch root {
	case "HKCU":
		rk = registry.CURRENT_USER
	case "HKLM":
		rk = registry.LOCAL_MACHINE
	case "HKCR":
		rk = registry.CLASSES_ROOT
	case "HKU":
		rk = registry.USERS
	default:
		return nil, nil, fmt.Errorf("host: unknown registry root %q", root)
	}
	key, err := registry.OpenKey(rk, path, registry.NOTIFY|registry.QUERY_VALUE)
	if err != nil {
		return nil, nil, err
	}
	ev, err := windows.CreateEvent(nil, 1, 0, nil)
	if err != nil {
		key.Close()
		return nil, nil, err
	}
	l, err := newLatch()
	if err != nil {
		windows.CloseHandle(ev)
		key.Close()
		return nil, nil, err
	}
	const filter = windows.REG_NOTIFY_CHANGE_NAME | windows.REG_NOTIFY_CHANGE_LAST_SET
	arm := func() error {
		windows.ResetEvent(ev)
		return windows.RegNotifyChangeKeyValue(windows.Handle(key), false, filter, ev, true)
	}
	out := make(chan struct{}, 1)
	armed := make(chan error, 1)
	go func() {
		// a notification is tied to the thread that asked for it, so this
		// goroutine keeps its thread
		runtime.LockOSThread()
		defer windows.CloseHandle(ev)
		defer l.close()
		defer key.Close()
		err := arm()
		armed <- err
		if err != nil {
			return
		}
		for {
			i, err := windows.WaitForMultipleObjects([]windows.Handle{ev, l.ev}, false, windows.INFINITE)
			if err != nil || i != windows.WAIT_OBJECT_0 {
				return // let go
			}
			signal(out)
			if arm() != nil {
				return // the key is gone
			}
		}
	}()
	if err := <-armed; err != nil {
		return nil, nil, err
	}
	var once sync.Once
	return out, func() { once.Do(l.wake) }, nil
}
