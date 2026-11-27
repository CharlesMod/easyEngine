package host

// netchange_windows.go — the network coming or going as windows says it:
// NotifyAddrChange with an overlapped event, which is set when an interface's
// address table changes. The goroutine waits on that event and on a latch,
// so the watch can be let go.

import (
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	iphlpapi             = windows.NewLazySystemDLL("iphlpapi.dll")
	procNotifyAddrChange = iphlpapi.NewProc("NotifyAddrChange")
	procCancelIPChange   = iphlpapi.NewProc("CancelIPChangeNotify")
)

// NetworkChanges is a channel of one coalesced signal for each time the
// system says a network link or address changed (the way "the network came
// back" is heard), and a func that lets the watch go. The channel is not
// closed. On windows it is NotifyAddrChange.
func NetworkChanges() (<-chan struct{}, func(), error) {
	if err := procNotifyAddrChange.Find(); err != nil {
		return nil, nil, err
	}
	l, err := newLatch()
	if err != nil {
		return nil, nil, err
	}
	ev, err := windows.CreateEvent(nil, 1, 0, nil) // manual reset
	if err != nil {
		l.close()
		return nil, nil, err
	}
	c := make(chan struct{}, 1)
	go func() {
		defer windows.CloseHandle(ev)
		defer l.close()
		var h windows.Handle
		ov := &windows.Overlapped{HEvent: ev}
		for {
			windows.ResetEvent(ev)
			r, _, _ := procNotifyAddrChange.Call(uintptr(unsafe.Pointer(&h)), uintptr(unsafe.Pointer(ov)))
			if r != uintptr(windows.ERROR_IO_PENDING) {
				return // it cannot be asked for
			}
			w, err := windows.WaitForMultipleObjects([]windows.Handle{ev, l.ev}, false, windows.INFINITE)
			if err != nil || w != windows.WAIT_OBJECT_0 {
				// let go (or failed): cancel the pending ask before ov leaves
				procCancelIPChange.Call(uintptr(unsafe.Pointer(ov)))
				return
			}
			signal(c)
		}
	}()
	var once sync.Once
	return c, func() { once.Do(l.wake) }, nil
}
