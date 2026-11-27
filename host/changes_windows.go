package host

import (
	"encoding/binary"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/sys/windows"
)

// Changes is a channel of one coalesced signal for each change of the file
// at path (written, replaced by a rename, created, removed), and a func that
// lets the watch go. The directory must exist; the file need not. On windows
// it is a ReadDirectoryChangesW watch of the directory, kept to the file's
// name; a buffer the system overflowed is told as a change, since which file
// it was is not known.
func Changes(path string) (<-chan struct{}, func(), error) {
	dir, base := filepath.Split(path)
	if dir == "" {
		dir = "."
	}
	p, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return nil, nil, err
	}
	dh, err := windows.CreateFile(p, windows.FILE_LIST_DIRECTORY,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil, windows.OPEN_EXISTING,
		windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OVERLAPPED, 0)
	if err != nil {
		return nil, nil, err
	}
	ovEv, err := windows.CreateEvent(nil, 1, 0, nil)
	if err != nil {
		windows.CloseHandle(dh)
		return nil, nil, err
	}
	l, err := newLatch()
	if err != nil {
		windows.CloseHandle(ovEv)
		windows.CloseHandle(dh)
		return nil, nil, err
	}
	out := make(chan struct{}, 1)
	ov := new(windows.Overlapped)
	buf := make([]byte, 64*1024)
	const mask = windows.FILE_NOTIFY_CHANGE_FILE_NAME | windows.FILE_NOTIFY_CHANGE_LAST_WRITE |
		windows.FILE_NOTIFY_CHANGE_SIZE | windows.FILE_NOTIFY_CHANGE_CREATION
	arm := func() error {
		*ov = windows.Overlapped{HEvent: ovEv}
		windows.ResetEvent(ovEv)
		return windows.ReadDirectoryChanges(dh, &buf[0], uint32(len(buf)), false, mask, nil, ov, 0)
	}
	if err := arm(); err != nil {
		l.close()
		windows.CloseHandle(ovEv)
		windows.CloseHandle(dh)
		return nil, nil, err
	}
	go func() {
		defer windows.CloseHandle(dh)
		defer windows.CloseHandle(ovEv)
		defer l.close()
		for {
			ev, err := windows.WaitForMultipleObjects([]windows.Handle{ovEv, l.ev}, false, windows.INFINITE)
			if err != nil || ev != windows.WAIT_OBJECT_0 {
				// let go (or the wait failed): cancel and wait for the
				// cancel to complete before the handles are closed
				windows.CancelIoEx(dh, ov)
				var n uint32
				windows.GetOverlappedResult(dh, ov, &n, true)
				return
			}
			var n uint32
			if err := windows.GetOverlappedResult(dh, ov, &n, false); err != nil {
				signal(out) // the watch failed: say a change, then end
				return
			}
			got := append([]byte(nil), buf[:n]...)
			rearmErr := arm() // before reading, so no change falls between
			if n == 0 || namesFile(got, base) {
				signal(out)
			}
			if rearmErr != nil {
				signal(out)
				return
			}
		}
	}()
	var once sync.Once
	return out, func() { once.Do(l.wake) }, nil
}

// namesFile says whether any FILE_NOTIFY_INFORMATION record in b names base.
func namesFile(b []byte, base string) bool {
	off := 0
	for off+12 <= len(b) {
		next := int(binary.LittleEndian.Uint32(b[off:]))
		n := int(binary.LittleEndian.Uint32(b[off+8:]))
		if off+12+n > len(b) {
			return true // malformed: say a change
		}
		u := make([]uint16, n/2)
		for i := range u {
			u[i] = binary.LittleEndian.Uint16(b[off+12+2*i:])
		}
		if strings.EqualFold(windows.UTF16ToString(u), base) {
			return true
		}
		if next == 0 {
			break
		}
		off += next
	}
	return false
}
