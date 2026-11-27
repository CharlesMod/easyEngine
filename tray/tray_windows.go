package tray

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"golang.org/x/sys/windows"
)

// Private window messages. The UI goroutine owns the icon and the menu, so other goroutines ask
// it to act with SendMessageW (which waits for the answer) and leave their arguments in the
// winTray under its mutex.
const (
	wmTray    = wmApp + 1 // the shell's callback message for the icon
	wmSetTip  = wmApp + 2
	wmNotify  = wmApp + 3
	wmShutOff = wmApp + 4
)

const trayIconID = 1

// winTray is the Windows backend: a hidden window plus a notification area icon.
type winTray struct {
	click func(string)
	done  chan struct{} // closed when the UI goroutine has fully exited

	hwnd           uintptr // set before newBackend returns
	hinst          uintptr
	class          *uint16
	icon           uintptr
	iconOwned      bool   // true when icon was made here and must be destroyed
	tip            string // UI goroutine only
	taskbarCreated uint32 // the "TaskbarCreated" broadcast, sent when Explorer restarts

	mu           sync.Mutex // guards the fields below
	items        []Item
	pendingTip   string
	pendingTitle string
	pendingBody  string
}

var (
	classCounter atomic.Uint32
	instances    sync.Map // hwnd (uintptr) -> *winTray
	procOnce     sync.Once
	wndProcPtr   uintptr
)

// newBackend starts the UI goroutine and waits for it to report whether the icon was added.
func newBackend(spec Spec, click func(string)) (backend, error) {
	var sid uint32
	if err := windows.ProcessIdToSessionId(windows.GetCurrentProcessId(), &sid); err == nil && sid == 0 {
		return nil, fmt.Errorf("tray: running in session 0 (a service): %w", ErrNoTray)
	}
	if h, _, _ := procFindWindow.Call(uintptr(unsafe.Pointer(utf16Ptr("Shell_TrayWnd"))), 0); h == 0 {
		return nil, fmt.Errorf("tray: no taskbar (Shell_TrayWnd) found: %w", ErrNoTray)
	}
	t := &winTray{click: click, done: make(chan struct{}), items: cloneItems(spec.Items), tip: spec.Tooltip}
	ready := make(chan error, 1)
	go t.run(spec, ready)
	if err := <-ready; err != nil {
		<-t.done
		return nil, err
	}
	return t, nil
}

// run is the UI goroutine: it makes the window and icon, then pumps messages until the window
// is destroyed. Everything that touches the window happens on this one locked thread.
func (t *winTray) run(spec Spec, ready chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer close(t.done)

	procOnce.Do(func() { wndProcPtr = windows.NewCallback(wndProc) })
	h, _, _ := procGetModuleHandle.Call(0)
	t.hinst = h
	t.class = utf16Ptr(fmt.Sprintf("easyEngineTray_%d_%d", windows.GetCurrentProcessId(), classCounter.Add(1)))
	wc := wndClassEx{
		cbSize:        uint32(unsafe.Sizeof(wndClassEx{})),
		lpfnWndProc:   wndProcPtr,
		hInstance:     t.hinst,
		lpszClassName: t.class,
	}
	if r, _, err := procRegisterClassEx.Call(uintptr(unsafe.Pointer(&wc))); r == 0 {
		ready <- fmt.Errorf("tray: RegisterClassExW: %w", err)
		return
	}
	defer procUnregisterClass.Call(uintptr(unsafe.Pointer(t.class)), t.hinst)

	// A hidden top-level window (not message-only) so it gets the TaskbarCreated broadcast.
	hwnd, _, err := procCreateWindowEx.Call(0, uintptr(unsafe.Pointer(t.class)),
		uintptr(unsafe.Pointer(utf16Ptr(spec.Title))), 0, 0, 0, 0, 0, 0, 0, t.hinst, 0)
	if hwnd == 0 {
		ready <- fmt.Errorf("tray: CreateWindowExW: %w", err)
		return
	}
	t.hwnd = hwnd
	instances.Store(hwnd, t)
	defer instances.Delete(hwnd)
	m, _, _ := procRegisterWindowMessage.Call(uintptr(unsafe.Pointer(utf16Ptr("TaskbarCreated"))))
	t.taskbarCreated = uint32(m)

	t.loadIcon(spec.Icon.ICO)
	if t.iconOwned {
		defer procDestroyIcon.Call(t.icon)
	}
	if !t.addIcon() {
		procDestroyWindow.Call(hwnd)
		ready <- fmt.Errorf("tray: Shell_NotifyIconW(NIM_ADD) failed: %w", ErrNoTray)
		return
	}
	ready <- nil

	var msg winMsg
	for {
		r, _, _ := procGetMessage.Call(uintptr(unsafe.Pointer(&msg)), 0, 0, 0)
		if int32(r) <= 0 { // 0 is WM_QUIT, -1 an error
			return
		}
		procDispatchMessage.Call(uintptr(unsafe.Pointer(&msg)))
	}
}

// loadIcon builds the icon from the ICO bytes at the small-icon size, and falls back to the
// stock application icon.
func (t *winTray) loadIcon(ico []byte) {
	if len(ico) > 0 {
		cx, _, _ := procGetSystemMetrics.Call(smCxSmIcon)
		cy, _, _ := procGetSystemMetrics.Call(smCySmIcon)
		if off, size, err := selectIcoEntry(ico, int(cx)); err == nil {
			h, _, _ := procCreateIconFromResourceEx.Call(uintptr(unsafe.Pointer(&ico[off])), uintptr(size),
				1, iconVersion, cx, cy, 0)
			if h != 0 {
				t.icon, t.iconOwned = h, true
				return
			}
		}
	}
	t.icon, _, _ = procLoadIcon.Call(0, idiApplication)
}

func (t *winTray) data(flags uint32) *notifyIconData {
	return &notifyIconData{
		cbSize: uint32(unsafe.Sizeof(notifyIconData{})),
		hWnd:   t.hwnd,
		uID:    trayIconID,
		uFlags: flags,
	}
}

func shell(op uint32, d *notifyIconData) bool {
	r, _, _ := procShellNotifyIcon.Call(uintptr(op), uintptr(unsafe.Pointer(d)))
	return r != 0
}

// addIcon puts the icon in the notification area and switches it to version 4 callbacks, where
// the low word of lParam is the event. It runs again after Explorer restarts.
func (t *winTray) addIcon() bool {
	d := t.data(nifMessage | nifIcon | nifTip | nifShowTip)
	d.uCallbackMessage = wmTray
	d.hIcon = t.icon
	copyUTF16(d.szTip[:], t.tip)
	if !shell(nimAdd, d) {
		return false
	}
	v := t.data(0)
	v.uVersion = notifyIconVersion4
	shell(nimSetVersion, v)
	return true
}

// wndProc is the window procedure, always called on the UI thread.
func wndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	v, ok := instances.Load(hwnd)
	if !ok {
		r, _, _ := procDefWindowProc.Call(hwnd, msg, wParam, lParam)
		return r
	}
	t := v.(*winTray)
	switch {
	case msg == wmTray:
		switch uint32(lParam & 0xFFFF) {
		// With version 4 a left click sends WM_LBUTTONUP and then NIN_SELECT: only the
		// latter opens the menu, so it opens once. Right click and the menu key send
		// WM_CONTEXTMENU.
		case wmContextMenu, ninSelect, ninKeySelect:
			t.popup()
		}
		return 0
	case msg == wmSetTip:
		t.mu.Lock()
		t.tip = t.pendingTip
		t.mu.Unlock()
		d := t.data(nifTip | nifShowTip)
		copyUTF16(d.szTip[:], t.tip)
		shell(nimModify, d)
		return 0
	case msg == wmNotify:
		t.mu.Lock()
		title, body := t.pendingTitle, t.pendingBody
		t.mu.Unlock()
		d := t.data(nifInfo)
		copyUTF16(d.szInfoTitle[:], title)
		copyUTF16(d.szInfo[:], body)
		d.dwInfoFlags = niifInfo
		if shell(nimModify, d) {
			return 1
		}
		return 0
	case msg == wmShutOff:
		shell(nimDelete, t.data(0))
		procDestroyWindow.Call(hwnd)
		return 0
	case msg == wmDestroy:
		procPostQuitMessage.Call(0)
		return 0
	case t.taskbarCreated != 0 && uint32(msg) == t.taskbarCreated:
		t.addIcon() // Explorer restarted and lost the icon
		return 0
	}
	r, _, _ := procDefWindowProc.Call(hwnd, msg, wParam, lParam)
	return r
}

// popup shows the menu at the cursor and reports the chosen item's ID.
func (t *winTray) popup() {
	t.mu.Lock()
	items := cloneItems(t.items)
	t.mu.Unlock()
	if len(items) == 0 {
		return
	}
	menu, _, _ := procCreatePopupMenu.Call()
	if menu == 0 {
		return
	}
	defer procDestroyMenu.Call(menu)
	for i, it := range items {
		var flags uintptr = mfString
		var text *uint16
		switch {
		case it.Separator:
			flags = mfSeparator
		default:
			if it.Disabled || it.ID == "" {
				flags |= mfGrayed
			}
			if it.Checked {
				flags |= mfChecked
			}
			text = utf16Ptr(escapeAmp(it.Title)) // an unescaped & would make a mnemonic
		}
		procAppendMenu.Call(menu, flags, uintptr(i+1), uintptr(unsafe.Pointer(text)))
	}
	var pt point
	procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	// The window must be in front or the menu will not close when the user clicks elsewhere.
	procSetForegroundWindow.Call(t.hwnd)
	id, _, _ := procTrackPopupMenu.Call(menu, tpmReturnCmd|tpmNoNotify|tpmRightButton|tpmBottomAlign,
		uintptr(pt.x), uintptr(pt.y), 0, t.hwnd, 0)
	procPostMessage.Call(t.hwnd, wmNull, 0, 0) // documented fix: lets the menu dismiss properly
	if id == 0 || int(id) > len(items) {
		return
	}
	it := items[id-1]
	if it.Separator || it.Disabled || it.ID == "" {
		return
	}
	t.click(it.ID)
}

func escapeAmp(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '&' {
			out = append(out, '&')
		}
		out = append(out, s[i])
	}
	return string(out)
}

func (t *winTray) setMenu(items []Item) {
	t.mu.Lock()
	t.items = items
	t.mu.Unlock()
}

func (t *winTray) setTooltip(s string) {
	t.mu.Lock()
	t.pendingTip = s
	t.mu.Unlock()
	procSendMessage.Call(t.hwnd, wmSetTip, 0, 0)
}

func (t *winTray) notify(title, body string) error {
	t.mu.Lock()
	t.pendingTitle, t.pendingBody = title, body
	t.mu.Unlock()
	if r, _, _ := procSendMessage.Call(t.hwnd, wmNotify, 0, 0); r == 0 {
		return errors.New("tray: Shell_NotifyIconW(NIM_MODIFY) refused the notification")
	}
	return nil
}

// close takes the icon down on the UI thread and waits for that thread to finish.
func (t *winTray) close() {
	procSendMessage.Call(t.hwnd, wmShutOff, 0, 0)
	<-t.done
}
