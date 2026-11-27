package tray

import (
	"unicode/utf16"

	"golang.org/x/sys/windows"
)

var (
	user32   = windows.NewLazySystemDLL("user32.dll")
	shell32  = windows.NewLazySystemDLL("shell32.dll")
	kernel32 = windows.NewLazySystemDLL("kernel32.dll")

	procRegisterClassEx          = user32.NewProc("RegisterClassExW")
	procUnregisterClass          = user32.NewProc("UnregisterClassW")
	procCreateWindowEx           = user32.NewProc("CreateWindowExW")
	procDestroyWindow            = user32.NewProc("DestroyWindow")
	procDefWindowProc            = user32.NewProc("DefWindowProcW")
	procGetMessage               = user32.NewProc("GetMessageW")
	procDispatchMessage          = user32.NewProc("DispatchMessageW")
	procPostMessage              = user32.NewProc("PostMessageW")
	procSendMessage              = user32.NewProc("SendMessageW")
	procPostQuitMessage          = user32.NewProc("PostQuitMessage")
	procRegisterWindowMessage    = user32.NewProc("RegisterWindowMessageW")
	procFindWindow               = user32.NewProc("FindWindowW")
	procGetSystemMetrics         = user32.NewProc("GetSystemMetrics")
	procCreateIconFromResourceEx = user32.NewProc("CreateIconFromResourceEx")
	procLoadIcon                 = user32.NewProc("LoadIconW")
	procDestroyIcon              = user32.NewProc("DestroyIcon")
	procCreatePopupMenu          = user32.NewProc("CreatePopupMenu")
	procAppendMenu               = user32.NewProc("AppendMenuW")
	procTrackPopupMenu           = user32.NewProc("TrackPopupMenu")
	procDestroyMenu              = user32.NewProc("DestroyMenu")
	procSetForegroundWindow      = user32.NewProc("SetForegroundWindow")
	procGetCursorPos             = user32.NewProc("GetCursorPos")
	procShellNotifyIcon          = shell32.NewProc("Shell_NotifyIconW")
	procGetModuleHandle          = kernel32.NewProc("GetModuleHandleW")
)

const (
	wmDestroy     = 0x0002
	wmNull        = 0x0000
	wmContextMenu = 0x007B
	wmApp         = 0x8000

	nimAdd        = 0
	nimModify     = 1
	nimDelete     = 2
	nimSetVersion = 4

	nifMessage = 0x01
	nifIcon    = 0x02
	nifTip     = 0x04
	nifInfo    = 0x10
	nifShowTip = 0x80

	notifyIconVersion4 = 4
	ninSelect          = 0x0400
	ninKeySelect       = 0x0401
	niifInfo           = 0x01

	mfString    = 0x0000
	mfGrayed    = 0x0001
	mfChecked   = 0x0008
	mfSeparator = 0x0800

	tpmRightButton = 0x0002
	tpmBottomAlign = 0x0020
	tpmNoNotify    = 0x0080
	tpmReturnCmd   = 0x0100

	smCxSmIcon     = 49
	smCySmIcon     = 50
	idiApplication = 32512
	iconVersion    = 0x00030000
)

// notifyIconData is NOTIFYICONDATAW in full (Vista and later). On amd64 its size is 976 bytes;
// Go pads the fields the same way C does, and a test checks the size.
type notifyIconData struct {
	cbSize           uint32
	hWnd             uintptr
	uID              uint32
	uFlags           uint32
	uCallbackMessage uint32
	hIcon            uintptr
	szTip            [128]uint16
	dwState          uint32
	dwStateMask      uint32
	szInfo           [256]uint16
	uVersion         uint32 // union with uTimeout
	szInfoTitle      [64]uint16
	dwInfoFlags      uint32
	guidItem         windows.GUID
	hBalloonIcon     uintptr
}

// wndClassEx is WNDCLASSEXW.
type wndClassEx struct {
	cbSize        uint32
	style         uint32
	lpfnWndProc   uintptr
	cbClsExtra    int32
	cbWndExtra    int32
	hInstance     uintptr
	hIcon         uintptr
	hCursor       uintptr
	hbrBackground uintptr
	lpszMenuName  *uint16
	lpszClassName *uint16
	hIconSm       uintptr
}

// winMsg is MSG.
type winMsg struct {
	hwnd    uintptr
	message uint32
	wParam  uintptr
	lParam  uintptr
	time    uint32
	ptX     int32
	ptY     int32
	private uint32
}

type point struct{ x, y int32 }

// copyUTF16 fills dst with s as UTF-16, cut to leave room for the closing NUL, never in the
// middle of a surrogate pair. The rest of dst is zeroed.
func copyUTF16(dst []uint16, s string) {
	u := utf16.Encode([]rune(s))
	if len(u) > len(dst)-1 {
		u = u[:len(dst)-1]
		if n := len(u); n > 0 && u[n-1] >= 0xD800 && u[n-1] < 0xDC00 {
			u = u[:n-1] // a lone high surrogate would be broken text
		}
	}
	n := copy(dst, u)
	for i := n; i < len(dst); i++ {
		dst[i] = 0
	}
}

func utf16Ptr(s string) *uint16 {
	u := utf16.Encode([]rune(s))
	for i, c := range u {
		if c == 0 {
			u = u[:i]
			break
		}
	}
	p := make([]uint16, len(u)+1)
	copy(p, u)
	return &p[0]
}
