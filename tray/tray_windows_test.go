package tray

import (
	"testing"
	"unsafe"
)

// The shell reads these structs by byte layout, so their sizes must match the C headers on a
// 64-bit build.
func TestWin32StructSizes(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("layout checked for 64-bit only")
	}
	if got := unsafe.Sizeof(notifyIconData{}); got != 976 {
		t.Errorf("NOTIFYICONDATAW is %d bytes, want 976", got)
	}
	if got := unsafe.Sizeof(wndClassEx{}); got != 80 {
		t.Errorf("WNDCLASSEXW is %d bytes, want 80", got)
	}
	if got := unsafe.Sizeof(winMsg{}); got != 48 {
		t.Errorf("MSG is %d bytes, want 48", got)
	}
}

func TestCopyUTF16Truncates(t *testing.T) {
	var d [4]uint16
	copyUTF16(d[:], "ab\U0001F600") // a pair would not fit in 3 units after "ab"
	if d[0] != 'a' || d[1] != 'b' || d[2] != 0 || d[3] != 0 {
		t.Errorf("got %v", d)
	}
}
