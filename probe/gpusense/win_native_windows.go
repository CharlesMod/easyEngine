//go:build windows

package gpusense

import (
	"fmt"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

func init() {
	nativeDXGI = osDXGI
	nativePDH = osPDH
	nativeProcImage = osProcImage
}

// ---- DXGI (COM through vtables) ----

var iidFactory1 = windows.GUID{Data1: 0x770aae78, Data2: 0xf26f, Data3: 0x4dba,
	Data4: [8]byte{0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}}

// DXGI_ADAPTER_DESC1 (304 bytes on 64-bit).
type dxgiDesc1 struct {
	Description                              [128]uint16
	VendorID, DeviceID, SubSysID, Revision   uint32
	DedicatedVideoMemory, DedicatedSystemMem uintptr
	SharedSystemMemory                       uintptr
	LuidLow                                  uint32
	LuidHigh                                 int32
	Flags                                    uint32
}

const (
	vtRelease       = 2
	vtEnumAdapters1 = 12 // IDXGIFactory1
	vtGetDesc1      = 10 // IDXGIAdapter1
	dxgiNotFound    = 0x887A0002
)

// comCall calls slot n of obj's vtable.
func comCall(obj uintptr, n int, args ...uintptr) uintptr {
	vt := *(*unsafe.Pointer)(unsafe.Pointer(&obj))
	fn := *(*uintptr)(unsafe.Add(vt, n*int(unsafe.Sizeof(uintptr(0)))))
	r, _, _ := syscall.SyscallN(fn, append([]uintptr{obj}, args...)...)
	return r
}

func osDXGI() ([]dxgiAdapter, error) {
	dll := windows.NewLazySystemDLL("dxgi.dll")
	proc := dll.NewProc("CreateDXGIFactory1")
	if err := proc.Find(); err != nil {
		return nil, fmt.Errorf("dxgi: %w", err)
	}
	var factory uintptr
	r, _, _ := syscall.SyscallN(proc.Addr(), uintptr(unsafe.Pointer(&iidFactory1)), uintptr(unsafe.Pointer(&factory)))
	if r != 0 || factory == 0 {
		return nil, fmt.Errorf("dxgi: CreateDXGIFactory1 failed: 0x%08X", uint32(r))
	}
	defer comCall(factory, vtRelease)
	var out []dxgiAdapter
	for i := 0; i < 64; i++ {
		var adapter uintptr
		r := comCall(factory, vtEnumAdapters1, uintptr(i), uintptr(unsafe.Pointer(&adapter)))
		if uint32(r) == dxgiNotFound {
			break
		}
		if r != 0 || adapter == 0 {
			return out, fmt.Errorf("dxgi: EnumAdapters1(%d) failed: 0x%08X", i, uint32(r))
		}
		var d dxgiDesc1
		r = comCall(adapter, vtGetDesc1, uintptr(unsafe.Pointer(&d)))
		comCall(adapter, vtRelease)
		if r != 0 {
			continue
		}
		out = append(out, dxgiAdapter{Name: windows.UTF16ToString(d.Description[:]), VendorID: int(d.VendorID),
			DedicatedBytes: uint64(d.DedicatedVideoMemory), SharedBytes: uint64(d.SharedSystemMemory),
			LUIDHigh: uint32(d.LuidHigh), LUIDLow: d.LuidLow, Flags: d.Flags})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("dxgi: no adapters")
	}
	return out, nil
}

// ---- PDH ----

const (
	pdhFmtLarge = 0x400
	pdhMoreData = 0x800007D2
)

// PDH_FMT_COUNTERVALUE_ITEM_W with a 64-bit (large) value.
type pdhItem struct {
	Name    *uint16
	CStatus uint32
	Value   int64
}

func osPDH() ([]pdhSample, error) {
	dll := windows.NewLazySystemDLL("pdh.dll")
	open, addEn := dll.NewProc("PdhOpenQueryW"), dll.NewProc("PdhAddEnglishCounterW")
	collect, getArr := dll.NewProc("PdhCollectQueryData"), dll.NewProc("PdhGetFormattedCounterArrayW")
	closeQ := dll.NewProc("PdhCloseQuery")
	for _, p := range []*windows.LazyProc{open, addEn, collect, getArr, closeQ} {
		if err := p.Find(); err != nil {
			return nil, fmt.Errorf("pdh: %w", err)
		}
	}
	var q uintptr
	if r, _, _ := syscall.SyscallN(open.Addr(), 0, 0, uintptr(unsafe.Pointer(&q))); r != 0 {
		return nil, fmt.Errorf("pdh: PdhOpenQueryW: 0x%08X", uint32(r))
	}
	defer syscall.SyscallN(closeQ.Addr(), q)
	paths := []string{pdhProcDedicated, pdhProcShared, pdhAdapterDedicated}
	counters := make([]uintptr, len(paths))
	added := 0
	var firstErr error
	for i, path := range paths {
		pp, err := windows.UTF16PtrFromString(path)
		if err != nil {
			return nil, err
		}
		r, _, _ := syscall.SyscallN(addEn.Addr(), q, uintptr(unsafe.Pointer(pp)), 0, uintptr(unsafe.Pointer(&counters[i])))
		if r != 0 {
			counters[i] = 0
			if firstErr == nil {
				firstErr = fmt.Errorf("pdh: PdhAddEnglishCounterW %s: 0x%08X (no GPU counters on this Windows?)", path, uint32(r))
			}
			continue
		}
		added++
	}
	if added == 0 {
		return nil, firstErr
	}
	if r, _, _ := syscall.SyscallN(collect.Addr(), q); r != 0 {
		return nil, fmt.Errorf("pdh: PdhCollectQueryData: 0x%08X", uint32(r))
	}
	var out []pdhSample
	for i, c := range counters {
		if c == 0 {
			continue
		}
		var size, count uint32
		r, _, _ := syscall.SyscallN(getArr.Addr(), c, pdhFmtLarge, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), 0)
		if uint32(r) != pdhMoreData || size == 0 {
			continue // no instances (or an error): this counter contributes nothing
		}
		buf := make([]uint64, (size+7)/8) // 8-aligned
		r, _, _ = syscall.SyscallN(getArr.Addr(), c, pdhFmtLarge, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&buf[0])))
		if r != 0 {
			continue
		}
		items := unsafe.Slice((*pdhItem)(unsafe.Pointer(&buf[0])), int(count))
		for _, it := range items {
			if it.Name == nil || it.CStatus != 0 && it.CStatus != 1 { // PDH_CSTATUS_VALID_DATA / NEW_DATA
				continue
			}
			out = append(out, pdhSample{Counter: paths[i], Instance: windows.UTF16PtrToString(it.Name), Value: it.Value})
		}
	}
	return out, nil
}

// ---- process image names ----

func osProcImage(pid int) string {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ""
	}
	defer windows.CloseHandle(h)
	buf := make([]uint16, 1024)
	n := uint32(len(buf))
	if err := windows.QueryFullProcessImageName(h, 0, &buf[0], &n); err != nil {
		return ""
	}
	return windows.UTF16ToString(buf[:n])
}
