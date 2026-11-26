//go:build windows

package run

import (
	"os"

	"golang.org/x/sys/windows"
)

type fileKey struct{ vol, idx uint64 }

// allocOf is the disk a file takes (its size rounded up to a 4 KiB cluster) and, for a file with
// more than one hard link, the key that identifies it.
func allocOf(p string, fi os.FileInfo) (int64, fileKey, bool) {
	n := roundBlock(fi.Size())
	if fi.IsDir() {
		return 4096, fileKey{}, false
	}
	h, err := windows.Open(p, windows.O_RDONLY, 0)
	if err != nil {
		return n, fileKey{}, false
	}
	defer windows.CloseHandle(h)
	var info windows.ByHandleFileInformation
	if windows.GetFileInformationByHandle(h, &info) != nil || info.NumberOfLinks < 2 {
		return n, fileKey{}, false
	}
	return n, fileKey{uint64(info.VolumeSerialNumber), uint64(info.FileIndexHigh)<<32 | uint64(info.FileIndexLow)}, true
}
