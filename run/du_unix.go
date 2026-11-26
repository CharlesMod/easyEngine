//go:build !windows

package run

import (
	"os"
	"syscall"
)

type fileKey struct{ dev, ino uint64 }

// allocOf is the disk a file takes (its allocated blocks) and, for a file with more than one
// hard link, the key that identifies it.
func allocOf(_ string, fi os.FileInfo) (int64, fileKey, bool) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return roundBlock(fi.Size()), fileKey{}, false
	}
	return int64(st.Blocks) * 512, fileKey{uint64(st.Dev), uint64(st.Ino)}, !fi.IsDir() && st.Nlink > 1
}
