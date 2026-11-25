//go:build !windows

package probe

import (
	"os"
	"path/filepath"
	"syscall"
)

func (OSSources) DiskFreeMiB(path string) (int, error) {
	for {
		if _, err := os.Stat(path); err == nil {
			break
		}
		parent := filepath.Dir(path)
		if parent == path {
			break
		}
		path = parent
	}
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return int(uint64(st.Bavail) * uint64(st.Bsize) / mib), nil
}
