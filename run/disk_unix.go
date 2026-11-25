//go:build !windows

package run

import "syscall"

func diskFree(dir string) (int, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, err
	}
	return int(uint64(st.Bavail) * uint64(st.Bsize) / (1 << 20)), nil
}
