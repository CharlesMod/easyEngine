package run

import (
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
)

func loadInt32(p *int32) int32 { return atomic.LoadInt32(p) }

// llamaEnv keeps the environment, puts the backend dir on the library path (the
// release builds ship their shared libraries next to the binary) and adds extras.
func llamaEnv(bin string, extra []string) []string {
	env := os.Environ()
	dir := filepath.Dir(bin)
	switch runtime.GOOS {
	case "linux":
		env = append(env, "LD_LIBRARY_PATH="+dir+string(os.PathListSeparator)+os.Getenv("LD_LIBRARY_PATH"))
	case "darwin":
		env = append(env, "DYLD_LIBRARY_PATH="+dir+string(os.PathListSeparator)+os.Getenv("DYLD_LIBRARY_PATH"))
	}
	return append(env, extra...)
}
