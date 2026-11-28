package run

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
)

func loadInt32(p *int32) int32 { return atomic.LoadInt32(p) }

// mediaEnv are llama-server's environment spellings of the projector and media flags the engine
// owns (they are refused in extra_args; the parent's environment must not carry them in either).
var mediaEnv = []string{
	"LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL", "LLAMA_ARG_MMPROJ_AUTO", "LLAMA_ARG_MMPROJ_OFFLOAD",
	"MTMD_BACKEND_DEVICE", "LLAMA_ARG_IMAGE_MIN_TOKENS", "LLAMA_ARG_IMAGE_MAX_TOKENS",
	"LLAMA_ARG_MTMD_BATCH_MAX_TOKENS", "LLAMA_ARG_MEDIA_ALLOW_PRIVATE_URLS", "LLAMA_MEDIA_MARKER",
	"LLAMA_ARG_VIDEO_FPS", "LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL", "LLAMA_ARG_VIDEO_FFMPEG_DIR",
}

// withoutMediaEnv is env less every mediaEnv variable.
func withoutMediaEnv(env []string) []string {
	out := make([]string, 0, len(env))
next:
	for _, kv := range env {
		for _, k := range mediaEnv {
			if strings.HasPrefix(kv, k+"=") {
				continue next
			}
		}
		out = append(out, kv)
	}
	return out
}

// llamaEnv keeps the environment (less the media variables), puts the backend dir on the library path (the
// release builds ship their shared libraries next to the binary) and adds extras.
func llamaEnv(bin string, extra []string) []string {
	env := withoutMediaEnv(os.Environ())
	dir := filepath.Dir(bin)
	switch runtime.GOOS {
	case "linux":
		env = append(env, "LD_LIBRARY_PATH="+dir+string(os.PathListSeparator)+os.Getenv("LD_LIBRARY_PATH"))
	case "darwin":
		env = append(env, "DYLD_LIBRARY_PATH="+dir+string(os.PathListSeparator)+os.Getenv("DYLD_LIBRARY_PATH"))
	}
	return append(env, extra...)
}
