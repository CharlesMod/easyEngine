package run

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
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

// envDropped says whether the parent's variable kv never reaches the model server: every
// LLAMA_ARG_* (llama-server reads its flags, including the model and its download, from them),
// and the media variables of mediaEnv.
func envDropped(kv string) bool {
	k, _, _ := strings.Cut(kv, "=")
	if strings.HasPrefix(k, "LLAMA_ARG_") {
		return true
	}
	return slices.Contains(mediaEnv, k)
}

// withoutMediaEnv is env less every variable envDropped names.
func withoutMediaEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		if !envDropped(kv) {
			out = append(out, kv)
		}
	}
	return out
}

// droppedEnvNames are the names the parent's environment has that llamaEnv leaves out, sorted.
func droppedEnvNames() []string {
	var out []string
	for _, kv := range os.Environ() {
		if envDropped(kv) {
			k, _, _ := strings.Cut(kv, "=")
			out = append(out, k)
		}
	}
	slices.Sort(out)
	return out
}

// llamaEnv keeps the environment (less every LLAMA_ARG_* and media variable; the extras the engine adds are kept), puts the backend dir on the library path (the
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
