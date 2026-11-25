package run

import (
	"context"
	"math"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"hermesengine/fit"
)

var helpCache sync.Map // bin path -> map[string]bool

var flagRe = regexp.MustCompile(`(?:^|[\s,\[|(])(--?[A-Za-z][A-Za-z0-9_-]*)`)

// knownFlags runs `llama-server --help` once per binary and returns the flag
// names it mentions. nil means the help could not be read (keep every flag).
func knownFlags(bin string, env []string) map[string]bool {
	if v, ok := helpCache.Load(bin); ok {
		m, _ := v.(map[string]bool)
		return m
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "--help")
	cmd.Dir = filepath.Dir(bin)
	cmd.Env = llamaEnv(bin, env)
	out, _ := cmd.CombinedOutput()
	var m map[string]bool
	if len(out) > 0 {
		m = map[string]bool{}
		for _, mm := range flagRe.FindAllStringSubmatch(string(out), -1) {
			m[mm[1]] = true
		}
		if len(m) < 5 {
			m = nil
		}
	}
	helpCache.Store(bin, m)
	return m
}

func has(known map[string]bool, names ...string) bool {
	if known == nil {
		return true
	}
	for _, n := range names {
		if known[n] {
			return true
		}
	}
	return false
}

// group turns a flat arg list into flag groups ("-c","4096" stays together).
func group(args []string) [][]string {
	var out [][]string
	for i := 0; i < len(args); i++ {
		g := []string{args[i]}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			g = append(g, args[i+1])
			i++
		}
		out = append(out, g)
	}
	return out
}

// BuildArgs is the llama-server command line (after the binary) for a shape.
// Flags the pinned release does not know are dropped and reported.
func BuildArgs(model string, shape fit.Shape, host string, port int, alias string, known map[string]bool) (args, dropped []string) {
	args = []string{"-m", model}
	required := map[string]bool{"-ngl": true, "-np": true, "-c": true}
	for _, g := range group(fit.Args(shape)) {
		if required[g[0]] || has(known, g[0], longName(g[0])) {
			args = append(args, g...)
		} else {
			dropped = append(dropped, g[0])
		}
	}
	fixed := [][]string{
		{"--kv-unified", "off"}, {"-fa", "on"}, {"--jinja"}, {"--cache-reuse", "256"},
		{"--no-webui"}, {"--metrics"}, {"--slots"},
	}
	for _, g := range fixed {
		switch g[0] {
		case "--kv-unified":
			if !has(known, "--kv-unified") && has(known, "--no-kv-unified") {
				args = append(args, "--no-kv-unified")
			} else if has(known, "--kv-unified") {
				args = append(args, g...)
			} else {
				dropped = append(dropped, g[0])
			}
		case "-fa":
			if has(known, "-fa", "--flash-attn") {
				args = append(args, g...)
			} else {
				dropped = append(dropped, g[0])
			}
		default:
			if has(known, g[0]) {
				args = append(args, g...)
			} else {
				dropped = append(dropped, g[0])
			}
		}
	}
	args = append(args, "--host", host, "--port", strconv.Itoa(port))
	if has(known, "-a", "--alias") {
		args = append(args, "-a", alias)
	} else {
		dropped = append(dropped, "-a")
	}
	return
}

func longName(f string) string {
	switch f {
	case "-ctk":
		return "--cache-type-k"
	case "-ctv":
		return "--cache-type-v"
	}
	return f
}

var (
	reBuf   = regexp.MustCompile(`failed to allocate\s+(?:\S+\s+)?buffer of size\s+(\d+)`)
	reAlloc = regexp.MustCompile(`allocating\s+([\d.]+)\s+MiB on device \d+`)
	reOOM   = regexp.MustCompile(`(?i)out of memory|cudaMalloc failed|failed to allocate|unable to allocate|ErrorOutOfDeviceMemory`)
)

// ParseOOM looks for a memory failure in llama-server's output. failedMiB is the
// size of the last buffer it could not allocate, 0 when none was printed.
func ParseOOM(text string) (oom bool, failedMiB int) {
	if !reOOM.MatchString(text) {
		return false, 0
	}
	if m := reBuf.FindAllStringSubmatch(text, -1); len(m) > 0 {
		b, _ := strconv.ParseFloat(m[len(m)-1][1], 64)
		return true, int(math.Ceil(b / (1 << 20)))
	}
	if m := reAlloc.FindAllStringSubmatch(text, -1); len(m) > 0 {
		f, _ := strconv.ParseFloat(m[len(m)-1][1], 64)
		return true, int(math.Ceil(f))
	}
	return true, 0
}
