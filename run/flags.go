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

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
)

var helpCache sync.Map // bin path -> *Help (a help that could not be read is not kept: read again next time)

var flagRe = regexp.MustCompile(`(?:^|[\s,\[|(])(--?[A-Za-z][A-Za-z0-9_-]*)`)

// helpOf runs `llama-server --help` once per binary and parses it. nil means the
// help could not be read (keep every flag); that is not kept, so the next launch reads it again.
func helpOf(bin string, env []string) *Help {
	if v, ok := helpCache.Load(bin); ok {
		h, _ := v.(*Help)
		return h
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "--help")
	cmd.Dir = filepath.Dir(bin)
	cmd.Env = llamaEnv(bin, env)
	out, _ := cmd.CombinedOutput()
	var h *Help
	if len(out) > 0 {
		h = ParseHelp(string(out))
	}
	if h != nil {
		helpCache.Store(bin, h)
	}
	return h
}

// knownFlags returns the flag names the binary's help mentions. nil means the
// help could not be read (keep every flag).
func knownFlags(bin string, env []string) map[string]bool {
	h := helpOf(bin, env)
	if h == nil {
		return nil
	}
	return h.Flags
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
// official, when set, becomes llama-server's own sampling defaults (--temp, --top-k, ...), so a
// client that sends no sampler fields gets the model's vendor values.
func BuildArgs(model string, shape fit.Shape, host string, port int, alias string, known map[string]bool, official *catalog.SamplerProfile) (args, dropped []string) {
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
		{"--no-kv-unified"}, {"-fa", "on"}, {"--jinja"}, {"--reasoning", "off"}, {"--cache-reuse", "256"},
		{"--metrics"}, {"--slots"},
	}
	for _, g := range fixed {
		switch g[0] {
		case "--no-kv-unified":
			// Independent slots: each gets exactly ctx_per_slot. llama-server's switch is the bare
			// --no-kv-unified (-no-kvu); "--kv-unified off" would turn unified KV ON and leave a stray arg.
			if has(known, "--no-kv-unified", "-no-kvu") {
				args = append(args, "--no-kv-unified")
			} else {
				dropped = append(dropped, g[0])
			}
		case "--reasoning":
			// Game lines are short: thinking would eat the whole token budget and leave an empty reply.
			// A client that wants it can still ask per request.
			if has(known, "--reasoning", "-rea") {
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
	if official != nil {
		sa, sd := SamplerArgs(*official, known)
		args = append(args, sa...)
		dropped = append(dropped, sd...)
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
	if strings.Contains(text, "pinned memory") { // a pinned host buffer that fell back to pageable is not a failure
		var keep []string
		for _, l := range strings.Split(text, "\n") {
			if !strings.Contains(l, "pinned memory") {
				keep = append(keep, l)
			}
		}
		text = strings.Join(keep, "\n")
	}
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
