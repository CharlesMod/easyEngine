package run

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Help is what one llama-server build's --help says about its flags.
type Help struct {
	Flags       map[string]bool     // every flag name the help mentions
	Groups      map[string][]string // spelling -> all spellings on its help line
	Placeholder map[string]string   // spelling -> the line's argument placeholder ("" if none)
}

var flagWordRe = regexp.MustCompile(`^--?[A-Za-z][A-Za-z0-9_-]*$`)

// ParseHelp reads a llama-server --help text. Flags is everything flagRe finds
// (what knownFlags always returned); Groups and Placeholder come from the flag
// lines. nil means too few flags to trust the text.
func ParseHelp(text string) *Help {
	h := &Help{Flags: map[string]bool{}, Groups: map[string][]string{}, Placeholder: map[string]string{}}
	for _, m := range flagRe.FindAllStringSubmatch(text, -1) {
		h.Flags[m[1]] = true
	}
	if len(h.Flags) < 5 {
		return nil
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, "\r \t")
		if !strings.HasPrefix(line, "-") || strings.HasPrefix(line, "----") {
			continue
		}
		names, ph := splitFlagLine(line)
		if len(names) == 0 {
			continue
		}
		for _, n := range names {
			h.Groups[n] = names
			h.Placeholder[n] = ph
			h.Flags[n] = true
		}
	}
	return h
}

// splitFlagLine takes the spellings off the front of a help line (words joined
// by commas), then the placeholder up to the description, which starts after a
// run of 2+ spaces. Spaces after a comma belong to the list, not the description.
func splitFlagLine(line string) (names []string, placeholder string) {
	i := 0
	for i < len(line) {
		j := i
		for j < len(line) && line[j] != ' ' && line[j] != ',' {
			j++
		}
		w := line[i:j]
		if !flagWordRe.MatchString(w) {
			break
		}
		names = append(names, w)
		if j < len(line) && line[j] == ',' {
			j++
			for j < len(line) && line[j] == ' ' {
				j++
			}
			i = j
			continue
		}
		i = j
		break
	}
	rest := line[i:]
	if strings.HasPrefix(rest, "  ") {
		return names, "" // the description starts right after the flags
	}
	rest = strings.TrimLeft(rest, " ")
	if k := strings.Index(rest, "  "); k >= 0 {
		rest = rest[:k]
	}
	return names, strings.TrimSpace(rest)
}

// Has reports whether the build knows any of the names, or any other spelling
// on the same help line. A nil Help knows nothing.
func (h *Help) Has(names ...string) bool {
	if h == nil {
		return false
	}
	for _, n := range names {
		if h.Flags[n] {
			return true
		}
		for _, s := range h.Groups[n] {
			if h.Flags[s] {
				return true
			}
		}
	}
	return false
}

// ownedFlagNames are the flags the engine sets itself (model, ports, memory
// shape, logging, tools, ...). A client's extra_args must never carry them in
// any spelling, or it could move the server, read other files or fight the fit.
var ownedFlagNames = []string{
	"-m", "-mu", "-hf", "-hff", "-hft", "-dr", "--ple", "-hfd", "-md", "-mm", "-mmu", "-c", "-np", "-ngl",
	"-ncmoe", "-cmoe", "-ncffn", "-ot", "--moe-cache-mib", "--moe-cache-cpu-share", "-ctk", "-ctv", "-fa",
	"-kvu", "-no-kvu", "--kv-unified-per-slot", "--swa-full", "-nkvo", "-kvo", "--no-host", "-fit", "-fitt",
	"-fitc", "-ub", "-dev", "-sm", "-ts", "-mg", "-devd", "-mmdev", "--rpc", "--host", "--port",
	"--reuse-port", "--path", "--api-prefix", "--api-key", "--api-key-file", "--ssl-key-file",
	"--ssl-cert-file", "--cors-origins", "--cors-methods", "--cors-headers", "--cors-credentials", "-to",
	"--threads-http", "--props", "--metrics", "--slots", "--no-slots", "--slot-save-path",
	"--sleep-idle-seconds", "--offline", "--embedding", "--rerank", "--models-dir", "--models-preset",
	"--models-max", "--models-memory-margin", "--models-autoload", "--tools", "--tools-runtime",
	"--allow-unauthenticated-tools", "--mcp-servers-config", "--mcp-servers-json", "-ag", "--connect",
	"--connect-code", "--ui", "--ui-config", "--ui-config-file", "--ui-mcp-proxy",
	"--ui-mcp-proxy-allow-private-urls", "--log-disable", "--log-file", "--log-jsonl", "--log-colors", "-v",
	"-lv", "--log-prefix", "--log-timestamps", "--log-prompts-dir", "-rea", "--jinja", "--cache-reuse", "-a",
	"-h", "--version", "-cl", "--list-devices", "--completion-bash", "--media-path",
	"--media-allow-private-urls", "--video-ffmpeg-dir", "--spec-default",
	// Long forms the engine knows without a help, so ownership holds with no build installed.
	"--model", "--model-url", "--hf-repo", "--hf-file", "--hf-token", "--docker-repo", "--shard",
	"--hf-repo-draft", "--model-draft", "--spec-draft-model", "--mmproj", "--mmproj-url", "--ctx-size",
	"--parallel", "--gpu-layers", "--n-gpu-layers", "--n-cpu-moe", "--cpu-moe", "--n-cpu-ffn",
	"--override-tensor", "--cache-type-k", "--cache-type-v", "--flash-attn", "--kv-unified",
	"--no-kv-unified", "--no-kv-offload", "--kv-offload", "--fit", "--fit-target", "--fit-ctx",
	"--ubatch-size", "--device", "--split-mode", "--tensor-split", "--main-gpu", "--device-draft",
	"--mmproj-device", "--timeout", "--embeddings", "--reranking", "--agent", "--no-agent", "--webui",
	"--no-ui", "--no-webui", "--webui-config", "--webui-config-file", "--log-verbose", "--verbose",
	"--verbosity", "--log-verbosity", "--reasoning", "--no-jinja", "--alias", "--help", "--usage",
	"--cache-list",
}

var ownedSet = func() map[string]bool {
	m := make(map[string]bool, len(ownedFlagNames))
	for _, n := range ownedFlagNames {
		m[n] = true
	}
	return m
}()

// allowedPathFlags take a local file path but are fine for a local client to
// pass: they read a file the client already has and cannot move the server.
var allowedPathFlags = []string{"--grammar-file", "-jf", "--chat-template-file", "--lora", "--lora-scaled",
	"--control-vector", "--control-vector-scaled", "-lcs", "-lcd", "--tts-lang"}

// normFlag lower-cases long names (llama-server's --Foo is not a thing we want
// to let slip past a check); single-dash names stay case-sensitive (-C is not -c).
func normFlag(name string) string {
	if strings.HasPrefix(name, "--") {
		return strings.ToLower(name)
	}
	return name
}

// ownedNamed is the name-only part of ownedFlag: the owned set and the preset
// patterns (any "...-default" or "--fim-..." shortcut loads a model or server config).
func ownedNamed(n string) bool {
	return ownedSet[n] || strings.HasSuffix(n, "-default") || strings.HasPrefix(n, "--fim-")
}

// ownedFlag reports whether the engine owns the flag. With a help, any spelling
// on the flag's help line counts, so an alias the list missed is still caught.
func ownedFlag(name string, h *Help) bool {
	n := normFlag(name)
	if ownedNamed(n) {
		return true
	}
	if h == nil {
		return false
	}
	for _, s := range h.Groups[n] {
		if ownedNamed(normFlag(s)) {
			return true
		}
	}
	return false
}

// ArgError is a refused extra_args list. Code is "bad_request", "flag_owned" or
// "flag_unknown"; Flag is the offending flag as given; Msg says why.
type ArgError struct{ Code, Flag, Msg string }

func (e *ArgError) Error() string { return e.Code + ": " + e.Msg }

const (
	maxExtraArgs     = 32
	maxExtraArgBytes = 256
)

// CheckExtraArgs vets a client's extra llama-server arguments. It bounds the
// list, refuses owned flags in every spelling (including --x=v and any case of
// a long name), and, when the build's help is known, refuses flags it does not
// have. It returns the flag names seen, normalised. With h == nil only the
// bounds and the owned check run.
func CheckExtraArgs(args []string, h *Help) (flags []string, err *ArgError) {
	if len(args) > maxExtraArgs {
		return nil, &ArgError{"bad_request", "", fmt.Sprintf("extra_args has %d tokens, at most %d", len(args), maxExtraArgs)}
	}
	for i, t := range args {
		if len(t) < 1 || len(t) > maxExtraArgBytes {
			return nil, &ArgError{"bad_request", "", fmt.Sprintf("extra_args[%d] must be 1 to %d bytes", i, maxExtraArgBytes)}
		}
		if !utf8.ValidString(t) {
			return nil, &ArgError{"bad_request", "", fmt.Sprintf("extra_args[%d] is not valid UTF-8", i)}
		}
		for _, r := range t {
			if unicode.IsControl(r) {
				return nil, &ArgError{"bad_request", "", fmt.Sprintf("extra_args[%d] has a control character", i)}
			}
		}
	}
	for _, t := range args {
		if !strings.HasPrefix(t, "-") {
			continue
		}
		if _, e := strconv.ParseFloat(t, 64); e == nil {
			continue // a negative number is a value
		}
		name := t
		if k := strings.Index(t, "="); k >= 0 {
			name = t[:k]
		}
		n := normFlag(name)
		if ownedFlag(name, h) {
			return nil, &ArgError{"flag_owned", name, name + " is set by the engine and cannot be passed"}
		}
		if h != nil && !h.Flags[n] && !h.Flags[name] {
			return nil, &ArgError{"flag_unknown", name, name + " is not a flag of this llama-server build"}
		}
		flags = append(flags, n)
	}
	return flags, nil
}
