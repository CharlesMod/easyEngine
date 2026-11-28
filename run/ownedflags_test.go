package run

import (
	"os"
	"strings"
	"testing"
)

func fixtureHelp(t *testing.T) (string, *Help) {
	t.Helper()
	b, err := os.ReadFile("testdata/help-b12651.txt")
	if err != nil {
		t.Fatal(err)
	}
	h := ParseHelp(string(b))
	if h == nil {
		t.Fatal("fixture help did not parse")
	}
	return string(b), h
}

func contains(l []string, s string) bool {
	for _, x := range l {
		if x == s {
			return true
		}
	}
	return false
}

func TestParseHelpGroupsAndPlaceholders(t *testing.T) {
	_, h := fixtureHelp(t)
	if g := h.Groups["-kvu"]; len(g) != 4 {
		t.Fatalf("-kvu group = %v", g)
	}
	g := h.Groups["--spec-draft-model"]
	if !contains(g, "-md") || !contains(g, "--model-draft") {
		t.Fatalf("--spec-draft-model group = %v", g)
	}
	for flag, want := range map[string]string{
		"--mmproj": "FILE", "--host": "HOST", "-fa": "[on|off|auto]",
		"--ui-config-file": "PATH", "--lora-scaled": "FNAME:SCALE,...", "-kvu": "",
	} {
		if got := h.Placeholder[flag]; got != want {
			t.Errorf("placeholder of %s = %q, want %q", flag, got, want)
		}
	}
}

func TestKnownFlagsSetUnchanged(t *testing.T) {
	text, h := fixtureHelp(t)
	want := map[string]bool{}
	for _, m := range flagRe.FindAllStringSubmatch(text, -1) {
		want[m[1]] = true
	}
	for k := range want {
		if !h.Flags[k] {
			t.Errorf("missing %s", k)
		}
	}
	for k := range h.Flags {
		if !want[k] {
			t.Errorf("extra %s", k)
		}
	}
	if ParseHelp("--a --b") != nil {
		t.Error("tiny help should give nil")
	}
}

func TestFixtureBoundary(t *testing.T) {
	_, h := fixtureHelp(t)
	allowed := map[string]bool{}
	for _, n := range allowedPathFlags {
		allowed[n] = true
		for _, s := range h.Groups[n] {
			allowed[s] = true
		}
	}
	seen := map[string]bool{}
	for name, group := range h.Groups {
		key := strings.Join(group, ",")
		if seen[key] {
			continue
		}
		seen[key] = true
		ph := strings.ToUpper(h.Placeholder[name])
		risky := false
		for _, w := range []string{"FNAME", "PATH", "FILE", "DIR"} {
			if strings.Contains(ph, w) {
				risky = true
			}
		}
		low := strings.ToLower(strings.Join(group, " ") + " " + h.Placeholder[name])
		for _, w := range []string{"host", "port", "url"} {
			if strings.Contains(low, w) {
				risky = true
			}
		}
		if !risky {
			continue
		}
		for _, s := range group {
			if !ownedFlag(s, h) && !allowed[s] {
				t.Errorf("help line %q: %s is neither owned nor an allowed path flag", key+" "+h.Placeholder[name], s)
			}
		}
	}
}

func TestCheckExtraArgsOwnedEverySpelling(t *testing.T) {
	_, h := fixtureHelp(t)
	n := 0
	for name, group := range h.Groups {
		if !ownedFlag(name, h) {
			continue
		}
		for _, s := range group {
			n++
			for _, args := range [][]string{{s, "v"}, {s + "=v"}} {
				for _, hh := range []*Help{h, nil} {
					if hh == nil && !ownedNamed(normFlag(s)) {
						continue // only the help knows this alias
					}
					_, err := CheckExtraArgs(args, hh)
					if err == nil || err.Code != "flag_owned" {
						t.Errorf("%v: err = %v", args, err)
					}
				}
			}
		}
	}
	if n < 50 {
		t.Fatalf("only %d owned spellings checked", n)
	}
}

func TestCheckExtraArgsCases(t *testing.T) {
	_, h := fixtureHelp(t)
	for _, a := range [][]string{
		{"--slot-save-path", "/tmp/x"}, {"--slot-save-path=/x"}, {"--SLOT-SAVE-PATH", "x"}, {"--Slot-Save-Path=x"}, {"-c", "4096"},
		{"--gpt-oss-20b-default"}, {"--fim-qwen-7b-default"},
	} {
		for _, hh := range []*Help{h, nil} {
			if _, err := CheckExtraArgs(a, hh); err == nil || err.Code != "flag_owned" {
				t.Errorf("%v: err = %v", a, err)
			}
		}
	}
	if _, err := CheckExtraArgs([]string{"--no-such-flag"}, h); err == nil || err.Code != "flag_unknown" {
		t.Errorf("unknown: %v", err)
	}
	if _, err := CheckExtraArgs([]string{"--no-such-flag"}, nil); err != nil {
		t.Errorf("nil help: %v", err)
	}
	ok := [][]string{
		{"--top-n-sigma", "1.5"},
		{"--spec-type", "ngram-mod", "--spec-ngram-mod-n-max", "16"},
		{"-C", "0xff"},
		{"--temp", "-0.5"},
		{"--lora", "a.gguf"},
	}
	for _, a := range ok {
		if _, err := CheckExtraArgs(a, h); err != nil {
			t.Errorf("%v: %v", a, err)
		}
	}
	fl, _ := CheckExtraArgs([]string{"--TEMP", "-0.5", "-C", "x"}, h)
	if len(fl) != 2 || fl[0] != "--temp" || fl[1] != "-C" {
		t.Errorf("flags = %v", fl)
	}
	bad := [][]string{
		make([]string, 33),
		{"--x", strings.Repeat("a", 257)},
		{"a\nb"},
		{""},
		{"\xff"},
	}
	for i := range bad[0] {
		bad[0][i] = "v"
	}
	for _, a := range bad {
		if _, err := CheckExtraArgs(a, h); err == nil || err.Code != "bad_request" {
			t.Errorf("%.20v: err = %v", a, err)
		}
	}
}

// Every projector and media flag of the recorded help is the engine's: a client's extra_args never
// carries one (flag_owned), and the parent environment's spellings of them never reach the child.
func TestMediaFlagsAreOwned(t *testing.T) {
	_, h := fixtureHelp(t)
	flags := []string{"-mm", "--mmproj", "-mmu", "--mmproj-url", "--mmproj-auto", "--no-mmproj", "--no-mmproj-auto",
		"--mmproj-offload", "--no-mmproj-offload", "-mmdev", "--mmproj-device", "--image-min-tokens",
		"--image-max-tokens", "--mtmd-batch-max-tokens", "--video-fps", "--video-timestamp-interval",
		"--video-ffmpeg-dir", "--media-path", "--media-allow-private-urls", "--vision-gemma-4b-default",
		"--vision-gemma-12b-default", "--image", "--audio", "--video"}
	for _, f := range flags {
		if !h.Has(f) {
			continue // not in this help (the server's help leaves --image/--audio/--video out): owned by name below
		}
		if _, err := CheckExtraArgs([]string{f}, h); err == nil || err.Code != "flag_owned" {
			t.Errorf("%s with the help: %v", f, err)
		}
	}
	for _, f := range flags { // and with no help at all, and as --flag=value
		if _, err := CheckExtraArgs([]string{f + "=x"}, nil); err == nil || err.Code != "flag_owned" {
			t.Errorf("%s without a help: %v", f, err)
		}
	}
}

func TestLlamaEnvDropsMediaVariables(t *testing.T) {
	for _, k := range mediaEnv {
		t.Setenv(k, "x")
	}
	t.Setenv("LLAMA_ARG_CTX_SIZE", "x")
	t.Setenv("LLAMA_ARG_HF_REPO", "x")
	t.Setenv("LLAMA_ARG_MODEL_URL", "x")
	t.Setenv("LLAMA_ARG_DOCKER_REPO", "x")
	t.Setenv("UNRELATED_VAR", "keep")
	env := llamaEnv("/bin/llama-server", []string{"LLAMA_ARG_MMPROJ=engine"})
	got := map[string]string{}
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		got[k] = v
	}
	for _, k := range append([]string{"LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_HF_REPO", "LLAMA_ARG_MODEL_URL", "LLAMA_ARG_DOCKER_REPO"}, mediaEnv...) {
		if k == "LLAMA_ARG_MMPROJ" {
			continue
		}
		if _, ok := got[k]; ok {
			t.Errorf("%s reached the child", k)
		}
	}
	if got["LLAMA_ARG_MMPROJ"] != "engine" || got["UNRELATED_VAR"] != "keep" {
		t.Errorf("the engine's own extra or an unrelated variable was lost: %v", got["LLAMA_ARG_MMPROJ"])
	}
}
