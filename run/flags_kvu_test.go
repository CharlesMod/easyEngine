package run

import (
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
)

// The real llama-server help lists "-kvu,  --kv-unified, -no-kvu, --no-kv-unified" (bare switches).
// Independent slots need the bare --no-kv-unified; "--kv-unified off" would turn unified KV on.
func TestNoKVUnifiedIsBareSwitch(t *testing.T) {
	known := map[string]bool{"-m": true, "-ngl": true, "-np": true, "-c": true, "-ctk": true, "-ctv": true,
		"--kv-unified": true, "-kvu": true, "--no-kv-unified": true, "-no-kvu": true, "-fa": true, "--flash-attn": true,
		"--jinja": true, "--cache-reuse": true, "--no-webui": true, "--metrics": true, "--slots": true,
		"--host": true, "--port": true, "-a": true, "--alias": true, "--n-cpu-moe": true}
	args, _ := BuildArgs("m.gguf", fit.Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}, "127.0.0.1", 1234, "x", known, nil)
	line := " " + strings.Join(args, " ") + " "
	if !strings.Contains(line, " --no-kv-unified ") {
		t.Fatalf("want --no-kv-unified in %q", line)
	}
	if strings.Contains(line, " --kv-unified ") || strings.Contains(line, " off ") {
		t.Fatalf("unified KV or stray 'off' in %q", line)
	}
}

func TestBuildArgsSamplerDefaults(t *testing.T) {
	known := map[string]bool{"-m": true, "-ngl": true, "-np": true, "-c": true, "--host": true, "--port": true,
		"--temp": true, "--top-k": true, "--top-p": true, "--min-p": true, "--repeat-penalty": true} // no --dry-multiplier
	f := func(v float64) *float64 { return &v }
	k := 64
	off := catalog.SamplerProfile{Temperature: f(1), TopK: &k, TopP: f(0.95), MinP: f(0), RepeatPenalty: f(1), DryMultiplier: f(0)}
	args, dropped := BuildArgs("m.gguf", fit.Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0"}, "127.0.0.1", 1, "x", known, &off)
	line := " " + strings.Join(args, " ") + " "
	for _, w := range []string{" --temp 1 ", " --top-k 64 ", " --top-p 0.95 ", " --min-p 0 ", " --repeat-penalty 1 "} {
		if !strings.Contains(line, w) {
			t.Errorf("lacks%s: %s", w, line)
		}
	}
	if strings.Contains(line, "--dry-multiplier") || !strings.Contains(strings.Join(dropped, " "), "--dry-multiplier") {
		t.Errorf("a flag the release does not know must be dropped and reported: %s / %v", line, dropped)
	}
	if args2, _ := BuildArgs("m.gguf", fit.Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0"}, "127.0.0.1", 1, "x", known, nil); strings.Contains(strings.Join(args2, " "), "--temp") {
		t.Error("no profile, no sampler flags")
	}
}

// The web UI is on: the engine serves it at /chat/ (api/chat.go), so --no-webui is never passed,
// even when the binary lists it.
func TestWebUIStaysOn(t *testing.T) {
	known := map[string]bool{"-m": true, "-ngl": true, "-np": true, "-c": true, "--host": true, "--port": true,
		"--no-webui": true, "--webui": true, "--no-ui": true, "--ui": true, "--metrics": true, "--slots": true}
	args, dropped := BuildArgs("m.gguf", fit.Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0"}, "127.0.0.1", 1, "x", known, nil)
	line := " " + strings.Join(args, " ") + " "
	for _, bad := range []string{"--no-webui", "--no-ui"} {
		if strings.Contains(line, bad) || strings.Contains(strings.Join(dropped, " "), bad) {
			t.Errorf("%s must not be passed or reported: %s / %v", bad, line, dropped)
		}
	}
	// The drop-unknown behaviour is unchanged for the flags that remain.
	delete(known, "--metrics")
	_, dropped = BuildArgs("m.gguf", fit.Shape{Slots: 1, CtxPerSlot: 4096, KVType: "q4_0"}, "127.0.0.1", 1, "x", known, nil)
	if !strings.Contains(strings.Join(dropped, " "), "--metrics") {
		t.Errorf("--metrics not listed by the binary must still be dropped: %v", dropped)
	}
}
