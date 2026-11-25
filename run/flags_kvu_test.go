package run

import (
	"strings"
	"testing"

	"hermesengine/fit"
)

// The real llama-server help lists "-kvu,  --kv-unified, -no-kvu, --no-kv-unified" (bare switches).
// Independent slots need the bare --no-kv-unified; "--kv-unified off" would turn unified KV on.
func TestNoKVUnifiedIsBareSwitch(t *testing.T) {
	known := map[string]bool{"-m": true, "-ngl": true, "-np": true, "-c": true, "-ctk": true, "-ctv": true,
		"--kv-unified": true, "-kvu": true, "--no-kv-unified": true, "-no-kvu": true, "-fa": true, "--flash-attn": true,
		"--jinja": true, "--cache-reuse": true, "--no-webui": true, "--metrics": true, "--slots": true,
		"--host": true, "--port": true, "-a": true, "--alias": true, "--n-cpu-moe": true}
	args, _ := BuildArgs("m.gguf", fit.Shape{Slots: 3, CtxPerSlot: 65536, KVType: "q4_0"}, "127.0.0.1", 1234, "x", known)
	line := " " + strings.Join(args, " ") + " "
	if !strings.Contains(line, " --no-kv-unified ") {
		t.Fatalf("want --no-kv-unified in %q", line)
	}
	if strings.Contains(line, " --kv-unified ") || strings.Contains(line, " off ") {
		t.Fatalf("unified KV or stray 'off' in %q", line)
	}
}
