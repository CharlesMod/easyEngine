//go:build !windows

package run

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// fakeLister is a llama-server that only answers --list-devices: all listed devices unpinned,
// pinned (CUDA_VISIBLE_DEVICES set) the given text.
func fakeLister(t *testing.T, all, pinned string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "llama-server")
	sh := "#!/bin/sh\necho 'Available devices:'\nif [ -n \"$CUDA_VISIBLE_DEVICES\" ]; then printf '" + pinned + "'; else printf '" + all + "'; fi\n"
	if err := os.WriteFile(bin, []byte(sh), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

func twoP40s(engine int) gpusense.GPUSummary {
	return gpusense.GPUSummary{EngineAdapter: engine, Adapters: []gpusense.Adapter{
		{Index: 0, Vendor: "nvidia", Name: "Tesla P40", DedicatedMiB: 24576, UUID: "GPU-90bd"},
		{Index: 1, Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3090", DedicatedMiB: 24576, UUID: "GPU-cca6"},
	}}
}

const twoListed = `  CUDA0: NVIDIA GeForce RTX 3090 (24135 MiB, 23000 MiB free)\n  CUDA1: Tesla P40 (24438 MiB, 23209 MiB free)\n`

var allKnown = map[string]bool{"--device": true, "--split-mode": true}

func TestResolvePinUUIDConfirmed(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	bin := fakeLister(t, twoListed, `  CUDA0: Tesla P40 (24438 MiB, 23209 MiB free)\n`)
	p := resolvePin(bin, "cuda", twoP40s(0), nil, allKnown)
	if !p.Pinned || p.Via != "uuid" || p.Device != "CUDA0" || !reflect.DeepEqual(p.Env, []string{"CUDA_VISIBLE_DEVICES=GPU-90bd"}) ||
		!reflect.DeepEqual(p.Args, []string{"--split-mode", "none"}) {
		t.Fatalf("%+v", p)
	}
	// a release without --split-mode: still held by CUDA_VISIBLE_DEVICES, the flag dropped
	if p := resolvePin(bin, "cuda", twoP40s(0), nil, map[string]bool{"--device": true}); !p.Pinned || len(p.Args) != 0 {
		t.Fatalf("%+v", p)
	}
}

// The UUID leaves no CUDA device (a driver that will not take it): matched by name instead.
func TestResolvePinUUIDRefusedFallsBackToName(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	bin := fakeLister(t, twoListed, ``)
	p := resolvePin(bin, "cuda", twoP40s(0), nil, allKnown)
	if !p.Pinned || p.Via != "name" || p.Device != "CUDA1" || p.Env != nil || !strings.Contains(p.Note, "left llama-server 0 CUDA devices") {
		t.Fatalf("%+v", p)
	}
	// and with no --device in this llama-server: its default, said so; the start goes on
	p = resolvePin(fakeLister(t, twoListed, ``), "cuda", twoP40s(0), nil, map[string]bool{"--split-mode": true})
	if p.Pinned || p.Args != nil || p.Env != nil || !strings.Contains(p.Note, "no --device") {
		t.Fatalf("%+v", p)
	}
}

// A CUDA_VISIBLE_DEVICES the user set is theirs: kept, and the card found in what it leaves.
func TestResolvePinKeepsUsersCUDAVisibleDevices(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "1")
	bin := fakeLister(t, twoListed, `  CUDA0: Tesla P40 (24438 MiB, 23209 MiB free)\n`)
	p := resolvePin(bin, "cuda", twoP40s(0), nil, allKnown)
	if p.Env != nil || !strings.Contains(p.Note, `CUDA_VISIBLE_DEVICES="1" is set`) || p.Device != "CUDA0" || p.Via != "name" {
		t.Fatalf("%+v", p)
	}
}

func TestResolvePinOneCardListsNothing(t *testing.T) {
	sum := gpusense.GPUSummary{EngineAdapter: 0, Adapters: []gpusense.Adapter{{Index: 0, Vendor: "nvidia", Name: "Tesla P40", DedicatedMiB: 24576, UUID: "GPU-1"}}}
	// a binary that cannot run: one card never needs it
	p := resolvePin(filepath.Join(t.TempDir(), "missing"), "cuda", sum, nil, allKnown)
	if p.Pinned || p.Args != nil || !strings.Contains(p.Note, "one adapter") {
		t.Fatalf("%+v", p)
	}
}

// --list-devices that does not answer says nothing against the UUID: the pin stands.
func TestResolvePinUnconfirmedUUIDStands(t *testing.T) {
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	bin := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 3\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	p := resolvePin(bin, "cuda", twoP40s(1), nil, allKnown)
	if !p.Pinned || p.Env[0] != "CUDA_VISIBLE_DEVICES=GPU-cca6" || !strings.Contains(p.Note, "not confirmed") {
		t.Fatalf("%+v", p)
	}
}
