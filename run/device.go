package run

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"hermesengine/plan"
	"hermesengine/probe/gpusense"
)

var listCache sync.Map // bin + "\x00" + extra env -> []plan.ListedDevice

// listDevicesTimeout is smokeTest's: a first CUDA run on Windows may wait on the antivirus.
var listDevicesTimeout = smokeDevicesTimeout

// listDevices is llama-server's --list-devices under the extra env, read once per binary and env
// (the devices a build sees do not change while it is installed). ok is false when it could not
// be run (not cached: the next start tries again); the list is then empty, not nil.
func listDevices(bin string, extra []string) (ds []plan.ListedDevice, ok bool) {
	key := bin + "\x00" + strings.Join(extra, "\x00")
	if v, ok := listCache.Load(key); ok {
		return v.([]plan.ListedDevice), true
	}
	ctx, cancel := context.WithTimeout(context.Background(), listDevicesTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "--list-devices")
	cmd.Dir = filepath.Dir(bin)
	cmd.Env = llamaEnv(bin, extra)
	out, err := cmd.CombinedOutput()
	ds = plan.ParseListDevices(string(out))
	if ds == nil {
		ds = []plan.ListedDevice{}
	}
	if err != nil {
		return ds, false
	}
	listCache.Store(key, ds)
	return ds, true
}

// resolvePin holds llama-server to the adapter the plan picked (plan.PinDevice), checked against
// the binary itself: a CUDA pin by UUID must leave exactly one CUDA device (where --list-devices
// answers at all; where it does not, the UUID stands), a --device pin needs a
// binary that knows --device. Whatever cannot be confirmed falls back to llama-server's own
// default devices, with the reason in the note; a start never fails for it. extra is the env the
// engine adds (tests); a CUDA_VISIBLE_DEVICES already in the engine's environment is the user's
// and is left as it is.
func resolvePin(bin, accel string, sum gpusense.GPUSummary, extra []string, known map[string]bool) plan.DevicePin {
	gpus := 0
	for _, a := range sum.Adapters {
		if !a.Unified {
			gpus++
		}
	}
	var listed []plan.ListedDevice
	if (accel == "cuda" || accel == "vulkan") && gpus > 1 && sum.EngineAdapter >= 0 {
		listed, _ = listDevices(bin, extra)
	}
	var prefix []string
	if accel == "cuda" {
		if cvd, set := lookupEnv("CUDA_VISIBLE_DEVICES", extra); set {
			sum = withoutUUID(sum)
			prefix = append(prefix, fmt.Sprintf("CUDA_VISIBLE_DEVICES=%q is set in the engine's environment, left as given", cvd))
		}
	}
	p := plan.PinDevice(sum, accel, listed)
	if p.Pinned && p.Via == "uuid" {
		var cuda []plan.ListedDevice
		ds, ran := listDevices(bin, append(append([]string{}, extra...), p.Env...))
		for _, d := range ds {
			if strings.HasPrefix(d.Name, "CUDA") {
				cuda = append(cuda, d)
			}
		}
		switch {
		case !ran: // nothing said against nvidia-smi's UUID: the pin stands
			p.Note += " (not confirmed: llama-server --list-devices did not answer)"
		case len(cuda) == 1:
			p.Device = cuda[0].Name
		default:
			prefix = append(prefix, fmt.Sprintf("%s left llama-server %d CUDA devices", p.Env[0], len(cuda)))
			p = plan.PinDevice(withoutUUID(sum), accel, listed)
		}
	}
	if p.Pinned && p.Via == "name" && !has(known, "--device", "-dev") {
		p = plan.DevicePin{Adapter: p.Adapter, Note: fmt.Sprintf("%s, but this llama-server has no --device: its default devices", p.Note)}
	}
	if p.Pinned && !has(known, "--split-mode", "-sm") {
		p.Args = dropFlag(p.Args, "--split-mode")
	}
	if len(prefix) > 0 {
		p.Note = strings.Join(append(prefix, p.Note), "; ")
	}
	return p
}

func withoutUUID(sum gpusense.GPUSummary) gpusense.GPUSummary {
	as := append([]gpusense.Adapter{}, sum.Adapters...)
	for i := range as {
		as[i].UUID = ""
	}
	sum.Adapters = as
	return sum
}

// lookupEnv finds key in extra (last wins) or else the process environment.
func lookupEnv(key string, extra []string) (string, bool) {
	for i := len(extra) - 1; i >= 0; i-- {
		if v, ok := strings.CutPrefix(extra[i], key+"="); ok {
			return v, v != ""
		}
	}
	v, ok := os.LookupEnv(key)
	return v, ok && v != ""
}

func dropFlag(args []string, flag string) []string {
	var out []string
	for i := 0; i < len(args); i++ {
		if args[i] == flag {
			i++ // its value
			continue
		}
		out = append(out, args[i])
	}
	return out
}
