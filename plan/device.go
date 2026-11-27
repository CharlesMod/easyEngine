package plan

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/probe/gpusense"
)

// ListedDevice is one line of `llama-server --list-devices`: "  CUDA0: Tesla P40 (24438 MiB, 24208 MiB free)".
type ListedDevice struct {
	Name     string `json:"name"` // llama.cpp's device name: CUDA0, Vulkan1, Metal ...
	Desc     string `json:"desc"` // the card's name as the backend reports it
	TotalMiB int    `json:"total_mib"`
	FreeMiB  int    `json:"free_mib"`
}

var listedRe = regexp.MustCompile(`^\s*([A-Za-z][A-Za-z_]*\d+)\s*:\s*(.*?)\s*(?:\((\d+)\s*MiB,\s*(\d+)\s*MiB free\))?\s*$`)

// ParseListDevices reads `llama-server --list-devices` output; lines that name no device are skipped.
func ParseListDevices(out string) []ListedDevice {
	var ds []ListedDevice
	for _, l := range strings.Split(out, "\n") {
		m := listedRe.FindStringSubmatch(strings.TrimRight(l, "\r"))
		if m == nil || m[2] == "" {
			continue
		}
		t, _ := strconv.Atoi(m[3])
		f, _ := strconv.Atoi(m[4])
		ds = append(ds, ListedDevice{Name: m[1], Desc: m[2], TotalMiB: t, FreeMiB: f})
	}
	return ds
}

// DevicePin is what llama-server is told about the adapter the plan picked
// (GPUSummary.EngineAdapter), so a model planned for one card never spreads to another.
type DevicePin struct {
	Adapter int    `json:"adapter"`          // Adapter.Index planned, -1 none (CPU)
	Pinned  bool   `json:"pinned"`           // llama-server is held to that one card
	Device  string `json:"device,omitempty"` // llama.cpp's name for it once held there: CUDA0, Vulkan1
	// Env and Args are what is added to llama-server's environment and command line.
	Env  []string `json:"env,omitempty"`
	Args []string `json:"args,omitempty"`
	// Via is how the card was matched: "uuid" (CUDA_VISIBLE_DEVICES=<GPU UUID>), "name" (the
	// backend's own name for it in --list-devices), or "" (not pinned).
	Via  string `json:"via,omitempty"`
	Note string `json:"note"` // what was given, or why nothing was (llama-server's own default then)
}

// AccelOf is the accelerator of a llama.cpp build id (catalog ids start with it: cuda-linux-x64,
// cuda13-win-x64, vulkan-win-x64, metal-mac-arm64, cpu-...).
func AccelOf(backend string) string {
	for _, a := range []string{"cuda", "vulkan", "metal", "cpu"} {
		if strings.HasPrefix(backend, a) {
			return a
		}
	}
	return ""
}

// devPrefix is the device name llama.cpp gives a backend's devices.
var devPrefix = map[string]string{"cuda": "CUDA", "vulkan": "Vulkan"}

// PinDevice decides how llama-server is held to the planned adapter, for a build of accel
// ("cuda", "vulkan", "metal", "cpu").
//
//   - CUDA: CUDA_VISIBLE_DEVICES=<the card's UUID from nvidia-smi>. CUDA itself hides every other
//     card, whatever order it would number them in (its default order is fastest first, not
//     nvidia-smi's PCI order), so no index is guessed. With no UUID (Windows without nvidia-smi),
//     the card is matched by name in listed and passed as --device CUDA<n>.
//   - Vulkan: --device Vulkan<n>, matched by name in listed (Vulkan has no UUID in the probe).
//   - Metal, CPU, or one adapter: nothing; llama-server's default is that one device.
//
// A pin also carries --split-mode none, so nothing spills to another card. listed is
// llama-server's --list-devices with no pin (nil when it was not read: the plan, which only
// predicts). Where the card cannot be matched the pin is empty and Note says why: llama-server
// then starts as it always has, never fails for it.
func PinDevice(sum gpusense.GPUSummary, accel string, listed []ListedDevice) DevicePin {
	p := DevicePin{Adapter: sum.EngineAdapter}
	e, ok := adapterByIndex(sum.Adapters, sum.EngineAdapter)
	if !ok {
		p.Adapter = -1
		p.Note = "no GPU planned: llama-server runs on the CPU"
		return p
	}
	gpus := 0
	for _, a := range sum.Adapters {
		if !a.Unified {
			gpus++
		}
	}
	switch {
	case accel == "metal" || e.Unified:
		p.Note = "one unified-memory GPU: llama-server's default device"
		return p
	case accel == "cpu" || accel == "":
		p.Note = "CPU build: no GPU device"
		return p
	case gpus <= 1:
		p.Note = fmt.Sprintf("one adapter (%s): llama-server's default device", e.Name)
		return p
	}
	prefix := devPrefix[accel]
	var devs []ListedDevice
	for _, d := range listed {
		if strings.HasPrefix(d.Name, prefix) {
			devs = append(devs, d)
		}
	}
	if accel == "cuda" {
		if e.Vendor != "nvidia" {
			p.Note = fmt.Sprintf("planned adapter %d (%s) is not an NVIDIA card: CUDA build left on its default", e.Index, e.Name)
			return p
		}
		if e.UUID != "" {
			p.Pinned, p.Via, p.Device = true, "uuid", prefix+"0"
			p.Env = []string{"CUDA_VISIBLE_DEVICES=" + e.UUID}
			p.Args = []string{"--split-mode", "none"}
			p.Note = fmt.Sprintf("adapter %d (%s) only: CUDA_VISIBLE_DEVICES=%s", e.Index, e.Name, e.UUID)
			return p
		}
	}
	if listed == nil {
		p.Note = fmt.Sprintf("adapter %d (%s): matched by name in llama-server --list-devices at launch", e.Index, e.Name)
		return p
	}
	same := 0
	for _, a := range sum.Adapters {
		if a.Vendor == e.Vendor {
			same++
		}
	}
	d, why := matchListed(e, devs, same == 1)
	if d == "" {
		p.Note = fmt.Sprintf("adapter %d (%s) not matched in llama-server's %s devices (%s): llama-server's default devices", e.Index, e.Name, prefix, why)
		return p
	}
	p.Pinned, p.Via, p.Device = true, "name", d
	p.Args = []string{"--device", d, "--split-mode", "none"}
	p.Note = fmt.Sprintf("adapter %d (%s) only: --device %s", e.Index, e.Name, d)
	return p
}

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

func normName(s string) string {
	s = strings.ToLower(s)
	s = strings.NewReplacer("(r)", " ", "(tm)", " ", "®", " ", "™", " ").Replace(s)
	return strings.TrimSpace(nonAlnum.ReplaceAllString(s, " "))
}

var vendorWords = map[string][]string{
	"nvidia": {"nvidia", "geforce", "quadro", "tesla", "rtx"},
	"amd":    {"amd", "radeon"},
	"intel":  {"intel", "iris", "arc", "uhd"},
}

// matchListed finds the adapter's device among the backend's listed ones: the one whose name
// matches exactly, else the one whose name holds the adapter's whole (never the other way: a
// "3060 Ti" is not a "3060"), else (byVendor: the adapter is
// the machine's only one of its vendor) the only listed one of that vendor. Two equally good candidates (two cards of one model) match none: their
// order under Vulkan is the driver's, not the probe's, so it is never guessed.
func matchListed(e gpusense.Adapter, devs []ListedDevice, byVendor bool) (string, string) {
	if len(devs) == 0 {
		return "", "none listed"
	}
	want := normName(e.Name)
	score := func(d ListedDevice) int {
		got := normName(d.Desc)
		switch {
		case want != "" && got == want:
			return 3
		case want != "" && strings.Contains(" "+got+" ", " "+want+" "): // the backend adds to the name: "(RADV NAVI23)"
			return 2
		}
		if !byVendor {
			return 0
		}
		for _, w := range vendorWords[e.Vendor] {
			if strings.Contains(" "+got+" ", " "+w+" ") {
				return 1
			}
		}
		return 0
	}
	best, bestName, n := 0, "", 0
	for _, d := range devs {
		switch s := score(d); {
		case s > best:
			best, bestName, n = s, d.Name, 1
		case s == best && s > 0:
			n++
		}
	}
	switch {
	case best == 0:
		return "", "no name matches"
	case n > 1:
		return "", fmt.Sprintf("%d devices match equally", n)
	}
	return bestName, ""
}
