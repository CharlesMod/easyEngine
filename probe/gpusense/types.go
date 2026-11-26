// Package gpusense reads the machine's graphics adapters and what each process holds on them.
// This file is the shared contract: the types the planner, the supervisor and the API use.
package gpusense

type Adapter struct {
	Index        int    `json:"index"`          // stable within one sense; matches probe.GPU.Index for NVIDIA where possible
	Vendor       string `json:"vendor"`         // "nvidia" | "amd" | "intel" | "apple" | "other"
	VendorID     int    `json:"vendor_id"`      // PCI id: 0x10de nvidia, 0x1002 amd, 0x8086 intel, 0 unknown
	Name         string `json:"name"`
	DedicatedMiB int    `json:"dedicated_mib"`
	SharedMiB    int    `json:"shared_mib"`        // system memory the adapter may borrow (DXGI SharedSystemMemory, Linux GTT); 0 unknown
	UsedMiB      int    `json:"used_mib"`          // card-wide dedicated use now, all processes; -1 unknown
	Integrated   bool   `json:"integrated"`
	Unified      bool   `json:"unified,omitempty"` // Apple: one pool
	LUID         string `json:"luid,omitempty"`    // Windows "0x%08X_0x%08X" (high_low), matches counter instance names
	PCI          string `json:"pci,omitempty"`     // Linux/NVIDIA bus id, normalised lower-case "0000:01:00.0"
	UUID         string `json:"uuid,omitempty"`    // NVIDIA
	Source       string `json:"source"`            // "nvidia-smi" | "dxgi" | "wmi" | "sysfs" | "sysctl"
}

type Process struct {
	PID          int    `json:"pid"`
	Name         string `json:"name"`    // executable base name, "" if unknown
	Adapter      int    `json:"adapter"` // Adapter.Index
	DedicatedMiB int    `json:"dedicated_mib"`
	SharedMiB    int    `json:"shared_mib"`
	Kind         string `json:"kind"`           // "compute" | "graphics" | "" (unknown)
	Ours         bool   `json:"ours,omitempty"` // engine's own children (llama-server)
	Host         bool   `json:"host,omitempty"` // the host app (profile/host pid, else parent pid)
}

type Picture struct {
	Adapters  []Adapter         `json:"adapters"`
	Processes []Process         `json:"processes"`
	Sources   map[string]string `json:"sources"` // source name -> "ok" | error text; missing sources degrade, never fail
}
