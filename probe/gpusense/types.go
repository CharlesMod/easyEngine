// Package gpusense reads the machine's graphics adapters and per-process graphics memory use.
// Every source is optional: a missing one is recorded in Picture.Sources and the rest carry on.
package gpusense

type Adapter struct {
	Index        int    `json:"index"`     // stable within one sense; matches probe.GPU.Index for NVIDIA where possible
	Vendor       string `json:"vendor"`    // "nvidia" | "amd" | "intel" | "apple" | "other"
	VendorID     int    `json:"vendor_id"` // PCI id: 0x10de nvidia, 0x1002 amd, 0x8086 intel, 0 unknown
	Name         string `json:"name"`
	DedicatedMiB int    `json:"dedicated_mib"`
	SharedMiB    int    `json:"shared_mib"` // system memory the adapter may borrow (DXGI SharedSystemMemory, Linux GTT); 0 unknown
	UsedMiB      int    `json:"used_mib"`   // card-wide dedicated use now, all processes; -1 unknown
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

type Reserve struct {
	OthersMiB   int    `json:"others_mib"`
	OthersBasis string `json:"others_basis"` // "processes"|"card"|"none"
	HostMiB     int    `json:"host_mib"`
	HostBasis   string `json:"host_basis"` // "measured"|"estimate"|"other-adapter"|"none"
	MarginMiB   int    `json:"margin_mib"`
	TotalMiB    int    `json:"total_mib"`
}

type GPUSummary struct {
	Adapters        []Adapter `json:"adapters"`
	EngineAdapter   int       `json:"engine_adapter"`  // -1 = CPU
	HostAdapter     int       `json:"host_adapter"`    // -1 = unknown
	HostIntegrated  string    `json:"host_integrated"` // "yes"|"no"|"unknown"
	Reserve         Reserve   `json:"reserve"`
	FreeForModelMiB int       `json:"free_for_model_mib"`    // dedicated(E) - others - host - margin, >=0
	OthersNote      string    `json:"others_note,omitempty"` // e.g. "Other apps are using 1.2 GB of your graphics card"
}

type GPUReport struct {
	GPUSummary
	Processes []Process         `json:"processes"`
	Sources   map[string]string `json:"sources"`
}
