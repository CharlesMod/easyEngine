// Package run is the supervisor: it installs the backend, model and agent,
// starts llama-server at the shape the ctx balancer picks, keeps it healthy,
// resizes it around the player's other GPU use, and keeps the agent running.
package run

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"hermesengine/catalog"
	"hermesengine/lifecycle"
	"hermesengine/plan"
	"hermesengine/probe"
	"hermesengine/probe/gpusense"
)

// Profile is the game profile (ENGINE.md section 8). Game-agnostic fields only.
type Profile struct {
	Game       string `json:"game"`
	Slots      int    `json:"slots"`
	MinSlots   int    `json:"min_slots"`
	CtxPerSlot struct {
		Max   int `json:"max"`
		Floor int `json:"floor"`
	} `json:"ctx_per_slot"`
	// GameReserveMiB is the host's estimated GPU use before it has any (the measured use wins
	// once above it; POST /engine/v1/host can change it).
	GameReserveMiB int `json:"game_reserve_mib"`
	// HostPID is the host app's process id (0: POST /engine/v1/host, else the first parent pid);
	// HostAdapter is the adapter it renders on, when it knows.
	HostPID     int               `json:"host_pid,omitempty"`
	HostAdapter *plan.HostAdapter `json:"host_adapter,omitempty"`
	Agent       *AgentSpec        `json:"agent,omitempty"`
	// Samplers are the game's own named sampler profiles. A request names one with the
	// X-Hermes-Sampler header; a name here also replaces the model's profile of that name.
	Samplers map[string]catalog.SamplerProfile `json:"samplers,omitempty"`
	// HostFilesMiB is how far the host's own files may still grow on the engine's disk (for
	// example the agent's call log: its cap times the copies it keeps). Every disk check counts
	// it in full, so a setup never fills the disk the host still writes to. 0 = none declared.
	HostFilesMiB int `json:"host_files_mib,omitempty"`
	// YieldExcept is the launcher app ids (Steam's) that never make the engine yield the card
	// under serve --yield-to-games: the supported host itself.
	YieldExcept []string `json:"yield_except,omitempty"`
}

// AgentSpec describes the agent the engine installs and runs.
type AgentSpec struct {
	ID     string            `json:"id"`
	Wheel  string            `json:"wheel"`
	Python string            `json:"python"`
	Args   []string          `json:"args"`
	Env    map[string]string `json:"env"`
	// Module overrides the module derived from the wheel's console script.
	Module string `json:"module,omitempty"`
	// LLMURLEnv names the variable that carries the engine URL
	// (default HERMESVOICE_LLM_URL, per ENGINE.md section 8).
	LLMURLEnv string `json:"llm_url_env,omitempty"`
	// DiskBytes is the measured disk the agent's setup takes (managed Python, uv cache, venv or
	// site dir) at its peak; 0 uses the catalog's measurement for the OS (agent_runtime).
	DiskBytes int64 `json:"disk_bytes,omitempty"`
}

// LoadProfile reads a profile file; an empty path gives the defaults.
func LoadProfile(path string) (Profile, error) {
	var p Profile
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return p, err
		}
		if err := json.Unmarshal(b, &p); err != nil {
			return p, err
		}
	}
	p.defaults()
	return p, nil
}

func (p *Profile) defaults() {
	if p.Slots <= 0 {
		p.Slots = 1
	}
	if p.MinSlots <= 0 || p.MinSlots > p.Slots {
		p.MinSlots = 1
	}
	if p.CtxPerSlot.Max <= 0 {
		p.CtxPerSlot.Max = 32768
	}
	if p.CtxPerSlot.Floor <= 0 || p.CtxPerSlot.Floor > p.CtxPerSlot.Max {
		p.CtxPerSlot.Floor = 8192
		if p.CtxPerSlot.Floor > p.CtxPerSlot.Max {
			p.CtxPerSlot.Floor = p.CtxPerSlot.Max
		}
	}
	if p.HostPID < 0 {
		p.HostPID = 0
	}
	if p.GameReserveMiB < 0 {
		p.GameReserveMiB = 0
	}
	if a := p.HostAdapter; a != nil && a.Name == "" && a.VendorID == 0 {
		p.HostAdapter = nil
	}
	if p.Agent != nil && p.Agent.Python == "" {
		p.Agent.Python = "3.12"
	}
}

// Config holds everything injectable. Zero values get production defaults.
type Config struct {
	DataDir string
	Profile Profile
	Version string

	Catalog *catalog.Catalog // nil: catalog.Load(DataDir)

	// Injectable system access.
	Probe  func(ctx context.Context) (probe.Hardware, error)
	Others func(ctx context.Context, gpu probe.GPU, ourPIDs []int) (int, error)
	// Sense reads the adapters and what each process holds on them; ours are the engine's own
	// pids, host the host app's.
	Sense    func(ctx context.Context, ours []int, host int) gpusense.Picture
	Measure  func(ctx context.Context, pid int) (int, error) // VRAM MiB used by our llama-server
	Seed     func(m catalog.Model, modelPath, kv string) (fit.Footprint, error)
	Group    *lifecycle.Group
	HTTP     *http.Client
	Now      func() time.Time
	DiskFree func(dir string) (int, error) // free MiB under dir; nil: the real filesystem

	ExtraEnv    []string // appended to llama-server's environment (tests)
	AgentEnv    []string // appended to the agent's environment (tests)
	LlamaPort   int      // 0: pick a free port
	NoAutoStart bool     // do not start llama-server or the agent by itself (CLI install/import)
	Parents     func() []int

	// Timing. Tests shrink these.
	WatchEvery        time.Duration // default 5s: watchdog tick
	HealthMisses      int           // default 6
	IdleChecks        int           // default 2
	GrowIdle          time.Duration // default 60s
	GrowBackoff       time.Duration // default 10 min after a failed grow
	HealthPoll        time.Duration // default 250ms while loading
	LoadTimeout       time.Duration // default 15 min
	StopGrace         time.Duration // default 5s
	AgentBackoffBase  time.Duration // default 1s
	AgentBackoffMax   time.Duration // default 30s
	AgentStable       time.Duration // default 60s of uptime resets the backoff
	CmdTimeout        time.Duration // default 5 min for uv commands
	UpgradeBackoff    time.Duration // default 30s: first wait before retrying the target download (doubles)
	UpgradeBackoffMax time.Duration // default 15 min
	SwapDrain         time.Duration // default 2 min: longest wait for requests in flight before a model swap
	// Crash recovery (run/recover.go).
	CrashBackoff []time.Duration // default 2s, 5s, 15s, 30s: wait before try n after failures in a row (the last repeats)
	CrashQuick   time.Duration   // default 2 min: a run shorter than this that crashes is a quick crash; one longer clears the record
	CrashSteps   int             // default 3: quick crashes in a row at one shape before one step down
	DrainMax     time.Duration   // default 2 min: longest wait for replies before a shrink that cannot wait for idle
	DrainMinMiB  int             // default 256: a held shortfall this large shrinks without waiting for idle
	OOMHold      time.Duration   // default 60s: out-of-memory lines in two ticks this close are a shrink signal

	// Yield, when set, gives the card to other programs that want it (run/yield.go).
	Yield *YieldConfig
}

func (c *Config) defaults() {
	if c.Probe == nil {
		c.Probe = probe.Probe
	}
	if c.Others == nil {
		c.Others = probe.Others
	}
	if c.Sense == nil {
		c.Sense = func(ctx context.Context, ours []int, host int) gpusense.Picture {
			return gpusense.Sense(ctx, probe.OSSources{}, ours, host)
		}
	}
	if c.Measure == nil {
		c.Measure = nvidiaMeasure
	}
	if c.Seed == nil {
		c.Seed = defaultSeed
	}
	if c.Group == nil {
		c.Group = lifecycle.NewGroup()
	}
	if c.HTTP == nil {
		c.HTTP = &http.Client{Timeout: 3 * time.Second}
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	if c.Version == "" {
		c.Version = "dev"
	}
	dur := func(p *time.Duration, d time.Duration) {
		if *p <= 0 {
			*p = d
		}
	}
	dur(&c.WatchEvery, 5*time.Second)
	dur(&c.GrowIdle, 60*time.Second)
	dur(&c.GrowBackoff, 10*time.Minute)
	dur(&c.HealthPoll, 250*time.Millisecond)
	dur(&c.LoadTimeout, 15*time.Minute)
	dur(&c.StopGrace, 5*time.Second)
	dur(&c.AgentBackoffBase, time.Second)
	dur(&c.AgentBackoffMax, 30*time.Second)
	dur(&c.AgentStable, 60*time.Second)
	dur(&c.CmdTimeout, 5*time.Minute)
	dur(&c.UpgradeBackoff, 30*time.Second)
	dur(&c.UpgradeBackoffMax, 15*time.Minute)
	dur(&c.SwapDrain, 2*time.Minute)
	if len(c.CrashBackoff) == 0 {
		c.CrashBackoff = []time.Duration{2 * time.Second, 5 * time.Second, 15 * time.Second, 30 * time.Second}
	}
	dur(&c.CrashQuick, 2*time.Minute)
	dur(&c.DrainMax, 2*time.Minute)
	dur(&c.OOMHold, 60*time.Second)
	if c.Yield != nil {
		c.Yield.defaults()
	}
	if c.CrashSteps <= 0 {
		c.CrashSteps = 3
	}
	if c.DrainMinMiB <= 0 {
		c.DrainMinMiB = 256
	}
	if c.DiskFree == nil {
		c.DiskFree = diskFree
	}
	if c.HealthMisses <= 0 {
		c.HealthMisses = 6
	}
	if c.IdleChecks <= 0 {
		c.IdleChecks = 2
	}
	c.Profile.defaults()
}

func defaultSeed(m catalog.Model, modelPath, kv string) (fit.Footprint, error) {
	if info, err := plan.ReadInfo(modelPath); err == nil {
		fp, err := plan.Seed(info, kv)
		if err != nil {
			return fit.Footprint{}, err
		}
		if m.Footprint != nil { // the header gives the sizes, the measurement the speed curve
			mf, err := plan.FromMeasured(m.Footprint, kv)
			if err != nil {
				return fit.Footprint{}, err
			}
			fp.TPS = mf.TPS
		}
		return fp, nil
	}
	return plan.SeedFromSize(m, kv)
}
