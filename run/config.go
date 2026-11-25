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

	"hermesengine/catalog"
	"hermesengine/fit"
	"hermesengine/gguf"
	"hermesengine/lifecycle"
	"hermesengine/probe"
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
	GameReserveMiB int        `json:"game_reserve_mib"`
	Agent          *AgentSpec `json:"agent,omitempty"`
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
	Probe   func(ctx context.Context) (probe.Hardware, error)
	Others  func(ctx context.Context, gpu probe.GPU, ourPIDs []int) (int, error)
	Measure func(ctx context.Context, pid int) (int, error) // VRAM MiB used by our llama-server
	Seed    func(m catalog.Model, modelPath, kv string) (fit.Footprint, error)
	Group   *lifecycle.Group
	HTTP    *http.Client
	Now     func() time.Time

	ExtraEnv    []string // appended to llama-server's environment (tests)
	AgentEnv    []string // appended to the agent's environment (tests)
	LlamaPort   int      // 0: pick a free port
	NoAutoStart bool     // do not start llama-server or the agent by itself (CLI install/import)
	Parents     func() []int

	// Timing. Tests shrink these.
	WatchEvery       time.Duration // default 5s: watchdog tick
	HealthMisses     int           // default 6
	IdleChecks       int           // default 2
	GrowIdle         time.Duration // default 60s
	GrowBackoff      time.Duration // default 10 min after a failed grow
	HealthPoll       time.Duration // default 250ms while loading
	LoadTimeout      time.Duration // default 15 min
	StopGrace        time.Duration // default 5s
	AgentBackoffBase time.Duration // default 1s
	AgentBackoffMax  time.Duration // default 30s
	AgentStable      time.Duration // default 60s of uptime resets the backoff
	CmdTimeout       time.Duration // default 5 min for uv commands
}

func (c *Config) defaults() {
	if c.Probe == nil {
		c.Probe = probe.Probe
	}
	if c.Others == nil {
		c.Others = probe.Others
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
	if c.HealthMisses <= 0 {
		c.HealthMisses = 6
	}
	if c.IdleChecks <= 0 {
		c.IdleChecks = 2
	}
	c.Profile.defaults()
}

func defaultSeed(m catalog.Model, modelPath, kv string) (fit.Footprint, error) {
	if info, err := gguf.Read(modelPath); err == nil {
		fp := fit.Seed(info, kv)
		return fp, nil
	}
	return fit.SeedFromSize(m, kv), nil
}
