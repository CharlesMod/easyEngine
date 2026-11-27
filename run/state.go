package run

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/lifecycle"
)

// External is the player's own OpenAI-compatible endpoint.
type External struct {
	BaseURL string `json:"base_url"`
	Model   string `json:"model"`
	APIKey  string `json:"api_key"`
}

// State is state.json.
type State struct {
	Mode     string   `json:"mode"` // local | external
	External External `json:"external"`
	LingerS  int      `json:"linger_s"`
	KVType   string   `json:"kv_type"`
	// SamplerPolicy is enforce (default) | fill | off (docs/SAMPLERS.md).
	SamplerPolicy string `json:"sampler_policy,omitempty"`
	Model         string `json:"model"`
	Backend       string `json:"backend"`
	// Target is the model a first run is still upgrading to while a smaller starter serves (run/starter.go).
	Target string `json:"target,omitempty"`
	// StarterOwned is the starter model this engine downloaded only to bridge a first run. It is
	// deleted once the target serves; a model that was already on disk is never deleted.
	StarterOwned string `json:"starter_owned,omitempty"`
	// Custom is the user's own GGUF file, served in place while Model is "custom" (run/custom.go).
	// Its path is outside models/ and nothing here ever copies, moves or deletes it.
	Custom *CustomFile `json:"custom,omitempty"`
	// CustomPrev is the catalog model that was selected before the custom file, for custom/clear.
	CustomPrev string   `json:"custom_prev,omitempty"`
	Book       fit.Book `json:"book"`
	// Previous is the catalog id of the model a switch replaced (its file is kept; run/switch.go).
	Previous string `json:"previous,omitempty"`
}

func (s *State) defaults() {
	if s.Mode != "external" {
		s.Mode = "local"
	}
	if !ValidPolicy(s.SamplerPolicy) {
		s.SamplerPolicy = PolicyEnforce
	}
	if s.KVType == "" {
		s.KVType = "q4_0"
	}
	if s.Book.Entries == nil {
		s.Book.Entries = map[string]fit.Calib{}
	}
}

// loadState reads state.json. A missing file is silently the defaults. A file that exists but
// does not parse is renamed aside to state.json.corrupt-<unix seconds> (never deleted, never
// overwritten) and note says so; the caller reports it.
func loadState(dir string) (State, string) {
	var st State
	var note string
	path := filepath.Join(dir, "state.json")
	if b, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(b, &st); err != nil {
			st = State{}
			aside := fmt.Sprintf("%s.corrupt-%d", path, time.Now().Unix())
			// A second corrupt file in the same second must not replace the first.
			for n := 1; ; n++ {
				if _, err := os.Lstat(aside); err != nil {
					break
				}
				aside = fmt.Sprintf("%s.corrupt-%d-%d", path, time.Now().Unix(), n)
			}
			if rerr := os.Rename(path, aside); rerr != nil {
				note = fmt.Sprintf("settings file was unreadable (%v) and could not be set aside (%v)", err, rerr)
			} else {
				note = fmt.Sprintf("settings file was unreadable and was set aside as %s", aside)
			}
		}
	}
	st.defaults()
	return st, note
}

// stateMu serialises every write of state.json in the process.
var stateMu sync.Mutex

func writeState(dir string, st State) error {
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	stateMu.Lock()
	defer stateMu.Unlock()
	return lifecycle.WriteFileAtomic(filepath.Join(dir, "state.json"), b, 0o600)
}

// Note puts a line in the status errors (something the player should read, for example a catalog
// override that was ignored).
func (s *Supervisor) Note(msg string) { s.addError("%s", msg) }

// LoadState reads state.json from a data dir (tests and tools).
func LoadState(dir string) State { st, _ := loadState(dir); return st }
