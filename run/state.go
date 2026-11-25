package run

import (
	"encoding/json"
	"os"
	"path/filepath"

	"hermesengine/fit"
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
	Model    string   `json:"model"`
	Backend  string   `json:"backend"`
	Book     fit.Book `json:"book"`
}

func (s *State) defaults() {
	if s.Mode != "external" {
		s.Mode = "local"
	}
	if s.KVType == "" {
		s.KVType = "q4_0"
	}
	if s.Book.Entries == nil {
		s.Book.Entries = map[string]fit.Calib{}
	}
}

func loadState(dir string) State {
	var st State
	if b, err := os.ReadFile(filepath.Join(dir, "state.json")); err == nil {
		json.Unmarshal(b, &st)
	}
	st.defaults()
	return st
}

func writeState(dir string, st State) error {
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	os.MkdirAll(dir, 0o755)
	tmp := filepath.Join(dir, "state.json.tmp")
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, "state.json"))
}

// LoadState reads state.json from a data dir (tests and tools).
func LoadState(dir string) State { return loadState(dir) }
