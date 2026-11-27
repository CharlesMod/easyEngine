package run

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/settings"
)

// External is the player's own OpenAI-compatible endpoint.
type External struct {
	BaseURL string `json:"base_url"`
	Model   string `json:"model"`
	APIKey  string `json:"api_key"`
}

// State is what the engine holds in memory: the keeper's settings (Mode to SamplerPolicy; they
// come from and go to settings.json, never state.json: json:"-") and what the engine itself
// learns (state.json, the rest).
type State struct {
	Mode     string   `json:"-"` // local | external
	External External `json:"-"`
	LingerS  int      `json:"-"`
	KVType   string   `json:"-"`
	// SamplerPolicy is enforce (default) | fill | off (docs/SAMPLERS.md).
	SamplerPolicy string `json:"-"`
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
	st, _, note, _ := loadAll(dir)
	return st, note
}

// loadAll is the start: settings.json (settings an old state.json held are first moved out of
// it) and state.json. serr is set when settings.json could not be used: the file is left as it
// is and the defaults stand.
func loadAll(dir string) (st State, file settings.File, note string, serr *settings.Error) {
	file, err := settings.Load(dir)
	if err != nil {
		if !errors.As(err, &serr) {
			serr = &settings.Error{Msg: err.Error()}
		}
	}
	st, note = loadEngineState(dir)
	st.applyFile(file)
	return st, file, note, serr
}

// applyFile sets the settings fields from the file.
func (s *State) applyFile(f settings.File) {
	s.Mode, s.LingerS, s.KVType, s.SamplerPolicy = f.Mode, f.LingerS, f.KVType, f.SamplerPolicy
	s.External = External{BaseURL: f.External.BaseURL, Model: f.External.Model, APIKey: f.External.APIKey}
}

func loadEngineState(dir string) (State, string) {
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

// ---- settings.json -------------------------------------------------------------------------

// settingsFile is what the engine knows of settings.json beyond the fields in State: the
// listener and yield options (run does not use them; the app does, at start), the file as it
// stood at start, and what is wrong with it now. It is under Supervisor.mu.
type settingsFile struct {
	listen, startListen settings.Listen
	yield, startYield   settings.Yield
	streamer            string // on | off: read by the proxy per request, so an edit applies at once
	keepContext         string // on | off | always: "" is on
	name, discovery     string // the engine's name and the mDNS announce (on | off): the app reads them at start and on each reload
	err                 string // settings.json cannot be used: the previous settings stand
	note                string // a listen or yield edit that waits for a restart
}

func (f *settingsFile) begin(file settings.File, serr *settings.Error) {
	f.listen, f.startListen, f.yield, f.startYield = file.Listen, file.Listen, file.Yield, file.Yield
	f.streamer = file.Streamer
	f.keepContext = file.KeepContext
	f.name, f.discovery = file.Name, file.Discovery
	if serr != nil {
		f.err = serr.Error() + "; using the defaults"
	}
}

// notes are the lines the status errors carry for the file.
func (f *settingsFile) notes() []string {
	var out []string
	if f.err != "" {
		out = append(out, f.err)
	}
	if f.note != "" {
		out = append(out, f.note)
	}
	return out
}

// settingsWriteMu serialises every write of settings.json in the process.
var settingsWriteMu sync.Mutex

func (s *Supervisor) fileLocked() settings.File {
	e := s.st.External
	return settings.File{
		Mode: s.st.Mode, External: settings.External{BaseURL: e.BaseURL, Model: e.Model, APIKey: e.APIKey},
		LingerS: s.st.LingerS, KVType: s.st.KVType, SamplerPolicy: s.st.SamplerPolicy,
		Streamer: s.sf.streamer, KeepContext: s.sf.keepContext, Name: s.sf.name, Discovery: s.sf.discovery,
		Listen: s.sf.listen, Yield: s.sf.yield,
	}.Normalize()
}

// EngineName is the settings name for this engine ("" means the host name; docs/ENGINE.md "Pairing").
func (s *Supervisor) EngineName() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sf.name
}

// Discovery reports whether a non-loopback listener announces itself by mDNS (settings "discovery").
func (s *Supervisor) Discovery() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sf.discovery != "off"
}

// Streamer reports whether the token streamer is on (settings.json "streamer"; docs/STREAMING.md).
func (s *Supervisor) Streamer() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sf.streamer != "off"
}

// KeepContext is the keep_context setting: "on", "off" or "always" (settings.json; "" is "on").
func (s *Supervisor) KeepContext() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sf.keepContext == "" {
		return "on"
	}
	return s.sf.keepContext
}

// SettingsFile is the settings as the engine holds them (the API key included: for the
// engine's own use, never for a client).
func (s *Supervisor) SettingsFile() settings.File {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fileLocked()
}

// InFlight is the number of proxied requests being answered.
func (s *Supervisor) InFlight() int { return int(s.inflightNow()) }

// saveSettings writes settings.json whole. A file the engine could not read is first set aside
// as settings.json.bad-<unix> (the engine never overwrites a file it could not read; a person
// who asks for a change through the API has chosen the engine's version).
func (s *Supervisor) saveSettings() {
	s.mu.Lock()
	f := s.fileLocked()
	bad := s.sf.err != ""
	s.mu.Unlock()
	settingsWriteMu.Lock()
	defer settingsWriteMu.Unlock()
	path := settings.Path(s.dir)
	if bad {
		aside, err := settings.SetAside(path)
		if err != nil {
			s.logf("settings.json is unreadable and could not be set aside: %v", err)
			return
		}
		s.logf("settings.json was unreadable; set aside as %s", aside)
		s.mu.Lock()
		s.sf.err = ""
		s.mu.Unlock()
	}
	if err := settings.Write(path, f); err != nil {
		s.logf("write settings: %v", err)
	}
}

// ReloadSettings follows settings.json: the app calls it when the file changes. A file that
// parses and differs from what the engine holds takes effect through the same code as PUT
// settings (mode, endpoint, linger, cache type and sampler policy at once); a changed listen
// or yield section is recorded and waits for a restart. A file that does not parse changes
// nothing and is said in the status errors until it is fixed. It never writes the file (a
// missing one is written again, whole).
func (s *Supervisor) ReloadSettings() {
	path := settings.Path(s.dir)
	f, err := settings.Read(path)
	if errors.Is(err, os.ErrNotExist) {
		s.saveSettings()
		return
	}
	if err != nil {
		var se *settings.Error
		if !errors.As(err, &se) {
			se = &settings.Error{Msg: err.Error()}
		}
		msg := se.Error() + "; using the previous settings"
		s.mu.Lock()
		changed := s.sf.err != msg
		s.sf.err = msg
		s.mu.Unlock()
		if changed {
			s.logf("error: %s", msg)
		}
		return
	}
	s.mu.Lock()
	s.sf.err = ""
	cur := s.fileLocked()
	s.mu.Unlock()
	same := f.Mode == cur.Mode && f.External == cur.External && f.LingerS == cur.LingerS &&
		f.KVType == cur.KVType && f.SamplerPolicy == cur.SamplerPolicy
	if !same {
		s.logf("settings.json changed: applying it")
		s.setSettings(Settings{
			Mode: f.Mode, External: External{BaseURL: f.External.BaseURL, Model: f.External.Model, APIKey: f.External.APIKey},
			ClearAPIKey: f.External.APIKey == "", LingerS: f.LingerS, KVType: f.KVType, SamplerPolicy: f.SamplerPolicy,
		}, false)
	}
	s.mu.Lock()
	s.sf.listen, s.sf.yield, s.sf.streamer = f.Listen, f.Yield, f.Streamer
	s.sf.keepContext = f.KeepContext
	s.sf.name, s.sf.discovery = f.Name, f.Discovery
	var what []string
	if f.Listen != s.sf.startListen {
		what = append(what, "listen address")
	}
	if !(settings.File{Yield: f.Yield}).Equal(settings.File{Yield: s.sf.startYield}) {
		what = append(what, "yield options")
	}
	s.sf.note = ""
	if len(what) > 0 {
		s.sf.note = "restart easy-engine to apply the new " + strings.Join(what, " and ")
	}
	s.mu.Unlock()
}
