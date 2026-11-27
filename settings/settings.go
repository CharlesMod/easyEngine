// Package settings is settings.json, the keeper's file in the data dir: what the person chose
// (mode and own endpoint, linger, cache type, sampler policy, the listener, the yield options),
// kept apart from state.json, which holds only what the engine itself learns.
//
// The file is always written whole, with every key and its default, so it documents itself.
// A file that does not parse or holds a value that is not allowed is an *Error: the engine
// never overwrites such a file (ENGINE.md section 2).
package settings

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/lifecycle"
)

// FileName is the file's name in the data dir.
const FileName = "settings.json"

// DefaultPort is the control API's first port.
const DefaultPort = 47480

// External is the player's own OpenAI-compatible endpoint.
type External struct {
	BaseURL string `json:"base_url"`
	Model   string `json:"model"`
	APIKey  string `json:"api_key"`
}

// Listen is where the control API listens. Host is 127.0.0.1 unless the keeper opens it up
// (then every request needs the token unless Open is true).
type Listen struct {
	Host string `json:"host"`
	Port int    `json:"port"`
	Open bool   `json:"open"`
}

// Yield is the card-sharing policy (serve --yield-* flags).
type Yield struct {
	ToGames  bool     `json:"to_games"`
	Except   []string `json:"except"`
	Ignore   []string `json:"ignore"`
	FloorMiB int      `json:"floor_mib"`
	MinMiB   int      `json:"min_mib"`
	GraceS   int      `json:"grace_s"`
	SteadyS  int      `json:"steady_s"`
	Strict   bool     `json:"strict"`
}

// Grace is GraceS as a duration.
func (y Yield) Grace() time.Duration { return time.Duration(y.GraceS) * time.Second }

// Steady is SteadyS as a duration.
func (y Yield) Steady() time.Duration { return time.Duration(y.SteadyS) * time.Second }

// File is the whole of settings.json.
type File struct {
	Mode          string   `json:"mode"` // local | external
	External      External `json:"external"`
	LingerS       int      `json:"linger_s"`
	KVType        string   `json:"kv_type"`
	SamplerPolicy string   `json:"sampler_policy"` // enforce | fill | off
	// Streamer is the token streamer (docs/STREAMING.md): on (default) | off. Off is the proxy's
	// plain byte copy, kept as a way back.
	Streamer string `json:"streamer"`
	Listen   Listen `json:"listen"`
	Yield    Yield  `json:"yield"`
}

// KVTypes are the cache types llama-server takes.
var KVTypes = []string{"q4_0", "q8_0", "f16", "q4_1", "q5_0", "q5_1", "iq4_nl", "bf16", "f32"}

// Policies are the sampler policies (docs/SAMPLERS.md).
var Policies = []string{"enforce", "fill", "off"}

// Streamers are the streamer setting's values.
var Streamers = []string{"on", "off"}

// Defaults is every setting at its default.
func Defaults() File {
	return File{
		Mode: "local", LingerS: 0, KVType: "q4_0", SamplerPolicy: "enforce", Streamer: "on",
		Listen: Listen{Host: "127.0.0.1", Port: DefaultPort},
		Yield:  Yield{Except: []string{}, Ignore: []string{}, FloorMiB: 128, GraceS: 15, SteadyS: 60},
	}
}

// Error is a settings.json that cannot be used: the line (0 when unknown) and what is wrong.
type Error struct {
	Line int
	Msg  string
}

func (e *Error) Error() string {
	if e.Line > 0 {
		return fmt.Sprintf("%s line %d: %s", FileName, e.Line, e.Msg)
	}
	return fmt.Sprintf("%s: %s", FileName, e.Msg)
}

func in(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

// Validate reports the first value that is not allowed, as the JSON key it came from and why.
func (f File) Validate() (key, why string) {
	switch {
	case f.Mode != "local" && f.Mode != "external":
		return "mode", fmt.Sprintf("mode must be local or external, not %q", f.Mode)
	case !in(KVTypes, f.KVType):
		return "kv_type", fmt.Sprintf("unknown kv_type %q", f.KVType)
	case !in(Policies, f.SamplerPolicy):
		return "sampler_policy", "sampler_policy must be enforce, fill or off"
	case !in(Streamers, f.Streamer):
		return "streamer", "streamer must be on or off"
	case f.LingerS < 0:
		return "linger_s", "linger_s must not be negative"
	case f.Listen.Port < 1 || f.Listen.Port > 65535:
		return "port", fmt.Sprintf("port %d is out of range (1-65535)", f.Listen.Port)
	case strings.TrimSpace(f.Listen.Host) == "":
		return "host", "host must not be empty"
	case f.Yield.FloorMiB < 0:
		return "floor_mib", "floor_mib must not be negative"
	case f.Yield.MinMiB < 0:
		return "min_mib", "min_mib must not be negative"
	case f.Yield.GraceS < 0:
		return "grace_s", "grace_s must not be negative"
	case f.Yield.SteadyS < 0:
		return "steady_s", "steady_s must not be negative"
	}
	return "", ""
}

// Normalize makes lists non-nil so the file writes [] and not null.
func (f File) Normalize() File {
	if f.Yield.Except == nil {
		f.Yield.Except = []string{}
	}
	if f.Yield.Ignore == nil {
		f.Yield.Ignore = []string{}
	}
	if f.Streamer == "" {
		f.Streamer = "on" // never written empty: an unset streamer is the default
	}
	return f
}

// Equal reports whether two settings hold the same values.
func (f File) Equal(g File) bool {
	a, _ := json.Marshal(f.Normalize())
	b, _ := json.Marshal(g.Normalize())
	return bytes.Equal(a, b)
}

// Encode is the file's text: indented, every key, a final newline.
func Encode(f File) []byte {
	b, _ := json.MarshalIndent(f.Normalize(), "", "  ")
	return append(b, '\n')
}

// Write writes the whole file atomically (a unique temp file, fsync, rename), mode 0600 (it can
// hold an API key).
func Write(path string, f File) error {
	return lifecycle.WriteFileAtomic(path, Encode(f), 0o600)
}

// Path is the file in a data dir.
func Path(dir string) string { return filepath.Join(dir, FileName) }

func lineOf(text []byte, off int64) int {
	if off > int64(len(text)) {
		off = int64(len(text))
	}
	if off < 0 {
		off = 0
	}
	return 1 + bytes.Count(text[:off], []byte("\n"))
}

// keyLine finds the line of the first `"key"` in text, 0 when absent.
func keyLine(text []byte, key string) int {
	i := bytes.Index(text, []byte(`"`+key+`"`))
	if i < 0 {
		return 0
	}
	return lineOf(text, int64(i))
}

// Parse reads the text of a settings file. Keys it lacks keep their defaults; keys it does not
// know are ignored. It returns an *Error for text that does not parse or holds a bad value.
func Parse(text []byte) (File, error) {
	f := Defaults()
	if err := json.Unmarshal(text, &f); err != nil {
		var se *json.SyntaxError
		var te *json.UnmarshalTypeError
		switch {
		case errors.As(err, &se):
			return Defaults(), &Error{Line: lineOf(text, se.Offset), Msg: se.Error()}
		case errors.As(err, &te):
			return Defaults(), &Error{Line: lineOf(text, te.Offset), Msg: fmt.Sprintf("%s must be %s, not %s", te.Field, te.Type, te.Value)}
		}
		return Defaults(), &Error{Msg: err.Error()}
	}
	if f.Mode == "" {
		f.Mode = "local"
	}
	if f.KVType == "" {
		f.KVType = "q4_0"
	}
	if f.SamplerPolicy == "" {
		f.SamplerPolicy = "enforce"
	}
	if f.Streamer == "" {
		f.Streamer = "on"
	}
	if f.Listen.Host == "" {
		f.Listen.Host = "127.0.0.1"
	}
	f = f.Normalize()
	if key, why := f.Validate(); key != "" {
		return Defaults(), &Error{Line: keyLine(text, key), Msg: why}
	}
	return f, nil
}

// Read reads path. A missing file is os.ErrNotExist (wrapped), never an *Error.
func Read(path string) (File, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Defaults(), err
	}
	return Parse(b)
}

// SetAside moves a settings file the engine cannot use to settings.json.bad-<unix seconds>
// (never deleted, never replaced) so that a write can follow. It returns the new name.
func SetAside(path string) (string, error) {
	aside := fmt.Sprintf("%s.bad-%d", path, time.Now().Unix())
	for n := 1; ; n++ {
		if _, err := os.Lstat(aside); err != nil {
			break
		}
		aside = fmt.Sprintf("%s.bad-%d-%d", path, time.Now().Unix(), n)
	}
	return aside, os.Rename(path, aside)
}

// settingKeys are the keys state.json used to hold.
var settingKeys = []string{"mode", "external", "linger_s", "kv_type", "sampler_policy"}

// Migrate moves the settings an old state.json holds into settings.json, once: settings.json
// is written (only when it does not exist yet) and state.json is rewritten without those keys.
// It says whether it moved anything. A state.json that does not parse is left alone (the
// engine sets it aside itself).
func Migrate(dir string) (bool, error) {
	sp := filepath.Join(dir, "state.json")
	raw, err := os.ReadFile(sp)
	if err != nil {
		return false, nil
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return false, nil
	}
	has := false
	for _, k := range settingKeys {
		if _, ok := m[k]; ok {
			has = true
		}
	}
	if !has {
		return false, nil
	}
	fp := Path(dir)
	if _, err := os.Stat(fp); errors.Is(err, os.ErrNotExist) {
		f := Defaults()
		json.Unmarshal(raw, &f) // the old keys; what is odd is made ordinary below
		if f.Mode != "external" {
			f.Mode = "local"
		}
		if !in(KVTypes, f.KVType) {
			f.KVType = "q4_0"
		}
		if !in(Policies, f.SamplerPolicy) {
			f.SamplerPolicy = "enforce"
		}
		if f.LingerS < 0 {
			f.LingerS = 0
		}
		if err := Write(fp, f); err != nil {
			return false, err
		}
	}
	for _, k := range settingKeys {
		delete(m, k)
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return false, err
	}
	return true, lifecycle.WriteFileAtomic(sp, b, 0o600)
}

// Load is what a start does: move old settings out of state.json, then read settings.json. A
// missing file is written with the defaults (whole). A file that is bad is left as it is and
// the defaults come back with the *Error.
func Load(dir string) (File, error) {
	Migrate(dir)
	p := Path(dir)
	f, err := Read(p)
	if errors.Is(err, os.ErrNotExist) {
		f = Defaults()
		Write(p, f) // best effort: a read-only dir still runs on the defaults
		return f, nil
	}
	return f, err
}
