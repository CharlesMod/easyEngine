package run

import (
	"fmt"
	"sort"
	"strconv"

	"hermesengine/catalog"
)

// Sampler policy modes (docs/SAMPLERS.md).
const (
	PolicyEnforce = "enforce" // every sampler field in a request is replaced by the profile's values
	PolicyFill    = "fill"    // only missing fields are filled
	PolicyOff     = "off"     // requests are not rewritten
)

// ValidPolicy reports whether m is one of the three modes.
func ValidPolicy(m string) bool { return m == PolicyEnforce || m == PolicyFill || m == PolicyOff }

// SamplerInfo is what the proxy and GET status know about sampling right now.
type SamplerInfo struct {
	Policy string `json:"policy"`
	// Active is true when requests can be rewritten: the model (local, or the external model
	// when we know it) has profiles. False means requests pass through untouched; Note says why.
	Active bool `json:"active"`
	// Model is the catalog id whose profiles apply ("" when none).
	Model string `json:"model,omitempty"`
	// Basis says how the model was found: "local" (the installed model) or "external" (the
	// external endpoint's model name matched a catalog model).
	Basis    string                            `json:"basis,omitempty"`
	Source   string                            `json:"source,omitempty"` // where the official values came from
	Note     string                            `json:"note,omitempty"`
	Profiles map[string]catalog.SamplerProfile `json:"profiles,omitempty"`
}

// catLocked returns the catalog. Caller holds s.mu.
func (s *Supervisor) catLocked() *catalog.Catalog {
	c := s.cat
	if c == nil {
		c = s.cfg.Catalog
	}
	if c == nil {
		var err error
		if c, err = catalog.Load(s.dir); err != nil {
			return nil
		}
		s.cat = c
	}
	return c
}

// Sampler returns the sampler policy and the profiles that apply to the current model.
func (s *Supervisor) Sampler() SamplerInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.samplerLocked()
}

func (s *Supervisor) samplerLocked() SamplerInfo {
	info := SamplerInfo{Policy: s.st.SamplerPolicy}
	if !ValidPolicy(info.Policy) {
		info.Policy = PolicyEnforce
	}
	var (
		m  catalog.Model
		ok bool
	)
	cat := s.catLocked()
	if s.st.Mode == "external" {
		info.Basis = "external"
		if s.st.External.Model == "" {
			info.Note = "external model not named: requests pass through untouched"
		} else if m, ok = cat.ModelByName(s.st.External.Model); !ok {
			info.Note = fmt.Sprintf("external model %q is not one we know: requests pass through untouched", s.st.External.Model)
		}
	} else {
		info.Basis = "local"
		if s.st.Model == "" {
			info.Note = "no model selected"
		} else if cat != nil {
			if m, ok = cat.Model(s.st.Model); !ok {
				info.Note = fmt.Sprintf("model %q is not in the catalog: requests pass through untouched", s.st.Model)
			}
		}
	}
	if ok && len(m.Sampler) == 0 {
		ok = false
		info.Note = fmt.Sprintf("catalog model %q has no sampler profiles: requests pass through untouched", m.ID)
	}
	if !ok && len(s.cfg.Profile.Samplers) == 0 {
		return info
	}
	info.Profiles = map[string]catalog.SamplerProfile{}
	if ok {
		info.Model, info.Source = m.ID, m.SamplerSource
		for k, v := range m.Sampler {
			info.Profiles[k] = v
		}
	}
	// A game's own named profiles (game profile "samplers") add to, or replace, the model's.
	for k, v := range s.cfg.Profile.Samplers {
		info.Profiles[k] = v
	}
	// Without a model's own values a game's profiles alone cannot stand in for official.
	info.Active = len(info.Profiles) > 0
	if info.Active {
		info.Note = ""
	}
	return info
}

// SamplerArgs turns a profile into llama-server's server-default flags, in a stable order.
// Flags the pinned release does not know are dropped and returned in dropped.
func SamplerArgs(p catalog.SamplerProfile, known map[string]bool) (args, dropped []string) {
	f := func(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }
	add := func(flag string, val string) {
		if has(known, flag) {
			args = append(args, flag, val)
		} else {
			dropped = append(dropped, flag)
		}
	}
	if p.Temperature != nil {
		add("--temp", f(*p.Temperature))
	}
	if p.TopK != nil {
		add("--top-k", strconv.Itoa(*p.TopK))
	}
	if p.TopP != nil {
		add("--top-p", f(*p.TopP))
	}
	if p.MinP != nil {
		add("--min-p", f(*p.MinP))
	}
	if p.TypicalP != nil {
		add("--typical", f(*p.TypicalP))
	}
	if p.RepeatPenalty != nil {
		add("--repeat-penalty", f(*p.RepeatPenalty))
	}
	if p.RepeatLastN != nil {
		add("--repeat-last-n", strconv.Itoa(*p.RepeatLastN))
	}
	if p.PresencePenalty != nil {
		add("--presence-penalty", f(*p.PresencePenalty))
	}
	if p.FrequencyPenalty != nil {
		add("--frequency-penalty", f(*p.FrequencyPenalty))
	}
	if p.DryMultiplier != nil {
		add("--dry-multiplier", f(*p.DryMultiplier))
	}
	if p.DryBase != nil {
		add("--dry-base", f(*p.DryBase))
	}
	if p.DryAllowedLength != nil {
		add("--dry-allowed-length", strconv.Itoa(*p.DryAllowedLength))
	}
	if p.DryPenaltyLastN != nil {
		add("--dry-penalty-last-n", strconv.Itoa(*p.DryPenaltyLastN))
	}
	return
}

// ProfileNames lists the names in a sampler info, sorted.
func (i SamplerInfo) ProfileNames() []string {
	var n []string
	for k := range i.Profiles {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// Logf writes a line to the engine log (the proxy uses it).
func (s *Supervisor) Logf(format string, a ...any) { s.logf(format, a...) }
