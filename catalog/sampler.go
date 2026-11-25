package catalog

import (
	"regexp"
	"strings"
)

// Sampler profile names every model carries. A game may name more in its profile.
const (
	ProfileOfficial   = "official"   // the vendor's published values
	ProfileRP         = "rp"         // spoken lines and creative text (the default)
	ProfileStructured = "structured" // grammar-constrained JSON: intent and tool steps
)

// SamplerProfile is one named set of sampling values. A nil field is "not set by this
// profile": enforce mode removes that field from a request so the server default applies.
type SamplerProfile struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	MinP             *float64 `json:"min_p,omitempty"`
	TypicalP         *float64 `json:"typical_p,omitempty"`
	RepeatPenalty    *float64 `json:"repeat_penalty,omitempty"`
	RepeatLastN      *int     `json:"repeat_last_n,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	DryMultiplier    *float64 `json:"dry_multiplier,omitempty"`
	DryBase          *float64 `json:"dry_base,omitempty"`
	DryAllowedLength *int     `json:"dry_allowed_length,omitempty"`
	DryPenaltyLastN  *int     `json:"dry_penalty_last_n,omitempty"`
}

// Fields is the profile as request-body fields (llama-server's names).
func (p SamplerProfile) Fields() map[string]any {
	m := map[string]any{}
	if p.Temperature != nil {
		m["temperature"] = *p.Temperature
	}
	if p.TopK != nil {
		m["top_k"] = *p.TopK
	}
	if p.TopP != nil {
		m["top_p"] = *p.TopP
	}
	if p.MinP != nil {
		m["min_p"] = *p.MinP
	}
	if p.TypicalP != nil {
		m["typical_p"] = *p.TypicalP
	}
	if p.RepeatPenalty != nil {
		m["repeat_penalty"] = *p.RepeatPenalty
	}
	if p.RepeatLastN != nil {
		m["repeat_last_n"] = *p.RepeatLastN
	}
	if p.PresencePenalty != nil {
		m["presence_penalty"] = *p.PresencePenalty
	}
	if p.FrequencyPenalty != nil {
		m["frequency_penalty"] = *p.FrequencyPenalty
	}
	if p.DryMultiplier != nil {
		m["dry_multiplier"] = *p.DryMultiplier
	}
	if p.DryBase != nil {
		m["dry_base"] = *p.DryBase
	}
	if p.DryAllowedLength != nil {
		m["dry_allowed_length"] = *p.DryAllowedLength
	}
	if p.DryPenaltyLastN != nil {
		m["dry_penalty_last_n"] = *p.DryPenaltyLastN
	}
	return m
}

// SamplerFields are every request field the proxy treats as a sampler setting. Enforce mode
// replaces or removes all of them; nothing else in a request is touched.
var SamplerFields = []string{
	"temperature", "top_k", "top_p", "min_p", "typical_p", "typical", "tfs_z", "top_n_sigma",
	"repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty", "penalize_nl",
	"dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers",
	"xtc_probability", "xtc_threshold",
	"mirostat", "mirostat_tau", "mirostat_eta",
	"dynatemp_range", "dynatemp_exponent",
	"samplers",
}

// Profile returns a named profile of the model's sampler block.
func (m Model) Profile(name string) (SamplerProfile, bool) {
	p, ok := m.Sampler[name]
	return p, ok
}

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

func normName(s string) string { return nonAlnum.ReplaceAllString(strings.ToLower(s), "") }

// ModelByName finds the catalog model that a free-form model name (an external endpoint's
// model id, a Hugging Face repo, a file name) refers to, by the model's sampler_match terms.
// Names are compared lower-cased with every character that is not a letter or digit removed,
// so "google/gemma-4-E4B-it", "gemma4:e4b" and "Gemma_4_E4B" all match "gemma4e4b".
// The longest matching term wins. A model without sampler_match is matched by its id only.
func (c *Catalog) ModelByName(name string) (Model, bool) {
	n := normName(name)
	if n == "" || c == nil {
		return Model{}, false
	}
	best, bestLen := Model{}, 0
	for _, m := range c.Models {
		terms := m.SamplerMatch
		if len(terms) == 0 {
			terms = []string{m.ID}
		}
		for _, t := range terms {
			t = normName(t)
			if t != "" && len(t) > bestLen && strings.Contains(n, t) {
				best, bestLen = m, len(t)
			}
		}
	}
	return best, bestLen > 0
}
