package main

import (
	"hermesengine/catalog"
)

func f64(v float64) *float64 { return &v }
func ival(v int) *int        { return &v }

// gemmaOfficial is the vendor's generation_config.json for every Gemma 4 build in the catalog
// (gemma-4-26B-A4B-it, gemma-4-E4B-it, gemma-4-E2B-it and both QAT repos are identical, checked
// 2026-11-25): temperature 1.0, top_k 64, top_p 0.95. The 26B model card says "Use the
// following standardized sampling configuration across all use cases". Unsloth's Gemma 4 docs
// give the same values with min_p 0 and the repetition penalty off (1.0). There is no official
// DRY, so DRY is explicitly off (multiplier 0).
func gemmaOfficial() catalog.SamplerProfile {
	return catalog.SamplerProfile{
		Temperature: f64(1.0), TopK: ival(64), TopP: f64(0.95), MinP: f64(0),
		RepeatPenalty: f64(1.0), DryMultiplier: f64(0),
	}
}

// samplerSpec is one model's sampler block. rp is what spoken lines and creative text get; it
// equals official (the GPU A/B of 2026-11-25 found the old temp 0.8 / min_p 0.05 no better).
// structured is for grammar-constrained JSON (intent, tool steps) and is greedy: measured on
// 2026-11-25 (first client's GPU bench, see docs/SAMPLERS.md), temperature 0 beat 1.0 on intent for E4B rp
// (+5 to +17 points) and removed invalid schema and free replies on the 26B and E4B rp, with
// reflection unchanged.
type samplerSpec struct {
	source                   string
	match                    []string
	official, rp, structured catalog.SamplerProfile
}

func gemmaSpec(source string, match ...string) samplerSpec {
	o := gemmaOfficial()
	s := gemmaOfficial()
	s.Temperature = f64(0) // greedy; top_k/top_p/min_p as official (inert at temperature 0)
	return samplerSpec{source: source, match: match, official: o, rp: o, structured: s}
}

// samplerSpecs is keyed by catalog model id. The commits are the Hugging Face revisions read
// on 2026-11-25.
var samplerSpecs = map[string]samplerSpec{
	// The reference build is a finetune of gemma-4-26B-A4B-it and takes its base model's values.
	refID: gemmaSpec("https://huggingface.co/google/gemma-4-26B-A4B-it/blob/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/generation_config.json"+
		" (base model of this build); https://unsloth.ai/docs (Gemma 4: same values)",
		"gemma4rpmaxxed", "gemma426b", "gemma4a4b"),
	"gemma4-e4b-rp-q40": gemmaSpec("https://huggingface.co/google/gemma-4-E4B-it/blob/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c/generation_config.json"+
		" (the QAT repo google/gemma-4-E4B-it-qat-q4_0-unquantized@476025a0 is identical); https://unsloth.ai/docs (Gemma 4: same values)",
		"gemma4e4b"),
	"gemma4-e2b-q4km": gemmaSpec("https://huggingface.co/google/gemma-4-E2B-it/blob/4adf15748de9e1a0097b6274cee9c15836c41c37/generation_config.json"+
		" (the QAT repo google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbaca is identical); https://unsloth.ai/docs (Gemma 4: same values)",
		"gemma4e2b"),
}

// applySamplers attaches each model's sampler block from samplerSpecs. It returns the ids of
// models that have no spec (every model must have one).
func applySamplers(models []catalog.Model) (missing []string) {
	for i := range models {
		sp, ok := samplerSpecs[models[i].ID]
		if !ok {
			missing = append(missing, models[i].ID)
			continue
		}
		models[i].Sampler = map[string]catalog.SamplerProfile{
			catalog.ProfileOfficial: sp.official, catalog.ProfileRP: sp.rp, catalog.ProfileStructured: sp.structured,
		}
		models[i].SamplerSource = sp.source
		models[i].SamplerMatch = sp.match
	}
	return missing
}
