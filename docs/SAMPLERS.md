# Sampler settings and how the engine enforces them

Checked 2026-11-25 at the sources named below. Contract context: `docs/ENGINE.md` §3, §5, §7.

## What each model is meant to use

| Catalog id | Model | temperature | top_k | top_p | min_p | repeat_penalty | DRY |
|---|---|---|---|---|---|---|---|
| `gemma4-rp-maxxed-q3kxl` | Gemma 4 26B RP build (finetune of gemma-4-26B-A4B-it) | 1.0 | 64 | 0.95 | 0 | 1.0 (off) | off |
| `gemma4-e4b-q4km` | Gemma 4 E4B | 1.0 | 64 | 0.95 | 0 | 1.0 (off) | off |
| `gemma4-e2b-q4km` | Gemma 4 E2B | 1.0 | 64 | 0.95 | 0 | 1.0 (off) | off |

Sources (Hugging Face revisions read on 2026-11-25; `generation_config.json` is identical in all):

- `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (the model card says
  "Use the following standardized sampling configuration across all use cases").
- `google/gemma-4-E4B-it` @ `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`.
- `google/gemma-4-E2B-it` @ `4adf15748de9e1a0097b6274cee9c15836c41c37`.
- QAT repos `google/gemma-4-E4B-it-qat-q4_0-unquantized` @ `476025a0` and
  `google/gemma-4-E2B-it-qat-q4_0-unquantized` @ `6befbaca`: same values.
- Unsloth's Gemma 4 docs and QAT page: same values, min_p 0, repetition penalty off, no official DRY.

The RP build is our own finetune of the 26B model; it takes its base model's values.

**Not adopted.** Community RP finetunes publish temperature 0.7, min_p 0.05, top_p 0.95,
repetition penalty 1.05 and DRY 0.8/1.75/2. Those are tuned for their own weights, not ours.
Our own bench (`docs/MODEL-BENCH.md` in the agent repo) found DRY made no clear difference on
our three models. Until an A/B test on the GPU says otherwise, `rp` equals `official`.

## The three profiles

Every catalog model has a `sampler` block (`catalog.json`, generated from the spec in
`tools/gencatalog/samplers.go`; `sampler_source` records URL and commit, `sampler_match` the name
fragments that recognise the model on an external endpoint):

| Profile | Used for | Today |
|---|---|---|
| `official` | the vendor's values; also llama-server's start defaults | as the table |
| `rp` | spoken lines and creative text; the default when no profile is named | = official |
| `structured` | grammar-constrained JSON: intent and tool steps | `temperature 0` (top_k 64, top_p 0.95, min_p 0 as official; inert at temperature 0) |

`rp` stays official: on 2026-11-25 the GPU A/B against the old temp 0.8 / min_p 0.05 found no
difference within noise on the 26B or E4B rp (no loops either way, equal voice accuracy, no
grammar or format difference; the one gap was a trailing intent-label leak, fixed in the agent's
sanitiser). `structured` is greedy: measured on the same day (the first client's GPU bench),
temperature 0 beats 1.0 on intent for E4B rp (grammar 82 to 86.7 %, schema 63 to 80 %, schema
valid 86 to 100 %) and removes invalid schema and free replies on the 26B (schema 88 to 97 %,
valid 90 to 100 %); reflection is unchanged.

## Enforcement

1. **Server defaults.** llama-server starts with the model's `official` profile as its own
   defaults: `--temp 1 --top-k 64 --top-p 0.95 --min-p 0 --repeat-penalty 1 --dry-multiplier 0`.
   A client that sends nothing gets the vendor sampling. Each flag is checked against the
   release's help and dropped (and logged) if unknown; all exist in b12680.
2. **Proxy policy** (any client: our agent, a player's tool, another game's agent). On every
   completion request (`/v1/chat/completions`, `/chat/completions`, `/v1/completions`,
   `/completions`, `/completion`) the proxy reads `X-Hermes-Sampler` (`rp`, `structured`,
   `official`, or a game profile's name; missing or unknown means `rp`, unknown is logged) and
   applies `sampler_policy`:
   - `enforce` (default): replaces every sampler field in the request with the profile's values
     and removes the ones the profile does not set. The fields: `temperature`, `top_k`,
     `top_p`, `min_p`, `typical_p`, `typical`, `tfs_z`, `top_n_sigma`, `repeat_penalty`,
     `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `penalize_nl`,
     `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`,
     `dry_sequence_breakers`, `xtc_probability`, `xtc_threshold`, `mirostat`, `mirostat_tau`,
     `mirostat_eta`, `dynatemp_range`, `dynatemp_exponent`, `samplers`.
   - `fill`: only missing fields are filled.
   - `off`: no rewriting.

   Nothing else is touched (`max_tokens`, `grammar`, `json_schema`, `stop`, `seed`,
   `cache_prompt`, `id_slot`, `n_probs`, `logit_bias`, ...), and untouched fields keep their
   exact bytes. A body that is not a JSON object is forwarded as it came. Streaming is
   unchanged (the body is rewritten before the forward).
3. **Visibility.** Every proxied response has `X-Hermes-Sampler-Applied: <profile>;<mode>`; when
   a request passed through the value is `none;<reason>` (`not-a-completion`, `off`,
   `<mode>;unknown-model`). `GET /engine/v1/status` has `sampler: {policy, active, model, basis,
   source, note, profiles}`; the setup page shows the profiles read-only and lets the policy be
   chosen; `PUT /engine/v1/settings {"sampler_policy": "enforce|fill|off"}` changes it (omitted
   keeps it).
4. **External mode.** The same policy applies only when the external endpoint's model name
   matches a catalog model (`sampler_match`, compared with punctuation and case ignored, so
   `google/gemma-4-E4B-it` and `gemma4:e4b` both match). Then its official values are used.
   Any other model: requests pass through untouched and `status.sampler.note` says so.

## How a game adds a profile

In the game profile (`--profile file.json`, ENGINE.md §8) add `samplers`:

```json
"samplers": {"cold": {"temperature": 0.2, "top_k": 20, "top_p": 0.9}}
```

The values use the catalog's field names (`temperature`, `top_k`, `top_p`, `min_p`, `typical_p`,
`repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `dry_multiplier`,
`dry_base`, `dry_allowed_length`, `dry_penalty_last_n`). A request then sends
`X-Hermes-Sampler: cold`. A name that equals a model profile (`rp`, `structured`) replaces
the model's for that game. Fields a profile omits fall back to the server defaults (official).

## What could not be confirmed without a GPU

The flags were confirmed in the b12680 source (`common/arg.cpp`), not by running its
`--help`; the engine's runtime check does that on every start. That llama-server b12680 honours
the request-level fields on `/v1/chat/completions` was not exercised against the real binary.
