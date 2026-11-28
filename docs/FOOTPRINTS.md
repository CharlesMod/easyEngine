# Measured footprints

What each catalog model really costs on a card, measured with llama.cpp b12680 on an RTX 5080
(16303 MiB, `-ngl 999 -fa on --no-kv-unified -ub 512`), and how the planner uses it. The numbers
live in `tools/gencatalog/measured.json` (carried into `catalog/catalog.json` as each model's
`footprint` block). They come from the sweep CSV (69 ok runs) by `tools/fitcheck -fit -emit`;
nothing here is typed by hand. Re-measure with the same tool after a llama.cpp bump.

```
go run ./tools/fitcheck -csv <sweep.csv> -models <dir of GGUF files> [-fit] [-emit measured.json]
```

## Per-model terms

VRAM for a shape is `fixed + slope(kv) * ctx_total/1k + state(kv) * slots + scratch - freed * n_cpu_moe`
(plus about 125-215 MiB of offload buffers once `n_cpu_moe > 0`). `ctx_total = slots * ctx_per_slot`.
Slope is the least-squares line through the q4_0 rows (residual under 30 MiB); q8_0 and f16 slopes come
from their rows, and any other KV type is interpolated by bytes per element.


> **2026-11-25:** the catalog's E4B is now `gemma4-e4b-rp-q40`, the keeper's QAT-exact Q4_0 rp build
> (cmod/gemma4-e4b-rp-GGUF, 3.65 GB). Measured on the RTX 5080 at b12680: fixed 2,994 MiB (stock Q4_K_M
> 3,318), the same KV slopes, host RAM 912 MiB (stock 1,848), decode at 3 slots 120 t/s (stock 92).
> The rows below for `gemma4-e4b-q4km` are the stock model, kept for comparison. The predictor
> under-predicts the rp rows by 2-4 %.

| model | fixed | GB per 1k ctx q4_0 / q8_0 / f16 | per-slot state q4_0 / q8_0 / f16 | scratch | per expert layer (VRAM / RAM) |
|---|---|---|---|---|---|
| gemma4-e2b-q4km | 1.76 GB | 0.0040 / 0.0054 / 0.0062 | 0 / 1.4 / 3.9 MiB | 54 MiB | dense |
| gemma4-e4b-q4km | 3.24 GB | 0.0087 / 0.0125 / 0.0157 | 0 / 11.9 / 34 MiB | 55 MiB | dense |
| gemma4-rp-maxxed-q3kxl | 9.73 GB | 0.0097 / 0.0146 / 0.0210 | 66 / 142 / 285 MiB | 60 MiB | 246 / 302 MiB (30 layers) |

The e2b and e4b "fixed" excludes 1.5 and 1.8 GB of per-layer embeddings that stay in RAM (add them to
CPU-only RAM, not to VRAM). The f16 per-slot cost rests on one row each and is the least certain.

## Speed (tokens/s per stream, q4_0, 65536 ctx per slot, everything on the card)

| model | 1 slot decode / prompt | 2 slots | 3 slots |
|---|---|---|---|
| gemma4-e2b-q4km | 199 / 22313 | 161 / 14398 | 136 / 9328 |
| gemma4-e4b-q4km | 134 / 11168 | 107 / 6979 | 92 / 4746 |
| gemma4-rp-maxxed-q3kxl | 143 / 5580 | 127 / 3349 | 105 / 2335 |

Decode against `--n-cpu-moe` at 3 slots (the planner's curve). Time per token is linear in the CPU layer
count (gemma 7.5 ms + 1.84 ms per layer), so the planner interpolates 1/tps.

| n_cpu_moe | 0 | 3 | 6 | 12 | 20 | 30 |
|---|---|---|---|---|---|---|
| gemma4-rp-maxxed-q3kxl | 105 | 71 | 54 | 36 | 25 | 15 |

## Predicted against measured

Header seeds (`fit.Seed` on the real GGUF header), predicted minus measured VRAM over every row.

| model | rows | before: mean / max MiB | before: max % | after: mean / max MiB | after: max % |
|---|---|---|---|---|---|
| gemma4-rp-maxxed-q3kxl | 20 | 460 / 848 | 8.3 | 57 / 115 | 1.0 |
| gemma4-e4b-q4km | 15 | 2482 / 2741 | 79.9 | 45 / 65 | 1.3 |
| gemma4-e2b-q4km | 15 | 2343 / 2481 | 131.8 | 73 / 93 | 3.9 |

Full row tables: `sweep/fitcheck-before.txt` and `sweep/fitcheck-after.txt` in the measurement directory.
What changed in `fit.Seed`: Gemma E-series per-layer embeddings are host-resident (not VRAM); the output and
vocab buffers scale with vocabulary size; a compute buffer grows with ctx (0.44 MiB per 1k ctx for the mask, plus a dequant buffer for quantised KV; the vocab buffer is 1.2 MiB per 1k vocabulary entries); SWA caches are capped at
`window + ubatch`; MoE offload adds a fixed scheduling buffer; the old CUDA-context and scratch guesses were
replaced by 135 MiB base overhead and 58 MiB runtime. All constants are global, none per model.

These constants are the core's default set (`fit.DefaultConsts`: backend `cuda@b12680`, card
`RTX 5080, 16303 MiB`), each with a word of where it was read. They hold for this build and this card:
the base a start takes before any buffer reads 168 to 220 MiB on a 4 GB card with another build. A
caller that has read its own gives them as a `fit.Consts` set to `fit.SeedWith`; every footprint and
shape says which set built it, so a figure from the default set on another card is a prediction. A
fixed part the book worked out from a reading is laid only on a seed of the set it was worked out under.

## Re-measured on llama.cpp b12651 (2026-11-26)

The catalog pin moved from b12680 to b12651 (Windows Smart App Control refused b12680's `llama.dll`).
The footprints above were measured at b12680 on the RTX 5080, so the same sweep (`sweep.py`, the 15-row
grid per model, plus the 30-layer offload rows for the 26B; 50 runs, all ok) was run on a Tesla P40
(24 GB, CUDA 12.8 Linux build, `-ngl 999 -fa on --no-kv-unified`, default `-ub 512`) at both builds,
so the build is the only thing that differs between the two P40 columns.

| model | fixed MiB: 5080 / b12680 (catalog) | P40 / b12680 | P40 / b12651 | largest per-row VRAM change b12680 to b12651 |
|---|---|---|---|---|
| gemma4-e2b-q4km | 1803 | 1671 | 1671 | 0 MiB (0 %) |
| gemma4-e4b-rp-q40 | 2994 | 2862 | 2862 | 0 MiB (0 %) |
| gemma4-rp-maxxed-q3kxl | 9961 | 9829 | 9829 | 16 MiB (0.15 %) |

Every KV slope (q4_0 / q8_0 / f16), per-slot state, host RAM and per-expert-layer figure is the same to
the printed digit at both builds; the scratch term is also the same at both builds (e2b 66, e4b 118, 26B 56 on
the P40, against 54, 55 and 60 in the catalog: a fit residual that depends on the card, not the build). The 132 MiB lower `fixed` on the P40 is the
card (its CUDA context is smaller than the 5080's), identical at both builds. Nothing moved by more than
2 % or 64 MiB because of the build, so `measured.json` is unchanged and still describes the RTX 5080
reference card; its `llamacpp` field keeps "b12680", where the numbers were read, and they hold at b12651.

Speed on the P40 (mean of the 15-row grid, b12651 / b12680): decode 1.012 (e2b), 0.987 (e4b rp), 1.024
(26B); prompt 1.006, 1.000, 1.009. E4B rp, 1 slot at 65536 ctx: 51.8 against 52.0 tok/s. Raw CSVs and
fitcheck output: `~/.local/share/rimworld-hermes-engine/pin/` (`sweep-b12651.csv`, `sweep-b12680.csv`,
`fit-b12651.txt`, `fit-b12680.txt`).

## Measured but not shipped

Ling 3.0-tiny (Q4_K_M, bailingmoe3: MLA plus linear attention, 4.76 GB, MoE with 23 expert layers) was
swept alongside the others (19 rows, predicted within 2.2 % after the fit changes, about 200 tok/s per stream
on the 5080 at 3 slots) and was dropped from the ladder on 2026-11-25 so every shipped model is Gemma. The
reader still understands the bailingmoe3 header (`gguf`, `fit.Seed`), so adding it back is a catalog
entry plus a `tools/gencatalog/measured.json` block.

## How to use these on another machine

Before a download the planner has only the catalog. Do this by hand, or read the same answer from
`/engine/v1/plan`:

1. `budget = VRAM total - other apps - game reserve (1500) - margin (max 512, 4 % of total)`.
2. VRAM at the wanted shape from the formula above (q4_0 KV, 3 slots x 65536 = 192k ctx).
3. If it exceeds the budget and the model is MoE, each expert layer moved to RAM frees its VRAM figure
   (minus about 125-215 MiB once) and costs its RAM figure; decode speed follows the curve.
4. Speed on a different card: the planner scales the measured speed by 0.5 (decode is memory-bandwidth
   bound; 8-12 GB cards have 30-45 % of the 5080's bandwidth) and keeps decode at or above 20 tok/s
   per stream, giving up ctx before speed. If nothing reaches 20 it still offers the shape, marked slow.
5. Dense models are never split. If the fixed part plus a small ctx does not fit, step down a model.

Worked examples (desktop uses about 800 MiB; RAM 24 GB free):

- **24 GB card.** Budget about 21.3 GB. The reference model needs 9961 + 9.97 x 192 + 66 x 3 + 60 = 12.1 GB, so
  it fits with room to spare; even f16 KV (15.0 GB) fits. Plan: reference model, 3 x 65536, all on the card.
- **16 GB card.** Budget about 13.3 GB. Reference model 12.1 GB fits at 3 x 65536 with all experts on the
  card, about 105 tok/s per stream. With a game holding 2800 MiB the plan drops to 3 x 35840, still no CPU layers.
- **12 GB card.** Budget about 9.5 GB. 12.1 GB does not fit. Ten expert layers to RAM free 2455 MiB (125 back),
  and ctx drops to 54272 per slot: about 9.5 GB. Speed is about 41 tok/s on the 5080, so about 20 estimated
  here, right at the floor. e4b (5.1 GB) fits whole at full ctx and is the better pick for a 12 GB card
  that has other things open.
- **8 GB card.** Budget about 5.6 GB. The reference model would need 28 of 30 layers in RAM (8.5 GB RAM) and
  about 8 tok/s, under the floor, so it is shown "tight (slow)". e4b (5.1 GB) fits whole at full ctx:
  recommended. e2b (2.6 GB) fits too, but it is the flagged fallback and is never preferred over e4b.
- **6 GB or less.** Only e2b fits whole (2.6 GB); e4b is tight (2 of 3 slots at 8192 ctx at 6 GB); the reference model is not usable. e2b is the catalog's
  `fallback` model (see ENGINE.md §3): the plan and status carry a warning for it.
  It is recommended only when no other model can run: where e4b fits at all (even tight), e4b is the pick.
  With no GPU, the planner picks the smallest non-fallback model whose RAM is at most 60 % of free RAM after a
  2 GiB OS reserve, else the smallest non-fallback model that fits (e4b), and e2b only when nothing else fits;
  it marks the plan as an estimate: no CPU speed was measured, so expect it slow.

Caveats: one card, one llama.cpp build, one driver. Other cards move `fixed` by a few hundred MiB
(CUDA context, allocator granularity); the margin covers that and `calibrate()` adjusts after the first
healthy load. The speed curve is a lower bound for fewer than 3 slots.
