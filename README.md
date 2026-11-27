# easyEngine

A one-click local LLM engine for game mods.

A mod ships one small binary, `easy-engine`. When the game starts, the mod launches it. The engine:

- **Sets itself up.** It reads the machine's GPU, VRAM, RAM and disk, recommends a model that fits,
  and downloads it together with a pinned llama.cpp build (resumable, sha256-checked). There is a
  setup page in the browser, and the same API can drive a setup window inside the game.
- **Runs the model.** Stock `llama-server` as a child process, with a context balancer that picks
  slots, context per slot and expert offload from measured free VRAM, leaves room for the game,
  shrinks when the game needs memory and grows back when idle.
- **Runs the mod's agent** (a Python wheel the mod names in its profile) in its own environment.
- **Goes away with the game.** When the game exits or crashes, everything stops.

A mod can instead point at any OpenAI-compatible endpoint; the engine then only proxies to it.

Models in the catalog: `gemma4-rp-maxxed` (16 GB and 12 GB cards), `Gemma 4 E4B` (8 GB), and
`Gemma 4 E2B`, a flagged fallback for small machines that front ends warn about.

## Build and test

```bash
go vet ./... && go test ./...
./core-check.sh   # the sizing core in core/, its own module at Go 1.23
```

```bash
./build.sh
```

`build.sh` cross-compiles Linux, Windows and macOS binaries into `dist/<os>-<arch>/`. Tests need no
GPU and no network; they use a fake llama-server.

The sizing core (footprint, shape chooser, calibration book, GGUF header reader) is a nested Go
module, `github.com/CharlesMod/easyEngine/core`, that other programs can require by its tag
(`core/vX.Y.Z`); see docs/ENGINE.md section 13.

## Docs

- [docs/ENGINE.md](docs/ENGINE.md): design and contract (control API, game profile, CLI, the
  context balancer).
- [docs/FOOTPRINTS.md](docs/FOOTPRINTS.md): what each catalog model costs on a card, as measured,
  and how the planner uses it.
- [docs/SAMPLERS.md](docs/SAMPLERS.md): the sampler settings each model is meant to use and how
  the proxy enforces them.

## Used by

- [RimWorldLLM](https://github.com/CharlesMod/RimWorldLLM) (HermesVoice for RimWorld).
