# easyEngine (design and contract)

**easyEngine** is a one-click local LLM runtime for game mods. It is game-agnostic: no source
file here may name a game (a test enforces it). HermesVoice for RimWorld
(https://github.com/CharlesMod/RimWorldLLM) is its first user and the running example in this file. A game tells the engine what it needs through a **game profile** (below). The
RimWorld part is the mod code that starts the engine, writes the profile and shows setup and
status in game.

Background: the survey `easy_local_engine.md` (on the keeper's machine under
`~/.local/share/rimworld-easy-engine-research/`, not in this repository)
and the keeper's goal (2026-11-24): a GUI installer, the model picked by available RAM/VRAM,
the ctx balancer, starting when the game starts and tucking away when it exits, and
any OpenAI-compatible endpoint as the alternative, with a Test light.

## 1. What it is

One native Go binary, `easy-engine` (this repo, Go module `github.com/CharlesMod/easyEngine`, Go 1.27,
toolchain at `~/.local/go/bin/go`; its sizing core is a nested module at Go 1.23 that other
programs import, §13). Cross-compiled for `linux-amd64`, `windows-amd64`,
`darwin-arm64` and `darwin-amd64`; no cgo. Not PyInstaller (AV false positives).

It does four jobs:

1. **Setup.** Probe the hardware, recommend a model and backend, download them (resumable,
   sha256-checked) and keep the install up to date. GUI: a setup page it serves itself
   (opened in the browser by `easy-engine setup`) **and** the same API driven from the
   game's own setup window. One API, two faces.
2. **Model server.** Run stock `llama-server` (official llama.cpp release build) as a child,
   with the **ctx balancer** choosing the shape (§5).
3. **Agent host.** Install and run the game's agent (for HermesVoice, the Python
   `hermesvoice-agent`) as a child, pointed at the engine.
4. **Lifetime.** Start when the game starts, stop everything when the game exits or crashes.

The engine always runs the agent. The llama-server part runs only in **local** mode; in
**external** mode the agent is pointed at the player's own endpoint and no model is
downloaded or started.

## 2. Files and places

Data dir (`dirs.Data()`): Linux `$XDG_DATA_HOME/easy-engine` (default
`~/.local/share/easy-engine`), Windows `%LOCALAPPDATA%\EasyEngine`, macOS
`~/Library/Application Support/EasyEngine`. `EASY_ENGINE_HOME` overrides it (tests always
set it to a temp dir; **tests never touch the real data dir and only remove what they made**).
The engine refuses a data dir it must not own: a relative path, a filesystem root, or the home
dir itself. `serve`, `setup`, `install`, `import` and `token` then print the reason and exit
with code 2 before writing anything.

```
<data>/
  settings.json         the keeper's file: mode, external endpoint, linger, cache type, sampler policy, listener, yield options (below)
  state.json            what the engine itself learns: model id, backend id, pending target (first-run upgrade), custom file, calibration book
  backends/<backend-id>@<tag>/  unpacked llama.cpp build (llama-server[.exe] + libs); backends/<backend-id>/ for one installed before tags were recorded
  models/<file>.gguf
  runtime/uv[.exe]            pinned uv binary
  runtime/venvs/<agent-id>/   agent virtualenv (uv-managed Python 3.12; on Windows site/ + base-python.txt, §8)
  downloads/<name>.part       in-progress downloads (resume with HTTP Range)
  logs/engine.log, logs/llama-server.log, logs/agent.log   (rotated at 10 MB, keep 2)
  logs/engine-stdio.log      serve mode: the engine's own fd 1/2 (a crash trace, a last exit
                             line); moved to .1 at start when 1 MiB or more
  engine.lock                 single-instance lock (pid, start time, port, host, child pids)
  tokens.json                 bearer token store, mode 0600 (§7 token)
```

### settings.json and state.json

`settings.json` is the keeper's file (package `settings`): what the person chose. `state.json`
keeps only what the engine learns (`model`, `backend`, `build`, `next_build`, `target`, `starter_owned`, `custom`,
`custom_prev`, `book`, `previous`). A first start writes `settings.json` **whole**, every key
with its default, indented, mode 0600 (it can hold an API key), through a unique temp file,
fsync and rename:

```json
{
  "mode": "local",
  "external": {
    "base_url": "",
    "model": "",
    "api_key": ""
  },
  "linger_s": 0,
  "kv_type": "q4_0",
  "sampler_policy": "enforce",
  "streamer": "on",
  "keep_context": "on",
  "name": "",
  "discovery": "on",
  "listen": {
    "host": "127.0.0.1",
    "port": 47480,
    "open": false
  },
  "yield": {
    "to_games": false,
    "except": [],
    "ignore": [],
    "floor_mib": 128,
    "min_mib": 0,
    "grace_s": 15,
    "steady_s": 60,
    "strict": false
  }
}
```

- `mode` local or external; `external` the own endpoint (§7); `linger_s` how long the model
  stays loaded after the last client (§6); `kv_type` one of `q4_0 q8_0 f16 q4_1 q5_0 q5_1 iq4_nl
  bf16 f32`; `sampler_policy` `enforce`, `fill` or `off`.
- `name` is this engine's human name for clients on the network (hello, the mDNS name; at most
  63 bytes); empty means the host name. `discovery` `on` (default) or `off`: the mDNS announce of
  a non-loopback listener (§7f). Both take effect when the file is saved (the responder is
  started, renamed or stopped on the reload).
- `listen` is the control API's address (`host`, first `port`, `open` = no token on a
  non-loopback host, like `--open`, §7); `yield` mirrors the `--yield-*` flags and their
  defaults (§6b). `except` and `ignore` are lists of strings. Missing keys keep their default,
  unknown keys are ignored.
- **The split.** On the first start after this change, the settings found in an old
  `state.json` (`mode`, `external`, `linger_s`, `kv_type`, `sampler_policy`) are moved to
  `settings.json` (only when that file does not exist yet) and `state.json` is rewritten without
  them. It happens once; the next start finds nothing to move.
- **A file that cannot be used** (does not parse, or an unknown `mode`, `kv_type` or
  `sampler_policy`, a negative `linger_s`, a `port` outside 1-65535) is **never overwritten**.
  The engine starts with the last good settings (the defaults at a first start) and says so in
  status `errors[]`: `settings.json line 3: unknown kv_type "x"; using the previous settings`.
  Fixing the file clears the error. (`PUT settings` is the one exception: it moves the unusable
  file aside as `settings.json.bad-<unix time>` and writes its own.)
- **Followed live.** The engine watches the file (`host.Changes`, an event) and compares
  content, not time: a valid edit takes effect through the same code as `PUT settings`
  (`mode`, `external`, `linger_s`, `kv_type`, `sampler_policy` at once). A changed `listen` or
  `yield` is recorded and takes effect at the next start; status `errors[]` says `restart
  easy-engine to apply the new listen address` until the file matches what runs. The engine's own
  writes come back as a change that changes nothing.
- The flags of `serve` and `app` win over the file for that run and are never written back.

## 3. Catalog

`catalog/catalog.json` is embedded in the binary (`go:embed`) and generated by
`tools/gencatalog` (pins exact URLs, sizes and sha256 from the GitHub and Hugging Face
APIs; never hand-typed hashes). `-repo owner/name` (default `ggml-org/llama.cpp`) picks the repo
whose releases carry the llama.cpp builds, and `-llama-tag` takes `bNNNN` or a patched
`bNNNN-easyN` (the full tag is what the catalog records), e.g. `-repo CharlesMod/llama.cpp
-llama-tag b12651-easy1`. A newer catalog may be placed at `<data>/catalog.json` and
wins when its `version` is higher. A catalog is checked when it is read (`catalog.Parse`): ids and
file names may use only letters, digits, `.`, `_` and `-` (no slash, no leading dot); a download
url must be `https` (plain `http` only for 127.0.0.1, localhost or ::1) and must come with a
64-character hex sha256 and a size above 0. A model with no url is import-only and is allowed.
An override that fails the check is ignored: the embedded catalog is used, the reason is logged
and it shows in status `errors[]`. Since version 8 every llama.cpp, cudart and uv asset carries
`unpacked_size`: gencatalog downloads the archive, unpacks it with the engine's own unpacker and
counts every file rounded up to 4 KiB plus 4 KiB per directory (`-cache` keeps the archives and
counts for reruns). `agent_runtime` lists the measured disk of the agent's setup per OS
(`tools/gencatalog/runtime.json`, measured by a real install; see §7c). A model's size and
sha256 are its LFS record at the repo's pinned commit; where the repo publishes a `SHA256SUMS`
that names the file, the two must agree or gencatalog stops. `go run ./tools/gencatalog
-models-only -out catalog/catalog.json` touches no GitHub: it rebuilds only `models` from
Hugging Face and keeps `llama`, `uv` and `agent_runtime` as they are in the file.

```json
{
  "version": 1,
  "llamacpp": {
    "tag": "b12xxx",
    "assets": [
      {"id": "cuda-linux-x64", "os": "linux", "arch": "amd64", "accel": "cuda",
       "url": "...", "sha256": "...", "size": 0, "unpacked_size": 0, "extra": []},
      {"id": "vulkan-linux-x64", "accel": "vulkan", ...},
      {"id": "cpu-linux-x64", "accel": "cpu", ...},
      {"id": "cuda-win-x64", "extra": ["cudart-win-x64"], ...},
      {"id": "metal-mac-arm64", "accel": "metal", ...}
    ]
  },
  "uv": [{"os": "linux", "arch": "amd64", "url": "...", "sha256": "...", "size": 0}],
  "models": [
    {"id": "gemma4-rp-maxxed-q3kxl", "name": "Gemma 4 26B RP", "codename": "Campfire", "codename_full": "Campfire-26B",
     "file": "gemma4-rp-maxxed-Q3_K_XL.gguf",
     "url": "https://huggingface.co/cmod/gemma4-rp-maxxed-GGUF/resolve/main/gemma4-rp-maxxed-Q3_K_XL.gguf",
     "sha256": "595eddca...", "size": 10001779136,
     "moe": true, "quality": 100, "chat": "gemma4", "notes": "reference model"},
    {"id": "gemma4-e4b-rp-q40", "file": "...", "url": "https://huggingface.co/...", "moe": false, "quality": 60},
    {"id": "gemma4-e2b-q4km", ..., "quality": 40,
     "fallback": true, "fallback_note": "Smallest model: expect repeated lines and weaker understanding of what was said."}
  ]
}
```

- **Tiers and code names.** The default models are a ladder of tiers, each with a `codename`
  and a `codename_full` (both optional, short plain text: at most 24 characters, letters,
  digits, spaces and `.` `-` `'` only, no leading or trailing space; a `codename_full` needs a
  `codename` and begins with it; checked by `catalog.Parse`). Where a model has one, every place
  that shows the model shows the full code name first (else the short one) and the model's name
  second ("Campfire-26B (Gemma 4 26B RP)": the setup page, the tray, the download and swap
  messages); a catalog without code names reads as before and shows the name alone. The names
  are display only: what each model does (starter, fallback, recommendation) is unchanged.

  | `codename` | `codename_full` | Model | Catalog id | Picked by |
  |---|---|---|---|---|
  | Flashlight | Flashlight-2B | Gemma 4 E2B (fallback) | `gemma4-e2b-q4km` | the ladder, last resort |
  | Lantern | Lantern-4B | Gemma 4 E4B RP | `gemma4-e4b-rp-q40` | the ladder; the usual first-run starter |
  | Campfire | Campfire-26B | Gemma 4 26B RP (MoE) | `gemma4-rp-maxxed-q3kxl` | the ladder; the pick on a 16 GB card |
  | Bonfire | Bonfire-27B | Qwen3.8 27B ASCII-Condensed Heretic Q3_K_XL (dense, not RP) | to come | a person only (`opt_in`) |

- **Opt-in.** `opt_in: true` marks a model a person picks: nothing chooses it by itself. The
  plan never makes it `recommended` (on a card or CPU-only), the first run never takes it as a
  starter, and the tier step-down below never steps to it. It is selected through the usual
  `POST install` / switch, and then serves like any other model; the plan lists it with
  `opt_in: true`.
- **Dense, never split.** `moe: false` (and, once the file is on disk, a header with no expert
  tensors) is a dense model: the sizing core never lays any of its layers off the card
  (`fit.Choose` moves only MoE expert layers to RAM), so a dense model either fits the card whole
  or does not fit (`limit: dense_never_split` in the plan). **Tier step-down** (`run/tiers.go`):
  when the selected catalog model would not fit this card and this machine even with nothing
  else on them (the card's whole memory, no other program, no game share, and the machine's
  whole RAM for any expert layers: what passes is left out), the engine serves the best
  installed catalog model below it (highest `quality` under the selected one's) that is not
  opt-in and fits the room now, and says so: `GET status` `model`, `model_name`,
  `model_codename` and `model_codename_full` are the tier that serves, `model_selected` stays
  the selection, `model_fallback` is true and `model_note` names both ("Bonfire-27B (...) is
  selected, but it does not fit this card whole, and a dense model is never split between the
  card and RAM: Campfire-26B (Gemma 4 26B RP) serves instead."). The `model` fact says `serving`
  beside `model` when a step is taken and again when it ends. The selection (`state.json`
  `model`) stays the person's; every start tries it first again, so a bigger card serves it.
  When the model would fit the card alone and only other programs or today's free memory stand
  in the way, nothing steps down: the engine waits as before (phase `error`, "this model does
  not fit"). With no installed tier below that fits, it waits too. The served tier's file is in
  use (storage cannot delete it). Selecting the tier that already serves for a step moves the
  selection to it and ends the step with no restart. A first-run swap never steps: a target
  the card cannot hold fails at the fit, it is never started, and the starter comes back with
  one load (`upgrade.state` `failed`).
- **Fallback.** `fallback: true` marks a model kept only as a last resort (today gemma4-e2b-q4km);
  `fallback_note` is a one-sentence, game-agnostic description of what to expect. The ladder is
  gemma4-rp-maxxed-q3kxl, gemma4-e4b-rp-q40, then the fallback, which is recommended only when no
  other model can run (§5, "The fallback is the last resort"). The flag is carried into the plan
  (`models[].fallback`, `fallback_note`) and, when the selected local model is a fallback, into
  `GET status` as `model_fallback` and `model_note`. Front ends show it as a warning; the setup page
  does. `tools/gencatalog` sets it from the model spec.
- **Sampler block.** Every model carries `sampler` (named profiles `official`, `rp`,
  `structured`; each a set of sampling values), `sampler_source` (URL and commit the official
  values were read from) and `sampler_match` (name fragments that recognise the model when an
  external endpoint reports it). `tools/gencatalog/samplers.go` holds the spec. `official` is
  the vendor's own values; today all three profiles of all three Gemma 4 models are
  temperature 1.0, top_k 64, top_p 0.95, min_p 0, repeat_penalty 1.0, DRY off. See
  `docs/SAMPLERS.md`.
- A model with an empty `url` is **import-only** (`easy-engine import <path>` copies or
  hard-links a local file after checking its sha256). The mechanism stays for future models
  that are not public. gemma4-rp-maxxed is public on Hugging Face (`cmod/gemma4-rp-maxxed-GGUF`)
  and downloads like any other entry; `plan.Verdict.import_only` is true only for entries
  without a `url`.
- The KV cost follows the architecture: MLA models (bailingmoe3) cache one latent per token
  (`kv_lora_rank + rope dims`, no V) on their full-attention layers; linear-attention/SSM layers
  hold a fixed f32 state per slot that does not grow with ctx; sliding-window layers are capped
  at the window. An architecture the reader does not know gets a conservative estimate (every
  layer full multi-head attention) and a log line, never zero.
- Architecture numbers for the footprint (layers, KV heads, head dims, sliding window,
  expert count, per-layer expert bytes) are **read from the GGUF header** (`gguf`)
  after download, never typed into the catalog. Before download the planner uses the model's
  optional **`footprint`** block (measured on a real card with `tools/fitcheck`, carried from
  `tools/gencatalog/measured.json`; see `docs/FOOTPRINTS.md`): fixed load, MiB per 1k ctx and
  per-slot state for each KV type, scratch, MiB per expert layer, and decode speed against
  `--n-cpu-moe`. A model with no block gets the catalog `size` and a conservative per-token
  guess, flagged `estimated`. Fit is re-checked from the header after download.
- Backend choice (`catalog.Backends`): NVIDIA GPU → `cuda` (with its `cudart` extra, Linux and
  Windows); AMD/Intel GPU on Linux/Windows → `vulkan`; Apple Silicon → `metal`; else `cpu`. If a
  backend fails its smoke test (`llama-server --version` and `--list-devices` show the GPU; a
  Vulkan build whose only device is Mesa's software one, llvmpipe, fails it) the next one is tried
  (cuda → vulkan → cpu). See "Which llama.cpp build" below for CUDA 12/13 and the rungs.

### Which llama.cpp build (catalog version 10)

The engine runs the patched build made for the hive (fork `CharlesMod/llama.cpp`, tag
`b12651-easy1`: upstream b12651 plus CUDA buffers in 256 MiB chunks, CUDA buffers laid by the
driver's VMM (`GGML_CUDA_VMM_BUFFERS`, on by default with one device; the engine pins one card),
the line `easy: slot N released, idle`, a Gemma 4 tool-args grammar fix; patch set e69b6a67), not
stock llama-server.

- **Rungs.** `llama` is an ordered list of releases, each `{tag, repo, patched, assets}`: the
  patched pin first, the stock upstream release second. A build of the stock rung is offered only
  for a platform and accelerator (os, arch, accel) the pin has no build for (today the arm64
  builds: `cpu-linux-arm64`, `vulkan-linux-arm64`, `cuda13-linux-arm64`, `cpu-win-arm64`,
  `vulkan-win-arm64`; the fork builds no arm64). It never stands in for a patched build that
  failed: a failed patched build is followed by the next accelerator's build of the ladder, and a
  failed start is reported. gencatalog writes both rungs from the two releases:
  `go run ./tools/gencatalog -out catalog/catalog.json -version 10 -repo CharlesMod/llama.cpp
  -llama-tag b12651-easy1` (the stock rung is `-stock-repo` default `ggml-org/llama.cpp`,
  `-stock-tag` default the pin's `b<N>`; only the builds the pin lacks are written). A catalog
  before version 10 (one `llamacpp` release) is still read, as a one-rung ladder (patched when its
  tag is `-easyN`).
- **CUDA 12 or 13.** The CUDA 13 build (`cuda13-*`) comes first for Blackwell, the CUDA 12 one for
  every other card; CUDA 13 is never offered when every NVIDIA card is older than Turing (CUDA 13
  dropped Maxwell, Pascal and Volta) or the driver is older than 580; CUDA 12 not when the driver
  is older than 525 (NVIDIA's compatibility floors; the driver is read with
  `nvidia-smi --query-gpu=driver_version`, an unknown driver passes). Linux x64 gets the fork's CUDA
  12.8 build (with its cudart tarball), whatever the card: there is no stock CUDA 13 rung on Linux
  x64. Windows: 12.4 and 13.5, each with its cudart zip.
- **A mixed machine.** An NVIDIA card gets CUDA whatever sits beside it (an AMD card, an Intel
  iGPU); an AMD card beside an Intel iGPU gets Vulkan, and the device pin (`plan.PinDevice`) holds
  the server to the planned adapter, never the iGPU (the probe leaves iGPUs without dedicated
  memory out on Linux).
- **On disk.** A build is installed under `backends/<id>@<tag>`; one installed before the engine
  recorded tags sits under `backends/<id>` (tag unknown). `state.json` `backend` and `build` name
  the build that runs, `next_build` the pin staged beside it. A fetched build carries `.unpacked` (its archive's sha256), a smoke-tested one `.installed`.
- **The pin moves.** When the selected build is not the one the catalog gives this machine for its
  backend (a stock b12651 or an unrecorded install, or an older pin), the engine fetches the pin
  beside it in the background (status `build.next`: downloading, installing, then `staged`):
  downloaded, checked against its sha256 and unpacked, never run, since a model server may hold
  the card. The disk is checked first (too little: `waiting`, the reason said). The next load of
  the model server (a start, a reshape, a yield's end, a wake), with the card free, smoke-tests the
  staged build as the engine's own process (inside `host.Own`, its pid among the engine's for the
  room watch) and runs it. A smoke test that fails there leaves the engine starting on the build it
  has, says `failed` and why, and is not run again for that tag until the pin, the driver or a
  person's or client's word (an API start, attach, wake, host, install, a settings change) changed;
  nothing is downloaded again, what was verified stays. A failed fetch (`waiting`, `error`, and
  status `errors`) is tried again on the network coming back or such a word, never on a request
  at the door alone and never on a clock. A move begins only while no install job, move or remove
  runs, and stages nothing if the selection changed or a remove ran meanwhile. An install job never
  replaces a complete selected build (main's layout included): only the move brings the pin. A
  planned stop with the pin staged saves no context (the context key carries the build, so nothing
  could be laid back), and contexts of another build are let go. Once the new build has served
  ready, the other builds of the same backend id are removed, by exact names only (`backends/<id>`,
  `backends/<id>@<tag>`, and main's old archives in `downloads/`; never past a symlink, never
  another backend's, not while an install, the move or a remove runs; said as `build.pruned`). A
  failure of the old build while a staged one waits is recorded under the old build's keys.
- **Keyed by the build.** What the engine remembers per build carries the full tag: the prompt
  cache key (`id@tag@sha256`), the out-of-memory cap's build (let go when it differs), and the
  book's key (`model|card|id@tag`), so b12651 and b12651-easy1, whose CUDA memory behaves
  differently, never share a measurement. A build installed before tags keeps its old keys.
- **The patched build's lines.** `easy: CUDA VMM buffers on|off (...)` is read from the server's
  stderr into status `build.vmm`. `easy: slot N released, idle` is not consumed: idle is already
  the door's own count reaching zero (a reply's end), which the line could not improve on, for every build.

## 4. Hardware probe

`probe` returns:

```go
type Hardware struct {
    OS, Arch     string
    CPUName      string
    Cores        int
    AVX2         bool
    RAMTotalMiB, RAMAvailMiB int
    DiskFreeMiB  int            // on the data dir's filesystem
    GPUs         []GPU
}
type GPU struct {
    Index       int
    Vendor      string // nvidia | amd | intel | apple | other
    Name        string
    VRAMTotalMiB, VRAMUsedMiB int  // used = all processes, ours included
    UnifiedMemory bool              // Apple: VRAM is RAM
    Source      string             // nvidia-smi | sysfs | sysctl | llama-list-devices | wmi
}
```

Sources: `nvidia-smi --query-gpu` (all OSes); Linux AMD/Intel from
`/sys/class/drm/card*/device/mem_info_vram_{total,used}`; macOS `sysctl hw.memsize` (unified,
usable ≈ 70 % of RAM); Windows non-NVIDIA from WMI `AdapterRAM` / `HardwareInformation.qwMemorySize`
(total only). Once a backend is installed, `llama-server --list-devices` refines free memory
for any vendor. Every source is behind an interface with a fake for tests.

`probe.Others(gpu)` = VRAM used by processes that are not ours (total used minus our
llama-server's own use, from `nvidia-smi --query-compute-apps` when available, else from the
calibration book). It is the fallback now: the reserve is measured from the sensed adapters.

### 4a. The GPU sense and the measured reserve (`probe/gpusense`, `plan/reserve.go`)

`gpusense.Sense` reads every adapter (`Adapter`: vendor, PCI vendor id, dedicated and shared
memory, used memory card-wide, integrated, unified, LUID/PCI/UUID) and, per process, what it
holds on which adapter (`Process`: pid, name, adapter, dedicated and shared MiB, `ours`, `host`).
A source that is missing degrades the picture and never fails it (`Picture.Sources`).

`plan.MeasureReserve(picture, hardware, host, estimate)` turns a picture into the reserve the
room is built from. The engine runs on one adapter **E**: the discrete one with the most
dedicated memory (of equal cards the emptier); an integrated one only when there is no discrete
adapter. Once the engine has started its model server it keeps that card for the session
(`MeasureReserveOn`): the card its server holds memory on, and between a stop and the next start
(a resize, a crash) the card it last started on, unless a larger one appears. Another app filling
the engine's card makes it shrink there; it never moves to an equal card that may belong to
someone else. A discrete card's capacity is the probe's total where the probe lists it (nvidia-smi on
NVIDIA: the figure the footprints and thresholds were set against; DXGI reports 3965 MiB for a
4096 MiB laptop 3050), else the sensed dedicated memory.

| Part | Rule |
|---|---|
| `others` | what processes on E hold that are neither the engine's own nor the host's (basis `processes`), or the card's used memory less ours and the host's where that is more (driver contexts, processes the sense cannot see); with no process listed on E, the card's used memory less ours (`card`); on a pooled adapter (below) the memory the rest of the system holds (`card`) |
| `host` | 0 when the host renders on another adapter (`other-adapter`); else the larger of what the host measurably holds on E and the estimate (`measured` when above it, else `estimate`); an unknown host adapter holds the estimate |
| `margin` | `fit.DefaultMargin`, unchanged |

The host's adapter is where its pid holds memory (the most, dedicated plus shared: on an
integrated adapter a renderer holds shared memory only), else the one the profile or
`POST host` names (matched by vendor id and name, case-insensitive substring either way), else
unknown. The estimate is the profile's `game_reserve_mib`, or `reserve_mib` of the last `POST
host`. The result is a `GPUSummary` (adapters, `engine_adapter` (-1 = CPU), `host_adapter` (-1
unknown), `host_integrated` yes/no/unknown, the `reserve`, `free_for_model_mib` = capacity less
others, host and margin, and `others_note` when others hold 256 MiB or more: "Other apps are
using 1.2 GB of your graphics card"). The note is information only: the engine never stops, asks
to close or touches another program.

An empty picture (nothing could be sensed) is read from `Hardware.GPUs`, the card's used memory
standing for the whole card: the numbers are the fixed reserve's, as before sensing. The
engine's `Config.Sense` hook lets tests inject pictures.

**Held to the planned adapter.** llama-server runs only on E (`plan.PinDevice`, resolved at each
start by `run.resolvePin`); on its own it would split a model across every card it sees. With one
GPU adapter, a Metal or a CPU build, nothing is added (its default is that one device). Otherwise:

- CUDA build, E an NVIDIA card with a UUID (nvidia-smi): `CUDA_VISIBLE_DEVICES=<UUID>`. CUDA then
  shows llama-server that card alone as `CUDA0`, whatever order it would number the cards in
  (CUDA's default order is fastest first, not nvidia-smi's PCI order, so an index is never
  guessed). The pin is confirmed by `--list-devices` under it: exactly one CUDA device, else it is
  dropped and the card matched by name as below (a `--list-devices` that does not answer within
  the smoke test's limit says nothing against the UUID, which then stands). A `CUDA_VISIBLE_DEVICES` already in the engine's
  environment is the user's and is kept.
- Otherwise (Vulkan; CUDA without a UUID): `--device <name>` (`Vulkan1`, `CUDA0`), the device in
  llama-server's own `--list-devices` whose name is E's (case and "(R)"/"(TM)" ignored), else the
  one whose name holds E's whole name ("... (RADV NAVI23)"), else, where E is the machine's only
  adapter of its vendor, the only listed device of that vendor. Two equally good candidates (two
  cards of one model) match none: Vulkan's order is the driver's.
- A pin adds `--split-mode none`, so nothing spills to another card (each flag only where the
  build's `--help` knows it).

What could not be matched or confirmed falls back to llama-server's default devices: a start
never fails for it. `GET status` `device` and `GET plan` `device` say what was given:
`{adapter, pinned, device, env, args, via: "uuid"|"name"|"", note}`. The plan's is a prediction
(from the installed build's accelerator, or the catalog's first backend before an install; a
Vulkan match is only known at launch), replaced by the running server's while one is up.

**Integrated adapters.** An integrated adapter's dedicated figure is never VRAM (WMI reports a
small or capped `AdapterRAM`). Where E is integrated and the build is `vulkan`, the adapter is a
pool of system memory like Apple's: `Room.Unified` (no expert offload), capacity
`min(dedicated + shared, 70 % of RAM)` (shared unknown: half the RAM), others = the pool less the
free RAM. Any other build runs on the CPU. From `Hardware.GPUs` alone an Intel GPU that is not an
Arc, and AMD's plain "Radeon Graphics", count as integrated.

## 5. The ctx balancer (`core/fit`, `plan`)

The footprint, the room and the choice of a shape are the sizing core's `fit` (§13); the
verdicts over the whole catalog, the room built from the probe and the speed scale are the
engine's `plan`. A generic port of the hive's measure-then-try launcher
(`~/hive-mind/serve/bin/hive-mind-llm-run`, design in
`~/.local/share/sprinter-llm/WORKLOG-adaptive-ctx.md`), with the machine-specific constants
replaced by numbers from the GGUF header and a calibration book. Pure functions; the runtime
loop lives in `run`.

### Inputs

- **Wants** (from the game profile): `slots` (HermesVoice: 3), `ctx_per_slot` max and floor
  (HermesVoice: 65536 and 8192), `kv_type` (default `q4_0`; q4_0 KV is allowed).
- **Footprint** of (model, device, backend):
  `vram(shape) = fixed + (per_token + compute_per_token)*ctx_total
  + swa_per_token*min(ctx_per_slot, window+ub)*slots + state*slots + scratch
  - expert_mib*n_cpu_moe + (n_cpu_moe>0 ? moe_sched : 0)`,
  with `per_token` from the header (`2 * n_layer_full * n_kv_head * head_dim * bytes(kv_type)`),
  `fixed` = dense + expert tensor bytes + the output/vocab buffers + the CUDA context, minus
  host-resident tensors (`per_layer_token_embd.weight` of Gemma E-series stays in RAM), and the
  compute/mask/dequant terms measured once on llama.cpp b12680 (`fit` constants; none per
  model). Header seeds land within 4 % or 120 MiB of the sweep on all four catalog models.
  The seed is then **calibrated** after every healthy load from
  the measured use (as `calibrate()` does in the hive launcher: accept a new `fixed` only
  within ±15 % of the seed). The book is keyed by model sha256 + GPU name + backend id and llama.cpp tag (`id@tag`) and
  kept in `state.json`.
- **Every figure marked.** `Footprint.Basis` (`basis`) and `Shape.VRAMBasis` (`vram_basis`) are
  `fit.Predicted` or `fit.Measured` (`""` only on a value built by hand). A footprint is measured
  where the book took a reading for its key (its `fixed` is that reading; the per-token terms stay
  predicted); `Seed`, `SeedFromBytes` and `plan.FromMeasured` say predicted. Every shape the
  planner makes is predicted; `Book.Mark(key, shape)` makes it measured, with `vram_mib` the
  reading, only where the book holds a reading of that very shape (`Calib.Readings`, one per
  shape that came up, kept whether or not the reading was taken). A card shape's `RAMMiB` (its
  expert layers in RAM) is always predicted; a CPU-only shape's is measured where the book holds
  its host reading (`Calib.HostReadings`).
  A reading passed over never marks the footprint or any other shape.
- **The machine's memory of a start (since core v0.1.4).** Every shape says the host memory its
  start holds in all, as the process's resident memory: `Shape.HostMiB` (`host_mib`) with
  `Shape.HostBasis` (`host_basis`). For a card shape it is predicted as the footprint's
  `HostBaseMiB` (`host_base_mib`: the runtime, the input embedding and the host compute buffers,
  from the set's constant `host_base_mib`, 1,536 MiB in `DefaultConsts`: a card start's resident
  memory less its host-resident weights was read as 974 to 1,520 MiB over the four Gemma E-series
  sweeps, every row) plus the weights the header keeps in RAM (`HostMiB`, Gemma E-series
  `per_layer_token_embd`) plus the expert layers moved (`RAMMiB`); it does not grow with ctx. For a
  CPU-only shape it is `RAMMiB`. Where the footprint cannot tell it (`HostBaseMiB` 0: a footprint
  built by hand, from measured numbers or by an older build, `SeedFromBytes`, which reads no
  header, or a set lacking the constant) the shape says `host_basis` `unknown` and `host_mib` 0,
  never a figure; `Footprint.HostNeed(shape)` answers the same figure, or `ErrNoHostFigure`.
  `Book.CalibrateCard(key, seed, shape, vramMiB, hostMiB, now)` is `Calibrate` with the process's
  resident memory read beside the card memory: the card memory is held to the seed exactly as
  before and the host memory is kept in the same reading (`Reading.HostMiB`, `host_mib`), and
  `Book.Mark` then gives that shape's `HostMiB` as measured (a CPU-only shape's host reading marks
  it too). A card reading without it leaves the prediction. `host_base_mib` has no part in the
  card's fixed part, so the book neither keeps it beside the fixed part nor keys on it: a book an
  older build wrote still overlays its fixed part, and an older build reads a newer book's set as
  the set it knows. The engine's `/status`
  `shape.vram_basis` says whether `vram_mib` was read from the running server.
- **A calibration says what it did.** `Book.Calibrate` answers a `fit.Calibration`: its
  `Verdict` (`taken`; `outside_gate`, more than ±15 % off the seed's `fixed`; `no_seed`;
  `no_reading`; `cpu_only`; `not_seed`), the `fixed` the reading implies and the seed's, so
  `OffMiB()` says by how much the seed is off. The gate and the seed's `fixed` are always the
  header's seed (`Basis` predicted), never the book's last reading: the engine keeps the seed
  beside the overlaid footprint and hands Calibrate the seed, so readings each near the last
  cannot walk the book away from it. A footprint marked measured offered as the seed is
  `not_seed` and moves nothing. Only `taken` moves the book's `fixed`; otherwise the footprint
  stays predicted. The engine logs the whole answer.
- **A cap names what lifts it.** `Book.LowerCap(key, fp, failed, ctxTotal, room, build, now)`
  keeps beside every cap it lowers a `CapLift` (`cap_lift`): the room it was measured under
  (`Room.Budget`: the card's, or the host's with no GPU), the miss (the memory `fp` predicts the failed
  shape needs above the same slots at `ctxTotal`; 0 where that cannot be told, as when the walk
  kept the ctx and moved experts) and the engine build. `Book.LetGo(key, room, build)`, with the
  room and build as they stand now, lets the cap go where the room is larger by more than the
  miss (any larger room for a miss of 0; of the same kind) or the build is another, and otherwise
  answers that it stands and its `CapLift`, never a standing cap with nothing named. A cap
  lowered beneath another keeps the one it moved from, with that one's own lift, in `cap_under`
  (`HeldCap`); letting the lower cap go falls back to the latest earlier cap that still stands,
  so a room back where an earlier out-of-memory was measured keeps that earlier cap. With
  neither changed the cap stands exactly as before. A cap from an older book recorded nothing:
  the first `LetGo` (or `LowerCap`) records the room and build as they stand then, and it lifts
  from there as any other. In the engine the build is the backend id, the llama.cpp tag and the sha256 of the
  archive installed under it (`backends/<id>@<tag>/.installed`; `id@tag@sha`), so a new llama.cpp build under the
  same id lets the cap go; another id is another book key altogether. The engine calls `LetGo`
  each time it reads the book for a start.
- **An unknown cache type is refused.** `Seed`, `SeedFromBytes`, `KVBytes`, and every entry that
  prices memory (`Choose`, `MissRetry`, `StepDown`, `Grow`, `Book.Calibrate`, `Book.LowerCap`,
  `Footprint.Need`, `Footprint.Largest`), for a type in the wants, in `KVTypes` or in a shape,
  answer `*fit.KVTypeError` naming it, never a price, and the book records nothing. The
  nine names (`f32`, `f16`, `bf16`, `q8_0`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`) price as
  before and `""` is still `q4_0`. `plan`'s verdict for such a type says so.
- **state.json both ways.** `Calib` keeps `fixed_mib`, `cap_ctx_total`, `loads` and `updated`
  with their meaning and adds `cap_lift`, `cap_under` and `readings`, each left out when empty
  (`cap_ctx_total` is always the cap that stands, the lowest). An older build
  reads a newer book with all four fields it knew (Go's decoder skips the rest) and drops the new
  ones when it writes; a newer build reads an older book as a book with no lifts and no
  readings: its caps stand as before until the first `LetGo` records what lifts them. `TestBookJSONBothWays` holds both.
  The fields of core v0.1.4 (`Shape.host_mib`, `host_basis`, `Room.card_host_free`,
  `Reading.host_mib`, `Footprint.host_base_mib`) are left out when zero, an older build skips
  them, and a newer build reads their absence as the behaviour of before (`TestHostMemJSONBothWays`).
- **Room:** `budget = VRAM total - others - game_reserve - margin`, with `others` and
  `game_reserve` (the host's share) measured from the sensed adapters (§4a): the game's own use
  once it is measured, its estimate (the profile's `game_reserve_mib`; HermesVoice sends 1500)
  before, and none when it renders on another adapter than the engine's. `margin` = max(512 MiB,
  4 % of total), never a guessed fit target.

### Wants and room another program may state

Every field below is new to the core's `Wants` or `Room`, and its zero value is the engine's own
behaviour exactly, so `plan`, `run` and the game profile set none of them.

| Field | Set | Unset |
|---|---|---|
| `Wants.Whole` (`whole`) | No layer and no expert off the card, for a dense or a MoE model, from `Choose`, `MissRetry`, `StepDown` and `Grow` (step 3 below is skipped; `StepDown` never moves an expert). Where only a split shape fits, `ErrTooBig`. | Expert layers may move to RAM. |
| `Wants.KVTypes` (`kv_types`) | The cache types, best first. The shape takes the first type at which the most ctx asked (`CtxMax`, under the book's cap) fits at `Slots`; where no type does, the last type with the most ctx that fits down to the floor, then fewer slots; under that, `ErrTooBig`. Each type is priced from the footprint, never tried: the footprint names the type it was seeded at (`Footprint.KVType`), the per-token cache scales with the bytes per element, and the f16 expansion of the widest layer (`DequantPerTokenMiB`) is added for a quantised type and taken off for f16, bf16 and f32. `Seed` and `SeedFromBytes` fill both; a footprint built from measured numbers (`plan.FromMeasured`) names no type and gets `ErrNoKVType`. `KVType` is then not read. The list is taken once wherever it is read: each type at its first place, repeats and empty names dropped, and (since core v0.1.4) ordered best first by the core itself, most bytes per element first (`Wants.Norm` shows the order; types of equal bytes, as f16 and bf16, keep the caller's order). A smaller type before a larger one could only ever have been planned as the last type's fallback, with less context than the smaller type gives, so the order passed never changes what fits, the walk down, or the least shape a refusal is worded from (the last type's floor). `MissRetry` and `StepDown` keep the type the current shape runs at (a shape's `""` is q4_0, whatever type the wants name) until its floor at `MinSlots`; there `StepDown` (and `MissRetry` through it) moves to the next type in the list with the largest shape that needs no more memory than the one that failed (predicted at its type) and is never better than it (no more slots, no more ctx per slot); where none fits there, on to the type after; `ErrTooBig` comes only at the last type's floor. A shape at a type the list does not hold (or, with no list, at any type) keeps it and ends at `ErrTooBig` at its floor. Only `Choose` plans the room afresh. The walk down is monotone: each step lowers the pair (types left after the shape's, then its predicted memory), so it ends for any list; the one step that does not lower it is the first expert layer moved to RAM where a layer weighs less than `MoESchedMiB`, and those moves are bounded by the expert layers. A type is chosen before ctx and experts: without `Whole`, f16 with experts in RAM comes before q4_0 with none. `Grow` counts a better type alone as no growth. | `KVType` alone: the footprint priced at it (or at a shape's own type), as given where it names no type. |
| `Wants.CtxFromModel` (`ctx_from_model`) | `CtxMax` is the model's trained context (`Footprint.CtxTrain`, which `Seed` takes from the header's `<arch>.context_length`, `gguf.Info.CtxTrain`); a floor above it, or unset, comes down to min(8192, trained). Where the header did not say it, `ErrNoCtxTrain`, never a guess; where it says less than the least context per slot (1,024), `ErrCtxTrainUnder`, never a shape above it. `MissRetry` and `StepDown` answer exactly as with that number named in `CtxMax`. | `CtxMax` as given. |
| `Room.Reserve` (`reserve`) | Host RAM kept for the OS when experts move or a run is CPU-only; negative is none. | `OSReserveMiB` (2048). |
| `Room.Margin` (`margin`) | The VRAM left free; negative is none. | 0 is `DefaultMargin(VRAMTotal)` (the engine's callers fill that value already). |
| `Room.LargestBlock` (`largest_block`) | The largest single allocation the card can still give. A shape's largest allocation (`Footprint.Largest`) must fit inside it: the greatest of the weights left on the card (`Footprint.WeightsMiB` less the moved expert layers), the cache (full-attention, sliding-window and recurrent state counted as one) and the compute buffers (the fixed load beyond the weights plus the scratch that grows with ctx), each taken as one allocation. A footprint with no `WeightsMiB` (measured numbers) gets `ErrNoWeights` where the block is positive. Negative: the card was read and has no free block left, so no shape on it is held and the answer is `ErrTooBig`, weights known or not. Not read for a CPU-only room. | Not bounded. |
| `Room.NoGPU` with `RAMAvail` | A pool that is one node of a processor's memory: no card, whatever the VRAM fields say; CPU-only shapes sized against `RAMAvail`, that node's free memory, less the reserve, and nothing else. | (as before) |
| `Room.CardHostFree` (`card_host_free`, since core v0.1.4) | The machine's free memory a start on the card is held to: the start's host memory (`Shape.HostMiB`, which does not grow with ctx) must fit in it less the reserve (`Reserve`). Where no card start of the model fits (the least, with no expert layer moved), `Choose`, `MissRetry`, `StepDown` and `Grow` answer `*fit.HostShortError` with `NeedMiB` (that least start's host memory), `FreeMiB` (the free figure less the reserve) and `Basis` (predicted); it is `ErrTooBig` to `errors.Is`, so a caller that only tells too big from the rest reads it as before, and `errors.As` gives the figures. Expert layers move to RAM only as far as it holds them, in `Choose` and in the walk down. Negative: read, and none free. A footprint that cannot tell the host memory of a card start (`HostBaseMiB` 0) is `ErrNoHostFigure`, never planned as zero. Not read for a CPU-only room (`RAMAvail` is the pool there). | Not read: a card start is held to nothing but `RAMAvail` for its expert layers, as before. |
| `Room.Unread` (`unread`) | The json names of fields the caller could not read. None is taken as zero: every entry that reads a room (`Choose`, `MissRetry`, `StepDown`, `Grow`, `Book.LowerCap`, `Book.LetGo`, `Room.Budget`) answers `*UnreadError` naming the first; no cap is let go or lowered and no room is recorded. | Every field was read. |

### Decode speed floor, and CPU-only

`Wants.MinTPS` (default 20 tok/s per stream; negative turns it off) outranks max ctx: step 3
stops adding CPU layers at the first `n` whose estimated speed falls under the floor, using the
model's measured `tps_by_n_cpu_moe` (time per token is linear in `n`, so 1/tps is
interpolated). The measured speed is multiplied by `SpeedScale`: 1 on the exact GPU the numbers
were measured on, 0.5 on any other hardware (decode is memory-bandwidth bound and common 8-12 GB
cards have 30-45 % of the reference card's bandwidth, so halving is conservative). If no shape
reaches the floor, the planner still returns the best shape, flagged slow ("tight ... (slow)",
`est_decode_tps` in the verdict) rather than refusing a model that fits.

With no usable GPU the recommendation is the smallest model whose estimated RAM is at most 60 %
of `RAMAvail - 2 GiB`, else the smallest that fits at all. Nothing was measured on a CPU, so
the verdict is marked `estimated` and says so.

**The fallback is the last resort** (`plan.lastResort`). A `fallback` model (today the E2B,
Flashlight) is a candidate for `recommended` only when no other model is: on a card, the best
non-fallback, non-opt-in model that fits "yes", else the best that fits "tight", and only with
neither the fallback ("yes", then "tight"); with no usable GPU, the CPU-only pick above among
the non-fallback models first (so the E4B wins whenever it fits, comfortable or not), and the
fallback only when nothing else fits. The fallback stays listed and installable everywhere.

### Choosing a shape, in order

1. Everything on GPU at `slots × ctx_per_slot.max`.
2. Shrink ctx per slot (1024-aligned) down to `max / 2`.
3. **MoE only:** move experts to RAM (`--n-cpu-moe N`), as many layers as host RAM allows
   (`RAMAvail - 2 GiB` OS reserve), then grow ctx back as far as it fits at that layer count.
   Take the shape with the most ctx; at equal ctx, the one with fewer CPU layers. (A few
   expert layers in RAM cost little speed; giving up long memory costs more.)
4. Shrink ctx per slot toward the floor (keeping the CPU-MoE layers). Dense models reach this
   step straight from step 2.
5. Fewer slots (3 → 2 → 1), each tried again from step 1. The profile says which slots are
   the minimum (HermesVoice: 1).
6. Still no fit: **dense models are never split** between card and RAM; the planner answers
   "too big" (`ErrTooBig`) and the setup UI recommends the next model down. A dense model on a
   machine with no usable GPU runs CPU-only, which is allowed.

The shape becomes explicit flags. **`--fit` is never used** (it turns itself off with `-ngl`
and gives no measured answer):

```
llama-server -m <gguf> -ngl 999 [--n-cpu-moe N] -np <slots> -c <slots*ctx_per_slot>
  --no-kv-unified  (independent slots: each one gets exactly ctx_per_slot)
  -ctk <kv> -ctv <kv> -fa on --jinja --reasoning off --cache-reuse 256 --metrics --slots
  --temp T --top-k K --top-p P --min-p M --repeat-penalty R --dry-multiplier 0   (the model's `official` sampler profile)
  --slot-save-path <data>/contexts   (the prompt cache across a restart, §5 below)
  --host 127.0.0.1 --port <free port> -a <model id>
```

The web UI stays on (llama-server's default; `--no-webui` is not passed): the engine serves it at
`/chat/` (see "The chat window" below), so llama-server's random port never shows.

The sampler flags make the model's official values llama-server's own defaults, so a client that
sends no sampler fields still samples as the vendor intends (`docs/SAMPLERS.md`). They are
checked against the release's help like every other flag (all exist in b12680: `common/arg.cpp`).

(Check every flag against `llama-server --help` of the pinned release at startup; drop and log
flags it does not know, as the hive's `ReadEngineHelp()` does.)

### At run time (`run`)

- **Measured retry:** on a load OOM, parse the failed buffer size from the log and retry once
  at the measured smaller shape (`miss_retry()`); a second failure steps the shape down one
  notch and is recorded.
- **Runtime OOM** lowers the stored cap for that (model, device) so the next start is smaller.
  A crash whose last words carry a memory error restarts one step smaller at once.
- **No clock in any of this** (docs/EVENTS.md): every reaction below is woken by the event that
  means it (`run/mind.go`), and the only timers are a watchdog for silence, a network timeout or
  a named moment, each named where it stands.
- **Crashes without a memory error** restart at once, through the normal choose path, which
  reads the room again. A failed start is tried again once at once. After 2 quick crashes in a
  row at one shape (each within 2 min of its start, `CrashQuick`/`CrashSteps`) it is read as a
  memory problem, most often another app taking the card: one step down at once, and the cap is
  lowered too when the sensed reserve grew past 64 MiB since the shape was chosen. A run that
  stays up 2 min (a named moment) clears the record. With nothing left to try (a second failed
  start in a row, the model does not fit, out of memory at every shape, crashes at the smallest
  shape) the engine waits for an **event** that could change the outcome: the room on the card
  moved, a request at the door, a client's word (`start`, `install`, a nudge). It never gives up
  for good and never retries on a cadence. The status `recovery` block has the failures and
  crashes in a row, the last error, `waiting_for` (what the next try waits for; empty while it
  tries at once), `retry_in_s` (always 0, kept for old clients), whether the crashes were read
  as memory, and the ctx of the shape the engine will not return to.
- **A shape that failed is a ceiling:** a shape that ran out of memory, crashed into a step
  down, or failed a grow is not chosen or grown back to for 10 min (`GrowBackoff`); a grow may
  then try once more. A shape shrunk away from because another app took room is not one: it
  grows back once the app lets go.
- **Out-of-memory the server survives:** llama-server can fail a request on a CUDA OOM and keep
  running. Its output is watched line by line (the memory patterns of `ParseOOM`, pinned host
  memory fallbacks excepted); out-of-memory lines said during two different requests within
  60 s (`OOMHold`, a named moment) step the model down, record the cap, at idle or as below.
  The lines of one failed request are let go.
- **Idle is exact:** every request reaches the model server through the engine's door (it
  listens on a private loopback port), so idle is the door's count of requests in flight
  reaching zero. `/slots` is not read for this (it is read once before a planned stop, to know
  which slots to save: "The prompt cache across a restart" below). Tok/s is read once from `/metrics` each time the count
  returns to zero.
- **Make room for the game:** the room is read when needed, never on a clock of its own: at each
  start, before a grow, on an out-of-memory line, when the holders of the engine's card device
  change (Linux, inotify: an open is a word that somebody touched the card, and the room is read
  only when the set of other programs holding its nodes differs from the last look; one scan of
  `/proc/*/fd` per word serves the room and the yield), when a program the last picture listed ends, and on the host's word
  (`POST host`). Another living program's memory gives no event (NVIDIA, and everything on
  Windows and macOS), so while a model is loaded, and while a start that failed for want of room
  waits for the room to change, the reading is also kept by the one watchdog for it, inside
  package `host` (`host.Level`, every 5 s, `RoomEvery`; the status `gpu` is said with each
  reading). The engine's own reads open the card too (nvidia-smi); `host.Own` keeps every device
  watch from hearing them, so a read never causes another. If the model's memory (`Footprint.Need` of the running shape) no longer
  fits the budget the reserve leaves (the player loaded a heavy save), restart at a smaller
  shape **when idle**; never in the middle of a line. The card's used memory is not the test: it
  holds the engine's own server too, and read alone would shrink the model with nothing else on
  the card. A change reshapes the model only when **held**: two consecutive reads of the reserve
  (others plus host) agree within 64 MiB. Other programs are never touched. The game's room wins
  over ctx.
  The shrink goes in one restart to the shape the room fits (`fit.Choose`), or, when no shape
  fits, to the smallest the steps reach; the model keeps serving there until the room comes back.
  A held shortfall of 256 MiB or more (`DrainMinMiB`) does not wait for a pause: llama-server
  holds its memory from load, so the reply in flight is safe to finish. New requests get
  "Loading model" (phase `resizing`) while it does, and the model shrinks once the count of
  requests in flight is zero, or after 2 min (`DrainMax`, a named moment) if a reply outlasts
  that. A smaller shortfall, or one read once, waits for idle. If the smaller shape fails to
  load on an out-of-memory, the engine records it and steps down from there, never back to the
  shape that no longer fit.
- **Grow back when idle:** idle for 60 s (`GrowIdle`, a named moment armed when the count
  reaches zero and cancelled when a request begins), the room is read again; if it agrees with
  the held reading and allows at least one more step (ctx or one fewer CPU-MoE layer), restart
  at the bigger shape. If that fails, restart at the old shape.
- **Ready by its own lines:** llama-server's output comes line by line; after the line that
  says it listens (`listening on http://…`, fixtures in `run/testdata/ready-*.txt`) `/health` is
  asked once per following line until it answers 200. The lines are the first word, not the
  only one: when the server has said nothing for 2 s (`ReadyQuiet`, a watchdog for silence)
  `/health` is asked once, listening line or not, and again only after a new line. A load that
  says nothing for 15 min (`LoadTimeout`) is asked once more and then fails. A build that prints
  forever and never answers 200 is not bounded.
- **A hung server:** no periodic `/health`. While requests are in flight a watchdog for silence
  (`HangSilence`, 2 min) is put back by every line of the server, every request beginning or
  ending and every byte of a response (the proxy calls `Heard`). When it fires, `/health` is
  asked once (3 s network timeout): no answer restarts the server, an answer puts it back. A
  quiet server with nothing in flight is never asked anything.
- **The prompt cache across a restart** (`run/context.go`; docs/EVENTS.md §6): below.

### The prompt cache across a restart (`run/context.go`)

Every planned restart of llama-server (a shrink, a grow, a yield and its return, a sleep and a
wake, a stop and a start, the restart after an install) would throw away what each slot has
read, so the next reply pays the whole prefill again: for a game's long conversation, tens of
thousands of tokens (a 32 000-token prompt is 65 s of prefill for Gemma 4 26B-A4B on a P40). The
engine keeps it instead, with llama-server's own slot files.

- **The files.** Every launch passes `--slot-save-path <data>/contexts` (the directory is the
  engine's, made 0700; the flag is the engine's own, like `-c`; a build whose `--help` does not
  list it runs without, and that is logged). Saved slots are `contexts/slot<i>-<pid>-<n>.bin`;
  `contexts/index.json` (0600) records each one: `{slot, tokens, bytes, client, file, key}`.
  The engine removes only the files its index names, by those names, in that directory.
- **Save** before a planned stop of a healthy server, after the drain (no request in flight) and
  before the stop: `GET /slots` says which slots hold tokens (`n_prompt_tokens` plus the
  `next_token[].n_decoded` of a slot that has served; a slot laid back and not used since is
  known from the lay-back, since llama-server says nothing of it), then one
  `POST /slots/{i}?action=save {"filename": ...}` per slot (answer
  `{id_slot, filename, n_saved, n_written, timings: {save_ms}}`; a slot with `n_saved` 0 is not
  kept). Each call has a network timeout of 2 min (`SlotTimeout`). A save that fails is said and
  that slot is not kept; the stop goes on. Nothing is saved: with `keep_context` off; with
  requests still in flight (a drain cut short); before a crash restart, a hang restart, an
  out-of-memory shrink (memory failures keep nothing), a KV-type change (nothing would fit) or a
  model swap (the key differs).
- **The context key.** A saved slot is laid back only into a server with the same model file
  (its catalog sha256; for a custom file, which has none, its path, size and time), the same
  llama build (the backend id and the sha256 of the archive installed under it, which names its
  patch set with it) and the same cache type.
- **Lay back** on the next start: once the new server answers `/health` and before the phase
  says `ready` (the door holds requests until then, §7e), each kept slot whose key matches, whose
  slot id exists in the new shape and whose tokens fit the new context per slot is restored
  (`POST /slots/{i}?action=restore`, answer `{id_slot, filename, n_restored, n_read, timings:
  {restore_ms}}`) and its file removed. Everything else is let go. A restore that fails is said,
  its file removed, and the start goes on: a lay-back never fails a start. The next request that
  continues the slot's conversation reads its prompt from the cache (`timings.cache_n` large,
  `prompt_n` only the new turn). A client keeps its slot by sending `id_slot` (llama-server's
  field); until clients own slots (a later slice) the engine keys a saved context by slot id.
- **Letting go.** Saved files are removed when laid back; when the model, the build or the KV
  type changes (on a KV change at once, else at the next start, by the key); at engine start
  (what a dead engine left: a sleep followed by an engine restart is not laid back) and at engine
  end; and when the disk runs short (an install's disk check lets kept contexts go first and
  reads the free space again).
- **Disk.** Before saving, the size is estimated from the sizing core's footprint of the running
  model (full-attention cache per token, the sliding-window cache up to its window three times
  over for the window and its checkpoints, the recurrent state, a header) and checked against
  the free disk the way installs are (`plan.NeedOf` with the logs' room, the host's files and the
  headroom); with too little, nothing is saved and that is said.
- **A yield is urgent.** A game waits for the card while a yield saves. Measured on poe (P40,
  llama.cpp b12680, q4_0 cache, 2 slots): Gemma 4 E2B, 3 043 tokens: save 17 ms, 10.6 MB, restore
  10 ms; 31 961 tokens: save 71 ms, 61 MB, restore 64 ms. Gemma 4 26B-A4B IQ3_S, 31 961 tokens:
  save 0.32 s, 362 MB, restore 0.24 s; the next turn read 31 949 tokens from the cache and
  processed 21 (379 ms) where a cold server processed 31 970 (65.6 s). Through the engine itself (E2B,
  1 x 32768): a 31 961-token slot saved at a sleep (61 MB), and the next turn, sent while asleep,
  was held, woke the model, and was answered 2.8 s later with 31 949 tokens from the cache and 21
  processed (the cold prompt took 18.7 s). A restore into a slot smaller than the saved tokens is
  refused by llama-server (400), which is why the engine checks the context per slot first. About 1.1 GB/s, so a full
  context of the models the engine serves is well under 2 s. **The rule:** `keep_context`
  `on` (default) saves on a yield only when the estimate is at most `YieldSaveMaxMiB` (2048 MiB,
  about 2 s at that speed; a dense model at f16 cache and a long context can pass it), else
  nothing is saved and that is said; `always` saves on a yield whatever the size; `off` never
  saves. Every other planned stop saves with `on` or `always`.
- **Said.** The happening `context` (keyed by slot; `all` for a save that did not happen for any
  slot): `{what: saved|laid_back|let_go|not_saved, slot, tokens, bytes, why}`. Status
  `contexts` is `{slots, tokens, bytes, kept: [{slot, tokens, bytes, client?}]}` while slots are
  kept (between a stop and the next start), null otherwise. The engine log has a `context:` line
  for each.

## 6. Lifetime

Two ways to run the engine. **`easy-engine app`** (and `easy-engine` with no command) is the
app: a tray icon, always open, no parent pid, waiting for a connection (§6a). **`serve`** is for
a service or a shell: no tray, and it ends with its parents, as described next.

- The game launches `easy-engine serve --profile <file> --parent-pid <game pid>`, with no
  console window (Windows `CREATE_NO_WINDOW`).
- **Single instance:** `engine.lock` plus a status probe on the port. If an engine is already
  running and healthy, a new `serve` adds the new parent pid to its watch list over the
  control API (`POST /engine/v1/attach`) and exits 0.
- **The lock is made whole:** the engine writes `engine.lock` to a unique temp file, syncs it and
  links it into place, so a reader never sees a half-written lock and two serves started
  together make one engine (the loser attaches). Where links are not supported it falls back to
  an exclusive create; a brand-new empty lock (under 5 s old) counts as held, not stale. The lock
  holds `host` (the listen host); a lock without it is a loopback engine. Child pids are
  rewritten with a unique temp and rename.
- **Stopping:** once the engine begins to shut down (a shutdown request, the last parent gone, a
  signal) its status phase is `stopping` and `POST attach` answers **409** `engine is stopping`.
  A new `serve` that gets the 409 waits for the dying engine to release the lock, then runs its
  own engine instead of attaching to it.
- **state.json** (and `settings.json`, §2) is written under one process-wide mutex through a unique temp file, fsync,
  rename and a directory sync. A corrupt `state.json` is renamed to
  `state.json.corrupt-<unix time>`, the engine starts from defaults, and status `errors[]` says
  so.
- **Parent watch:** the engine follows each parent pid by the system's word of its end (Linux a
  pidfd, macOS kqueue, Windows a wait on the process handle: `host.ExitsOf`, with the pid's start
  time so pid reuse cannot fool it). When the last parent is gone, it stops its children and
  exits within 5 s. `--linger <dur>` (setting, default 0) keeps the model loaded for a while for
  quick restarts.
- **Children die with the engine:** Linux `Pdeathsig=SIGKILL` + own process group; Windows a
  Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`; macOS process group + the children
  are killed on every exit path, and the engine's own crash is covered by the next start
  killing any stale pids recorded in `engine.lock`.
- **Pause:** `POST /engine/v1/sleep` unloads the model (stops llama-server) and keeps the
  agent; any proxied model request or `POST /engine/v1/wake` starts it again. The game may
  call it in the main menu. Not used by default.
- Stop order: agent first (SIGTERM, 5 s, then kill: a watchdog for silence on the child), then
  llama-server.
- **The agent** restarts once at once when it exits; a second quick exit in a row (a run
  shorter than 60 s, `AgentStable`) waits for a request or a client's word, and its status says
  so (phase `error`).
- **Downloads** have no retry cadence: an attempt that moved bytes resumes at once; one that
  moved none waits for the network coming back (`host.NetworkChanges`), a client's `install`
  or nudge, or the engine's next start. The 60 s stall timer on a stream is a watchdog for
  silence. Progress is said per 1 % of the file (or per 4 MiB with no size), never per time.

### 6a. The app (`easy-engine app`)

- **What it is.** The engine plus a tray icon (Linux StatusNotifierItem, Windows notification
  area, macOS menu bar; no cgo). The listener, linger and yield options come from
  `settings.json` (§2); the `serve` flags are accepted as overrides for one run. Same lock, same
  rule as `serve`: if an engine already runs, a second `app` says so and exits 0 (it attaches no
  pid, and shows no second icon).
- **The menu, for now exactly four items:** *Chat* (above Settings; on only while a model is ready,
  see "The chat window" below), *Launch at Startup* (a checkmark; the user's own
  startup entry named `easy-engine`: Windows `HKCU\...\Run`, macOS a LaunchAgent, Linux
  `~/.config/autostart/easy-engine.desktop`; it starts `<this program> app`, on Windows the
  windowless `easy-enginew.exe` beside it when there is one; the checkmark shows what the system
  says after the click), *Settings* (opens `<data dir>/settings.json` in the system's editor) and
  *Quit* (drains and stops everything as SIGTERM does, exit 0). The tooltip says the phase and
  the model.
- **No tray here.** With no tray to show in (no desktop session, GNOME without the AppIndicator
  extension, a Windows service session) the app logs one line and runs without an icon.
  `easy-engine` typed with no command in a terminal, where no tray can be shown, prints the usage
  instead of starting a server.
- **Waits for the connection.** The app starts with no model loaded (phase `sleeping`, "Model
  not loaded"; `Config.StartSleeping`); the agent and an unfinished upgrade wait for the first
  load too. The app never ends because clients are gone. A client that
  attaches a pid (`POST attach`) wakes the model if it sleeps; a request at the proxy wakes it too,
  and the door **holds** that request until the model is up (§7e). The clients are the attached
  pids, the attachments (§7g: held or by pid) and the requests at the door (in flight or held: a
  remote client that never attached counts by its requests). Once a client has come, when no pid
  and no attachment stands and the door is empty while a model is up, the keeper's `linger_s` (a named moment) runs; a request or an
  attach cancels it, and it runs again from the door's next empty. When it ends with nobody
  there the model is **put to sleep** (`POST sleep`'s unload); the engine stays. A model a person
  started with no client ever coming is not put to sleep by the linger. The tooltip follows the
  hub (phase, model, door), not only clicks, and says `(N waiting)` while the door holds N
  requests. `serve --parent-pid` keeps its behaviour: the engine ends with its parents.

#### Launch commands per OS (for a mod's launcher)

| OS | Command | Notes |
|---|---|---|
| Linux | `easy-engine app` | start it detached (own session, stdio to /dev/null) |
| Windows | `easy-enginew.exe app` | the windowless build: no console window |
| macOS | `easy-engine app` | start it detached |

A mod does not pass a parent pid any more. It finds a running engine (`engine.lock` in the data
dir, then `GET /engine/v1/status`), attaches its own pid (`POST /engine/v1/attach`) and uses it,
and only when none runs on its own machine starts the app detached with the command above and
then attaches (docs/EVENTS.md §8).

### 6b. Yield the card to other programs (`serve --yield-to-games`)

Off by default. With it, the engine unloads llama-server when another program wants the card
and loads again once it is free. The engine process, its port and its API stay up.

**Signals, earliest first** (code: `yield/`, `run/yield.go`):

1. **A launcher starts a game.** Linux: a process whose argv is Steam's
   `reaper SteamLaunch AppId=<n> -- …` (it lives exactly as long as the game), a Lutris
   `lutris-wrapper: <title>`, or `umu-run` (Heroic and Lutris run Proton games through it;
   best effort). argv gives no event: it is read through the yield's `host.Level` watchdog
   (every `--yield-every`, default 1 s, only while the watch is armed), and only new or young
   pids have their cmdline read. Windows: `HKCU\Software\Valve\Steam\RunningAppID` (0 = none),
   re-read on the registry's change notification. macOS: `RunningAppID` in
   `~/Library/Application Support/Steam/registry.vdf`, re-read when the file changes. A launch usually comes
   before the game opens the GPU at all. gamescope is not a signal: in Steam's game mode it is
   the compositor and always runs (games inside it still show as `reaper`).
2. **A process opens the engine's GPU** (Linux only). A program must open `/dev/nvidia<minor>`
   or a DRM card/render node before it can allocate. inotify `IN_OPEN` on the engine adapter's
   nodes (`host.Opens`, no time cap) wakes the watch at once. Which nodes: one rule
   (`adapterNodes` in `yield/os_linux.go`), used by the yield's watch and the room's alike and by
   the scan's `wanted` set. An NVIDIA adapter (its DRM nodes driven by `nvidia`) is listened to at
   `/dev/nvidia<minor>` alone, minor from `/proc/driver/nvidia/gpus/<pci>/information` (every
   `/dev/nvidia<n>` if it cannot be read), never its dri nodes: the NVIDIA driver's clients open
   the device node, so a holder of a dri node that does not hold `/dev/nvidia<minor>` is no holder
   of the card, and a compositor's opens of the dri nodes would only add words. Another driver
   (amdgpu, i915, xe) has its own dri nodes only. While the adapter's PCI address is not known,
   every GPU node. The set is made again by the event that says the adapter is known or changed
   (the room's reading, `measure`: the room renews its listening and the yield loop wakes on
   `room`), never by a clock. The watch then wakes at once; it then reads `/proc/<pid>/fd` of every
   process (about 25 ms) and reads the level again at once. A process that still holds one of the engine adapter's nodes (mapped from
   `/proc/driver/nvidia/gpus/<pci>/information` and `/sys/class/drm/<node>/device`) and is not
   in the baseline is a newcomer. `nvidiactl` alone does not count (NVML monitors open it),
   and neither does a short-lived probe such as `nvidia-smi` that is gone by the second look.
3. **Memory on the engine's adapter.** No event exists for it, so it is read by the yield's
   `host.Level` watchdog, every second while the watch is armed (a model loaded or the card
   yielded) (Linux `nvidia-smi -q -d PIDS`, about
   0.02 s of CPU against 0.11 s for `-q -x`; AMD/Intel fdinfo; Windows the GPU Process Memory
   counters): a process not in the baseline holding any memory there (`--yield-min-mib`,
   default 0), or a baseline process grown by more than `--yield-floor-mib` (default 128),
   each held in two reads in a row (inside `host.Level`). A newcomer's end is heard by
   `host.Exits`. On an idle desktop (an RTX 5080, 600 one-second reads),
   baseline processes moved by 0 to 2 MiB.

Memory can be seen only after the allocation has been served, so this signal always lets the
newcomer's first allocations come out of free memory. Signals 1 and 2 come before any
allocation. Nothing here can hold the allocation itself. That would take a Vulkan layer or an
eBPF probe on the driver's allocation ioctl, and neither is built.

**Baseline.** This is what holds the card when the watch starts, when nothing of the engine's
is loaded, and again after each resume. A process whose pid changes under the same name, at no
more memory than before, keeps its place in the baseline. Never counted: the engine and its
children (on Linux, anything descended from the engine's pid; elsewhere llama-server's and
the agent's pids), the host pid, `--yield-ignore` names, launches whose app id is in
`--yield-except` or in the profile's `yield_except`, and anything those launches started.

**Yield.** New requests get the 503 below at once. Replies already running finish for up to
20 s (the door's count reaching zero, bounded by that named moment). Then llama-server stops (a load in progress is cancelled), and the phase becomes
`yielded` with `status.yield = {reason, kind: game|gpu, since, draining}`. While draining,
the phase is still `ready` and `yield.draining` is true. A yield never touches the model book,
the caps or the recovery record, and it counts no restart. A yield is not a crash and not a
shortfall. `start` answers an error while the card is yielded, and `sleep` does nothing.
`stop` ends the yield and keeps the engine stopped; the watch comes back with the next `start`.

**Coming back.**
- A game keeps the engine away until 15 s after its launcher process is gone (`--yield-grace`,
  a named moment). If the game comes back during that time, the wait starts again.
- When no newcomer is left, the engine loads again. The shape is chosen afresh, which gives
  the same shape when the game left nothing behind; if it left something, the usual fit
  logic applies.
- A non-game newcomer that still runs lets the engine back once its use has held within the
  floor for 60 s (`--yield-steady`, a named moment). The engine then loads only a shape that fits the room the
  newcomer leaves; if none fits, it stays yielded and checks again after another steady
  period. A later growth yields again. `--yield-strict` waits until every newcomer has exited.

## 7. Control API and proxy

Listens on **`127.0.0.1:47480`** (falls back to 47481–47489; the port it took is written in
`engine.lock` and in the status). Not on the LAN unless `--host` says so.

**Anti-CSRF:** every non-GET `/engine/v1/*` request must carry the header
`X-Easy-Engine: 1` and, if it has an `Origin`, the origin must be the engine's own. Browsers
cannot send the custom header cross-origin without a preflight that the engine refuses.

**The gate.** Every path (the control API, the setup page and the proxy) passes one gate, in this
order:

1. On a loopback listener the `Host` must be `127.0.0.1`, `localhost` or `[::1]` (any port);
   else **403**. This stops DNS rebinding.
2. An `Origin` that is not the engine's own, or a `Sec-Fetch-Site` other than `same-origin` or
   `none`, is refused with **403**, GET included. Every `OPTIONS` request is refused with 403.
3. A non-loopback listener needs a bearer token (below); else **401** with
   `WWW-Authenticate: Bearer` and the body
   `{"error":"bearer token required (easy-engine token prints it)","code":"<code>"}`, where code is
   `token_required` (no bearer sent), `token_revoked` (a token this engine issued and then revoked:
   pair again) or `token_invalid` (a token it does not know). The status and the message are as
   before; `code` is a new field. The keyless paths of §7f (hello, and the client's side of
   pairing) are the exception: they need no token and never answer 401.

A loopback listener never asks for a token from this machine's own callers. **A request on the
loopback listener that carries a forwarding header** (`Forwarded`, `X-Forwarded-For`, `X-Real-IP`
and the rest listed in §7g) is another machine's, passed on by something on this machine: it is
remote everywhere and needs a token as on a non-loopback listener (none: 401 `token_required`; a
paired token: scoped as a paired token is, with the reduced status; the master key: everything,
as `master@<ip>`), and it is never the identity `loopback`. **A reverse proxy in front of the
loopback listener must pass a token** (the master key, or a paired one per client). The engine
reads the token file for this and never makes one on loopback; with no file, such a request is
401. A plain loopback request, and the setup page in a local browser, are unchanged. The proxy drops every `Access-Control-*` header the
upstream sent, so a model server cannot grant a web page access, and it never passes the
client's `Authorization` header to the local llama-server.

**Token.** When the listener is not loopback (`--host` names another address), the engine makes
a token on first use: 32 random bytes, base64url (43 characters), stored in the data dir as
`tokens.json` at mode 0600. The file holds `master` (the token in clear) and `tokens`, a list of
`{id, name, created, sha256}` entries for the tokens pairing issued (§7f; the token itself is
never kept, only its hash), and `revoked`, the same entries with a `revoked` time, so a revoked
token is told apart from an unknown one. Every request must send `Authorization: Bearer <token>`;
every comparison is constant-time. `easy-engine token` prints the master token (making it if
needed); `easy-engine token --new` replaces it (paired clients keep their own tokens);
`easy-engine token --list` lists the paired clients (id, name, created; never a token or a hash;
last use is not recorded, it would mean a file write per request); `easy-engine token --revoke
<id>` revokes one, at once (the file is read on every check). The engine's
own clients (`status`, `stop`, `gpu`, `install`, a second `serve`) read the token from the file and
find the listen host in the lock. In external mode the engine's token is never sent on to the
external endpoint.

**Scope of a paired token.** The master token, the loopback listener, the setup page and an open
listener may call everything. A token that pairing issued (§7f) is a client on another machine,
such as a game mod: it may use the model and read how the engine is doing, and nothing else. The
gate (`api/scope.go`, one check, one place) holds an allow list; whatever it does not name is
refused with 403 `{"error":"...","code":"not_allowed"}`:

- `/engine/v1`: `GET hello`; the keyless pairing paths (`POST pair`, `GET pair/{id}`,
  `DELETE pair/{id}`); `GET status`, in the reduced view below; `POST attach`;
  `DELETE attach/{id}`.
- The proxy, on the cleaned path (`//v1/chat/completions`, a trailing slash and encoded forms
  are judged as what they spell; a path with a `.` or `..` segment is refused): `POST`
  `/v1/chat/completions`, `/chat/completions`, `/v1/completions`, `/completions`, `/completion`,
  `/tokenize`, `/detokenize`, `/apply-template`; `GET` `/v1/models`, `/models`, `/health`,
  `/v1/health`. Not `/slots`, `/props`, `/metrics`, `/lora*`, embeddings, rerank or infill; the
  list can grow when a client needs one.
- Refused to it: shutdown, settings, install, remove, model choice, `host`, start, stop, sleep,
  wake, logs, the facts, events and book streams (they carry pids, paths and other clients), the
  pairing list and answer, the chat window, and every other proxied path.

**The reduced status.** For a paired token `GET status` has the same JSON names and shapes with
fields removed (`api.PairedStatus`, copied field by field, so a field added to the full status
stays hidden until it is classified; a test fails until it is). Kept: `version`, `api`, `mode`,
`phase`, `message` (blank in the `error` and `yielded` phases), `progress`, `model`, `backend`,
`shape`, `tok_s`, `model_tier`, `model_stage`, `kv_type`, `restarts`, `model_fallback`,
`model_note`, `sampler`, `model_name`, `model_codename`, `model_codename_full`, `model_selected`, `model_params`,
`model_size_class`, `held`. Removed:
`port`, `server_url`, `agent`, `parents`, `errors`, `external`, `linger_s`, `upgrade`,
`contexts`, `custom`, `previous`, `gpu`, `device`, `recovery`, `yield`, `build` (paths, pids, the local
model server's address, error texts, other clients, host program detail). Everything else needs
the master token or the engine's own machine.

`serve --open` makes a non-loopback listener with **no** token, and logs one warning line:
`easy-engine: WARNING: listening on <host>:<port> with no token (--open); ...`. It makes no
token file. What a caller on it may do depends on where it is, judged by the real peer address
(a loopback address, or one of this machine's own on the listener) and never by a header (a
forwarding header only ever makes a caller remote):

- **from this machine**: everything, as on the loopback listener (except the person's side of
  pairing, which answers only on the loopback listener or with the master token).
- **from another machine**: exactly what a paired token may. On the control API: `hello` and the
  client's side of pairing (`POST pair`, `GET`/`DELETE pair/{id}`), `GET status` in its reduced
  form (above: no ports, paths, errors, contexts or other clients), `POST attach` and `DELETE
  attach/{id}`; every other route, including any added later, is 403 `not_allowed`. On the model
  server's paths: the same allow list as a paired token (§7g). Administration from another
  machine (sleep, stop, wake, start, install, remove, settings, storage, logs, events and the rest)
  needs the master key.

The one exception to the token on a non-loopback listener is the setup page itself (`GET /`,
`/index.html`): a browser navigation cannot send a header, and the page's script has to load to
read the token. It is a fixed file with nothing secret in it. Open it as
`http://<host>:<port>/#token=<token>`; the page keeps the token for the tab (sessionStorage),
removes it from the address bar and sends it on every call. With a missing or wrong token the page
says to run `easy-engine token`. `easy-engine setup` opens that address itself.

**Timeouts.** The server reads request headers for at most 10 s (`ReadHeaderTimeout`). It has no
other timer: streams and long model calls run as long as the client keeps them.

### `/engine/v1/*` (JSON)

| Method, path | What |
|---|---|
| `GET facts` | everything that stands on the engine's hub: `{life, seq, facts: [Event], readings: [Event]}` (§7e) |
| `GET events?life=L&since=N[&prefix=P]` | the held NDJSON stream of events after seq N of life L (§7e); 409 for another life |
| `GET book` | every event name the engine says: `[{name, kind, keyed?, what}]` |
| `GET hello`, `POST pair`, `GET`/`DELETE pair/{id}`, `GET pairing`, `POST pairing/answer` | who this engine is, and pairing a client on the network (§7f); the first four need no token |
| `GET status` | `{version, api, port, held, mode: "local"\|"external", phase, message, progress, model, backend, build, shape, server_url, agent, parents, errors[], model_fallback, model_note, model_tier, model_stage, upgrade, previous, sampler, gpu, device, recovery, model_name, model_codename, model_codename_full, model_selected, model_params, model_size_class}`; `model_name` is the human name of the model that serves, `model_codename` and `model_codename_full` its tier's short and full code names (§3; `""` when it has none, as for a custom file); `model_selected` is the person's selection (the catalog id, `custom:<name>` for a custom file, `""` when none): it equals `model` except after a tier step-down (§3), where `model` and the names are the tier that serves; `build` is the llama.cpp build `{backend, tag, patched, rung, why, pin, vmm, next}` or absent with none installed (§3 "Which llama.cpp build"; `next` is the move to the pin `{backend, tag, state: downloading\|installing\|waiting\|staged\|failed, error, attempt, progress}`); `api` is the control API's version (1; it says the stream and the hold exist), `held` the requests the door holds now (§7e); `recovery` is `{failures, crashes, last_error, retry_in_s, suspect_memory, ceiling_ctx}` while the engine recovers from crashes or failed starts, null once a run has stayed up 2 min (§5 at run time); `device` is what the last llama-server start was given to hold it to the planned adapter (§4a), null before the first start; `model_params`/`model_size_class` are in §7d; `gpu` is the `GPUSummary` of §4a (no processes) as the watchdog last read it, null before the first reading; `previous` is `{id, name, file, bytes}` or null (§7b); `model_tier`, `model_stage` and `upgrade` are the first-run staging fields (§7a); `sampler` is `{policy, active, model, basis, source, note, profiles}` (§7 sampler policy) |
| `GET gpu[?refresh=1]` | `GPUReport`: `{adapters, engine_adapter, host_adapter, host_integrated, reserve: {others_mib, others_basis, host_mib, host_basis, margin_mib, total_mib}, free_for_model_mib, others_note?, processes: [{pid, name, adapter, dedicated_mib, shared_mib, kind, ours?, host?}], sources}` (§4a); cached 5 s, `?refresh=1` senses now |
| `POST host {pid?, adapter?: {name, vendor_id, memory_mib}, reserve_mib?}` | what the host app says about itself; 204; 400 for a negative or dead pid, a negative figure, an adapter with neither name nor vendor id. A zero or missing field keeps what is held. The next watchdog tick re-reads with it |
| `GET hardware` | `Hardware` (§4), cached; `?refresh=1` re-probes |
| `GET plan` | the recommendation and every catalog model with its verdict: `{recommended, models: [{id, name, codename, codename_full, opt_in, size, fits: "yes"\|"tight"\|"no", why, limit, shape, est_vram_mib, est_ram_mib, installed, disk_need_bytes, fallback, fallback_note}], backend, disk_free_bytes, gpu, device}` (`gpu` is the `GPUReport` the verdicts were made around; `device` the adapter pin of §4a); `limit`, on a "no", names what stops the model, and `why` says it in words with the figures: `vram` (the card, for an MoE model even with every expert layer in system RAM, or a model kept whole), `ram` (system RAM: a CPU-only run, the expert layers the card cannot hold, or a GPU that borrows system memory), `disk` (the download), `dense_never_split` (a dense model too big for the card: dense models are never split between card and RAM), `vram_block` (no single free block large enough), `kv_type` (an unknown cache type), `model` (the header lacks what the planner needs); the custom file's check carries the same `limit`; `disk_need_bytes` is 0 for an installed model, else what setting it up still needs from the current state (§7c), with `disk_need` `{steps: [{what, peak_bytes, keep_bytes}], peak_bytes, logs_bytes, host_bytes, headroom_bytes, total_bytes}` saying what needs it; the plan's disk verdict uses the same count against the free space read at the time of the request; `disk_free_bytes` is -1 when unknown |
| `POST install {model, backend?}` | starts the download/unpack/agent install job; 409 if one runs. While another model is ready and serving it is a **switch** (§7b): the current model keeps serving |
| `POST cancel` | cancels the running job (keeps `.part` files for resume) |
| `POST shutdown` | answers 200, then stops the agent and llama-server and exits the engine |
| `GET storage[?refresh=1]` | what the engine keeps on disk (§7b): `{data_dir, total_bytes, free_bytes (-1 unknown), parts: [{name: "models"\|"backends"\|"runtime"\|"downloads"\|"logs", path, bytes}], models: [{file, id, bytes, in_use, partial}], approximate}` |
| `POST remove-downloads {confirm: true}` | 202; job that deletes the downloaded files (§7b); 400 without `confirm: true`; 409 if one runs |
| `POST delete-model {id\|file}` | deletes one file in `models/`: 200 `{freed_bytes}`, 404 no such file, 409 it is serving or downloading |
| `POST import {path}` | import-only models: checks sha256, hard-links or copies |
| `POST start`, `POST stop` | the model server (local mode) |
| `POST sleep`, `POST wake` | §6 |
| `PUT settings {mode?, external?:{base_url?, model?, api_key?, clear_api_key?}, linger_s?, kv_type?, sampler_policy?, streamer?: "on"\|"off", keep_context?: "on"\|"off"\|"always"}` | **partial**: only the fields sent change, the rest keep their saved value. An explicit empty `base_url` clears it; an empty or missing `api_key` keeps the key, and `clear_api_key: true` removes it. Written to `settings.json` (whole, with the `listen` and `yield` keys it holds; §2); a mode change takes effect at once, and so does `streamer` (the next request; 400 for any value but on or off) and `keep_context` (the next planned stop; 400 for any value but on, off or always; §5 "The prompt cache across a restart"). `GET settings` shows `streamer` and `keep_context`. `listen` and `yield` are edited in the file, not here |
| `POST test` | tests the current endpoint (local or external): `GET /models` then an 8-token chat; returns `{light: "green"\|"amber"\|"red", message, latency_ms, model}` using the same rules as the mod's `LlmProbe` |
| `POST attach {pid?, name?, hold?, profile?}` | adds a parent pid (the old wire, `{pid}` alone, answers as before plus fields) and an **attachment** with the client's profile; `hold: true` keeps the answer open as its presence (§7g); **409** once the engine is stopping |
| `DELETE attach/{id}` | detaches an attachment: 204; 404 `attachment_unknown`; 403 `not_yours` (§7g) |
| `GET logs?name=engine\|llama-server\|agent&tail=200` | plain text |

`phase` is one of: `not_installed`, `downloading`, `installing`, `starting`, `loading`,
`ready`, `resizing`, `sleeping`, `yielded` (§6b), `removing` (§7b), `external`, `stopping` (the
engine is shutting down; it wins over every other phase), `error`. `progress` is `{done_bytes, total_bytes,
rate_bps, eta_s, step}` while downloading, else null. `shape` is `{slots, ctx_per_slot,
n_cpu_moe, kv_type, vram_mib, vram_basis}`; `vram_basis` is `measured` where `vram_mib` was read
from the running server, else `predicted`. `contexts` is the prompt cache kept across a restart:
`{slots, tokens, bytes, kept: [{slot, tokens, bytes, client?, client_slot?}]}` while saved slots
wait for the next start, else absent (§5, "The prompt cache across a restart"); `client` is the
key of the attachment the slot served and `client_slot` its own number (§7g). `attachments` lists
every attachment as its `attachment` fact says it, the base first; `wants` is the `wants` fact
(§7g).

### 7a. First run: a starter model, then the target

Downloading the 10 GB reference model takes much longer than a player will wait for a first
line, so a first install stages it:

1. `POST install {model}` where `model` (the **target**) is not on disk: the engine also picks a
   **starter**, the smallest installable catalog model that fits this machine (verdict not "no"),
   is at most half the target's size and is **not a fallback model** (normally gemma4-e4b, about
   3.6-5 GB; a fallback such as E2B is too degraded to be a first impression), with room on disk. It
   downloads the starter first, starts serving it (`phase` goes `ready` as usual) and downloads
   the target in the background (same resume and sha256 rules as any download).
2. When the target is on disk and verified, the engine **swaps** at a safe moment: it waits until
   no load or resize is under way, takes the operation lock (so the watchdog stays out), flips the
   phase to `resizing` (message "Switching to the full model"), waits for requests in flight and
   busy slots to finish (at most `SwapDrain`, default 2 min), stops the old server and starts the
   target with a fresh fit (`fit.Choose` with the live room). Nothing is cut off mid-reply.
3. **Clients during the swap are held** at the door (§7e) and answered by the target once it is
   ready: a slower reply, not an error. (Before API 1 they got a 503 "Switching to the full
   model".)
4. No starter when: the target is already on disk; the target is the smallest that fits (a
   CPU-only machine picks the smallest) or less than twice the next smaller model; the starter
   would not fit the disk; the install runs from the CLI (`easy-engine install` exits with the
   job and never stages); the only smaller model is a fallback (so a target of E4B downloads
   alone).
   After a successful swap the starter's file is **deleted**, but only when this engine downloaded
   it for the first run (`state.json` `starter_owned`); a model already on disk is kept.
5. Failures never touch what serves. A failed or corrupt target download (sha256 mismatch drops
   the `.part`) is retried with backoff (`UpgradeBackoff` 30 s doubling to 15 min); a partial
   `.part` resumes with HTTP Range. If the free disk space cannot take what the target still needs
   (§7c: the rest of its download, the logs to their caps, the headroom), `upgrade.state` is `no_space`, the starter stays, and
   the check repeats on the same backoff. If the target downloads but will not load, the starter
   is started again and `upgrade.state` is `failed`.
6. `state.json` keeps `target` while the upgrade is unfinished; the next engine start resumes it.
   `POST cancel` stops the background download (the `.part` stays, state `cancelled`); an install
   of the target again (or an engine restart) resumes. An install of any other model replaces the
   upgrade; installing the same target while it runs answers 409.

Status fields (all in `GET status`; `null` or `""` when nothing is staged):

| Field | Values |
|---|---|
| `model_tier` | `"starter"`: the answering model is a stand-in, show "simpler speech while the full model downloads"; `"full"`: the chosen model; `""`: external mode or no model |
| `model_stage` | `""` (no staging), `starter_downloading`, `target_downloading`, `target_retrying`, `target_no_space`, `swapping`, `starter_serving` (upgrade cancelled or the target would not load), `target_serving` (the swap finished this run) |
| `upgrade` | `{target, target_name, target_codename, target_codename_full, state, progress, error, attempt, retry_in_s}` (`target_codename` and `target_codename_full` are the target tier's short and full code names, `""` when it has none); `state` is `queued` (starter still downloading), `downloading`, `retrying`, `no_space`, `swapping`, `done`, `cancelled`, `failed`; `progress` is the target download as `{done_bytes, total_bytes, rate_bps, eta_s, step}` |
| `model_fallback`, `model_note` | also set while `model_tier` is `starter` (the starter's own `fallback_note`, else a generic sentence), the same warning idea as for a fallback model |

### 7b. Storage, removing downloads, switching models

Data dir layout: `state.json`, `models/` (`<file>.gguf`, partial `<file>.gguf.part`), `backends/`,
`runtime/` (uv, venvs, python), `downloads/` (archives), `logs/`.

- `GET storage` sizes each directory with a walk capped at 2 s (`approximate: true` when the cap
  hit); complete results are cached 10 s, `?refresh=1` re-walks. `models[].in_use` is true for the
  selected model, the target of a running download, and partial files while an install job runs.
- `POST remove-downloads {confirm:true}` runs a job: cancels any install/upgrade, stops the agent
  and llama-server, deletes the contents of `models/`, `backends/`, `runtime/`, `downloads/`
  (symlinks and junctions are unlinked, never followed; nothing outside the data dir is touched;
  `logs/` and `state.json` stay), clears `model`, `target`, `starter_owned`, `backend`, `previous`
  and the fit book; mode, external, linger, kv_type and sampler_policy stay. Phase becomes
  `not_installed` and `status.message` reads "Removed downloaded files (X GB freed)", plus the
  paths that could not be deleted (a file still in use on Windows; each delete is retried for about a
  second). `POST install` then works from scratch. Phase is `removing` while it runs and installs
  answer 409.
- `POST delete-model {id|file}` deletes one file in `models/` (and its `.part`). 409 when it is the
  selected model or an active download. A plain file name only; no paths.
- **Switching:** `POST install {model: B}` while A is ready and serving keeps A serving and
  downloads B through the upgrade path (`status.upgrade` with `switch: true`, `model_stage`
  `target_downloading`, ...), then swaps (A stopped, B started with a fresh fit). `model_tier`
  stays `full` and no starter warning is set. If B is already on disk the swap is immediate. A is
  never deleted; after the swap `status.previous` is `{id, name, file, bytes}` of A (null when A
  was a first-run starter that was deleted, or after `delete-model` of it or the next switch).
  `POST cancel` during the download keeps A serving and forgets B (`state.target` cleared). If there
  is not enough disk the install answers 400 `not enough disk space for B: needs X GB, Y GB free`.
  The CLI `install` command keeps the old, non-switching path.

`status.progress` keeps showing the install job (the starter, or the model server); the target's
download is `upgrade.progress`. `status.model` is the id of the model answering right now. A client
shows its "simpler speech" notice when `model_tier == "starter"`, and a progress bar from
`upgrade.progress` when `model_stage` is `target_downloading`.

### 7d. Your own GGUF file (`custom/*`)

The engine can serve a local GGUF file in place. The file is never copied, moved or deleted (not by
`custom/clear`, not by any cleanup); `state.custom.path` lies outside `models/`.

- `POST custom/check {path}` -> 200 `CustomCheck`: `ok`, `error` ("file not found", "not a GGUF file",
  "header unreadable: ..."), `path`, `name` (base name without `.gguf`), `size`, `arch`, `params`
  (sum of tensor elements), `ctx_train`, `quant` (from `general.file_type`), `moe`, `known`,
  `estimated`, `fits` (`yes|tight|no`), `why`, `shape`, `est_vram_mib`, `est_ram_mib`,
  `est_decode_tps`, `chat_template`. Reads the header only; nothing is stored.
- `POST custom/use {path}` -> 202 `{started:true}` | 400 `{error}` | 409 (an install job runs). Saves
  `state.custom = {path, name, size, mtime}` and `state.model = "custom"` (the catalog model it
  replaced is kept as `custom_prev`), stops the running server and loads the file in the background.
  On a machine with no backend yet it first runs an install job for llama.cpp and the agent only (no
  catalog model, no starter download), reported in status phase/progress like a normal install and
  answering 409 while another job runs; when the job ends the file is served. An engine restart with
  `state.model = "custom"` and a backend resumes serving the file.
- `POST custom/clear {}` -> 200. Forgets the file; the `custom_prev` model serves again when installed,
  else `not_installed`.

Serving: alias `custom:<name>`, `--jinja` and the other fixed flags as for any model. Footprint: a
catalog model with a measured footprint whose file has exactly the same byte size lends its measured
footprint and sampler profiles (the catalog records no arch/params/quant, and a ten GB hash is not
cheap), and the check reports `known: true`. Otherwise the footprint is the header seed (`fit.Seed`),
`estimated: true`, and the room is narrowed by an extra 768 MiB margin (`plan.CustomExtraMarginMiB`).
The ctx balancer and verdicts are those of a catalog model. Sampler: a catalog model matched by size
or by name lends its profiles; else the generic rp values (temperature 1.0, top_k 64, top_p 0.95,
min_p 0) as both `official` and `rp`. The calibration book keys on `custom-<hash of path+size+mtime>`
instead of a sha256.

Status: `custom` is `{path, name, size, missing, params, size_class}` or null; `model` is `"custom:<name>"` while it
serves; `model_name` is the human name of the model that serves (the catalog name, or the file name), `model_codename`/`model_codename_full` its tier's code names (`""` for a custom file).
`GET plan` gains `custom` (a `CustomCheck` of the current file, or null).

Size class: status has top-level `model_params` (int64, total parameters, 0 unknown) and
`model_size_class` (`"small"` below 8e9 parameters, `"large"` at or above, `""` unknown); a
mixture-of-experts model counts every expert. While a custom file serves they are read from its
header (`general.parameter_count` when present, else the sum of the tensor element counts) and
repeated inside `custom` as `params` / `size_class`; `custom/check` carries the same `params` and
`size_class`. For a catalog model they are 0 and `""` (the catalog holds no parameter count).

A missing file (at `custom/use`, at engine start, at restart or by the watchdog): `custom/use` answers
400 "file not found"; an engine start or restart goes to phase `error`, message
`Model file not found: <path>`, with no retry; `custom/use` with another path or `custom/clear`
recovers.

### Proxy

Every other path (`/v1/*`, `/completion`, `/slots`, `/props`, `/health`, `/metrics`, `/tokenize`, …)
is forwarded unchanged to llama-server (streaming included). While the model is on its way up
the door **holds** the request and answers it when the model is ready (§7e). Where a request is
answered at once without a model (no model installed, the model server's error, a person's stop,
a download, the engine stopping) the engine answers **503** with `Retry-After: 2` and
`{"error":{"code":503,"type":"unavailable_error","message":"..."}}` (llama-server's own shape, so
clients that already retry on 503 work). While the card is yielded (§6b) it
answers **503** with `Retry-After: 10` and
`{"error":{"code":"yielded","type":"unavailable_error","message":"Paused while <reason> runs"}}`,
where `<reason>` is, for example, `AppId 123 (game.exe)`. The string code tells "paused for
another program" apart from "loading". In external mode, `/v1/*` is forwarded
to the external base URL with its key added, so the agent always talks to the engine.
Streamed completion replies go through the token streamer, on by default (`"streamer": "off"` in settings.json turns it off): see `docs/STREAMING.md`.

### Sampler policy

The proxy applies a sampler policy to every completion request (`POST /v1/chat/completions`,
`/chat/completions`, `/v1/completions`, `/completions`, `/completion`), for any client. The
request names a profile with `X-Easy-Sampler: rp|structured|official` (or a name the game
profile adds); missing or unknown means `rp` (unknown is logged). The header is removed before
forwarding. `sampler_policy` (a setting, saved in `settings.json`) is:

- `enforce` (default): every sampler field in the request (`temperature`, `top_k`, `top_p`,
  `min_p`, `typical_p`, `typical`, `tfs_z`, `top_n_sigma`, `repeat_penalty`, `repeat_last_n`,
  `presence_penalty`, `frequency_penalty`, `penalize_nl`, `dry_*`, `xtc_*`, `mirostat*`,
  `dynatemp_*`, `samplers`) is replaced with the profile's values; fields the profile does not
  set are removed, so the server defaults (the official values, §5) apply.
- `fill`: only missing fields are filled from the profile.
- `off`: no rewriting.

Non-sampler fields (`max_tokens`, `grammar`, `json_schema`, `stop`, `seed`, `cache_prompt`,
`id_slot`, `n_probs`, `logit_bias`, ...) are never touched. Every proxied response carries
`X-Easy-Sampler-Applied: <profile>;<mode>` (`none;<reason>` when the request passed through).
The header tells what happened: when the body is not JSON or is empty the request passes
through and it says `none;<policy>;body-not-json` or `none;<policy>;no-body`. A body that cannot
be read gets a **400**.
In external mode the same policy applies only when the external model name matches a catalog
model (`sampler_match`); otherwise requests pass through and `status.sampler.note` says so.
Design, table and how a game adds a profile: `docs/SAMPLERS.md`.

### Setup page

`GET /` serves an embedded single-page setup UI (plain HTML/JS, no CDN, works offline): the
hardware, the recommended model with a one-click **Install**, the other models with fit
verdicts, a progress bar with speed and ETA, the live phase and shape, the Local/External
switch with a **Test** button and a green/amber/red light, the sampler policy (selectable) with the
profiles in force (read-only), and the three log tails. On a non-loopback listener it takes the
token from the address fragment (see Token above). It has no timer: it reads `facts` once and
follows `events` with a stream reader (§7e), re-reading `status` once per batch of events and the
open log tails on an event; on a 409, a gap or the stream's end it reads the facts again. An
engine it cannot reach is shown with a **Reconnect** button (the click, the browser going online
and the page becoming visible again are the words that try again).

### 7e. The event stream and the door (API 1)

The engine says what happens on its hub (docs/EVENTS.md §2): facts that stand until replaced or
said gone, readings (levels said when they change) and happenings (history). The names are in
`GET book` (`run/book.go`): `engine` `{version, api, port}`, `phase` `{phase, message}`, `model`,
`door` `{held}`, `mind.shape`, `mind.device`, `job`, `install.progress`, `upgrade`, `yield`,
`recovery`, `room`, `agent` (keyed), `pairing` (keyed, §7f), `client` (keyed, §7f), `attachment` (keyed, §7g),
`wants` (§7g), and the happenings `request`, `error`, `nudge`, `mind.exit`,
`mind.hang`, `mind.oom`, `network`, `mark`. No secret is ever on the stream (the external API key
and the tokens are in no fact). The three paths pass the same gate and token as every other.

An event: `{"seq":12,"at":"2026-11-27T14:09:47.11-06:00","life":"3f9c0a1b2d4e5f60","name":"phase","kind":"fact","body":{"phase":"ready","message":"Ready"}}`
(`key` where a name is keyed, `gone: true` when a fact stops standing; `at` is data only).

- `GET /engine/v1/facts` → 200 `{"life":L,"seq":N,"facts":[...],"readings":[...]}`, sorted by
  name and key.
- `GET /engine/v1/events?life=L&since=N[&prefix=P]` → 200 `application/x-ndjson`, one event per
  line, oldest first, flushed as said, held open for as long as the client keeps it. `prefix`
  keeps names beginning with it (`phase`, `mind.`). **409** `{"error":"not this life's seq: read
  the facts whole","life":L2,"seq":N2}` when L is not this run of the engine or N is past its
  seq; **400** for a bad `since` or an undeclared prefix. When the history ring (4096 events) has
  let go of N the stream ends with a last line `{"gap":true}`. The stream ends when the engine
  ends.
- `GET /engine/v1/book` → 200 `[{"name","kind","keyed"?,"what"}]`.

**A client's recipe.** Read `facts` (keep `life`, `seq` and what stands). Follow
`events?life=<life>&since=<seq>`, applying each line (a fact replaces the one of its name and key,
or removes it when `gone`), and keep the last seq. On a 409, a `{"gap":true}` line or the
stream's end, read `facts` again and follow again; a refused connection means the engine is gone.
Never decide from a happening's history, only from what stands.

The `request` happening: `{"what":"begin","inflight":1,"id":"rid-7"}` when a request passes the
door, `{"what":"end","inflight":0,"id":"rid-7", ...}` when it ends (the id is the reply's
`X-Request-Id`, so a client pairs them), and for a streamed reply the end carries what the
streamer read: `end` (done, upstream_error, ended_early, client_gone), `events`, `finish_reason`,
`prompt_tokens`, `cached_tokens`, `completion_tokens`, `tok_s`, `prompt_ms`, `predicted_ms`,
`slot`. A streamed reply's `tok_s` is status `tok_s`; otherwise `/metrics` is read once when the
door goes idle (`{"what":"idle","tok_s":...}`).

**The door holds.** A proxied request that arrives while the model is on its way up is held and
answered when the model is ready, instead of a 503:

- **Held:** phase `starting`, `loading`, `resizing` (a reshape, or the swap to the full model),
  `ready` while a swap has begun, and `sleeping` unless a person stopped it (the request is also
  the word that wakes it).
- **Answered at once, exactly as before:** yielded (503, `"code":"yielded"`, `Retry-After: 10`),
  `not_installed`, `error`, a person's Stop (`sleeping`, message "Stopped"), `downloading`,
  `installing`, `removing`, `stopping`, every external-mode error, and everything the gate
  refuses.
- **How a hold ends:** it waits on the hub (the `phase`, `model`, `yield` and `upgrade` facts),
  never on a clock: when the phase becomes `ready` the request is forwarded; when it becomes one
  of the answered-at-once phases the request gets that answer then; when the client goes away
  the hold ends silently; when the engine ends it gets 503 "The engine is stopping". There is no
  engine timeout on a hold: the client's own timeout bounds it.
- **A held streamed completion** (`"stream": true`, the streamer on) gets its stream opened as
  for a slow upstream (docs/STREAMING.md §4): after 2 s, 200 `text/event-stream` and `: easy
  waiting`, repeated every 15 s; the reply follows in the same stream. A hold that ends in an
  answered-at-once phase after the stream was opened ends it with the terminal event, code
  `yielded` or `unavailable`, `"status":503` (STREAMING.md §5); before the stream opened, the
  JSON 503 above.
- A held request is counted in the `door` fact `{held}` and status `held`, **not** in flight: the
  hang watchdog does not arm for a request the model server has not seen, and a drain (a reshape,
  a swap, a yield) does not wait for a request that waits for it. Its `request` begin is said when
  it passes the door. The resident app counts held requests as clients (§6a).
- A request held while the model loads is a word: the load asks `/health` once on it (one ask
  per held request), so a build whose `/health` turns 200 long after its last log line is found
  ready by the next request, not by `LoadTimeout`.
- The phase is said **before** a stop begins (yield, sleep, a person's stop, shutdown): a request
  that arrives during the stop is answered by that phase (yielded, held to wake again, 503), never
  proxied to a server being stopped. `stopping` is now said on the stream by the engine itself.

### 7f. Hello, discovery and pairing (a client on another machine)

A game on one machine and the engine on another (the PC with the card): the client finds the
engine on the local network and pairs by comparing a six-digit code. Nothing is typed from one
device into another. Packages `discover` (mDNS), `pair` (the requests), `api/pair.go` (the paths).

**Hello.** `GET /engine/v1/hello` needs no token on any listener and never answers 401 (the
host, origin and preflight rules of the gate still apply). 200:

```json
{"name":"den-pc","engine":"easy-engine","api":1,"version":"0.4.0","engine_id":"9f2c...32 hex",
 "model_name":"Gemma 4 26B RP","model_codename":"Campfire","model_codename_full":"Campfire-26B","model_id":"gemma4-rp-maxxed-q3kxl","phase":"ready","card":"NVIDIA GeForce RTX 4080",
 "pairing":true,"token_required":true}
```

`name` is the `name` setting or the host name; `engine_id` is a stable random id made once and
kept in `<data dir>/engine-id`; `model_name` the human name of the model that serves (its id when it
has none); `model_codename` and `model_codename_full` its tier's short and full code names (§3; `""` when it has none, a custom file or an
older catalog), shown before the name; `model_id` its catalog id (`""` when none is selected, or when a custom GGUF serves,
which has no catalog id); `card` the adapter the engine uses ("" before the first reading, or on CPU);
`pairing` is true only on a non-loopback listener with tokens in force; `token_required` the same
condition without the pairer. With an `Authorization: Bearer` header the engine checks: a valid
token adds `"paired":true,"client_id":"<id>"` (`"master"` for the master token); a revoked one
`"paired":false,"revoked":true`; an unknown one `"paired":false`. Hello carries no paths, no user
name, no tokens and no client list.

**Discovery.** While the listener is not loopback (not `127.0.0.0/8`, `::1` or `localhost`) and
the `discovery` setting is on, the engine announces itself by mDNS/DNS-SD and answers queries,
on `224.0.0.251:5353` and `[ff02::fb]:5353` on every up, multicast-capable interface:

- service `_easy-engine._tcp.local.`; instance `<name> [<engine_id[:6]>]._easy-engine._tcp.local.`;
  SRV target `easy-engine-<engine_id[:8]>.local.` on the listening port (never the machine's own
  host name, which the system's responder owns), with A/AAAA records for it;
- TXT `id=<engine_id>`, `api=<n>`, `name=<name>`, nothing else.

It announces at start and when the system says the network changed (`host.NetworkChanges`),
answers each query that asks for it (by unicast to a querier on another port than 5353, RFC 6762
§6.7, or that sets the QU bit), re-announces on a rename, and says goodbye (TTL 0) when turned off
and on Quit. There is no timed repeat. A loopback listener announces nothing.
`easy-engine find` sends one query from an ephemeral port and lists what answers within
`--wait` (a network timeout):

```
NAME    ADDRESS             ENGINE ID                         API
den-pc  192.168.1.20:47480  9f2c4e...                         1
```

(`--json`: `[{"name","engine_id","api","address","host"}]`; with none: `No engine answered on
this network.`)

**Pairing.** Offered only when `hello.pairing` is true (a non-loopback listener with tokens; with
`--open` there is nothing to pair for, and `POST pair` answers 404 `pairing_off`). The client's
side needs no token; non-GET calls carry `X-Easy-Engine: 1` like every other.

| Method, path | Answers |
|---|---|
| `POST pair {name}` | 200 `{"id":"<32 hex, secret>","code":"493201"}`. `name` is the client's name for itself, shown to the person: control and format characters dropped, spaces folded, at most 40 characters (`unnamed client` when empty). 404 `{"error","code":"pairing_off"}` where pairing is not offered; 400 `{"error","code":"bad_request"}` for a body that is not `{name}` JSON (capped at 4 KiB); 410 `pair_gone` while the engine ends; 500 `{"error","code":"pair_failed"}` if the engine could not open it. A new ask from an address that already has a request waiting replaces it: the old `id`'s GET gets 410 |
| `GET pair/{id}` | held until the person answers. 200 `{"token","client_id","engine_id"}` on Allow: the token is issued now and shown this once; 403 `{"error","code":"pair_denied"}` on Deny; 410 `{"error","code":"pair_gone"}` when the request is gone: collected already, withdrawn, pushed out by the cap, never known, replaced by a newer ask from the same address, or the engine restarted or quit; 500 `{"error","code":"issue_failed"}` when the token could not be written (the Allow stands: GET again). A client that hangs up leaves its request waiting and may GET it again |
| `DELETE pair/{id}` | the client withdraws for good: 204; 410 `pair_gone` if unknown |
| `GET pairing` | the person's side (loopback listener or the master token only; else 403 `not_allowed`, and 401 with no token on the network): 200 `{"offered":bool,"requests":[{"key","name","code","from","quiet"?}]}`, oldest first; `quiet: true` on a request from an address the person denied in this life |
| `POST pairing/answer {key?, code, allow}` | the same rule: the answer names the `code` the person compared, with the request's `key` beside it when the caller has it (the page and the tray do; `easy-engine pair` answers by code alone). 200 `{"key","name","code","allowed"}`; 404 no waiting request with that key or code; 409 `{"error","code":"pair_changed"}` when the key's request now shows another code (its source asked again since): nothing is answered, look again; 409 a code two requests show (by code alone); 400 without `code` or `allow` |

A request waits until it is answered, withdrawn, replaced or pushed out. One source address has
at most one waiting request: a new ask from it replaces the old one (whose held GET gets 410)
under the same public `key`, with a new code and id, so the person is not notified again. Because
the key outlives a replacement, every answer binds to the code the person saw: an Allow or Deny
named for the old code is refused (`pair_changed`), ends nothing and issues nothing; the person
looks again at the new code. At most
4 wait across sources, and a new one over that pushes the oldest out. Answered requests not yet
collected are capped at 4 the same way. Once the person has denied a request from an address, that
address's later requests in this life are still listed but marked `quiet` and raise no
notification. There is no expiry timer and no rate clock: counts bound it. The code is drawn so
no two waiting requests show the same one. `key` is a public handle for the person's side (the
fact is keyed by it); the secret `id` is never said on the hub. A keyless path's error bodies are
fixed text with a code; the detail of an internal error (which could name the data dir) goes only
to the engine's log.

Where the person answers:

- the setup page: a Pairing card with the code, the name, the address it came from, and Allow /
  Deny, shown while a request waits (it reads `GET pairing` when a `pairing` event arrives);
- the tray: one notification per request not yet answered (none for a `quiet` one, none for a
  replacement by the same address), `Allow <name> (493 201)?`, and while it waits two menu
  items above the standing four, `Allow <name> (493 201)` and `Deny <name> (493 201)`;
- headless: `easy-engine pair` lists them (`CODE  NAME  FROM`, or `No pairing request is
  waiting.`), `easy-engine pair --approve 493201` / `--deny 493201` answer one (spaces in the code
  are ignored; exit 1 with `pair: no waiting request shows 493 201`, or `pair: two waiting requests
  show 493 201; answer on the setup page or in the tray`). They reach the running engine through
  the lock with the master token.

The hub says a waiting request as the fact `pairing` (key: its `key`; body `{name, code, from,
quiet}`; said again on a replacement; gone when answered, withdrawn or pushed out), and a paired
client the first time the gate accepts its token in a life as the fact `client` (key: its client
id; body `{name}`), gone as soon as its token no longer stands: the app follows `tokens.json`
(`host.Changes`), so `easy-engine token --revoke` takes it down at once. Neither ever carries a
token.

**A client's recipe, from find to paired.**

1. Find: browse `_easy-engine._tcp.local.` (or ask the person to run `easy-engine find`); take
   the address and port from SRV/A, and `id` from TXT. Remember the engine by its `engine_id`, not
   its address.
2. `GET /engine/v1/hello` there: check `engine == "easy-engine"`, `api >= 1`. If you hold a token
   for this `engine_id`, send it: `paired: true` means use it; `revoked: true` means drop it and
   pair again; `paired: false` means it is not known here.
3. If `pairing` is true: `POST /engine/v1/pair {"name":"<your name>"}` with `X-Easy-Engine: 1`;
   show the `code` large ("493 201") with "Compare this code with the one on the engine's screen
   and press Allow there".
4. `GET /engine/v1/pair/<id>` and wait (no timeout of your own needed; if the connection drops,
   GET again). 200: keep `token` with `engine_id`, send it as `Authorization: Bearer` on every
   call. 403: the person denied it. 410: start again from 3. When the person cancels on your
   side, `DELETE /engine/v1/pair/<id>`.
5. A later 401 with `code: token_revoked` means pair again; `token_invalid` means the engine does
   not know this token (another engine at this address, or its file was replaced).

**Security.** What an unauthenticated host on the LAN can now do, and its bound:

- `hello`: read the engine's name, id, version, phase, model name and card name. Nothing secret;
  no paths, user name, tokens or client list.
- mDNS: see the same name, id and port; its queries are answered, each with one small packet.
- `POST pair`: open a request, which shows a name it chose on the person's screen. One address
  has one waiting request (a flood from it only replaces its own, under the same key) and raises
  one notification until the person answers; after a Deny, none at all. At most 4 wait across
  addresses. The name is cleaned and capped. (A host that can take many addresses on the LAN can
  still raise one notification per address; that is the bound.)
- Error bodies on these paths are fixed text: no file system or OS error reaches the network.
- It cannot answer a request: answering needs a loopback listener or the master token; a paired
  client's token is refused (403), and so is anything without a token from the network (401).
- A paired token is scoped: it may use the model and read the reduced status, nothing more
  ("Scope of a paired token", §7).
- The six-digit code is a comparison code, not a secret: the client and the engine both show
  it so the person can tell the right request from another; nothing accepts it as proof, and the
  only path that takes a code (`pairing/answer`) is closed to the network.
- The request `id` (128 random bits) is what collects the token; it is given only to the client
  that asked, never said on the hub, never logged. A token travels once, in the 200 body of the
  held GET (`Cache-Control: no-store`), and is never in a URL, a log line or an event. Only its
  SHA-256 is kept. Every token comparison is constant-time.
- On Quit every waiting request ends: held GETs are answered 410, the facts go, no token is
  issued after, and the network hears the mDNS goodbye before the listener closes.

### 7g. Clients: one engine, several programs (ee-clients)

Several mods and programs share one engine; each attaches with its profile (docs/EVENTS.md §5).
A **client** is who connects (a paired token's `client_id`, or `loopback`); an **attachment** is
one use of the engine by a client, keyed by `<identity>/<name>` (identity `loopback` for a local
caller). The `--profile` file is the **base** attachment (`attachment: "profile"`), first in every
order, for the engine's life. A caller is **local** when its TCP peer is this machine (loopback,
or the listener's own address), the gate's client is the loopback listener, the setup page,
`--open` or the master token, and the request carries no forwarding header (`Forwarded`,
`X-Forwarded-For`, `X-Forwarded-Host`, `X-Forwarded-Proto`, `X-Real-IP`, `Via`, any `Tailscale-*`):
a header can only take privilege away. Every paired token is **remote**. A forwarder on this
machine that adds no such header (`ssh -L`, `socat`) makes its callers local: put such a forwarder
only where everyone who can reach it may act as this machine. A remote caller with no id of its own
is told apart by address: `open@<ip>` on an `--open` listener, `master@<ip>` for the master token
from another machine.

**Local callers are trusted among themselves.** Every local program has the one identity
`loopback`: any local program can use, replace or detach any local attachment. It is the owner's
machine. The **master key from another machine** is the owner too: it is not held to the paired
allow list on the proxy and its attachments are not budgeted (it still gets the remote caps, and
the door still refuses it what cannot be confined while clients hold slots, and the stream paths).
An **`--open` caller from another machine** is held on the proxy to the same allow list as a
paired token (`api/scope.go`: completions, tokenize, detokenize, apply-template, models, health;
403 `not_allowed` for the rest, so `GET /slots`, `/props`, `/metrics`, embeddings and the like
never reach the model server from it), and on the control API too (§7, `serve --open`): it can
attach, detach and read the reduced status, never sleep, stop or administer the engine. Every `open@<ip>` shares one restart budget and one set of
remote caps (each address keeps its own key space).

**Attach.** `POST /engine/v1/attach` (with `X-Easy-Engine: 1`):

```json
{"pid": 4242, "name": "narrator-mod", "hold": false,
 "profile": {"slots": 3, "min_slots": 1, "ctx_per_slot": {"max": 65536, "floor": 8192},
   "samplers": {"cold": {"temperature": 0.2}},
   "server": {"ubatch": 256, "reasoning": "off", "extra_args": ["--top-n-sigma", "1.5"],
              "required": ["reasoning"]},
   "models": {"prefer": ["gemma4-rp-maxxed-q3kxl"], "can_use": ["gemma4-rp-maxxed-q3kxl", "gemma4-e4b-rp-q40"]},
   "game_reserve_mib": 1500, "host_pid": 4242,
   "agent": {"id": "my-agent", "wheel": "/abs/agent.whl", "args": ["run"]}}}
```

The profile is the `--profile` file's shape (§8) plus `server`, `models` and `catalog`. A profile with no
`slots` states no wants (it counts for presence only, as `{pid}` alone does). 200 (with `hold`,
the first NDJSON line):

```json
{"status":"attached","pid":4242,"attachment":"9f2c4e01a7b3d655","name":"narrator-mod",
 "key":"loopback/narrator-mod","client_id":"loopback","local":true,"presence":"pid",
 "slots":{"want":3,"min":1,"given":3,"first":0,"pending":false},
 "ctx_per_slot":{"asked":65536,"given":32768},
 "server":{"taken":["reasoning","--top-n-sigma"],"not_taken":[],"checked":true},
 "model":{"selected":"gemma4-e4b-rp-q40","usable":true,"advice":"prefers gemma4-rp-maxxed-q3kxl"},
 "agent":"my-agent"}
```

(`"resumed": true` and `"replaced": true` appear when they happened.)

- `slots.given` and `ctx_per_slot.given` are what the running model gives it now; `pending: true`
  while a reshape for it is coming (its requests wait at the door meanwhile), with `why`
  `waiting_for_idle` (a remote client's grow waits for the door's next empty moment) or
  `restart_budget_spent` (below); `unserved` says why
  it has none when the card has no room for it beside the attachments before it (its requests get
  503 `no_room` until the next start). `first` is its first engine slot, for logs only: a client
  numbers its own slots `0..given-1`.
- `server.not_taken[]` is `{option, asked, standing?, held_by?, why}`: `why` is `conflict` (an
  attachment earlier in order holds another value), `this build has no such flag`, `the engine
  sets this flag`, `no build to check against` (no `--help` could be read: `extra_args` are never
  passed unchecked; the help is read again at the next launch) or `the model server did not start
  with them` (below). `checked: false` when no build is installed yet.
- **Presence.** `pid` (local callers only): the attachment leaves when that process ends. `hold:
  true`: the response stays open (`application/x-ndjson`), a new line of the same shape each time
  its slots, the context per slot or its options change, a last line `{"end":"detached"}`,
  `{"end":"replaced"}`, `{"end":"revoked"}` (its paired token was revoked, or the master token a
  `master@<ip>` attachment came with was replaced: every attachment of that credential ends at
  once, its slots go vacant; a token revoked between the gate and the attach's registration makes
  the attach 401 `token_revoked`) or `{"end":"engine_ending"}`; the attachment leaves when the connection
  ends. A remote client must hold. Both given: it leaves at the first to end. The listener sets
  TCP keepalive (idle 30 s, 3 probes 10 s apart; a network timeout), so a peer that vanished is
  found in about a minute; it helps the held `events` and `pair/{id}` too.
- **Resume.** The running model's slot map changes only at a start. A leave leaves its slots
  vacant (`wants.vacant`); an attach of the same key with the same slot count takes them back,
  cache and all, with no restart (`"resumed": true`), and any reshape its leave had asked for is
  cancelled. An attach of a key that stands replaces its presence (`"replaced": true`, the same id
  and slots). A new attachment is served from vacant slots when they suffice, the running ctx is
  at least its floor, its options change nothing and the RAM prompt cache is off (below); a
  remote one never from another client's vacant range (that client's prompts are still in it:
  slots no client held, or its own, only); else the model grows. A **local** caller's grow drains: requests are held at the door while the
  replies in flight finish, and when `DrainMax` (2 min) comes first the ones still running are cut,
  each stream with its terminal event. A **remote** caller's grow or option change never drains:
  it waits for the door's next empty moment (`pending`, `why: waiting_for_idle`); its own requests
  held meanwhile are not in flight, so with no other traffic it grows at once.
- **The last client's leave reshapes nothing.** The shape and its vacant range stay until (a) the
  same key returns (it resumes from its standing range: no restart, no charge), (b) the model server stops for another
  reason (an unload, sleep, a yield, a crash, the person's change: the next start uses the wants
  that stand then), or (c) a local caller with no slots of its own that the stale shape
  under-serves sends a request (the base has no range in it, or it has more slots than the
  standing wants and less context per slot than their max): that request is held and the engine
  reshapes then, as a local grow does. A remote anonymous request never restarts the engine. A
  **local** leave that is not the last reshapes, when idle, only when that gives someone something
  (an unserved one served, an option handed over, more ctx to one below its max). A **remote**
  leave never reshapes by itself: an attachment left out or waiting for room is then given a
  reshape of its own, when idle, charged to its own budget (a local one's to none); one whose
  budget is spent waits for the next start made for another reason.
- **Never held without an event.** A request is held only while something is coming that ends
  the hold: the model on its way up (the phase's word), a reshape asked for its attachment (the
  start that serves it), a stale shape's reshape (that start). A client with no slots in the
  running model and nothing coming for it (its budget spent, its floor short, no room beside the
  others) is answered at once, 503 `no_room` with the why.
- **Restart budget.** The budget limits how often a remote client can make a **running** model
  server stop and start (other callers lose their caches and their time), not ordinary use.
  Every restart of a running model server is one of: a local caller's want (its attach, change,
  leave or under-served request), the engine's own reasons (a load, the person's change, a yield,
  memory, a crash), or a remote identity's attach or change of wants or options, charged to it
  (one restart costs an identity one, however many of its attachments asked for it): 4
  per **waking period** (`restarts_left` in its answer and held lines; every `--open` address
  shares one). **A start from down is never charged**: with no model server running (never
  started, unloaded by the linger, asleep, stopped, yielded), an attach, a return or a change of
  wants is folded into the wants and costs nothing; the next start (a request wakes the model)
  takes them. A **return** (the same key again) while the model runs is free when it is served
  from its own standing range; one that needs a restart is charged as an attach is. Leaves and
  revokes are never charged and never restart by themselves. **The budget is whole again** when
  the model server stops for the engine's own reason or the person's: the linger's unload or a
  sleep asked from this machine, the person's stop, a yield of the card. The stop is the event; no
  clock. Never by a stop a remote client could cause: a sleep or stop asked from another machine, a
  crash, an out-of-memory walk or a failed start (after its reshape or any other), a reshape's own
  stop. Within one waking period, restarts of a running server caused by remote clients alone never
  exceed what they were charged. An attachment that comes while a start is loading is folded in by
  a grow once the model is up, and that grow is charged (else each start's loading window would let
  one more uncharged restart in); so is a change of wants or options an attachment makes while a
  start is loading (the merge was taken before it). A start whose port another process took
  before llama-server bound it is started again at once on a new port: the same restart, charged
  once, no failure counted. The master key from another machine is not budgeted.
  With the budget
  spent a change is not made (`why: restart_budget_spent`): the attachment keeps what it has, or,
  with nothing, its requests get 503 `no_room` until a start that comes for another reason.
- **Floors and ceilings.** A remote client's ctx floor never raises the others': it is clamped to
  the merged floor of the local attachments and the base (the defaults' when there are none). Its
  `ctx_per_slot.max` is clamped to the largest max among the base, the local attachments and the
  default profile: the owner raises the ceiling by the base profile. When the
  context per slot given is below its own floor it is unserved (`the context per slot given (n) is
  below its floor (m)`) and never the cause of a reshape. When the fit refuses the merge, one
  attachment is left out at a time: remote before local, later before earlier, never the base.
- **Extra args that stop the server.** If the model server exits while starting with clients'
  `extra_args` in its launch, for any reason other than running out of memory (a bad value, a
  crash, anything), the one retry runs at once without them; they are told not taken
  (`the model server did not start with them`) in the attach fact and the held stream, and stay
  out while the same `extra_args` stand.
- **Errors** `{"error","code",...}`: 400 `bad_request` (shape or bounds: slots 1-16 per attachment,
  64 in all, ctx 1024-1048576, 16 samplers named `[a-z0-9_-]{1,32}` (stored lower-cased), 16 model
  ids, 8 required names), 400 `pid_remote` (a remote `pid` or `host_pid`), 400 `agent_not_allowed`
  (a remote agent), 400 `path_not_allowed` (a remote `host_files_mib`), 400 `extra_args_not_allowed`
  (any remote `extra_args`), 400 `not_offered` (`server.mmproj`, `server.speculative`: a projector
  will come from the catalog with its model; ngram speculation goes in a local `extra_args`), 400
  `flag_owned` (`{"flag"}`), 400 `flag_unknown` (not in the build's `--help`), 409 `conflict`
  (`{option, asked, standing, held_by, why}`: a `required` option not taken), 409
  `model_unsupported` (`"model"` required and the selection not in `can_use`), 413 (body over 1 MiB
  local, 64 KiB remote). Caps: a local caller's are counted over local attachments only (32
  attachments, 64 slots with the base: 429 `too_many_attachments`, 400 `bad_request`); a remote
  one's over its own share (4 attachments and 16 slots per client, every `open@<ip>` one client)
  and over all remote attachments (16 attachments, 32 slots): 429 `too_many_attachments` or
  `too_many_slots`. Remote attachments never count against a local caller. Nothing is
  registered on any error. The old wire keeps its answers: `{}` is 400 `body must be {"pid": n}`,
  `{pid}` with no watcher is 501; **an intended change**: `{pid}` from a caller that is not local is
  400 `pid_remote` where it was a parent before (a pid named from another machine means nothing
  here); where the system gives no word of a process's end, `{pid}` makes
  no attachment at all (a parent only, as before clients) and meets no cap. A `pid` is made a
  parent before the attachment: a refused attach leaves it a parent, and a parent that cannot be
  made changes no attachment.
- **Engine-owned flags** (refused in `extra_args` in every spelling the build's help gives the
  flag, plain or `=value`, any case of a long name, `_` or `-` between its words as llama-server
  reads them): the model and its sources, sizing, devices, the listener and API, roles (embedding,
  rerank, router, tools, web UI), the logs the engine reads, every flag that writes a file (the
  lookup caches `-lcs`/`-lcd`, logs, slot saves), the RAM prompt cache (`-cram`/`--cache-ram`,
  `--cache-idle-slots`) and the batch size (`-b`/`--batch-size`), `--jinja`, `--cache-reuse`, `-a`, every
  projector, media and draft-model flag, presets, and info-and-exit flags (`run/ownedflags.go`). A
  test reads a recorded b12651 `--help` and fails when a flag taking a file, path, directory, host,
  port or URL is neither owned nor in the local allow list (`--grammar-file`, `-jf`,
  `--chat-template-file`, `--lora*`, `--control-vector*`, `--tts-lang`).
- **Options merge.** In order (the base, then by attach): the first to state an option holds it, a
  later same value shares it, a later other value is not taken. `reasoning` replaces the engine's
  `--reasoning off`. The merge is made again from scratch at every start, so an option passes to
  the next asker when its holder leaves.
- **RAM prompt cache.** llama-server keeps a RAM cache of prompt states (`--cache-ram`, default
  8192 MiB, common/common.h:664) saved from every slot and loads the best match into whichever
  slot asks (server-slot-sched.cpp:167-206): one client's state could be loaded into another's
  slot, and cached-token counts would tell a client whether its prompt shares a prefix with
  someone else's; an erase does not clear it (server-context.cpp:393-399 clears the slot only). The
  rule: **the RAM prompt cache is on only when the engine has no listener but loopback and the slot
  map has exactly one owner** (the base and one attachment are two; the commons count as one).
  Otherwise each start passes `--cache-ram 0`: `serve --host` and `--open` never run it
  (`run.Config.RemoteListener`), so a remote caller never reads a cache hit on a prompt that was
  not its own. While it is on, a new owner is never served from vacant slots (it needs a restart,
  the grow path, which turns it off), except a return under the same key, and an untrusted caller
  (one forwarded to the loopback listener) is never served in the commons (409
  `attach_required`). `wants.ram_cache` says whether it is on. (Its 8 GiB of host RAM is not
  counted by the sizing.)
- **Agent.** One: the base profile's (as before), else the first attachment in order that brings
  one; it runs while that attachment stands. Remote clients bring none.
- **Model.** The person's selection; `models` is advice in the answer, never a swap. An attachment
  that requires `model` (`server.required`) is judged against the model that serves at every
  start, not only at its attach: a later start that serves another (a step down a tier, §5) leaves
  it unserved ("the model that serves (camp) is not one it can use"; its requests 503 `no_room`),
  and a start back on a model it can use serves it again.
- **Catalog overlay** (local callers and the `--profile` file only): `"catalog": {"models": [...]}`,
  up to 16 entries of the catalog's own model shape (`id`, `name`, `file`, `url`, `sha256`,
  `size`, and optionally `footprint`, `sampler`, ...). Each is checked by the catalog's rules (a
  plain id and file name; an https URL, or a loopback http one, with its sha256 and size; or no URL
  for an import-only entry). It adds models, never replaces one: an id already in the catalog is
  400 `catalog_invalid` unless the entry is the very same (an overlay brought again adds nothing),
  and so is a file name another model uses. Taken models stay for the engine's life (the selection
  and installs stay the person's). A remote client's `catalog` is 400 `catalog_not_allowed`: it
  would make this machine download what its person did not choose.
- **Host.** The host pid is `POST host`'s, else the base profile's `host_pid`, else the first parent
  pid, else the first local attachment's `host_pid` or pid; other programs' card use is measured.

**Whose slots.** Isolation never rests on a header the client may leave out: the engine decides
whom every request that touches a slot belongs to.

1. `X-Easy-Client: <id>` names an attachment of the caller's own identity: that attachment's
   slots, in its own numbering. Another client's id is 403 `not_yours` (a local caller may name the
   base, `profile`); an id that left is 404 `attachment_unknown`.
2. No header, a remote caller, exactly one attachment of its own with slots: that one. More than
   one: 400 `client_required` (say which in `X-Easy-Client`). (Local callers share one identity,
   `loopback`, so this rule is not theirs: the resident chat and a local tool are not a game.)
3. None of its own with slots (an attachment with no slots counts as none): the base's range when
   the base has one. The base's range is **the commons**: callers that hold nothing share it, in
   the engine's own slot numbers; a slot outside it is 400 `slot_not_offered`.
4. No base: the slots no attachment owns, if the running model has any (the commons again).
5. No commons (every slot is an attachment's, standing or vacant): a caller of this machine is
   held while the engine reshapes to give the commons a range (a local grow: drained, never
   charged), then served there; if the card has no room for it beside the attached clients it is
   503 `no_room`. A remote caller gets 409 `attach_required`: "this caller has no slots of its own
   in the running model: attach to be given slots" (also after the one client it was detached).

**The invariant.** A request runs only in slots of its own range: its attachment's, or the
commons. While the slot map holds any attachment's range, standing or vacant, no caller, local or
remote, is run unconfined, and no slot serves two owners in turn while it still holds the other's
prompt. Each slot carries whose prompt it holds (its tag: an attachment's key; `commons` for this
machine and the master key, which trust each other; `anon:<identity>` for a remote caller in the
commons). A slot changes owner only in two ways: the server restarted in between (a new process:
nothing held but what the lay-back puts back into its owner's slot), or the slot is erased
first: the door picks the slot (one already holding the caller's own prompt first, then one that
needs no erase, then the least recently used) and, when it holds another's, sends `POST
/slots/{n}?action=erase` and waits for its answer before the request goes (llama-server: the
erase waits while the slot is processing, server-context.cpp:3307-3332; it clears the slot's
tokens and its memory, :393-399, and needs `--slot-save-path`, :5834-5837, which the engine
always passes). An erase that fails is 502 `slot_not_cleared`, and the slot is taken to hold
anyone's (every next owner erases it). So a remote client may be given another's vacant range
without a restart: each slot is erased on its first use.

Nothing is confined only while the map holds no attachment's range and no untrusted caller's
prompt, and then only for a trusted caller: exactly as before clients. A remote caller with no
attachment is always confined to the commons under its own tag. What a request does with slots comes from llama-server's own routes (b12651, read
from its source; `api/paths.go`):

| Paths | Slot | At the door |
|---|---|---|
| `/completion`, `/completions`, `/v1/completions`, `/chat/completions`, `/v1/chat/completions`, `/v1/messages`, `/infill` | one, by the body's `id_slot` (a prompt array: every task on that same slot) | confined; `n`/`n_cmpl` other than absent, `null` or the literal `1` is 400 `n_not_offered` (its child tasks take any free slots, server-task.h:329; the server casts the value to int32, so `-4294967294` would be 2); `extra_params` in a `/v1/messages` body is 400 `extra_params_not_offered` (llama-server unpacks it over the converted body, server-chat.cpp:680-685, so past the door's `id_slot` and `n`) |
| `POST /slots/{n}?action=erase` | one, by the path | confined; save and restore refused (403) |
| `/embedding(s)`, `/v1/embeddings`, `/rerank(ing)`, `/v1/rerank(ing)`, `/tts`, `/(v1/)audio/speech`, `/(v1/)audio/transcriptions` (multipart), `GET /v1/realtime`, `/v1/systemone` | one the server picks | not confinable |
| `/responses`, `/v1/responses` | one the server picks (its conversion drops `id_slot`, server-chat.cpp:32) | not confinable |
| `/v1/chat/completions/control`, `POST /lora-adapters`, `POST /props`, `/predict`, any POST not in this table | another request, every slot, or unknown | not confinable |
| `GET`/`DELETE /v1/stream`, `POST /v1/streams/lookup` | none: another request's stream | conversation ids owned (below); refused to every remote caller (403) |
| `/tokenize`, `/detokenize`, `/apply-template`, the `input_tokens`/`count_tokens` paths, every `GET` (`/health`, `/metrics`, `/props`, `/models`, `/slots`, ...) | none | passed |

A path that cannot be confined is refused to every caller while the map holds an attachment's
range or an untrusted caller's prompt (403 `path_not_offered`): the server picks its slot, which
the door can neither keep to a range nor erase first. Otherwise it is passed for a trusted caller
(this machine, the master key) and refused to any other: its slot may hold another's prompt. (A paired token reaches only the completion,
tokenize and model paths at all: the gate's scope, `api/scope.go`; an `--open` caller from
another machine the same. The table holds for the master key from another machine, which the gate
does not scope.)

A confined request **fails closed**: its body must be exactly one JSON object (a byte-order mark,
an empty body, an array, trailing data, a top-level key given twice or nesting deeper than 64 is
400 `bad_request`: two readers may keep different ones of a doubled key); `id_slot`
must be an integer (or -1, or absent); `/slots/{n}` must be ASCII digits (`1x`, ` 1`, `+1` are 400:
llama-server would read each as 1). `id_slot` n in the client's numbering goes to its engine slot;
outside `0..given-1` is 400 `slot_out_of_range`. Absent or -1 takes the least recently used slot of
its range with no request on it, the pick and its mark made in one step: **send `id_slot` if you
want your cache hit**. The reply's `id_slot` is mapped back (a JSON reply whole, an event stream
line by line); a JSON reply over 64 MiB is 502 `reply_too_large`, never cut short. The header is
never forwarded. 503 `no_room` while unserved.

A body the door must look at (a JSON POST it confines, and the stream lookup it rewrites) is read
whole before the request reaches the door (a slow sender holds no slot, no count in flight, no
drain): at most 64 MiB from a local caller and 16 MiB from a remote one (413 `body_too_large`),
each read bounded by a network timeout on silence (30 s, `API.BodySilence`) and the whole read by
a second one (5 min, `API.BodyTotal`; 400 `body_unread`).

**What the engine keeps for other machines is bounded**, whatever they send and however long they
are held (`api/clientslots.go`), and each remote client (a token's identity; every `--open`
address counts as one) has its own share, so one client cannot take the others' room:

| | one remote client | all remote clients |
|---|---|---|
| bodies being read at once (each at most 16 MiB) | 2 | 8 |
| bodies read whose requests have not ended (held at the door or in flight) | 32 MiB | 128 MiB |
| requests held at the door | 8 | 32 |

So at most 8 x 16 + 128 = 256 MiB of remote bodies in all. Over any of these: 429 `busy`, saying
which (for example "32 MiB of one client", "128 MiB in all"). Every other body (multipart uploads, GETs, the unconfinable paths) streams
through as it comes, as before.

**Conversation ids.** llama-server's resumable streams key a session by the `X-Conversation-Id`
header (any case); a new request with a known id replaces and cancels that session, `GET
/v1/stream?conv_id=` replays it and `DELETE /v1/stream?conv_id=` cancels it; `POST
/v1/streams/lookup` takes `conversation_ids` (server-stream.cpp:235-253, :456-470, :510-546,
:563-604). The engine owns that name space: every id a request gives (the header, the `conv_id`
query, the lookup's list) is rewritten to one that carries its owner, hashed: the attachment's
key; else, for a caller of this machine, the commons; else the remote client's own (a remote
caller with no attachment never shares the commons, where its id could replace and so cancel a
local stream). Only the part before the first `::` is owned: the lookup also matches a stored id
that is the asked one then `::` (`<id>::<model>`, server-stream.cpp:533-536), so the rest is kept
and a lookup of an id still finds its variants. The lookup's answer is mapped back; a client
replays, cancels or finds only its own. A remote caller is refused the stream paths in every
method.

**A departed client's prompts.** A vacant range still carries the prompts of the client that held
it. It is never the commons (rules 3-4 skip it); another client given it is served only after each
slot's erase (the invariant), and with the RAM prompt cache on, only its own return is served from
it without a restart. Callers of this machine and the master key trust each other: no erase
between them.

A context saved from a client's slot goes back into that client's slot of the same number after a
reshape; one saved from a slot no client owned goes back only into the same number when no client
owns it in the new map, else it is let go (`that slot belongs to a client now`).

- samplers: `X-Easy-Sampler` is looked up in the attachment's own profiles first.
- the `request` happening's begin and end say `attachment`; the end says `client_slot` (its own
  number) beside `slot` (the engine's).

**Facts.** `attachment` (keyed by id: the answer's shape, gone when it leaves); `wants` (`{slots,
min_slots, ctx_max, ctx_floor, from, unserved: [{attachment, why}], vacant: [{first, count, key}],
server}`); `client` (§7f) gains `attachments: [ids]` once a paired client attaches; `context`
gains `client` (the key) and `client_slot`. Status says `attachments` and `wants` to the owner;
both are withheld from a paired token (they describe other clients; `api/scope_status.go`), and
facts are not offered to a paired token at all: a paired client learns of its own attachment from
the attach answer and its held stream.

### The chat window

llama-server's web UI ("Llama Chat") is on (`--no-webui` is not passed). Its conversations live in
browser storage keyed by origin and llama-server's port changes on every launch, so the engine
serves it at a fixed address, on the control port:

- `GET /chat/` is a small embedded page (`web/chat.html`): a same-origin frame holding the UI. The
  page remembers the frame's address hash (`#/chat/<id>`, how the UI names a conversation) in
  `localStorage` and restores it when it opens (an address given to the page wins), so Chat
  reopens to the same conversation (the UI's own sidebar, with its conversation list, search,
  delete and export, works inside the frame; deleting the open conversation leaves the UI on a new
  chat, and the page remembers that). The page is the top document, so the browser tab says
  *easyEngine* and shows the engine's icon (`GET /chat/icon.png`, the tray's picture); the
  frame's own title and icon do not reach the tab. While no local model is ready it is a short plain page saying
  so (no timer: the tray's Chat item is off then, and the page says to reload).
- `/chat/ui/*` is llama-server, reverse-proxied through the engine's proxy with the `/chat/ui`
  prefix stripped (streaming is flushed at once). The UI builds every asset and API address
  relative to where it was loaded, and routes by the address hash, so nothing else is needed;
  llama-server's own routes (health, metrics, slots) are not moved, which is why `--api-prefix`
  is not used.
- The tray's *Chat* item opens `http://<host>:<port>/chat/` in a new window of the user's default
  browser: no app mode, no small window, no second browser. Where the browser's launch line is
  known and the browser takes `--new-window <url>`, it is started that way: Linux resolves the
  line from `xdg-settings get default-web-browser` and the `.desktop` Exec line (Flatpaks
  included), Windows from the `https` UserChoice ProgId and its `shell\open\command` (the
  `-osint -url`, `--single-argument` and `--` words that hand a link over are dropped), macOS from
  the LaunchServices handler (`open -na "<app>" --args --new-window <url>`, for the apps in the
  `macBrowsers` table). The names that take it are the Chromium family (Chrome, Chromium, Brave,
  Edge, Vivaldi, Opera, Thorium) and the Firefox family (Firefox, LibreWolf, Waterfox, Floorp,
  Zen). Any other browser (Safari, Epiphany), an unresolved default, or a launch that fails, gets
  the page from the system's ordinary opener (`xdg-open`, `open`, `ShellExecute`), which may be a
  tab. A listener beyond loopback needs a bearer token a page cannot send, so chat is not offered
  there.

### 7c. Disk need

Every disk check counts what the setup will write from the current state, in install order,
each step at its own peak: the llama.cpp build (archive + unpacked tree side by side, then the
tree), each extra such as cudart (archive + tree, then two trees while its files are copied into
the server's dir), each model still missing less its `.part`, uv (archive + tree), and the
agent's runtime (managed Python, uv cache, venv; the catalog's `agent_runtime` for the OS or the
profile's `agent.disk_bytes`, less what of it is on disk). The need is the running maximum over
the steps (what earlier steps keep + the current step's peak), plus room for each rotating log
and the stdio log to reach its ceiling from its current size (3 x (10 MiB + 32 KiB) per rotating
log, 4 MiB for the stdio log), plus the profile's `host_files_mib`, plus `plan.DiskHeadroomMiB`
(125 MiB). One function (`run.Supervisor.setupNeed`, built on `plan.NeedOf`) serves the plan's
verdict, the install's up-front check (before a byte is written), the check before each fallback
build, the starter choice (a starter only when starter + target + the rest fit together), the
background upgrade, a switch, `disk_need_bytes`, and the import's copy fallback (when a hard link
is not possible, the whole file is checked first). A custom GGUF is used in place and writes
nothing. A refusal names the parts: `not enough disk space for <model>: needs 4.8 GB: model
server 0.3 GB, graphics runtime 0.8 GB, <model> 3.4 GB, Python installer 38 MB, agent runtime
0.1 GB, logs 95 MB, headroom 0.1 GB, 4.8 GB free (1 MB short)`. A free space that cannot be read
is an `errors[]` line in status, and the install goes on unchecked.

The headroom covers only what no count sees (other programs writing meanwhile). Evidence, poe,
2026-11-26, fresh ext4 loop filesystem, free space polled every 20 ms, a full fresh install of
E4B RP with cuda + cudart b12651, uv 0.13.10 and the agent: counted peak 4966006784 B, measured
peak 4965629952 B (the count is 368 KiB over before any headroom). With free space set to the
need + 1 MiB the install finished; at the need - 1 MiB it was refused up front with 4 KiB
written.
Windows (DESKTOP, NTFS, cuda-win + cudart-win, no venv): the files the install left (rounded
to 4 KiB) were 5192 MB against 5218 MB counted, and the cudart step's sampled peak stayed under
its count; free space on that system disk moved by up to 300 MB either way during a run from
other programs (remote-desktop traces, a WSL disk), which is what the headroom is for there.

## 8. Game profile

Written by the game's mod, passed with `--profile`. Game-agnostic fields only:

```json
{
  "game": "RimWorld",
  "slots": 3, "min_slots": 1,
  "ctx_per_slot": {"max": 65536, "floor": 8192},
  "game_reserve_mib": 1500,
  "host_pid": 12345,
  "host_adapter": {"name": "Intel(R) Iris(R) Xe Graphics", "vendor_id": 32902, "memory_mib": 128},
  "samplers": {"cold": {"temperature": 0.2, "top_k": 20}},
  "agent": {
    "id": "hermesvoice-agent",
    "wheel": "/path/inside/mod/Engine/agent/hermesvoice_agent-0.1.0-py3-none-any.whl",
    "python": "3.12",
    "args": ["run"],
    "env": {"HERMESVOICE_BRIDGE_JSON": "/path/to/bridge.json"}
  }
}
```

`server` (optional: `ubatch` 32-512, `reasoning` on/off/auto, `extra_args`, `required`) and
`models` (optional: `prefer`, `can_use`) are the launch options and model advice of §7g; the
file's are the base attachment's.

`host_files_mib` (optional) is how far the host's own files on the engine's disk may still grow,
e.g. the agent's call log at its cap times its copies; every disk check counts it in full (§7c).
`agent.disk_bytes` (optional) replaces the catalog's measured agent runtime size.

`game_reserve_mib` is the host's estimated GPU use before it has any; once the host's measured
use is above it, the measurement wins (§4a). `host_pid` is the host app's process id (omitted or
0: the pid of `POST /engine/v1/host`, else the first `--parent-pid`); `host_adapter` is the
adapter it renders on, when it knows (optional): with it the engine reserves nothing on its own
adapter when the host is elsewhere, as on a laptop whose game runs on the integrated GPU. A
`POST /engine/v1/host` overrides both at run time.

The engine installs the agent wheel into `runtime/venvs/<agent.id>` with the pinned `uv`
(`uv venv --python 3.12`, `uv pip install <wheel>`; reinstall when the wheel's sha256
changes) and runs it as `<venv>/bin/python -m <module> <args>` (the module name is derived
from the wheel's console script entry; for HermesVoice `hermesvoice_agent.cli`). uv only ever
uses its own managed Python (`UV_PYTHON_PREFERENCE=only-managed`), never one on the player's PATH.
On Windows there is no venv: a venv's `python.exe` is an unsigned launcher copy that Smart App
Control blocks, so the engine runs `uv python install --no-bin 3.12`, installs the wheel with
`uv pip install --python <base> --target runtime\venvs\<agent.id>\site`, records the base
interpreter in `base-python.txt` there, and runs `<base python> -m <module> <args>` with
`PYTHONPATH=<that site>` and `PYTHONNOUSERSITE=1`. It adds
`EASY_ENGINE_LLM_URL=http://127.0.0.1:<engine port>` (`/v1` appended in openai mode) and
`EASY_ENGINE_URL=http://127.0.0.1:<engine port>` to the agent's env (the first name is the
default of the profile's `llm_url_env`; in openai mode the same prefix also gets `_MODE=openai`
and `_MODEL=<model>`). The old names are kept too, see "Renamed from hermes-engine" below. Agent crash → restart with backoff
(1, 2, 4 … 30 s).

## 9. CLI

```
easy-engine [app]   [--port N] [--host H [--open]] [--linger D] [--yield-to-games ...]   a tray icon; stays open, waits for a connection (§6a)
                      the defaults are settings.json (§2); the flags override them for one run, never written back
easy-engine serve   --profile F --parent-pid P [--port N] [--host H [--open]] [--linger D]
                      [--yield-to-games [--yield-except IDS] [--yield-ignore NAMES] [--yield-floor-mib N]
                       [--yield-min-mib N] [--yield-grace D] [--yield-steady D] [--yield-strict]]   §6b
easy-engine setup   [--profile F]        serve + open the setup page in the browser; no parent pid needed
easy-engine probe                         prints Hardware as JSON
easy-engine plan    [--profile F]         prints the plan as JSON
easy-engine gpu     [--profile F] [--host-pid N] [--json]   adapters, processes, reserve and free memory (§4a); asks a running engine, else senses here
easy-engine install --model ID [--yes]    headless install with a text progress bar
easy-engine import  PATH
easy-engine token   [--new | --list | --revoke ID]   prints the master token for a non-loopback listener (§7); --new replaces it;
                      --list shows the paired clients (id, name, created); --revoke ID revokes one (§7f)
easy-engine find    [--wait D] [--json]   engines that answer one mDNS query on this network (default wait 2s) (§7f)
easy-engine pair    [--approve CODE | --deny CODE]   lists the pairing requests waiting at the running engine, or answers one (§7f)
easy-engine status | stop
easy-engine version
```

## 10. In game: the example integration (HermesVoice for RimWorld, in the RimWorldLLM repo)

- The mod ships the binaries in `Engine/<os>-<arch>/easy-engine[.exe]` and the agent wheel
  in `Engine/agent/`. `build.sh` here cross-compiles into `dist/<os>-<arch>/`; the csproj deploy target copies them.
- `Mod/Source/Bridge/EngineLauncher.cs` and `EngineClient.cs` (no Verse references): pick the
  binary for the platform, set the exec bit on Unix, write the profile next to `bridge.json`,
  attach to a running engine or spawn one with `--parent-pid` (hidden window), and wrap the
  control API.
- Settings, Connection section: **LLM source: Local engine (default) | My own endpoint.**
  Local shows the engine's phase light and one line (model, slots × ctx, tok/s) and a
  **Set up / Change model…** button that opens `Window_EngineSetup` (the same content as the
  setup page: hardware, recommendation, fit verdicts, Install with progress, Cancel). My own
  endpoint keeps today's fields and Test button, saved to the engine as external mode and to
  `bridge.json` as today.
- First run: if the engine says `not_installed`, the mod shows one dismissible letter
  ("HermesVoice can set up a local model: Set up / Use my own endpoint / Later").
- Nothing here blocks the game thread: all engine calls are async with short timeouts, results
  marshalled to the main thread.

## 11. Tests and done-when

- `go test ./...` with fakes for nvidia-smi output, sysfs, HTTP downloads (resume,
  corrupt hash, cancel), GGUF headers (small generated files), the fit planner (tables of
  hardware × model → shape, including the 16 GB/12 GB/8 GB/CPU cases and "dense never split"),
  and a **fake llama-server** binary (`internal/fakellama`) that honours `-c/-np/-ngl`,
  reports `/health` 503→200, says the lines the real server says in the same order (the
  listening line, a line per request taken and released), and can be told by control files
  (followed by `host.Changes`) to OOM at load or at run time or to crash. Lifecycle tests:
  parent death stops children; second `serve` attaches; stale pids killed.
- **Tests keep the clock law too:** they wait on the supervisor's hub (`testkit.WaitFor`,
  `WaitPhase`), bounded by a watchdog for silence; a test that something did not happen does the
  thing, makes the engine look (`React`, a marker on the hub, a counted read) and then asserts.
  No sleep stands in for an event.
- The sizing core (`core/`) is its own module, so the root's `go test ./...` does not reach it:
  `./core-check.sh` (run first by `build.sh`) tests it, and builds and vets it for Linux,
  Windows and macOS, at its own go line (§13).
- A purity test: no source file contains `rimworld`/`RimWorld`/`pawn`/`colon`
  (case-insensitive), as the agent core's test does.
- `Mod.Tests`: EngineClient against a fake engine; launcher binary choice per platform; profile
  JSON shape.
- **Real check on sprinter (RTX 5080 16 GB):** download the reference model from Hugging Face (or `easy-engine import` a local copy), start
  with the HermesVoice profile, reach `ready` with 3 slots, three concurrent completions answer,
  a VRAM hog (`~/.local/share/sprinter-llm/probe/vram-hog.py`) makes it shrink when idle and grow
  back after, and killing the parent stops everything. Live numbers go in `docs/ENGINE-RESULTS.md`
  (not written yet; the measured footprints are in `docs/FOOTPRINTS.md`, the disk evidence in §7c).

## 12. Not in this round

- Signing and notarising the binaries (needs the keeper's certificates); a tray icon; Player2 as a provider; Steam
  launch-option wrappers.

## 13. The sizing core (`core/`)

The footprint model, the shape chooser, the calibration book and the GGUF header reader are a
Go module of their own, so another program can size a model exactly as the engine does without
taking the engine's downloads, processes or server:

- Module `github.com/CharlesMod/easyEngine/core` in `core/`, with its own `go 1.23` line and **no
  `require` at all**. Packages: `core/fit` (Seed, SeedFromBytes, SeedWith, SeedFromBytesWith,
  Choose, MissRetry, StepDown, Grow, Args, the Book) and `core/gguf` (ReadHeader, Parse and Info,
  with the trained context, and the chat template).
- **Constants by engine build and kind of card.** The constants a footprint is built from that were
  measured, not derived (the base overhead, the vocabulary's buffers, the runtime, the mask, the
  f16 expansion, the offload buffers, the host RAM per expert layer, and since core v0.1.4 the host
  base of a card start, `host_base_mib`, which a set may lack: the host memory of a card start is
  then unknown, §5), are one set, `fit.Consts`,
  keyed by the backend (the engine build) and the kind of card it was read on, each constant with
  a word of where its reading came from. `DefaultConsts` is the set read on the RTX 5080 at b12680
  (docs/FOOTPRINTS.md) and is used wherever a caller gives none, so `Seed` and `SeedFromBytes` are
  unchanged; `SeedWith` and `SeedFromBytesWith` take a caller's own set. The footprint carries its
  set (`Footprint.Consts`; nil, as in a footprint built by hand, is the default) and every shape
  priced from it names the set (`Shape.ConstsBackend`, `Shape.ConstsCard`), through JSON, so a
  caller can tell a figure built with a set read elsewhere (a prediction) from one read on its
  own build and card. The seed builds the figures from its set once; every entry then prices the
  figures alone (the measured retry's runtime is the footprint's own `ScratchMiB`), never a
  constant read beside them, so no price mixes two sets. The book keeps the set a fixed part was
  worked out under (`Calib.Consts`; nil, as an older build wrote it, is the default) and
  `Book.Footprint` lays that fixed part only on a seed of the same set: on another set's seed the
  seed's own fixed part stands, Predicted, until a reading is taken against it. `settle` reads the
  set whole; a set lacking a constant, or its word, is `*ConstError` naming it from every entry
  that prices, never a zero.
- **The header's facts.** `gguf.ReadHeader(r, size, tensors)` walks the header once and keeps every
  metadata key whose value is an integer of any width (`Value.Int`, `Value.Uint`, its sign kept), a
  bool, a string, or an array of integers or bools of at most `MaxIntArray` (65536) elements; it
  counts what it walks past (floats, arrays of strings or floats, longer arrays) and, where asked,
  reads every tensor's name, dimensions, type, offset and byte size. `Parse` is it with the tensor
  table, and `Info` is derived from that one reading. `File.ChatTemplate` gives the chat template
  exactly as the header holds it, or says there is none, and `File.ChatTemplates` the named ones.
  A header cut short (`ErrCut`), with a wrong magic (`ErrMagic`), an unknown version
  (`*VersionError`), a string, array or count past the size given or past `MaxString` (256 MiB)
  (`ErrPastSize`), or a type the format does not have (`ErrMalformed`) is refused; a template is
  read whole or not at all, and a long string is read as it arrives, so a header never makes the
  reader allocate what it only claims. Since core v0.1.4 a header whose counts could size an
  allocation or wrap a sum is refused too (`ErrMalformed`), never a panic, an out-of-memory or a
  negative size: `<arch>.block_count` above 65,536 or below 0 (0 is read as not said, and the
  tensor names tell, as before); a block index in a tensor name (`blk.N.`) of 65,536 or more, its
  digits read with the overflow checked (an index at or past a stated `block_count` is kept: a file
  may carry extra blocks, as nextn/MTP layers, and nothing is sized from it); a tensor dimension of
  0 or past the int64 range, or an element count, byte size, offset, or sum of tensor bytes past
  it (`File.ParamCount` answers 0, not known, there).
- **What another program may ask beyond a game.** Whole (no layer or expert off the card), the
  cache's types best first, the model's own trained context as the most, and the room as given
  (the host's reserve, the margin, the largest free block, a processor node as the pool, the
  machine's free memory a card start is held to, and a field marked unread): §5, "Wants and room
  another program may state". Unset, each is the engine's own behaviour; each refusal is a typed
  error (`ErrTooBig`, `ErrNoCtxTrain`, `ErrCtxTrainUnder`, `ErrNoKVType`, `ErrNoWeights`,
  `ErrNoHostFigure`, `*HostShortError`, which is `ErrTooBig` to `errors.Is`, `*UnreadError`,
  `*KVTypeError`).
- **One checked place.** Every exported entry of `core/fit` that prices memory or reads a room
  hands its footprint, shape, room and wants to one unexported function, `settle`, before it
  prices anything, and takes every number from what it answers: the footprint priced at the
  type the shape runs at (a shape's `""` is the default q4_0, with or without wants, the type
  `Args` renders and the engine runs; with no shape, the wants' type; a footprint naming no type
  is taken as given), the room's budgets, the wants resolved and never rewritten from a shape.
  It refuses, in this order: a room field marked unread (`*UnreadError`), a largest block the
  footprint cannot be held to (`ErrNoWeights`, where wants are planned), a free host memory it
  cannot be held to (`ErrNoHostFigure`, where wants are planned on a card), wants it cannot resolve
  (`ErrNoCtxTrain`, `ErrCtxTrainUnder`, `*KVTypeError`), a shape's unknown type
  (`*KVTypeError`), a type the footprint cannot be priced at (`ErrNoKVType`). A `Footprint` and a
  `Room` themselves answer no price and no budget; the entries that do are `Choose`, `MissRetry`,
  `StepDown`, `Grow` (each refuses all five), `Book.Calibrate`, `Book.CalibrateCard`,
  `Footprint.Need`, `Footprint.HostNeed` and
  `Footprint.Largest` (a shape's type, the footprint's), `Book.LowerCap` (the room and the
  shape's type), `Book.LetGo` and `Room.Budget` (the room). `TestEntriesTableIsComplete` reads the
  package's source and fails on an exported function or method with no row in the table that
  `TestEveryEntryRefusesAlike` feeds an unread field, an unknown type in the wants, the list and a
  shape, a shape at a type not the seed's, and a shape naming no type under wants at another type
  or a list without the default.
- **What it says it knows.** Every memory figure is marked predicted or measured, a calibration
  answers what it did with a reading and by how much the seed is off, a cap carries the room and
  build it was measured under and is let go by `Book.LetGo`, and an unknown cache type is a typed
  error: §5, the calibration book.
- **What it must never do.** Its non-test code imports only pure standard-library packages
  (`errors`, `fmt`, `math`, `sort`, `strconv`, `strings`, `io`, `bufio`, `bytes`,
  `encoding/binary`, `slices`, `maps`, and `time` as a type only): never `os`, `os/exec`, `net`,
  `syscall`, `unsafe`, `log`, and never a package of the engine's root module. It never reads the
  clock or arms a timer (`Book.Calibrate` and `Book.LowerCap` take the stamp from the caller),
  never logs (`fit.Seed` returns its notes beside the footprint), and never opens a file or
  runs a process (`gguf.Parse` takes an `io.Reader` and the file's size). `core/fit`'s
  `TestCoreStaysPure` reads the core's own source and `go.mod` and fails on any of these, on a
  raised go line, and on a `require`.
- **Checked at Go 1.23.** `./core-check.sh` (run first by `build.sh`) refuses a go line other
  than 1.23, runs the core's tests, and builds and vets it with `GOOS` set to `linux`, `windows`
  and `darwin`. The root (which says `go 1.27`) is checked at 1.23 only in a scratch copy with its
  go line lowered: it builds for the three systems and its packages pass there, except the tests
  of package `run`, which use `t.Context` (Go 1.24) and do not compile at 1.23; `run` itself builds
  and vets. vet's `stdversion` check reports a standard-library call newer than the go line
  even on a newer toolchain. It never fetches a toolchain (`GOTOOLCHAIN=local`).
- **What left `fit` and `gguf`, and where it went.** The engine's package `plan` holds the
  planner of a whole catalog (`Plan`, `PlanWith`, `PlanResult`, `Verdict`, `RoomFor`, the
  CPU-only pick, the 2048 MiB least usable card), the footprint from a catalog's measured block
  (`FromMeasured`, `SeedFromSize`, which falls back to the core's `SeedFromBytes`), the speed
  scale (`SpeedScaleFor`, `OtherHardwareSpeed`), `Seed` (the core's, with its notes logged) and
  `ReadInfo`, which opens a model file and hands it to `gguf.Parse`. `catalog.TPSPoint` is the
  core's `fit.TPSPoint`, so `catalog.json` decodes as before. `run`, `tools/fitcheck` and
  `internal/testkit` import the core by its path; `catalog`, `probe` and `dirs` stay outside it.
- **In this repository** the root requires the core and points the requirement at the
  directory (`go.mod`: `require github.com/CharlesMod/easyEngine/core v0.1.0` and
  `replace github.com/CharlesMod/easyEngine/core => ./core`), so a plain checkout builds offline
  with no `go.sum` line for it. The root's `./...` does not include
  `core/`. Because of the `replace`, the engine itself is built from a checkout (`build.sh`), not
  with `go install` of the root module.
- **Tagging.** A version of the core is a tag with the directory as its prefix:
  `git tag core/v0.1.0 && git push origin core/v0.1.0` (the tag names a commit whose `core/` is
  that version; v0 and v1 take no `/vN` suffix in the path). Keep the root's `require` line on
  the newest tag. A change to the core is made here, under the core's tests, and tagged; every
  program that moves its own requirement then has it.
- **Another program** requires it as any module: `require github.com/CharlesMod/easyEngine/core
  v0.1.0` (its `go.sum` gets the module's hash on the first fetch), imports
  `github.com/CharlesMod/easyEngine/core/fit` and `.../core/gguf`, and needs only Go 1.23. With
  the module cache filled once (or a vendor directory), it builds with no network.

## 14. Renamed from hermes-engine

The engine was called `hermes-engine`. Every name that carried that product name was renamed;
the ones that are a wire or disk contract with an installed mod or a running service keep
working. All of the old names live in one package, `internal/legacy`, and nowhere else in the
code (`purity_test.go` enforces it).

| Old | New | Compatibility kept |
|---|---|---|
| Go module `hermesengine` | `github.com/CharlesMod/easyEngine` | none needed (the nested `core/` module path and its exported names did not change) |
| binary / command `hermes-engine` (`dist/<os>-<arch>/hermes-engine[.exe]`, `cmd/hermes-engine`) | `easy-engine` | none: a launcher must run the new file name |
| request header `X-Hermes-Engine: 1` | `X-Easy-Engine: 1` | the engine accepts either; the engine's own clients send both, so a new client can attach to a running old engine |
| request header `X-Hermes-Sampler` | `X-Easy-Sampler` | read from the new one, else the old one; both are removed before the request goes upstream |
| response header `X-Hermes-Sampler-Applied` | `X-Easy-Sampler-Applied` | both are sent, with the same value |
| `HERMES_ENGINE_HOME` | `EASY_ENGINE_HOME` | the new one wins; the old one is honoured when only it is set |
| agent env `HERMES_ENGINE_URL` | `EASY_ENGINE_URL` | both are set, same value |
| agent env `HERMESVOICE_LLM_URL` (default of the profile's `llm_url_env`) | `EASY_ENGINE_LLM_URL` | when the profile names no `llm_url_env`, both are set, same value; in openai mode also `..._LLM_MODE=openai` and `..._LLM_MODEL=<model>` under both prefixes. A profile that names its own `llm_url_env` gets only that variable and its prefix, as before |
| data dir Linux `$XDG_DATA_HOME/hermes-engine` | `$XDG_DATA_HOME/easy-engine` | see below |
| data dir Windows `%LOCALAPPDATA%\HermesEngine` | `%LOCALAPPDATA%\EasyEngine` | see below |
| data dir macOS `~/Library/Application Support/HermesEngine` | `~/Library/Application Support/EasyEngine` | see below |
| catalog model names "... (HermesVoice build)" | "Gemma 4 26B RP", "Gemma 4 E4B RP" (catalog `version` 9) | none (display names only) |

Unchanged on purpose: the control API paths (`/engine/v1/...`), the default port 47480, the
file names `engine.lock` and `state.json`, and the shape of `GET /engine/v1/status`.

**Data dir.** `dirs.Data()` has no side effects: with no env override it returns the new
default dir if it exists, else the old default dir if it exists, else the new one. The move is
a separate step, `dirs.Migrate()`, called once when `serve` or `setup` starts, and only when no
env override (either name) is set and the new dir does not exist but the old one does:

- if the old dir's `engine.lock` names a live engine (pid alive, start time matches), the old
  dir is used in place for this run, so the new binary attaches to the running old engine
  instead of starting a second one;
- otherwise the old dir is moved to the new path with one `os.Rename` (same file system); on
  Linux and macOS a symlink is left at the old path so an older client still finds
  `engine.lock`; if the rename fails for any reason the old dir is used in place and the
  reason is logged.

The migration never deletes anything. A client that spoke only the old names needs no change
to keep working; it should move to the new names, and a launcher must call `easy-engine`.
