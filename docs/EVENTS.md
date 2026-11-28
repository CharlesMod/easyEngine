# Events, not clocks (design and contract)

Status: design, 2026-11-27; built through ee-stream (the stream on the wire, the door holds, the
law binds every file) except where a row below says otherwise. This file lands in the repo as `docs/EVENTS.md`. It is the contract the
event-driven redesign is built to. Where it and `docs/ENGINE.md` disagree, this file is the newer word
and ENGINE.md is brought into line by the step that builds the part.

## 1. The law

Taken from Hive 0.2 (`CLAUDE.md`, "Events, not clocks"), in the same words, so one engine serves a
game mod and the hive under one rule.

- Logic is driven by the event that means the thing: a call ending, a child's exit, a pipe's end, a
  line said, a file changed (the kernel's word), a fact on the stream, a route the other side answers
  when the thing happens.
- **Forbidden:** a sleep in a loop, a poll of a file, a port or a route, a retry cadence, a settle
  delay, a quiet window, a lease or a time-to-live standing in for a fact.
- **The honest timers are three,** each named as that one in a comment where it stands:
  a **watchdog for silence** on a child process or a stream; a **network timeout** on one call; and
  a **named moment**, a time somebody chose (the keeper's `--linger`, `--yield-grace`,
  `--yield-steady`).
- Where a system gives no event for a thing (another program's use of a graphics card on NVIDIA and
  on Windows), the read stays inside package `host`, named as a watchdog, armed only while its answer
  can change what the engine does, and nothing outside `host` knows there is one.
- If a case is found where no event exists, adding the event is the work.
- *Checked by* `laws.Clock`: a test that fails any call to `time.Sleep`, `time.Tick`,
  `time.NewTicker`, `time.After`, `time.AfterFunc` or `time.NewTimer`, in code and in tests, unless
  the call's own line or one of the two lines above carries a comment naming it `watchdog for
  silence`, `network timeout` or `named moment` (and `PageClock`: no `setInterval`/`setTimeout` in
  the page unless so named). The tree is converted (ee-stream): the ratchet list
  `laws/unconverted.txt` is empty and deleted, and the law binds every file.

## 2. Events

One shape, the hive's (`hum.Event`), so the hive's adapter can follow an engine as it follows a frame.

```go
type Event struct {
    Seq  uint64          `json:"seq"`   // per life, from 1
    At   time.Time       `json:"at"`    // data only: no choice depends on it
    Life string          `json:"life"`  // one run of the engine process
    Name string          `json:"name"`  // declared in the book, e.g. "phase", "mind.shape"
    Kind Kind            `json:"kind"`  // "fact" | "reading" | "happening"
    Key  string          `json:"key,omitempty"` // which one, where a name has several (a client, a slot)
    Gone bool            `json:"gone,omitempty"`
    Body json.RawMessage `json:"body,omitempty"`
}
```

- A **fact** stands until it is replaced or said gone. A **reading** is a level a sense says when it
  changes. A **happening** is history (a request ended, a load failed); no choice may depend on the
  history ring, only on what stands.
- Every name is declared in one **book** (name, kind, whether keyed, what the body is). Saying an
  undeclared name is an error. The book is the API's schema: `GET /engine/v1/book` serves it.
- In process: `Hub.Say(e, body)`, `Hub.SayIf(e, body, since)` (compare-and-say), and
  `Hub.Wake(prefixes...)` giving a `Waker` whose channel holds one signal: signals coalesce and a
  writer never blocks. A reader wakes, then reads what stands (`Facts`, `Fact`, `Readings`,
  `Since(seq, prefix)`). There is no subscriber queue to overflow and no slow-reader case.
- On the wire: `GET /engine/v1/facts` answers everything that stands with `life` and `seq`;
  `GET /engine/v1/events?life=L&since=N[&prefix=P]` is a held NDJSON stream, one event per line,
  flushed as said; 409 when the life differs (the client reads the facts whole and follows again);
  the stream ends, with a last line `{"gap":true}`, when the history ring (4096) has let go of
  `since`, and when the engine ends, which
  is how a client hears the engine is gone. `GET /engine/v1/status` stays, computed from the facts,
  for clients not yet moved. *Built* (ee-stream): the three paths are mounted behind the gate
  (`api/api.go`, `event/door.go`, unchanged: they agreed with this section); the `engine` fact and
  status carry `api: 1`; the `door` fact `{held}` is declared; docs/ENGINE.md §7e is the client's
  contract. `pairing` and `client` are said since ee-pair (§7); `context` is built (ee-cache, §6).
- The first names: `engine` (version, api, port, build), `phase`, `model`, `mind.shape`,
  `mind.device`, `agent` (keyed by client), `client` (keyed), `install.progress` (reading, said per
  step of bytes, never per time), `upgrade`, `build` (the llama.cpp build that runs and the move to the catalog's pin; ENGINE.md §3), `room` (reading: adapters, reserve, free for the model),
  `yield`, `recovery`, `error` (happening), `request` (happening: begin, end, slot, tokens, tok/s),
  `attachment` (keyed) and `wants` (§5),
  `context` (happening: saved, laid back, let go), `vision` (fact: the projector loaded, and why
  not) and `vision.progress` (reading: its download, said per step of bytes).

## 3. What replaces each poll

| Today | Becomes | Status |
|---|---|---|
| Watchdog tick every 5 s (`/health`, `/slots`, `/metrics`, probe, sense) | No tick. Child exit is `Proc.Done`. A hung server is a watchdog for silence on a request in flight. Idle is the door's own count reaching zero (the patched build's `easy: slot N released, idle` line tells the engine nothing the count has not and is not consumed: ENGINE.md §3). Tok/s is read from the replies the door forwards. | Built (ee-mind): exit by `Proc.Done`; hang by `HangSilence` while requests are in flight, put back by the server's lines, Begin/end and the proxy's `Heard` on each write, made again after it fires; idle by the door's count. Tok/s from the replies: a streamed reply's own `timings` (the streamer's summary, said in the `request` end, ee-stream); a non-streamed one still has one `/metrics` read when the count reaches zero. |
| `/health` every 250 ms while loading | The server's own lines: after `listening on`, `/health` is asked once per following line; silence is a watchdog. | Built (ee-mind); fixtures in `run/testdata/ready-*.txt`. A request held at the door while loading is a word too: `/health` is asked once on it (ee-stream). |
| Parent pid poll every 1 s | `host.Exits(pid)`: pidfd (Linux), kqueue `EVFILT_PROC` (macOS), a wait on the process handle (Windows). | Built (`host.ExitsOf`). |
| Yield loop 250 ms / 1 s | Opens of the engine adapter's device node by inotify with no cap (Linux; an NVIDIA adapter at `/dev/nvidia<minor>` alone, see ENGINE.md); Steam's `RunningAppID` by registry notification (Windows) and file change (macOS); a stranger's end by `host.Exits`. What has no event (a launcher's argv on Linux, a process's memory on NVIDIA) is a `host` watchdog. | Built (ee-mind); the Windows and macOS watches are compile-checked only. |
| Sense the card every 5 s | Read when needed, never watched: at a start, before a grow, on a refused start or an out-of-memory line, on a change of the programs holding the card's node (an open is the word, a scan of who holds it the test) or a stranger's end, on the host's word (`POST host`), on a client's ask. The one `host` watchdog left is the level of the card while a mind shares it with a living program that gives no event. | Built (ee-mind), except that the `host.Level` watchdog (`RoomEvery`) is armed for as long as a model is loaded, not only while a stranger shares the card. Device opens off Linux are compile-checked only. |
| Sleep loops waiting on our own state (drains, phase waits, job waits) | A `Waker` on the fact that means it. | Built (ee-mind). |
| Crash, agent, download and upgrade backoff | No cadence. One restart at once; a second failure of the same kind waits for an event that could change the outcome: the room changed, a request arrived, a client or the keeper said so, the network came back (`host`). The fact `recovery` says what it waits for. | Built (ee-mind): `recovery.waiting_for`, `retry_in_s` always 0. The words: a request at the door (said as a nudge while something waits), a client's attach, settings, start, wake, install or `POST host` (api calls `Nudge`), the room (watched while a start waits for it), `host.NetworkChanges` (Windows and macOS compile-checked only). Downloads listen before each attempt. |
| `503` + `Retry-After` during a load or reshape | The door holds the request and answers it when the mind is up (the client's own timeout bounds it). `yielded`, `not_installed` and `error` are facts and are answered at once. | Built (ee-stream): `Route.Hold`, `Supervisor.Enter/Hold`, `api.hold` waits on the hub and the client's context; a held streamed request gets the waiting word; held requests are not in flight. The phase is said before a stop (yield, sleep, stop, shutdown). Rules: docs/ENGINE.md §7e. |
| Clients polling `status` (page 1 s, mod 2 s, CLI 300 ms, agent 60 s) | The stream of §2. | Built for the engine's own clients (ee-stream): the setup page and the CLI follow `events`; the tray's tooltip and the resident linger wake on the hub; the app's lock loop waits on `host.Changes` and `host.Exits`. The mod and the agent are theirs to move. |

## 4. Parts

- `event`: the hub and the book. Pure standard library.
- `host`: every word the OS gives (process exits, file and registry changes, device opens, the
  network coming back) and the only watchdogs for what it does not give. Nothing outside `host`
  shells out on a clock.
- `laws`: the law test.
- `mind`: one model server on one card, as a value with no package globals: sense, choose (`core/fit`),
  start, ready, serve, drain, save, stop. Several can live in one process. This is what another
  program embeds.
- `door`: the one place a call reaches a mind: admission, the in-flight count, holding during a
  reshape, the sampler policy, the saved contexts.
- `engine`: clients and their profiles, wants merged across clients, installs, the catalog, the
  agents, and which minds exist. The binary is `engine` plus the HTTP face.

## 5. Clients (one engine, several mods)

*Built* (ee-clients); the wire is docs/ENGINE.md §7g. Two words: a **client** is who connects
(a paired token's `client_id`, or the local machine, `loopback`); an **attachment** is one use of
the engine by a client, with its profile, keyed by (identity, name). The `--profile` file is the
**base** attachment, first in every order, for the engine's life.

- **Presence, by events only.** A local caller may give a pid: the attachment leaves when it ends
  (`host.ExitsOf`). Any caller may hold its attach (`hold: true`): the POST's response stays open
  as NDJSON and the attachment leaves when that connection ends; a peer that vanished is found by
  TCP keepalive on the listener (the network timeout) and by a write that fails. `DELETE
  attach/{id}` detaches; a revoked paired token detaches all its client's attachments at once
  (`{"end":"revoked"}`). No lease, no clock.
- **Wants.** The attachments' merged: slots summed, min slots summed, the largest ctx and floor
  (llama-server gives every slot one ctx, so the largest floor is everyone's), a remote client's
  floor clamped to the local attachments' and the base's (it never raises theirs; a shape below
  its own floor leaves it unserved, never a reshape's cause) and its ctx max to the largest max of
the base, the local attachments and the defaults. Caps: one remote identity at most 16 slots,
all remote attachments at most 32 slots and 16 attachments (every `open@` caller one identity,
one budget), never counted against a local caller's. None stating any: the profile (the
  base, else the defaults), exactly as before. A merge the fit refuses even at the floor leaves
  one attachment out (remote before local, later before earlier, never the base; unserved, said
  with why) until the next start.
- **The running mind's slot map changes only at a start.** Each attachment gets a range (each its
  want when all fit, else its min slots, then the rest in order). One that leaves is said gone and
  counts for nothing in the next merge, but its range stays, **vacant**, until the next start;
  each of its slots is erased in the server only when another owner's request first uses it. An attach of the same key with the same count takes it back
  (`resumed`, the cache in it); an attach of a key that stands replaces its presence
  (`replaced`, the old held stream ends `{"end":"replaced"}`); otherwise vacant slots serve a new
  attachment when they suffice, the running ctx is at least its floor and its options change
  nothing and the RAM prompt cache is off (with one owner it is on, and a new owner restarts the
  server, where it is then off). Only otherwise does the mind grow. A local caller's grow drains
  (requests held at the door, those in flight finish or are cut when the named moment `DrainMax`
  comes, each stream with its terminal event), then a start with the new merge; a remote caller's
  waits for the door's next empty moment (the `idle` cause), never a drain. The last client's
  leave reshapes nothing: its vacant range waits for its return (no restart, no charge), for the
  model to stop for another reason, or for a local caller the stale shape under-serves (whose
  request is held while it reshapes). Another local leave reshapes, when idle, only if that gives
  someone something; a remote leave never reshapes by itself (a waiting attachment's own reshape is
  charged to it). Every restart of a running server is a local caller's want, the engine's own reason, or
  charged to the remote identity whose attach or change caused it, 4 per waking period: a start
  from down is never charged (an attach, return or change while no server runs is folded into the
  next start), and a linger unload, a sleep asked from this machine, the person's stop or a yield
  makes every budget whole again (never a stop a remote client could cause). A return while the
  model runs is free only when served from its own standing range. A vacant range is never the
  commons; another client given it is served after each slot's erase. One restart charges an
  identity once; a change of wants made while a start loads is charged as a late attach is; a start
  whose port another process took is started again at once on a new port (the same restart, no
  failure counted: the launch's exit is the event). Nothing is held
  without an event that ends the hold: a client with nothing coming is answered 503 at once.
- **Slots belong to clients.** The engine decides whose every slot-touching request is: the
  attachment it names (of the caller's own identity), else a remote caller's one attachment with
  slots, else the commons (the base's range, or the slots nobody owns), else, for a local caller,
  held while the engine reshapes to give the commons a range, and for a remote one 409
  `attach_required`. The invariant: a request runs only in its own range (its attachment's or the
  commons); while the map holds any attachment's range no caller is run unconfined; a slot changes
  owner only by a restart or an erase the door sends and waits for before the request (each slot
  tagged with whose prompt it holds). The RAM prompt cache is on only with no listener but loopback
  and one owner. Local callers and the master key trust each other; an `--open` remote
  caller is held to the paired allow list; the master key from another machine is unrestricted.
  `n`/`n_cmpl` other than absent, null or the literal 1 in a confined body is 400 `n_not_offered`
  (llama-server would take free slots for the children; it casts the value to int32), and a
  top-level key given twice is 400. A request's slot work is read from llama-server's own
  routes; a path that cannot be held to one range is refused to every caller while an attachment
  holds a range, and always to an untrusted one. `extra_params` in a `/v1/messages` body is 400. A confined body that is not exactly one JSON object is refused (fail
  closed). `id_slot` in its own numbering is mapped to its engine slot, the reply's mapped back;
  absent or -1 takes its least recently used free slot (pick and mark one step). Bodies are read
  before the door (only the JSON POSTs the door looks at; 64 MiB local, 16 MiB remote), each read
  under a network timeout on silence and a total bound. Each remote client has its own share: 2
  bodies read at once (8 in all), 32 MiB kept until their requests end (128 MiB in all), 8 requests
  held at the door (32 in all): 429 `busy` past any; 256 MiB of remote bodies at most. Conversation ids of resumable
  streams carry their owner (a remote caller with no attachment its own, never the commons; an
  id's `::<model>` tail kept). Saved contexts are keyed by the
  attachment's key and its own slot number, and laid back into the slot that key has after a
  start that renumbered; a key with no standing attachment at the lay-back is let go, and a slot
  no client owned goes back only into the same unowned number.
- **Launch options** (`server`: `ubatch`, `reasoning`, a local client's `extra_args`): the first in
  order to state one holds it, a later different value is told `not_taken` (409 when it marked the
  option `required`); each is checked against the installed build's `--help` and never passed
  blind (no readable help: `extra_args` are not taken). A server that exits while starting with
  clients' `extra_args` is started once more at once without them (told not taken) after any exit
  that is not out of memory. `--cache-ram` is the engine's: 0 when the slot map has more than one
  owner. Engine-owned flags are refused in every spelling the help gives. One agent: the base's,
  else the first attachment that brings one. The model stays the person's selection; a client's
  `models` is advice.
- **Facts**: `attachment` (keyed), `wants` (with `unserved` and `vacant`), `client` gains
  `attachments`, `request` and `context` gain the attachment and the client's slot number. A
  local profile may bring a catalog overlay: models added (never replacing one), checked by the
  catalog's own rules, kept for the engine's life.

## 6. The prompt cache across a reshape

Every planned stop of a healthy mind (shrink, grow, yield, sleep, a person's stop, the restart
after an install) saves each slot that holds tokens (`GET /slots` says which;
`POST /slots/{i}?action=save`, into `--slot-save-path`, the engine's own `<data>/contexts/`),
after the drain and before the stop, and the next start lays each one back (`action=restore`)
once the server is up and before the phase says ready, where its **context key** (model file
hash, llama build with its patch set, cache type) is the same, its slot exists and its tokens fit
the new context per slot; else it is let go. Saved files are let go when laid back, when the
model, build or cache type changes, at engine start and end, and when the disk runs short; the
engine removes only the files its index (`contexts/index.json`) names. The size is estimated
from the sizing core and checked against the free disk; with too little nothing is saved. A
crash, a hang and an out-of-memory restart save nothing; neither does a drain cut short. A yield
saves only what fits `YieldSaveMaxMiB` unless `keep_context` is `always` (`off` saves nothing).
With the door holding requests (§3), a reshape is: drain, save, stop, start, lay back, release; a
client sees a slower reply, not an error and not a cold prompt. Each step is said as the
happening `context` and status `contexts` lists what is kept. *Built* (ee-cache), with the
measurements and the yield rule in docs/ENGINE.md §5, "The prompt cache across a restart".
Sliding-window models (Gemma) keep their context checkpoints across a save and restore on the
llama.cpp b12651, stock and the patched pin built on it, b12651-easy1 (upstream "server : preserve context checkpoints across slot
save/restore", #26004); on poe a laid-back Gemma 4 slot was continued with only the new turn
processed.

## 7. Security

- Every path, the proxy's included, checks `Host` (loopback names, or the bound address) and
  refuses a request whose `Origin` is not the engine's own. `Access-Control-*` headers from the
  model server are dropped.
- A listener that is not loopback requires a token on every request (`Authorization: Bearer`),
  made at first use and kept in the data dir (0600), shown by `easy-engine token`. `--open` turns
  that off by the keeper's own word.
- A 401 body says why with a code beside its message: `token_required`, `token_revoked` (revoked
  hashes are remembered in `tokens.json`) or `token_invalid`.
- *Built* (ee-pair): **hello, discovery and pairing** (docs/ENGINE.md §7f). `GET hello` needs no
  token and never 401s; it leaks nothing secret. A non-loopback listener announces
  `_easy-engine._tcp` by mDNS (TXT `id`, `api`, `name`), at start, on `host.NetworkChanges` and
  in answer to a query, never on a clock; the `discovery` setting turns it off. A client pairs
  with `POST pair {name}` -> `{id, code}` and a held `GET pair/{id}`; the person compares the
  six-digit code and answers on the setup page, in the tray or with `easy-engine pair`. Answering
  needs loopback or the master token, never a paired client's token. No expiry timer and no rate
  clock: one waiting request per source address (a new ask replaces it under the same key and is not notified again, so an answer names the code
  the person compared and a stale one is refused, `pair_changed`; a
  denied address is listed, never notified), at most 4 across addresses, the oldest pushed out. The token is issued per
  client (hashed, named, listed, revocable) when the client collects an Allow, shown once, and
  is never logged, in a URL or on the stream. The facts `pairing` (keyed by a public key, never
  the secret request id) and `client` (keyed by client id, `{name}`), gone at once on a revoke (the app
  follows `tokens.json`) say who waits and who
  connected.
- `PUT settings` changes only the fields it names. The lock is created with its content in one
  step. `state.json` has one writer, a unique temp file and an fsync; a file that does not parse is
  set aside and said as an error, never silently replaced.

## 8. The app: always open, a tray icon, and a mind on demand (keeper, 2026-11-27)

The engine no longer starts and ends with a game. It is an app that stays open and waits for a
connection.

- `easy-engine` with no command (and `easy-engine app`) runs the engine with a tray icon on Linux,
  Windows and macOS. `serve` stays, for a service or a shell with no desktop; it shows no tray. On
  Windows `easy-enginew.exe` is the same program built windowless, for login start.
- The tray menu, for now, is four items: **Chat** (llama-server's web UI in a new browser window; on only
  while a model is ready; `docs/ENGINE.md` "The chat window"), **Launch at Startup** (a checkmark; package `autostart`:
  the Run key, a LaunchAgent, an XDG autostart entry, all for the current user), **Settings** (opens
  `settings.json` with the system's own editor) and **Quit** (drain, stop the mind and the agents,
  end). The tooltip says what stands (the phase and the model). With no tray to show in, the engine
  runs without one.
- **`settings.json`** in the data dir is the keeper's file: the mode and external endpoint, linger,
  cache type, sampler policy, the listener (host, port, open or token), the yield options. It is
  written whole with every key, read at start, and followed by `host.Changes`: a valid edit takes
  effect when the file is saved; an edit that does not parse is said as an error and the last good
  settings stand; the engine never overwrites a file it could not read. `PUT settings` writes the
  same file. `state.json` keeps only what the engine itself learns (the book, the chosen model,
  the upgrade in flight). On the first start after this change the settings held in `state.json`
  move to `settings.json`.
- **A mind on demand.** The app starts with no model loaded. A client connecting (an `attach`, a
  paired client's first call, a request at the door) is the event that loads it; the door holds
  that first request until the mind is up. When the last client is gone (its pid ended, by
  `host.Exits`, or its connection closed for a remote client) and nothing is in flight, the keeper's
  `linger` (a named moment) runs and the mind is unloaded; the engine stays. *Built* (ee-stream):
  a remote client is counted by its requests at the door (in flight or held) and, since
  ee-clients, by its attachments (held or by pid, §5); the linger is armed when the door is empty
  and no pid or attachment stands, and a request or an attach cancels it. A mod does not start
  the engine with a parent pid any more: it finds the engine (the lock, hello), attaches with its
  profile, and only if none runs on its own machine starts the app detached.
- The tray is also where a pairing request is answered (§7): a notification "Allow <client>
  (493 201)?", and while one waits an Allow and a Deny item above the standing four. *Built*
  (ee-pair).
