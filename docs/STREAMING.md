# The token streamer (contract)

The engine's proxy reads every streamed completion reply event by event on its way to the
client. It is on by default; `"streamer": "off"` in `settings.json` turns it off (the proxy then
copies bytes exactly as before the streamer, and only `X-Request-Id` remains). The setting
applies to the next request, without a restart.

Code: package `stream` (the reader), `api/stream.go` (the proxy's side and the watchdog).

## 1. Which requests

A request is streamed when all of these hold, and only then:

- `POST` to a completion path: `/v1/chat/completions`, `/chat/completions`, `/v1/completions`,
  `/completions`, `/completion`;
- its JSON body says `"stream": true`;
- the streamer is on.

Everything else (`stream` false or missing, a body that is not JSON, any other path, any other
method) is forwarded exactly as before: the same status, headers and bytes.

Of a streamed request's replies, the reader reads only a `200` with
`Content-Type: text/event-stream` and no `Content-Encoding`. A JSON error, a whole JSON reply or
a compressed reply goes through untouched (but see §4 for a reply that comes after the engine
opened the stream itself).

## 2. What the client sees in the usual case

Every event the upstream sends goes to the client **byte for byte**, the moment its blank line
arrives (`\n\n` and `\r\n\r\n` both end an event). Nothing is rewritten, re-ordered, merged or
held back: role chunks, `prompt_progress` chunks, usage chunks with `"choices": []`, the
server's own `:` pings, `event:`/`id:`/`retry:` lines and unknown JSON all pass as they came. A
reply ends as the upstream ends it: chat and text completions with `data: [DONE]\n\n`; native
`/completion` with its last event carrying `"stop": true` (llama-server sends no `[DONE]` there,
and the engine adds none).

Headers of a stream the upstream opened (from a test, external mode, through the whole handler):

```
HTTP/1.1 200 OK
Cache-Control: no-cache
Content-Type: text/event-stream
X-Easy-Sampler-Applied: none;enforce;unknown-model
X-Request-Id: rid-6
X-Upstream-Request-Id: up-1
```

(plus the legacy copy of the sampler header, `Date` and the transfer headers). `Cache-Control:
no-cache` is added only when the upstream set none.

## 3. X-Request-Id

Every reply through the proxy, streamed or not, carries `X-Request-Id`: the client's own when it
sent one that is printable ASCII and at most 128 bytes, else 16 random hex digits
(`3f9c0a1b2d4e5f60`). An id the upstream gave its reply (a player's own endpoint often does) is
kept as `X-Upstream-Request-Id`, so the reply carries exactly one `X-Request-Id`. The id is in
every terminal event (`request_id`).

## 4. The waiting word and the quiet word

llama-server sends no byte at all while a request waits for a free slot, and many clients give up
on a connection that has said nothing for 30 s. Two comments keep the line visibly alive. SSE
clients ignore comment lines, so they are never taken for tokens.

- **Opening the stream.** If the upstream has sent no response header **2 s** after the request
  was handed to it, the engine answers itself: status 200 with

  ```
  HTTP/1.1 200 OK
  Cache-Control: no-cache
  Content-Type: text/event-stream
  X-Accel-Buffering: no
  X-Request-Id: rid-4
  ```

  (plus `X-Easy-Sampler-Applied` and `Date`), then at once:

  ```
  : easy waiting\n\n
  ```

  When the upstream then answers `200` with an event stream, its events follow in the same reply
  (its own headers are no longer needed). When it answers anything else (a 4xx/5xx JSON error, or
  the engine's own 502 when the upstream cannot be reached), that answer becomes the terminal
  event with code `upstream_error` (§5) and the stream ends.

- **Fast answers are unchanged.** An upstream that answers within 2 s (a free slot sends its
  headers in milliseconds; a bad request is refused at once) gets its own status and body through:
  a `400` stays a `400` with the upstream's JSON body.

- **Silence.** On any streamed reply, after **15 s** with no byte written, the engine writes a
  comment: `: easy waiting\n\n` before the upstream's first body byte, `: easy quiet\n\n` after it.
  A comment is only ever written between two events, never inside one. It repeats every 15 s of
  silence. A stream whose reply declared a `Content-Length` gets no comment (there is no room for
  a byte more than was promised).

Both periods are a *watchdog for silence* in the engine's law (`docs/EVENTS.md`): one timer per
streamed request, re-armed from the time of the last write, stopped when the request ends.

## 5. The terminal error event

When the **engine** ends a started stream early, the last bytes the client receives are one
event:

```
data: {"error":{"code":"<code>","type":"<type>","message":"<message>","status":<n>,"request_id":"<id>"}}\n\n
```

and the reply ends. **No `data: [DONE]` follows** (the same as llama-server's own mid-stream error
and OpenAI's). `status` is present only where an HTTP status applied. Any unfinished trailing
bytes of a cut event are dropped first, so the terminal event always begins at an event boundary.
The exact bytes, copied from the tests:

`yielded`: local mode, the engine gave the card to a game mid-reply (code and message as the 503
for that state):

```
data: {"error":{"code":"yielded","type":"unavailable_error","message":"Paused while AppId 999 (x) runs","status":503,"request_id":"rid-3"}}\n\n
```

`model_server_gone`: local mode, the model server ended the reply early for any other reason
(crash, restart, reshape, stop):

```
data: {"error":{"code":"model_server_gone","type":"server_error","message":"The model server stopped before the reply was finished","request_id":"rid-1"}}\n\n
```

`upstream_ended`: external mode, the endpoint ended the stream early:

```
data: {"error":{"code":"upstream_ended","type":"upstream_error","message":"The endpoint ended the reply before it was finished","request_id":"rid-2"}}\n\n
```

`upstream_error`: the engine had opened the stream with the waiting word and the upstream then
answered an error (here a 500 with `{"error":{"code":500,"message":"boom",...}}`; the message is
the upstream's own, or the engine's `upstream: <error>` with status 502 when it could not be
reached):

```
data: {"error":{"code":"upstream_error","type":"upstream_error","message":"boom","status":500,"request_id":"rid-5"}}\n\n
```

"Early" means: chat and text completions ended (cleanly or by a broken connection) before
`data: [DONE]`, and before any `finish_reason` (a server that sends a `finish_reason` and then
ends without `[DONE]` has finished); native `/completion` ended before an event with
`"stop": true`.

What never gets a terminal event:

- an upstream that sends its own error event (`data: {"error":{...}}`): it is forwarded as it is
  and nothing is added;
- a client that has left: nothing is written to it, and the upstream call is cancelled (llama-server
  frees the slot when the connection closes);
- a reply the reader does not read (§1), and an event longer than 4 MiB: from there the reader
  stops framing and passes the rest through raw, adding nothing.

## 6. For a client

- Treat `data: {"error":...}` as the end of the reply, whatever the code. `yielded` and
  `model_server_gone` are worth a retry after the engine is `ready` again (`GET /engine/v1/status`
  or the events stream); `upstream_ended` and `upstream_error` come from the player's endpoint.
- A reply that ends without `[DONE]` (or `"stop": true`) and without an error event did not come
  through the streamer (it is off, or the reply was compressed).
- Ignore lines that start with `:`.
