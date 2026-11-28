// fakellama is a fake llama-server for the engine's tests. It accepts the real
// flags, serves the endpoints the engine and its clients use, and has knobs
// (environment variables and control files) to misbehave on demand:
//
//	FAKELLAMA_CTL              dir with control files: busy (slots show processing),
//	                           oom (die with an out-of-memory error), and the outputs
//	                           starts.log and vram.<pid> (fake VRAM use for fake nvidia-smi)
//	FAKELLAMA_HEALTH_DELAY     how long /health answers 503 (Go duration, default 0)
//	FAKELLAMA_READY_LINES      "none": no listening line is ever printed (a build worded
//	                           otherwise); "early": the ready lines are printed at once and
//	                           /health turns 200 only after FAKELLAMA_HEALTH_DELAY, with no
//	                           line after it
//	FAKELLAMA_OOM_ABOVE        at load, -c above this value prints llama.cpp's allocation
//	                           failure (with a buffer size) and exits 1
//	FAKELLAMA_BYTES_PER_TOKEN  buffer bytes per ctx token for that message (default 52428)
//	FAKELLAMA_VRAM_BASE_MIB / FAKELLAMA_VRAM_PER_TOKEN  fake VRAM = base + per_token * ctx
//	FAKELLAMA_HIDE             comma list of flags the help text omits and the parser rejects
//	FAKELLAMA_DIE_ON           a flag the server takes but exits on while starting
//	FAKELLAMA_CACHE_RAM        set: the help lists -cram, --cache-ram (the RAM prompt cache)
//	FAKELLAMA_STREAM_DELAY     gap between streamed chunks (default 100ms)
//	FAKELLAMA_CHAT_DELAY       delay of a non-streaming completion (default 0)
//	FAKELLAMA_TIMINGS          set: a stream ends with llama-server's last chunk (finish_reason,
//	                           id_slot 0, timings with predicted_per_second 50) before [DONE]
//	FAKELLAMA_SLOT_BYTES       bytes a saved slot file holds per token (default 16)
//	FAKELLAMA_ID_SLOT          set: every reply says the slot it used (id_slot in the JSON reply, in
//	                           every streamed chunk and in the timings chunk)
//	(every completion request body is saved to <ctl>/lastbody.json, every /slots/{id} action
//	asked is logged to <ctl>/slotacts.log; /infill answers as /completion)
//	FAKELLAMA_DEVICES          text printed by --list-devices (default "CUDA0: Fake GPU"); ";" separates lines
//	FAKELLAMA_DEVICES_PINNED   what --list-devices prints when CUDA_VISIBLE_DEVICES is set (default: the
//	                           first listed device as CUDA0; "none" prints no device)
//	(starts.log records CUDA_VISIBLE_DEVICES as cvd=)
//	more control files in FAKELLAMA_CTL:
//	  crash      one shot: print a non-memory fault and exit 1 (removed first)
//	  crashloop  every start whose -c is above the number in the file (empty: every start) exits 1
//	             with a non-memory fault shortly after it turns healthy; the file stays
//	  oomlog     one shot: print CUDA out-of-memory lines and keep serving (a survived OOM)
//	  oomreq     while present, completions fail with HTTP 500 and an out-of-memory body, and
//	             print the CUDA out-of-memory lines (the server survives)
//	  oom_above  like FAKELLAMA_OOM_ABOVE, read at each start (it wins over the env)
//	  loadfail   while present, every start fails while loading with a non-memory error
//	  loadgate   while present (a FIFO the test made), a start waits while loading until the test
//	             writes to it and closes it: a start held in its loading window, by an event
//	  savefail / restorefail  while present, slot save / restore answer 500 (nothing written)
//	  slots.log  written: one line per slot save ("save <id> <name> <n>") and restore attempt
//	             ("restore <id> <name> ok|fail"); needs --slot-save-path to be served at all
//	  healthy    written (with the pid) when /health turns 200
//
// It says the lines a real llama-server says, in the same order (run/testdata/ready-*.txt): once
// the model is loaded and it listens, "llama_server: model loaded", "llama_server: listening on
// http://..." and "update_slots: all slots are idle"; per request "launch_slot_ ... processing
// task", then "release: ... stop processing" and, when none is left, "all slots are idle". The
// engine hears that it is ready from those lines, as it does with the real server. Control files
// are followed by the kernel's word of their change (host.Changes), never polled.
package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CharlesMod/easyEngine/host"
)

var withValue = map[string]bool{
	"-m": true, "-ngl": true, "--n-cpu-moe": true, "-np": true, "-c": true, "-ctk": true, "-ctv": true,
	"-fa": true, "--cache-reuse": true, "--host": true, "--port": true, "-a": true, "--reasoning": true,
	"--alias": true, "--device": true, "--split-mode": true,
	"--temp": true, "--top-k": true, "--top-p": true, "--min-p": true, "--repeat-penalty": true, "--dry-multiplier": true,
	"--typical": true, "--repeat-last-n": true, "--presence-penalty": true, "--frequency-penalty": true,
	"--dry-base": true, "--dry-allowed-length": true, "--dry-penalty-last-n": true,
	"--slot-save-path": true,
	"-ub":              true, "--ubatch-size": true, "--top-n-sigma": true, "--spec-type": true,
	"--cache-ram": true, "-cram": true,
}
var bare = map[string]bool{"--jinja": true, "--no-webui": true, "--metrics": true, "--slots": true, "--no-kv-unified": true}

func helpText(hidden map[string]bool) string {
	var b strings.Builder
	b.WriteString("----- common params -----\n\n")
	lines := []string{
		"-h,    --help                  print usage and exit",
		"--version                      show version and build info",
		"--list-devices                 print list of available devices and exit",
		"-m,    --model FNAME           model path",
		"-ngl,  --gpu-layers N          max. number of layers to store in VRAM",
		"--n-cpu-moe N                  keep the MoE weights of the first N layers in CPU",
		"-np,   --parallel N            number of parallel sequences",
		"-c,    --ctx-size N            size of the prompt context",
		"-ctk,  --cache-type-k TYPE     KV cache data type for K",
		"-ctv,  --cache-type-v TYPE     KV cache data type for V",
		"-fa,   --flash-attn [on|off|auto]  set Flash Attention use",
		"-kvu,  --kv-unified, -no-kvu, --no-kv-unified   use single unified KV buffer",
		"-rea,  --reasoning [on|off|auto]        Use reasoning/thinking in the chat",
		"--jinja                        use jinja template for chat",
		"--cache-reuse N                min chunk size to attempt reusing from the cache",
		"--no-webui                     disable the Web UI",
		"--metrics                      enable prometheus compatible metrics endpoint",
		"--slots                        enable slots monitoring endpoint",
		"--slot-save-path PATH          path to save slot kv cache (default: disabled)",
		"-dev,  --device <dev1,dev2,..> comma-separated list of devices to use for offloading",
		"-sm,   --split-mode {none,layer,row}  how to split the model across multiple GPUs",
		"--host HOST                    ip address to listen",
		"--port PORT                    port to listen",
		"-a,    --alias STRING          set alias for model name",
		"--temp, --temperature N        temperature (default: 0.8)",
		"--top-k N                      top-k sampling (default: 40)",
		"--top-p N                      top-p sampling (default: 0.95)",
		"--min-p N                      min-p sampling (default: 0.05)",
		"--typical, --typical-p N       locally typical sampling",
		"--repeat-last-n N              last n tokens to consider for penalize",
		"--repeat-penalty N             penalize repeat sequence of tokens",
		"--presence-penalty N           repeat alpha presence penalty",
		"--frequency-penalty N          repeat alpha frequency penalty",
		"--dry-multiplier N             set DRY sampling multiplier",
		"--dry-base N                   set DRY sampling base value",
		"--dry-allowed-length N         set allowed length for DRY sampling",
		"--dry-penalty-last-n N         set DRY penalty for the last n tokens",
		"-ub,   --ubatch-size N         physical maximum batch size",
		"-cram, --cache-ram N           RAM prompt cache size in MiB (FAKELLAMA_CACHE_RAM: listed)",
		"--top-nsigma, --top-n-sigma N  top-n-sigma sampling",
		"--spec-type TYPE               speculative decoding type",
	}
	for _, l := range lines {
		skip := strings.Contains(l, "--cache-ram") && os.Getenv("FAKELLAMA_CACHE_RAM") == ""
		for h := range hidden {
			if strings.Contains(l, h+" ") || strings.HasSuffix(l, h) || strings.Contains(l, " "+h+",") {
				skip = true
			}
		}
		if !skip {
			b.WriteString(l + "\n")
		}
	}
	return b.String()
}

var (
	inflight int32
	tasks    int32
	outMu    sync.Mutex
)

// say prints lines together, as one goroutine of the server would.
func say(lines ...string) {
	outMu.Lock()
	defer outMu.Unlock()
	for _, l := range lines {
		fmt.Println(l)
	}
}

// begin and finish say a request's lines as the real server does.
func begin() int32 {
	atomic.AddInt32(&inflight, 1)
	t := atomic.AddInt32(&tasks, 1) - 1
	say(fmt.Sprintf("slot launch_slot_: id  0 | task %d | processing task, is_child = 0", t))
	return t
}

func finish(t int32) {
	n := atomic.AddInt32(&inflight, -1)
	l := []string{fmt.Sprintf("slot      release: id  0 | task %d | stop processing: n_tokens = 8, truncated = 0", t)}
	if n == 0 {
		l = append(l, "srv  update_slots: all slots are idle")
	}
	say(l...)
}

// slot is one slot's token state: a token count, and the prompt text it last held (so a cache
// hit is a common prefix, as llama-server's is: a prompt another caller never sent hits nothing).
// A restored slot holds a count with no text: any prompt as long hits it, as before.
type slot struct {
	tokens  int    // tokens the slot holds now
	decoded int    // predicted tokens of the last answer
	held    bool   // has held a task
	task    int32
	text    string // the prompt text it last held ("" unknown: restored, or never used)
}

var (
	slotMu       sync.Mutex
	slots        []slot
	slotSavePath string // --slot-save-path
)

// promptText is a request body's prompt text: the prompt, then every message's text content.
func promptText(raw []byte) string {
	var b struct {
		Prompt   any `json:"prompt"`
		Messages []struct {
			Content any `json:"content"`
		} `json:"messages"`
	}
	json.Unmarshal(raw, &b)
	var t strings.Builder
	if p, ok := b.Prompt.(string); ok {
		t.WriteString(p)
	}
	for _, m := range b.Messages {
		if c, ok := m.Content.(string); ok {
			t.WriteString(c)
		}
	}
	return t.String()
}

// promptTokens is the fake's token count of a request body: ceil(bytes of prompt text / 4).
func promptTokens(raw []byte) int {
	var b struct {
		Prompt   any `json:"prompt"`
		Messages []struct {
			Content any `json:"content"`
		} `json:"messages"`
	}
	json.Unmarshal(raw, &b)
	n := 0
	if p, ok := b.Prompt.(string); ok {
		n += len(p)
	}
	for _, m := range b.Messages {
		if c, ok := m.Content.(string); ok {
			n += len(c)
		}
	}
	return (n + 3) / 4
}

// account books a completion request on its slot and returns what the answer's timings say.
func account(raw []byte, task int32, predicted int) (cacheN, promptN, id int) {
	var b struct {
		IDSlot *int `json:"id_slot"`
	}
	json.Unmarshal(raw, &b)
	id = 0
	if b.IDSlot != nil && *b.IDSlot >= 0 {
		id = *b.IDSlot
	}
	slotMu.Lock()
	defer slotMu.Unlock()
	if id >= len(slots) {
		id = 0
	}
	sl := &slots[id]
	nt, text := promptTokens(raw), promptText(raw)
	switch {
	case sl.text != "": // a hit is the common prefix of what the slot holds and the new prompt
		p := 0
		for p < len(sl.text) && p < len(text) && sl.text[p] == text[p] {
			p++
		}
		cacheN = min(p/4, sl.tokens)
		if p == len(sl.text) && nt >= sl.tokens { // the whole held prompt (and its answer) again
			cacheN = sl.tokens
		}
		promptN = nt - cacheN
	case sl.tokens > 0 && nt >= sl.tokens:
		cacheN, promptN = sl.tokens, nt-sl.tokens
	default:
		promptN = nt
	}
	sl.text = text
	sl.tokens = nt + predicted
	sl.decoded = predicted
	sl.held = true
	sl.task = task
	return cacheN, promptN, id
}

// recordBody writes the last completion request body to <ctl>/lastbody.json so tests can see
// what the engine's proxy forwarded.
func recordBody(raw []byte) {
	if ctl := os.Getenv("FAKELLAMA_CTL"); ctl != "" {
		os.WriteFile(filepath.Join(ctl, "lastbody.json"), raw, 0o644)
	}
}

// printOOM prints what a CUDA llama-server prints when a compute allocation fails mid-run.
func printOOM() {
	say("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 512.00 MiB on device 0: cudaMalloc failed: out of memory",
		"ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 536870912",
		"decode: failed to decode, ret = -3")
}

func main() {
	hidden := map[string]bool{}
	for _, h := range strings.Split(os.Getenv("FAKELLAMA_HIDE"), ",") {
		if h = strings.TrimSpace(h); h != "" {
			hidden[h] = true
		}
	}
	args := os.Args[1:]
	for _, a := range args {
		switch a {
		case "--version":
			fmt.Println("version: 1 (fake)")
			return
		case "--list-devices":
			d := os.Getenv("FAKELLAMA_DEVICES")
			if d == "" {
				d = "CUDA0: Fake GPU (16384 MiB, 16000 MiB free)"
			}
			if os.Getenv("CUDA_VISIBLE_DEVICES") != "" {
				first, _, _ := strings.Cut(d, ";")
				_, desc, _ := strings.Cut(first, ":")
				d = "CUDA0:" + desc
				if p := os.Getenv("FAKELLAMA_DEVICES_PINNED"); p == "none" {
					d = ""
				} else if p != "" {
					d = p
				}
			}
			fmt.Println("Available devices:")
			for _, l := range strings.Split(d, ";") {
				if l = strings.TrimSpace(l); l != "" {
					fmt.Println("  " + l)
				}
			}
			return
		case "--help", "-h":
			if ctl := os.Getenv("FAKELLAMA_CTL"); ctl != "" { // each help read is logged: <ctl>/help.log
				if f, err := os.OpenFile(filepath.Join(ctl, "help.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
					fmt.Fprintln(f, "help")
					f.Close()
				}
			}
			fmt.Print(helpText(hidden))
			return
		}
	}
	vals := map[string]string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if hidden[a] {
			fmt.Fprintf(os.Stderr, "error: invalid argument: %s\n", a)
			os.Exit(1)
		}
		switch {
		case withValue[a]:
			if i+1 >= len(args) {
				fmt.Fprintf(os.Stderr, "error: missing value for %s\n", a)
				os.Exit(1)
			}
			vals[a] = args[i+1]
			i++
		case bare[a]:
			vals[a] = "1"
		default:
			fmt.Fprintf(os.Stderr, "error: invalid argument: %s\n", a)
			os.Exit(1)
		}
	}
	np, _ := strconv.Atoi(vals["-np"])
	if np == 0 {
		np = 1
	}
	c, _ := strconv.Atoi(vals["-c"])
	slots = make([]slot, np)
	slotSavePath = vals["--slot-save-path"]
	ctl := os.Getenv("FAKELLAMA_CTL")
	if ctl != "" {
		f, err := os.OpenFile(filepath.Join(ctl, "starts.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err == nil {
			fmt.Fprintf(f, "pid=%d np=%d c=%d ngl=%s moe=%s ctk=%s cvd=%s bin=%s args=%s\n", os.Getpid(), np, c, vals["-ngl"], vals["--n-cpu-moe"], vals["-ctk"], os.Getenv("CUDA_VISIBLE_DEVICES"), os.Args[0], strings.Join(args, " "))
			f.Close()
		}
	}
	// FAKELLAMA_DIE_ON: a flag this build takes but dies on while starting (a client's extra_args
	// the model server does not start with)
	if die := os.Getenv("FAKELLAMA_DIE_ON"); die != "" {
		if _, ok := vals[die]; ok {
			fmt.Fprintf(os.Stderr, "error: %s: failed to start\n", die)
			os.Exit(1)
		}
	}
	fmt.Println("srv  llama_server: initializing ...")
	fmt.Println("fakellama: loading model", vals["-m"])
	if ctl != "" {
		if _, err := os.Stat(filepath.Join(ctl, "loadfail")); err == nil {
			fmt.Println("llama_model_load: error loading model: tensor 'blk.0.attn_q.weight' data is not within the file bounds")
			os.Exit(1)
		}
		if f, err := os.Open(filepath.Join(ctl, "loadgate")); err == nil { // blocks until the test writes
			io.ReadAll(f)
			f.Close()
		}
	}
	above, _ := strconv.Atoi(os.Getenv("FAKELLAMA_OOM_ABOVE"))
	if ctl != "" {
		if b, err := os.ReadFile(filepath.Join(ctl, "oom_above")); err == nil {
			above, _ = strconv.Atoi(strings.TrimSpace(string(b)))
		}
	}
	if above > 0 && c > above {
		bpt, _ := strconv.Atoi(os.Getenv("FAKELLAMA_BYTES_PER_TOKEN"))
		if bpt == 0 {
			bpt = 52428
		}
		fmt.Printf("ggml_backend_cuda_buffer_type_alloc_buffer: allocating %.2f MiB on device 0: cudaMalloc failed: out of memory\n", float64((c-above)*bpt)/(1<<20))
		fmt.Printf("alloc_tensor_range: failed to allocate CUDA0 buffer of size %d\n", (c-above)*bpt)
		fmt.Println("llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache")
		os.Exit(1)
	}
	if ctl != "" {
		base, _ := strconv.ParseFloat(os.Getenv("FAKELLAMA_VRAM_BASE_MIB"), 64)
		per, _ := strconv.ParseFloat(os.Getenv("FAKELLAMA_VRAM_PER_TOKEN"), 64)
		mib := int(base + per*float64(c))
		os.WriteFile(filepath.Join(ctl, fmt.Sprintf("vram.%d", os.Getpid())), []byte(strconv.Itoa(mib)), 0o644)
	}
	delay, _ := time.ParseDuration(os.Getenv("FAKELLAMA_HEALTH_DELAY"))
	var isReady atomic.Bool
	alias := vals["-a"]
	if alias == "" {
		alias = vals["--alias"]
	}
	if alias == "" {
		alias = "fake"
	}
	exists := func(name string) bool {
		if ctl == "" {
			return false
		}
		_, err := os.Stat(filepath.Join(ctl, name))
		return err == nil
	}
	var looped atomic.Bool // ready + 150 ms: from then on, a crashloop file ends this server
	crashloop := func() {
		if !looped.Load() {
			return
		}
		if b, err := os.ReadFile(filepath.Join(ctl, "crashloop")); err == nil {
			if lim, _ := strconv.Atoi(strings.TrimSpace(string(b))); lim == 0 || c > lim {
				say("fakellama: GGML_ASSERT failed: illegal memory access")
				os.Exit(1)
			}
		}
	}
	controls := map[string]func(){
		"oom": func() {
			if exists("oom") {
				os.Remove(filepath.Join(ctl, "oom")) // one shot: the restarted server must not die again
				say("CUDA error: out of memory",
					"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1024.00 MiB on device 0: cudaMalloc failed: out of memory")
				os.Exit(1)
			}
		},
		"crash": func() {
			if exists("crash") {
				os.Remove(filepath.Join(ctl, "crash"))
				say("fakellama: Segmentation fault (core dumped)")
				os.Exit(1)
			}
		},
		"oomlog": func() {
			if exists("oomlog") && os.Remove(filepath.Join(ctl, "oomlog")) == nil {
				printOOM()
			}
		},
		"crashloop": crashloop,
	}
	if ctl != "" {
		for name, f := range controls {
			ch, _, err := host.Changes(filepath.Join(ctl, name))
			if err != nil {
				fmt.Fprintln(os.Stderr, "fakellama: cannot follow", name, err)
				os.Exit(1)
			}
			go func() {
				for range ch {
					f()
				}
			}()
			f() // a control file already there when the server starts
		}
	}
	loading := func(w http.ResponseWriter) bool {
		if !isReady.Load() {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(503)
			fmt.Fprint(w, `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`)
			return true
		}
		return false
	}
	js := func(w http.ResponseWriter, v any) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(v)
	}
	mux := http.NewServeMux()
	// A stand-in for llama-server's web UI: one page and one asset, both naming the path they
	// were asked for, so a test can see what the engine's /chat/ proxy forwarded.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" && r.URL.Path != "/_app/x.js" {
			http.NotFound(w, r)
			return
		}
		// Like the real UI: gzip only (415 without it), isolation headers on the page.
		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			http.Error(w, "Error: gzip is not supported by this browser", http.StatusUnsupportedMediaType)
			return
		}
		w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
		w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
		w.Header().Set("Content-Encoding", "gzip")
		zw := gzip.NewWriter(w)
		fmt.Fprintf(zw, "fake llama web ui path=%s", r.URL.Path)
		zw.Close()
	})
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if !loading(w) {
			js(w, map[string]string{"status": "ok"})
		}
	})
	mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
		if loading(w) {
			return
		}
		busy := exists("busy") || atomic.LoadInt32(&inflight) > 0
		var s []map[string]any
		slotMu.Lock()
		for i := 0; i < np; i++ {
			m := map[string]any{"id": i, "n_ctx": c / np, "speculative": false, "is_processing": busy && i == 0}
			if sl := slots[i]; sl.held {
				m["id_task"] = sl.task
				m["n_prompt_tokens"] = sl.tokens - sl.decoded
				m["n_prompt_tokens_processed"] = 0
				m["n_prompt_tokens_cache"] = 0
				m["next_token"] = []map[string]any{{"has_next_token": false, "has_new_line": false, "n_remain": -1, "n_decoded": sl.decoded}}
			}
			s = append(s, m)
		}
		slotMu.Unlock()
		js(w, s)
	})
	mux.HandleFunc("/slots/", func(w http.ResponseWriter, r *http.Request) {
		if loading(w) {
			return
		}
		if ctl != "" { // every slot action asked is logged: <ctl>/slotacts.log, "<path>?<query>"
			if f, err := os.OpenFile(filepath.Join(ctl, "slotacts.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
				fmt.Fprintf(f, "%s?%s\n", r.URL.Path, r.URL.RawQuery)
				f.Close()
			}
		}
		slotAction(w, r, np, c/np, ctl, exists)
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, r *http.Request) {
		if !loading(w) {
			js(w, map[string]any{"total_slots": np, "model_alias": alias, "default_generation_settings": map[string]any{"n_ctx": c / np}})
		}
	})
	models := func(w http.ResponseWriter, r *http.Request) {
		if !loading(w) {
			js(w, map[string]any{"object": "list", "data": []map[string]any{{"id": alias, "object": "model"}}})
		}
	}
	mux.HandleFunc("/v1/models", models)
	mux.HandleFunc("/models", models)
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds 42.5\n")
	})
	chat := func(w http.ResponseWriter, r *http.Request) {
		if loading(w) {
			return
		}
		t := begin()
		defer finish(t)
		if exists("oomreq") {
			printOOM()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(500)
			fmt.Fprint(w, `{"error":{"code":500,"message":"CUDA error: out of memory","type":"server_error"}}`)
			return
		}
		var body struct {
			Stream bool `json:"stream"`
		}
		raw, _ := io.ReadAll(r.Body)
		recordBody(raw)
		json.Unmarshal(raw, &body)
		cacheN, promptN, slot := account(raw, t, 5)
		// FAKELLAMA_ID_SLOT: every reply says the slot it used, as llama-server's native replies do
		withSlot := os.Getenv("FAKELLAMA_ID_SLOT") != ""
		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			gap, _ := time.ParseDuration(os.Getenv("FAKELLAMA_STREAM_DELAY"))
			if gap == 0 {
				gap = 100 * time.Millisecond
			}
			gated := os.Getenv("FAKELLAMA_STREAM_GATE") != "" && ctl != ""
			for i := 0; i < 5; i++ {
				if withSlot {
					fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"t%d \"}}],\"id_slot\":%d}\n\n", i, slot)
				} else {
					fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"t%d \"}}]}\n\n", i)
				}
				fl.Flush()
				if gated && i < 4 {
					// the next chunk only once the test has read this one: <ctl>/go.<i+1>, by
					// the kernel's word of its making (a proxy that buffered would never let
					// the test read, and the test fails on its own watchdog)
					gate := filepath.Join(ctl, fmt.Sprintf("go.%d", i+1))
					ch, stop, err := host.Changes(gate)
					if err != nil {
						return
					}
					for !exists(filepath.Base(gate)) {
						select {
						case <-ch:
						case <-r.Context().Done():
							stop()
							return
						}
					}
					stop()
					continue
				}
				time.Sleep(gap) // named moment: FAKELLAMA_STREAM_DELAY, the fake's pace between tokens
			}
			if os.Getenv("FAKELLAMA_TIMINGS") != "" { // the last chunk llama-server sends: finish, slot, timings
				tm := map[string]any{"cache_n": cacheN, "prompt_n": promptN, "prompt_ms": 12.5, "predicted_n": 5, "predicted_ms": 100, "predicted_per_second": 50}
				said := 0
				if withSlot {
					said = slot
				}
				last, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]any{}, "finish_reason": "stop"}}, "id_slot": said, "timings": tm})
				fmt.Fprintf(w, "data: %s\n\n", last)
				fl.Flush()
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		if d, _ := time.ParseDuration(os.Getenv("FAKELLAMA_CHAT_DELAY")); d > 0 {
			time.Sleep(d) // named moment: FAKELLAMA_CHAT_DELAY, how long the fake takes to answer
		}
		reply := map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ready"}}}, "timings": map[string]any{"cache_n": cacheN, "prompt_n": promptN, "predicted_n": 5}}
		if withSlot {
			reply["id_slot"] = slot
		}
		js(w, reply)
	}
	mux.HandleFunc("/v1/chat/completions", chat)
	completion := func(w http.ResponseWriter, r *http.Request) {
		if loading(w) {
			return
		}
		t := begin()
		defer finish(t)
		raw, _ := io.ReadAll(r.Body)
		recordBody(raw)
		cacheN, promptN, slot := account(raw, t, 5)
		reply := map[string]any{"content": "ready", "stop": true, "timings": map[string]any{"cache_n": cacheN, "prompt_n": promptN, "predicted_n": 5}}
		if os.Getenv("FAKELLAMA_ID_SLOT") != "" {
			reply["id_slot"] = slot
		}
		js(w, reply)
	}
	mux.HandleFunc("/completion", completion)
	// resumable streams, as far as a test needs: every conversation id asked is logged to
	// <ctl>/convs.log ("<what> <id>"); the lookup answers each id it was given
	// the sessions a header opened, matched by lookup as llama-server does (server-stream.cpp:533-536:
	// the id itself, or the id then "::")
	var sessMu sync.Mutex
	sessions := map[string]bool{}
	convLog := func(what, id string) {
		if what == "HEADER" {
			sessMu.Lock()
			sessions[id] = true
			sessMu.Unlock()
		}
		if ctl == "" {
			return
		}
		if f, err := os.OpenFile(filepath.Join(ctl, "convs.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			fmt.Fprintf(f, "%s %s\n", what, id)
			f.Close()
		}
	}
	mux.HandleFunc("/v1/stream", func(w http.ResponseWriter, r *http.Request) {
		convLog(r.Method, r.URL.Query().Get("conv_id"))
		if r.Method == http.MethodDelete {
			w.WriteHeader(204)
			return
		}
		http.Error(w, `{"error":"Stream not found or expired"}`, 404)
	})
	mux.HandleFunc("/v1/streams/lookup", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			IDs []string `json:"conversation_ids"`
		}
		json.NewDecoder(r.Body).Decode(&in)
		var out []map[string]any
		for _, id := range in.IDs {
			convLog("LOOKUP", id)
			sessMu.Lock()
			for sid := range sessions {
				if sid == id || strings.HasPrefix(sid, id+"::") {
					out = append(out, map[string]any{"conversation_id": sid, "alive": true})
				}
			}
			sessMu.Unlock()
		}
		js(w, map[string]any{"sessions": out})
	})
	mux.HandleFunc("/infill", completion)
	addr := vals["--host"] + ":" + vals["--port"]
	// Like the real server, reflect Origin in CORS headers; and say what Authorization arrived,
	// so the engine's proxy tests can check it neither leaks nor lets CORS through.
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if id := r.Header.Get("X-Conversation-Id"); id != "" {
			convLog("HEADER", id)
		}
		if o := r.Header.Get("Origin"); o != "" {
			w.Header().Set("Access-Control-Allow-Origin", o)
		}
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("X-Fake-Saw-Authorization", r.Header.Get("Authorization"))
		mux.ServeHTTP(w, r)
	})
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if v := os.Getenv("FAKELLAMA_VMM"); v != "" { // the patched build's word on its CUDA buffers ("on (1 device)")
		say("easy: CUDA VMM buffers " + v)
	}
	readyLines := os.Getenv("FAKELLAMA_READY_LINES")
	sayReady := func() {
		say("srv  llama_server: model loaded",
			"srv  llama_server: listening on http://"+addr,
			"srv  update_slots: all slots are idle")
	}
	becomeReady := func() {
		isReady.Store(true)
		if ctl != "" { // a test hears that /health answers 200 now (a word, not a poll)
			os.WriteFile(filepath.Join(ctl, "healthy"), []byte(strconv.Itoa(os.Getpid())), 0o644)
		}
		switch readyLines {
		case "none":
			say("srv  llama_server: model loaded", "srv  update_slots: all slots are idle")
		case "early": // printed already, before /health answered
		default:
			sayReady()
		}
		// named moment: a crashloop server ends 150 ms after it turns ready
		time.AfterFunc(150*time.Millisecond, func() { looped.Store(true); crashloop() })
	}
	if readyLines == "early" {
		sayReady()
	}
	if delay > 0 {
		time.AfterFunc(delay, becomeReady) // named moment: FAKELLAMA_HEALTH_DELAY, how long the load takes
	} else {
		becomeReady()
	}
	if err := http.Serve(ln, wrapped); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// slotAction serves POST /slots/{id}?action=save|restore the way llama-server does with
// --slot-save-path: the file is a header line and N*FAKELLAMA_SLOT_BYTES zero bytes.
func slotAction(w http.ResponseWriter, r *http.Request, np, ctxPer int, ctl string, exists func(string) bool) {
	w.Header().Set("Content-Type", "application/json")
	fail := func(code int, msg, typ string) {
		w.WriteHeader(code)
		json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": code, "message": msg, "type": typ}})
	}
	dir := slotSavePath
	if dir == "" {
		fail(501, "This server does not support slots action.", "not_supported_error")
		return
	}
	if r.Method != http.MethodPost {
		fail(405, "Method Not Allowed", "invalid_request_error")
		return
	}
	action := r.URL.Query().Get("action")
	if action != "save" && action != "restore" && action != "erase" {
		fail(400, "Invalid action", "invalid_request_error")
		return
	}
	id, err := strconv.Atoi(strings.TrimPrefix(r.URL.Path, "/slots/"))
	if err != nil || id < 0 || id >= np {
		fail(400, "Invalid slot ID", "invalid_request_error")
		return
	}
	if action == "erase" { // as llama-server's: the slot's tokens and its cache are cleared
		slotMu.Lock()
		n := slots[id].tokens
		slots[id] = slot{}
		slotMu.Unlock()
		if ctl != "" {
			if lf, err := os.OpenFile(filepath.Join(ctl, "slots.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
				fmt.Fprintf(lf, "erase %d\n", id)
				lf.Close()
			}
		}
		json.NewEncoder(w).Encode(map[string]any{"id_slot": id, "n_erased": n})
		return
	}
	var body struct {
		Filename string `json:"filename"`
	}
	raw, _ := io.ReadAll(r.Body)
	json.Unmarshal(raw, &body)
	name := body.Filename
	if name == "" || strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
		fail(400, "Invalid filename", "invalid_request_error")
		return
	}
	logLine := func(f string, a ...any) {
		if ctl == "" {
			return
		}
		if lf, err := os.OpenFile(filepath.Join(ctl, "slots.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			fmt.Fprintf(lf, f+"\n", a...)
			lf.Close()
		}
	}
	path := filepath.Join(dir, name)
	if action == "save" {
		if exists("savefail") {
			fail(500, "fake save failure", "server_error")
			return
		}
		slotMu.Lock()
		n := slots[id].tokens
		slotMu.Unlock()
		per, _ := strconv.Atoi(os.Getenv("FAKELLAMA_SLOT_BYTES"))
		if per == 0 {
			per = 16
		}
		data := append([]byte(fmt.Sprintf("fakeslot tokens=%d ctx=%d\n", n, ctxPer)), make([]byte, n*per)...)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			fail(500, err.Error(), "server_error")
			return
		}
		logLine("save %d %s %d", id, name, n)
		json.NewEncoder(w).Encode(map[string]any{"id_slot": id, "filename": name, "n_saved": n, "n_written": len(data), "timings": map[string]any{"save_ms": 0.5}})
		return
	}
	if exists("restorefail") {
		logLine("restore %d %s fail", id, name)
		fail(500, "fake restore failure", "server_error")
		return
	}
	data, err := os.ReadFile(path)
	n, ctx := 0, 0
	if err == nil {
		line, _, _ := strings.Cut(string(data), "\n")
		_, perr := fmt.Sscanf(line, "fakeslot tokens=%d ctx=%d", &n, &ctx)
		if perr != nil {
			err = perr
		}
	}
	if err != nil || n < 0 || n > ctxPer {
		logLine("restore %d %s fail", id, name)
		fail(400, "Unable to restore slot: No available space in KV cache or invalid slot save file", "invalid_request_error")
		return
	}
	slotMu.Lock()
	slots[id].tokens = n
	slotMu.Unlock()
	logLine("restore %d %s ok", id, name)
	json.NewEncoder(w).Encode(map[string]any{"id_slot": id, "filename": name, "n_restored": n, "n_read": len(data), "timings": map[string]any{"restore_ms": 0.5}})
}
