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
//	FAKELLAMA_STREAM_DELAY     gap between streamed chunks (default 100ms)
//	FAKELLAMA_CHAT_DELAY       delay of a non-streaming completion (default 0)
//	(every completion request body is saved to <ctl>/lastbody.json)
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
//
// It says the lines a real llama-server says, in the same order (run/testdata/ready-*.txt): once
// the model is loaded and it listens, "llama_server: model loaded", "llama_server: listening on
// http://..." and "update_slots: all slots are idle"; per request "launch_slot_ ... processing
// task", then "release: ... stop processing" and, when none is left, "all slots are idle". The
// engine hears that it is ready from those lines, as it does with the real server. Control files
// are followed by the kernel's word of their change (host.Changes), never polled.
package main

import (
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
	}
	for _, l := range lines {
		skip := false
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
	ctl := os.Getenv("FAKELLAMA_CTL")
	if ctl != "" {
		f, err := os.OpenFile(filepath.Join(ctl, "starts.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err == nil {
			fmt.Fprintf(f, "pid=%d np=%d c=%d ngl=%s moe=%s ctk=%s cvd=%s args=%s\n", os.Getpid(), np, c, vals["-ngl"], vals["--n-cpu-moe"], vals["-ctk"], os.Getenv("CUDA_VISIBLE_DEVICES"), strings.Join(args, " "))
			f.Close()
		}
	}
	fmt.Println("srv  llama_server: initializing ...")
	fmt.Println("fakellama: loading model", vals["-m"])
	if ctl != "" {
		if _, err := os.Stat(filepath.Join(ctl, "loadfail")); err == nil {
			fmt.Println("llama_model_load: error loading model: tensor 'blk.0.attn_q.weight' data is not within the file bounds")
			os.Exit(1)
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
		fmt.Fprintf(w, "fake llama web ui path=%s", r.URL.Path)
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
		for i := 0; i < np; i++ {
			s = append(s, map[string]any{"id": i, "n_ctx": c / np, "is_processing": busy && i == 0})
		}
		js(w, s)
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
		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			gap, _ := time.ParseDuration(os.Getenv("FAKELLAMA_STREAM_DELAY"))
			if gap == 0 {
				gap = 100 * time.Millisecond
			}
			gated := os.Getenv("FAKELLAMA_STREAM_GATE") != "" && ctl != ""
			for i := 0; i < 5; i++ {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"t%d \"}}]}\n\n", i)
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
			fmt.Fprint(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		if d, _ := time.ParseDuration(os.Getenv("FAKELLAMA_CHAT_DELAY")); d > 0 {
			time.Sleep(d) // named moment: FAKELLAMA_CHAT_DELAY, how long the fake takes to answer
		}
		js(w, map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ready"}}}})
	}
	mux.HandleFunc("/v1/chat/completions", chat)
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		if loading(w) {
			return
		}
		t := begin()
		defer finish(t)
		raw, _ := io.ReadAll(r.Body)
		recordBody(raw)
		js(w, map[string]any{"content": "ready", "stop": true})
	})
	addr := vals["--host"] + ":" + vals["--port"]
	// Like the real server, reflect Origin in CORS headers; and say what Authorization arrived,
	// so the engine's proxy tests can check it neither leaks nor lets CORS through.
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	readyLines := os.Getenv("FAKELLAMA_READY_LINES")
	sayReady := func() {
		say("srv  llama_server: model loaded",
			"srv  llama_server: listening on http://"+addr,
			"srv  update_slots: all slots are idle")
	}
	becomeReady := func() {
		isReady.Store(true)
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
