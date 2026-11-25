// fakellama is a fake llama-server for the engine's tests. It accepts the real
// flags, serves the endpoints the engine and its clients use, and has knobs
// (environment variables and control files) to misbehave on demand:
//
//	FAKELLAMA_CTL              dir with control files: busy (slots show processing),
//	                           oom (die with an out-of-memory error), and the outputs
//	                           starts.log and vram.<pid> (fake VRAM use for fake nvidia-smi)
//	FAKELLAMA_HEALTH_DELAY     how long /health answers 503 (Go duration, default 0)
//	FAKELLAMA_OOM_ABOVE        at load, -c above this value prints llama.cpp's allocation
//	                           failure (with a buffer size) and exits 1
//	FAKELLAMA_BYTES_PER_TOKEN  buffer bytes per ctx token for that message (default 52428)
//	FAKELLAMA_VRAM_BASE_MIB / FAKELLAMA_VRAM_PER_TOKEN  fake VRAM = base + per_token * ctx
//	FAKELLAMA_HIDE             comma list of flags the help text omits and the parser rejects
//	FAKELLAMA_STREAM_DELAY     gap between streamed chunks (default 100ms)
//	FAKELLAMA_CHAT_DELAY       delay of a non-streaming completion (default 0)
//	FAKELLAMA_DEVICES          text printed by --list-devices (default "CUDA0: Fake GPU")
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

var withValue = map[string]bool{
	"-m": true, "-ngl": true, "--n-cpu-moe": true, "-np": true, "-c": true, "-ctk": true, "-ctv": true,
	"-fa": true, "--cache-reuse": true, "--host": true, "--port": true, "-a": true, "--kv-unified": true,
	"--alias": true,
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
		"--kv-unified                   use single unified KV buffer",
		"--jinja                        use jinja template for chat",
		"--cache-reuse N                min chunk size to attempt reusing from the cache",
		"--no-webui                     disable the Web UI",
		"--metrics                      enable prometheus compatible metrics endpoint",
		"--slots                        enable slots monitoring endpoint",
		"--host HOST                    ip address to listen",
		"--port PORT                    port to listen",
		"-a,    --alias STRING          set alias for model name",
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

var inflight int32

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
			fmt.Println("Available devices:\n  " + d)
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
			fmt.Fprintf(f, "pid=%d np=%d c=%d ngl=%s moe=%s ctk=%s args=%s\n", os.Getpid(), np, c, vals["-ngl"], vals["--n-cpu-moe"], vals["-ctk"], strings.Join(args, " "))
			f.Close()
		}
	}
	fmt.Println("fakellama: loading model", vals["-m"])
	if above, _ := strconv.Atoi(os.Getenv("FAKELLAMA_OOM_ABOVE")); above > 0 && c > above {
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
	ready := time.Now().Add(delay)
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
	go func() {
		for {
			time.Sleep(20 * time.Millisecond)
			if exists("oom") {
				os.Remove(filepath.Join(ctl, "oom")) // one shot: the restarted server must not die again
				fmt.Println("CUDA error: out of memory")
				fmt.Println("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1024.00 MiB on device 0: cudaMalloc failed: out of memory")
				os.Exit(1)
			}
		}
	}()
	loading := func(w http.ResponseWriter) bool {
		if time.Now().Before(ready) {
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
		atomic.AddInt32(&inflight, 1)
		defer atomic.AddInt32(&inflight, -1)
		var body struct {
			Stream bool `json:"stream"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			gap, _ := time.ParseDuration(os.Getenv("FAKELLAMA_STREAM_DELAY"))
			if gap == 0 {
				gap = 100 * time.Millisecond
			}
			for i := 0; i < 5; i++ {
				fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"t%d \"}}]}\n\n", i)
				fl.Flush()
				time.Sleep(gap)
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}
		if d, _ := time.ParseDuration(os.Getenv("FAKELLAMA_CHAT_DELAY")); d > 0 {
			time.Sleep(d)
		}
		js(w, map[string]any{"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ready"}}}})
	}
	mux.HandleFunc("/v1/chat/completions", chat)
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		if loading(w) {
			return
		}
		atomic.AddInt32(&inflight, 1)
		defer atomic.AddInt32(&inflight, -1)
		js(w, map[string]any{"content": "ready", "stop": true})
	})
	addr := vals["--host"] + ":" + vals["--port"]
	fmt.Println("fakellama: listening on", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
