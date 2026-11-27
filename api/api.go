// Package api serves the control API (/engine/v1), the embedded setup page and
// the reverse proxy to llama-server.
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/CharlesMod/easyEngine/internal/legacy"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/web"
)

// API wires the supervisor to HTTP.
type API struct {
	S *run.Supervisor
	// Attach adds a parent pid to the watch list.
	Attach func(pid int) error
	// Shutdown asks the process to exit (POST shutdown).
	Shutdown func()
	// AnyHost marks a non-loopback listener (--host): the Host header is not restricted, and a
	// bearer token is required (Tokens) unless Open is set. A loopback listener checks the Host
	// name and never asks for a token. See gate.go.
	AnyHost bool
	// Tokens decides which bearer tokens a non-loopback listener accepts.
	Tokens Checker
	// Open is the keeper's explicit choice (serve --open) to run a non-loopback listener with no token.
	Open bool
	// stopping is set once shutdown has begun: status says so and attach is refused.
	stopping atomic.Bool
	// settingsMu serialises the read-merge-write of a partial settings update.
	settingsMu sync.Mutex
	// heard, when set, stands in for S.Heard (a test counts the calls).
	heard func()
	// Client is used by POST test (default: a plain client with no overall timeout).
	Client *http.Client
}

// Listen binds host:base, falling back to base+1 .. base+9.
func Listen(host string, base int) (net.Listener, int, error) {
	var last error
	for p := base; p < base+10; p++ {
		l, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(p)))
		if err == nil {
			return l, p, nil
		}
		last = err
	}
	return nil, 0, last
}

// SetStopping marks the engine as shutting down: GET status then says phase "stopping" and
// POST attach answers 409, so a new serve waits for this engine to end instead of attaching to it.
func (a *API) SetStopping() { a.stopping.Store(true) }

// Handler is the whole HTTP surface; every path goes through the gate (gate.go).
func (a *API) Handler() http.Handler {
	return a.gate(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/engine/"):
			a.engine(w, r)
		case r.URL.Path == "/chat" || strings.HasPrefix(r.URL.Path, "/chat/"):
			a.chat(w, r)
		case r.URL.Path == "/" || r.URL.Path == "/index.html":
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-store")
			w.Write(web.Index)
		default:
			a.proxy(w, r)
		}
	}))
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// csrfOK: every non-GET request needs X-Easy-Engine: 1 (a cross-site Origin is refused by the gate).
func (a *API) csrfOK(r *http.Request) bool {
	if r.Method == http.MethodGet || r.Method == http.MethodHead || r.Method == http.MethodOptions {
		return true
	}
	return legacy.CSRFOK(r.Header, "X-Easy-Engine")
}

func decode(r *http.Request, v any) error {
	d := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<20))
	if err := d.Decode(v); err != nil {
		return fmt.Errorf("bad JSON: %v", err)
	}
	return nil
}

func (a *API) engine(w http.ResponseWriter, r *http.Request) {
	if !a.csrfOK(r) {
		fail(w, http.StatusForbidden, "missing X-Easy-Engine header")
		return
	}
	name := strings.TrimPrefix(r.URL.Path, "/engine/v1/")
	if name == r.URL.Path {
		fail(w, http.StatusNotFound, "unknown path")
		return
	}
	need := func(method string) bool {
		if r.Method != method {
			w.Header().Set("Allow", method)
			fail(w, http.StatusMethodNotAllowed, "use "+method)
			return false
		}
		return true
	}
	ctx := r.Context()
	if strings.HasPrefix(name, "custom/") && a.customRoute(w, r, ctx, name, need) {
		return
	}
	switch name {
	case "status":
		if need("GET") {
			st := a.S.Status()
			if a.stopping.Load() {
				st.Phase = "stopping"
			}
			writeJSON(w, 200, st)
		}
	case "hardware":
		if need("GET") {
			hw, err := a.S.Hardware(ctx, r.URL.Query().Get("refresh") == "1")
			if err != nil {
				fail(w, 500, err.Error())
				return
			}
			writeJSON(w, 200, hw)
		}
	case "gpu":
		if need("GET") {
			rep, err := a.S.GPUReport(ctx, r.URL.Query().Get("refresh") == "1")
			if err != nil {
				fail(w, 500, err.Error())
				return
			}
			writeJSON(w, 200, rep)
		}
	case "host":
		if need("POST") {
			var in run.HostSet
			if err := decode(r, &in); err != nil {
				fail(w, 400, "body must be {\"pid\": n, \"adapter\": {\"name\", \"vendor_id\", \"memory_mib\"}, \"reserve_mib\": n}")
				return
			}
			if err := a.S.SetHost(in); err != nil {
				fail(w, 400, err.Error())
				return
			}
			a.S.Nudge("host")
			w.WriteHeader(http.StatusNoContent)
		}
	case "plan":
		if need("GET") {
			p, err := a.S.Plan(ctx)
			if err != nil {
				fail(w, 500, err.Error())
				return
			}
			writeJSON(w, 200, p)
		}
	case "install":
		if need("POST") {
			var in struct{ Model, Backend string }
			if err := decode(r, &in); err != nil || in.Model == "" {
				fail(w, 400, "body must be {\"model\": id, \"backend\": optional id}")
				return
			}
			switch err := a.S.Install(in.Model, in.Backend); err {
			case nil:
				writeJSON(w, 202, map[string]string{"status": "started"})
			case run.ErrBusy:
				fail(w, 409, err.Error())
			default:
				fail(w, 400, err.Error())
			}
		}
	case "storage":
		if need("GET") {
			writeJSON(w, 200, a.S.Storage(r.URL.Query().Get("refresh") == "1"))
		}
	case "remove-downloads":
		if need("POST") {
			var in struct{ Confirm bool }
			if err := decode(r, &in); err != nil || !in.Confirm {
				fail(w, 400, "body must be {\"confirm\": true}")
				return
			}
			switch err := a.S.RemoveDownloads(); err {
			case nil:
				writeJSON(w, 202, map[string]any{"started": true})
			case run.ErrBusy:
				fail(w, 409, err.Error())
			default:
				fail(w, 400, err.Error())
			}
		}
	case "delete-model":
		if need("POST") {
			var in struct{ ID, File string }
			if err := decode(r, &in); err != nil || in.ID == "" && in.File == "" {
				fail(w, 400, "body must be {\"id\": catalog id} or {\"file\": name}")
				return
			}
			freed, err := a.S.DeleteModel(in.ID, in.File)
			switch {
			case err == nil:
				writeJSON(w, 200, map[string]any{"freed_bytes": freed})
			case errors.Is(err, run.ErrInUse):
				fail(w, 409, err.Error())
			case errors.Is(err, run.ErrNoModel):
				fail(w, 404, err.Error())
			default:
				fail(w, 400, err.Error())
			}
		}
	case "cancel":
		if need("POST") {
			a.S.Cancel()
			writeJSON(w, 200, map[string]string{"status": "cancelled"})
		}
	case "import":
		if need("POST") {
			var in struct{ Path string }
			if err := decode(r, &in); err != nil || in.Path == "" {
				fail(w, 400, "body must be {\"path\": file}")
				return
			}
			switch err := a.S.Import(in.Path); err {
			case nil:
				writeJSON(w, 202, map[string]string{"status": "started"})
			case run.ErrBusy:
				fail(w, 409, err.Error())
			default:
				fail(w, 400, err.Error())
			}
		}
	case "start":
		if need("POST") {
			a.S.Nudge("start")
			go a.S.Start()
			writeJSON(w, 202, map[string]string{"status": "starting"})
		}
	case "stop":
		if need("POST") {
			a.S.Stop()
			writeJSON(w, 200, map[string]string{"status": "stopped"})
		}
	case "sleep":
		if need("POST") {
			a.S.Sleep()
			writeJSON(w, 200, map[string]string{"status": "sleeping"})
		}
	case "wake":
		if need("POST") {
			a.S.Nudge("wake")
			go a.S.Wake()
			writeJSON(w, 202, map[string]string{"status": "waking"})
		}
	case "settings":
		switch r.Method {
		case "GET":
			writeJSON(w, 200, a.S.Settings())
		case "PUT", "POST":
			var patch settingsPatch
			if err := decode(r, &patch); err != nil {
				fail(w, 400, err.Error())
				return
			}
			a.settingsMu.Lock()
			err := a.S.SetSettings(patch.apply(a.S.Settings()))
			a.settingsMu.Unlock()
			if err != nil {
				fail(w, 400, err.Error())
				return
			}
			// SetSettings itself says the "settings" nudge, after it has set what the reactor reads
			writeJSON(w, 200, a.S.Settings())
		default:
			fail(w, 405, "use GET or PUT")
		}
	case "test":
		if need("POST") {
			writeJSON(w, 200, a.test())
		}
	case "attach":
		if need("POST") {
			if a.stopping.Load() {
				fail(w, 409, "engine is stopping")
				return
			}
			var in struct{ Pid int }
			if err := decode(r, &in); err != nil || in.Pid <= 0 {
				fail(w, 400, "body must be {\"pid\": n}")
				return
			}
			if a.Attach == nil {
				fail(w, 501, "attach not available")
				return
			}
			if err := a.Attach(in.Pid); err != nil {
				fail(w, 400, err.Error())
				return
			}
			a.S.Nudge("attach")
			writeJSON(w, 200, map[string]any{"status": "attached", "pid": in.Pid})
		}
	case "facts":
		if need("GET") {
			a.S.Hub().ServeFacts(w, r)
		}
	case "events":
		if need("GET") {
			a.S.Hub().ServeEvents(w, r)
		}
	case "book":
		if need("GET") {
			a.S.Hub().ServeBook(w, r)
		}
	case "logs":
		if need("GET") {
			n, _ := strconv.Atoi(r.URL.Query().Get("tail"))
			txt, err := a.S.Logs(r.URL.Query().Get("name"), n)
			if err != nil {
				fail(w, 400, err.Error())
				return
			}
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			w.Write([]byte(txt))
		}
	case "shutdown":
		if need("POST") {
			a.SetStopping()
			writeJSON(w, 200, map[string]string{"status": "shutting down"})
			if a.Shutdown != nil {
				go a.Shutdown()
			}
		}
	default:
		fail(w, 404, "unknown endpoint")
	}
}

// test checks the current endpoint (local or external).
func (a *API) test() TestResult {
	c := a.Client
	if c == nil {
		c = &http.Client{}
	}
	st := a.S.Status()
	send := httpSender(c)
	if st.Mode == "external" {
		rt := a.S.Route()
		res := RunTest(rt.URL, st.External.Model, rt.Key, send)
		if res.Code == string("ok_fixed_base") && res.EffectiveURL != "" {
			a.S.FixExternalBase(res.EffectiveURL)
		}
		return res
	}
	rt := a.S.Route()
	if rt.Kind != "local" {
		light := "amber"
		if st.Phase == "not_installed" || st.Phase == "error" {
			light = "red"
		}
		if st.Phase == "sleeping" {
			light = "amber"
		}
		return TestResult{Light: light, Message: rt.Message, Model: st.Model, Code: "not_ready"}
	}
	done := a.S.Begin()
	defer done()
	return RunTest(rt.URL+"/v1", st.Model, "", send)
}

// settingsPatch is a PUT settings body: only the fields it names change. An explicit empty
// external.base_url clears it; clear_api_key: true clears the key; external.api_key "" leaves it.
type settingsPatch struct {
	Mode     *string `json:"mode"`
	External *struct {
		BaseURL *string `json:"base_url"`
		Model   *string `json:"model"`
		APIKey  *string `json:"api_key"`
	} `json:"external"`
	ClearAPIKey   *bool   `json:"clear_api_key"`
	LingerS       *int    `json:"linger_s"`
	KVType        *string `json:"kv_type"`
	SamplerPolicy *string `json:"sampler_policy"`
	Streamer      *string `json:"streamer"`
	KeepContext   *string `json:"keep_context"`
}

// apply lays the named fields over the current settings (whose api key is blank: SetSettings
// keeps the stored key unless it is given a new one or told to clear it).
func (p settingsPatch) apply(cur run.Settings) run.Settings {
	if p.Mode != nil {
		cur.Mode = *p.Mode
	}
	if e := p.External; e != nil {
		if e.BaseURL != nil {
			cur.External.BaseURL = *e.BaseURL
		}
		if e.Model != nil {
			cur.External.Model = *e.Model
		}
		if e.APIKey != nil {
			cur.External.APIKey = *e.APIKey
		}
	}
	cur.ClearAPIKey = p.ClearAPIKey != nil && *p.ClearAPIKey
	if p.LingerS != nil {
		cur.LingerS = *p.LingerS
	}
	if p.KVType != nil {
		cur.KVType = *p.KVType
	}
	if p.SamplerPolicy != nil {
		cur.SamplerPolicy = *p.SamplerPolicy
	}
	if p.Streamer != nil {
		cur.Streamer = *p.Streamer
		if cur.Streamer == "" {
			cur.Streamer = "-" // an explicit empty value is refused, not taken as "keep"
		}
	}
	if p.KeepContext != nil {
		cur.KeepContext = *p.KeepContext
		if cur.KeepContext == "" {
			cur.KeepContext = "-" // an explicit empty value is refused, not taken as "keep"
		}
	}
	return cur
}
