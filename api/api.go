// Package api serves the control API (/engine/v1), the embedded setup page and
// the reverse proxy to llama-server.
package api

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"hermesengine/run"
	"hermesengine/web"
)

// API wires the supervisor to HTTP.
type API struct {
	S *run.Supervisor
	// Attach adds a parent pid to the watch list.
	Attach func(pid int) error
	// Shutdown asks the process to exit (POST shutdown).
	Shutdown func()
	// AnyHost disables the Host-header check (set with --host for LAN use).
	AnyHost bool
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

// Handler is the whole HTTP surface.
func (a *API) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/engine/"):
			a.engine(w, r)
		case r.URL.Path == "/" || r.URL.Path == "/index.html":
			if !a.hostOK(r) {
				http.Error(w, "bad host", http.StatusForbidden)
				return
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Header().Set("Cache-Control", "no-store")
			w.Write(web.Index)
		default:
			a.proxy(w, r)
		}
	})
}

func (a *API) hostOK(r *http.Request) bool {
	if a.AnyHost {
		return true
	}
	h := r.Host
	if hh, _, err := net.SplitHostPort(h); err == nil {
		h = hh
	}
	h = strings.Trim(h, "[]")
	return h == "127.0.0.1" || h == "localhost" || h == "::1"
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// csrfOK: every non-GET request needs X-Hermes-Engine: 1, and an Origin, if
// there is one, must be the engine's own.
func (a *API) csrfOK(r *http.Request) bool {
	if r.Method == http.MethodGet || r.Method == http.MethodHead || r.Method == http.MethodOptions {
		return true
	}
	if r.Header.Get("X-Hermes-Engine") != "1" {
		return false
	}
	if o := r.Header.Get("Origin"); o != "" {
		u, err := url.Parse(o)
		if err != nil || u.Host != r.Host {
			return false
		}
	}
	return true
}

func decode(r *http.Request, v any) error {
	d := json.NewDecoder(http.MaxBytesReader(nil, r.Body, 1<<20))
	if err := d.Decode(v); err != nil {
		return fmt.Errorf("bad JSON: %v", err)
	}
	return nil
}

func (a *API) engine(w http.ResponseWriter, r *http.Request) {
	if !a.hostOK(r) {
		fail(w, http.StatusForbidden, "bad host")
		return
	}
	if !a.csrfOK(r) {
		fail(w, http.StatusForbidden, "missing X-Hermes-Engine header or foreign origin")
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
	switch name {
	case "status":
		if need("GET") {
			writeJSON(w, 200, a.S.Status())
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
			go a.S.Wake()
			writeJSON(w, 202, map[string]string{"status": "waking"})
		}
	case "settings":
		switch r.Method {
		case "GET":
			writeJSON(w, 200, a.S.Settings())
		case "PUT", "POST":
			var in run.Settings
			if err := decode(r, &in); err != nil {
				fail(w, 400, err.Error())
				return
			}
			if err := a.S.SetSettings(in); err != nil {
				fail(w, 400, err.Error())
				return
			}
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
			writeJSON(w, 200, map[string]any{"status": "attached", "pid": in.Pid})
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
