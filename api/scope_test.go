package api_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/run"
)

// scopeEnv is an engine on a non-loopback listener with a master token and one paired token.
func scopeEnv(t *testing.T) (pe pairEnv, paired string) {
	t.Helper()
	pe = pairServer(t)
	_, tok, err := pe.st.Issue("Kitchen PC")
	if err != nil {
		t.Fatal(err)
	}
	return pe, tok
}

func notAllowed(c int, b string) bool { return c == 403 && strings.Contains(b, `"code":"not_allowed"`) }

type route struct {
	method, path string
	allowed      bool // for a paired token
	stream       bool // not asked of the master or loopback: the answer is a stream
}

// Every control route the router knows (api/api.go engine, custom.go, pair.go), with the
// methods that answer, and what a paired token may do with it.
var scopeRoutes = []route{
	{"GET", "hello", true, false},
	{"POST", "pair", true, false},
	{"GET", "pair/abc", true, false},
	{"DELETE", "pair/abc", true, false},
	{"GET", "pairing", false, false},
	{"POST", "pairing/answer", false, false},
	{"GET", "status", true, false},
	{"POST", "attach", true, false},
	{"DELETE", "attach/abc", true, false},
	{"GET", "hardware", false, false},
	{"GET", "gpu", false, false},
	{"POST", "host", false, false},
	{"GET", "plan", false, false},
	{"POST", "install", false, false},
	{"GET", "storage", false, false},
	{"POST", "custom/check", false, false},
	{"POST", "custom/use", false, false},
	{"POST", "custom/clear", false, false},
	{"GET", "settings", false, false},
	{"PUT", "settings", false, false},
	{"POST", "settings", false, false},
	{"POST", "test", false, false},
	{"GET", "facts", false, true},
	{"GET", "events", false, true},
	{"GET", "book", false, true},
	{"GET", "logs", false, true},
	{"POST", "cancel", false, false},
	{"POST", "import", false, false},
	{"POST", "start", false, false},
	{"POST", "stop", false, false},
	{"POST", "sleep", false, false},
	{"POST", "wake", false, false},
	{"POST", "remove-downloads", false, false},
	{"POST", "delete-model", false, false},
	{"POST", "shutdown", false, false},
	{"GET", "nothing-here", false, false},
}

func TestPairedScopeOverControlRoutes(t *testing.T) {
	pe, paired := scopeEnv(t)
	hdr := with(csrf, bearer(paired))
	for _, r := range scopeRoutes {
		c, b, _ := do(t, r.method, pe.srv.URL+"/engine/v1/"+r.path, "{}", hdr)
		if r.allowed && notAllowed(c, b) {
			t.Errorf("paired %s %s refused: %d %s", r.method, r.path, c, b)
		}
		if !r.allowed && !notAllowed(c, b) {
			t.Errorf("paired %s %s: %d %s, want 403 not_allowed", r.method, r.path, c, b)
		}
	}
	// the wrong method on an allowed name is not allowed either
	for _, m := range []string{"POST", "PUT", "DELETE"} {
		if c, b, _ := do(t, m, pe.srv.URL+"/engine/v1/status", "{}", hdr); !notAllowed(c, b) {
			t.Errorf("paired %s status: %d %s", m, c, b)
		}
	}
	if c, b, _ := do(t, "GET", pe.srv.URL+"/engine/v1/attach", "", hdr); !notAllowed(c, b) {
		t.Errorf("paired GET attach: %d %s", c, b)
	}
	if c, b, _ := do(t, "DELETE", pe.srv.URL+"/engine/v1/attach/a/b", "", hdr); !notAllowed(c, b) {
		t.Errorf("paired DELETE attach/a/b: %d %s", c, b)
	}
	for _, p := range []string{"/engine/v1//status", "/engine/v1/./shutdown", "/engine/v1/status/../shutdown", "/engine/v1/", "/engine/v1", "/engine/"} {
		if c, b, _ := do(t, "POST", pe.srv.URL+p, "{}", hdr); !notAllowed(c, b) {
			t.Errorf("paired POST %s: %d %s", p, c, b)
		}
	}
}

// The master token and the loopback listener are not scoped: no route answers not_allowed
// because of who asks. (Streams are left out: they never end.)
func TestMasterAndLoopbackAreNotScoped(t *testing.T) {
	pe, _ := scopeEnv(t)
	_, _, loop := readyEnv(t)
	for name, c := range map[string]struct {
		url string
		hdr map[string]string
	}{
		"master":   {pe.srv.URL, with(csrf, bearer(pe.master))},
		"loopback": {loop.URL, csrf},
	} {
		for _, r := range scopeRoutes {
			if r.stream || r.path == "shutdown" {
				continue
			}
			code, b, _ := do(t, r.method, c.url+"/engine/v1/"+r.path, "{}", c.hdr)
			if notAllowed(code, b) && r.path != "pairing/answer" {
				t.Errorf("%s %s %s: %d %s", name, r.method, r.path, code, b)
			}
		}
		for _, p := range []string{"/slots", "/props", "/v1/models"} {
			if code, b, _ := do(t, "GET", c.url+p, "", c.hdr); code != 200 {
				t.Errorf("%s GET %s: %d %s", name, p, code, b)
			}
		}
	}
}

func TestPairedScopeOverProxy(t *testing.T) {
	pe, paired := scopeEnv(t)
	hdr := with(csrf, bearer(paired), map[string]string{"Content-Type": "application/json"})
	body := `{"messages":[{"role":"user","content":"hi"}]}`
	for _, p := range []string{"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions",
		"/completion", "/tokenize", "/detokenize", "/apply-template",
		"//v1/chat/completions", "/v1/chat/completions/", "/x/../v1/chat/completions"} {
		if c, b, _ := do(t, "POST", pe.srv.URL+p, body, hdr); notAllowed(c, b) {
			if strings.Contains(p, "/x/..") {
				continue // a dot segment is refused on purpose
			}
			t.Errorf("paired POST %s refused: %s", p, b)
		}
	}
	for _, p := range []string{"/v1/models", "/models", "/health", "/v1/health", "/health/", "//health"} {
		if c, b, _ := do(t, "GET", pe.srv.URL+p, "", hdr); notAllowed(c, b) {
			t.Errorf("paired GET %s refused: %s", p, b)
		}
	}
	type call struct{ m, p string }
	for _, k := range []call{
		{"GET", "/slots"}, {"GET", "/slots/0"}, {"POST", "/slots/0?action=erase"}, {"GET", "/props"}, {"POST", "/props"},
		{"GET", "/metrics"}, {"POST", "/v1/embeddings"}, {"POST", "/embeddings"}, {"POST", "/infill"}, {"POST", "/rerank"},
		{"GET", "/lora-adapters"}, {"POST", "/lora-adapters"},
		{"GET", "/v1/../slots/0"}, {"GET", "/v1/%2e%2e/slots/0"}, {"GET", "/%73lots"}, {"GET", "/v1/../props"},
		{"GET", "//props"}, {"GET", "/props/"}, {"GET", "/v1/models/../../props"}, {"GET", "/health/../props"},
		{"POST", "/v1/chat/completions/../../slots/0?action=save"}, {"GET", "/v1/chat/completions"},
		{"POST", "/v1/models"}, {"DELETE", "/v1/chat/completions"}, {"PUT", "/completion"},
		{"GET", "/"}, {"POST", "/"}, {"GET", "/chat/"}, {"GET", "/chat/ui/slots"}, {"POST", "/chat/v1/chat/completions"},
	} {
		if c, b, _ := do(t, k.m, pe.srv.URL+k.p, "{}", hdr); !notAllowed(c, b) && !(k.p == "/" && k.m == "GET") {
			t.Errorf("paired %s %s: %d %s, want 403 not_allowed", k.m, k.p, c, b)
		}
	}
}

// The fake model server shows a refused call never reached it: /slots on it answers 200 and
// /props answers with its props, and a paired client gets neither.
func TestPairedClientDoesNotReachSlotsOrProps(t *testing.T) {
	pe, paired := scopeEnv(t)
	hdr := bearer(paired)
	for _, p := range []string{"/slots", "/props"} {
		if c, b, _ := do(t, "GET", pe.srv.URL+p, "", hdr); !notAllowed(c, b) || strings.Contains(b, "slots") && strings.Contains(b, "id") {
			t.Errorf("paired GET %s: %d %s", p, c, b)
		}
		if c, _, _ := do(t, "GET", pe.srv.URL+p, "", bearer(pe.master)); c != 200 {
			t.Errorf("master GET %s: %d", p, c)
		}
	}
	if c, b, _ := do(t, "POST", pe.srv.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, with(csrf, hdr)); c != 200 {
		t.Errorf("paired chat: %d %s", c, b)
	}
	if c, b, _ := do(t, "GET", pe.srv.URL+"/v1/models", "", hdr); c != 200 {
		t.Errorf("paired models: %d %s", c, b)
	}
}

// What a paired token sees of status, populated: the full status has the removed fields and the
// reduced one has none of them and every kept key a client reads.
func TestPairedStatusIsReduced(t *testing.T) {
	pe, paired := scopeEnv(t)
	// populate: a custom file with a path, parents, and the agent are all in the full status
	full := func(tok string) map[string]json.RawMessage {
		c, b, _ := do(t, "GET", pe.srv.URL+"/engine/v1/status", "", bearer(tok))
		if c != 200 {
			t.Fatalf("status: %d %s", c, b)
		}
		m := map[string]json.RawMessage{}
		if err := json.Unmarshal([]byte(b), &m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	fm, pm := full(pe.master), full(paired)
	for _, k := range api.StatusRemoved {
		if _, ok := fm[k]; !ok && k != "contexts" {
			t.Errorf("master status lacks %q", k)
		}
		if _, ok := pm[k]; ok {
			t.Errorf("paired status has %q", k)
		}
	}
	for _, k := range []string{"phase", "sampler", "model", "model_tier", "model_size_class", "model_name"} {
		if _, ok := pm[k]; !ok {
			t.Errorf("paired status lacks %q, which a client reads", k)
		}
	}
	for _, k := range api.StatusKept {
		if _, ok := pm[k]; !ok && k != "model_note" {
			t.Errorf("paired status lacks kept %q", k)
		}
		if string(pm[k]) != string(fm[k]) && k != "tok_s" && k != "message" && k != "progress" && k != "held" {
			t.Errorf("%q changed shape: %s vs %s", k, pm[k], fm[k])
		}
	}
}

// Populated by hand: the copy withholds every removed field and blanks a message that could
// name a path or another program.
func TestPairedStatusCopyWithholdsPopulatedFields(t *testing.T) {
	st := run.Status{Phase: "error", Message: "Model file not found: /home/u/x.gguf", ServerURL: "http://127.0.0.1:5000",
		Errors: []string{"open /home/u/x: denied"}, Parents: []run.Parent{{Pid: 7, Alive: true}},
		Agent: run.AgentStatus{Pid: 9}, Port: 4480, Custom: &run.CustomInfo{Path: "/home/u/x.gguf"},
		Recovery: &run.RecoveryInfo{LastError: "boom /home/u"}, Model: "m", ModelTier: "full"}
	b, _ := json.Marshal(api.PairedStatusOf(st))
	for _, bad := range []string{"/home/u", "127.0.0.1", "\"pid\"", "4480", "boom"} {
		if strings.Contains(string(b), bad) {
			t.Errorf("reduced status leaks %q: %s", bad, b)
		}
	}
	if !strings.Contains(string(b), `"model":"m"`) || !strings.Contains(string(b), `"model_tier":"full"`) {
		t.Errorf("kept fields missing: %s", b)
	}
}

// A field added to run.Status must be put in the keep list or the removed list on purpose.
func TestStatusFieldsAreClassified(t *testing.T) {
	seen := map[string]bool{}
	for _, k := range append(append([]string{}, api.StatusKept...), api.StatusRemoved...) {
		if seen[k] {
			t.Errorf("%q is in both lists or twice", k)
		}
		seen[k] = true
	}
	rt := reflect.TypeOf(run.Status{})
	fields := map[string]bool{}
	for i := 0; i < rt.NumField(); i++ {
		name := strings.Split(rt.Field(i).Tag.Get("json"), ",")[0]
		fields[name] = true
		if !seen[name] {
			t.Errorf("run.Status field %q (json %q) is in neither api.StatusKept nor api.StatusRemoved", rt.Field(i).Name, name)
		}
	}
	for k := range seen {
		if !fields[k] {
			t.Errorf("%q is listed but run.Status has no such field", k)
		}
	}
	// the reduced type carries exactly the kept names
	pt := reflect.TypeOf(api.PairedStatus{})
	got := map[string]bool{}
	for i := 0; i < pt.NumField(); i++ {
		got[strings.Split(pt.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	for _, k := range api.StatusKept {
		if !got[k] {
			t.Errorf("PairedStatus lacks kept %q", k)
		}
		delete(got, k)
	}
	for k := range got {
		t.Errorf("PairedStatus has %q, not in StatusKept", k)
	}
}
