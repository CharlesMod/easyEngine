package catalog

import (
	"encoding/json"
	"strings"
	"testing"
)

const (
	projSHA   = "418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f"
	projSize  = 1193058784
	projURL   = "https://huggingface.co/o/r/resolve/c099eb48e663fd284577b04978a94ffccb261841/mmproj-F16.gguf"
	otherSHA2 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)

func goodProj() *Projector {
	return &Projector{File: "m-mmproj.gguf", URL: projURL, SHA256: projSHA, Size: projSize,
		LoadMiB: 1296, ImageMiB: 130, MaxImageTokens: 512, Detail: Detail{Low: 512, High: 768, Max: 1024}}
}

func projModel(id string, p *Projector) Model {
	return Model{ID: id, Name: id, File: id + ".gguf", URL: "https://example.com/" + id + ".gguf",
		SHA256: otherSHA2, Size: 5, Projector: p}
}

// parseModels runs models through Parse (the whole catalog check) via JSON.
func parseModels(t *testing.T, ms ...Model) error {
	t.Helper()
	b, err := json.Marshal(Catalog{Version: 1, Models: ms})
	if err != nil {
		t.Fatal(err)
	}
	_, err = Parse(b)
	return err
}

func TestProjectorChecks(t *testing.T) {
	mut := func(f func(p *Projector)) *Projector { p := goodProj(); f(p); return p }
	cases := []struct {
		name string
		p    *Projector
		bad  bool
	}{
		{"good", goodProj(), false},
		{"import-only", &Projector{File: "m-mmproj.gguf", Detail: Detail{Low: 1, High: 1, Max: 1}}, false},
		{"zero image tokens cap", mut(func(p *Projector) { p.MaxImageTokens = 0 }), false},
		{"equal detail levels", mut(func(p *Projector) { p.Detail = Detail{Low: 512, High: 512, Max: 512} }), false},
		{"detail at the bound", mut(func(p *Projector) { p.Detail.Max = MaxDetailPx }), false},
		{"partial name", mut(func(p *Projector) { p.File = "m-mmproj.gguf.part" }), true},
		{"empty name", mut(func(p *Projector) { p.File = "" }), true},
		{"slash in name", mut(func(p *Projector) { p.File = "a/b.gguf" }), true},
		{"dotdot name", mut(func(p *Projector) { p.File = "../b.gguf" }), true},
		{"leading dot", mut(func(p *Projector) { p.File = ".b.gguf" }), true},
		{"backslash name", mut(func(p *Projector) { p.File = `a\b.gguf` }), true},
		{"http remote", mut(func(p *Projector) { p.URL = "http://example.com/x.gguf" }), true},
		{"ftp", mut(func(p *Projector) { p.URL = "ftp://example.com/x.gguf" }), true},
		{"loopback http", mut(func(p *Projector) { p.URL = "http://127.0.0.1:9/x.gguf" }), false},
		{"url without sha", mut(func(p *Projector) { p.SHA256 = "" }), true},
		{"short sha", mut(func(p *Projector) { p.SHA256 = "abc" }), true},
		{"url without size", mut(func(p *Projector) { p.Size = 0 }), true},
		{"negative size", mut(func(p *Projector) { p.Size = -1 }), true},
		{"negative load", mut(func(p *Projector) { p.LoadMiB = -1 }), true},
		{"negative image", mut(func(p *Projector) { p.ImageMiB = -1 }), true},
		{"negative tokens", mut(func(p *Projector) { p.MaxImageTokens = -1 }), true},
		{"zero detail", mut(func(p *Projector) { p.Detail = Detail{} }), true},
		{"low above high", mut(func(p *Projector) { p.Detail = Detail{Low: 800, High: 768, Max: 1024} }), true},
		{"high above max", mut(func(p *Projector) { p.Detail = Detail{Low: 512, High: 2048, Max: 1024} }), true},
		{"max over the bound", mut(func(p *Projector) { p.Detail.Max = MaxDetailPx + 1 }), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := projModel("m", tc.p)
			err := parseModels(t, m)
			if (err != nil) != tc.bad {
				t.Fatalf("Parse: bad=%v err=%v", tc.bad, err)
			}
			c, perr := Parse(mustJSON(t, Catalog{Version: 1}))
			if perr != nil {
				t.Fatal(perr)
			}
			_, err = c.WithModels([]Model{m})
			// an overlay's projector is always a checked download: no import-only one
			if bad := tc.bad || tc.name == "import-only"; (err != nil) != bad {
				t.Fatalf("WithModels: bad=%v err=%v", tc.bad, err)
			}
		})
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestProjectorFileCollisions(t *testing.T) {
	// the projector is its own model's file
	self := projModel("m", goodProj())
	self.Projector.File = self.File
	if err := parseModels(t, self); err == nil {
		t.Error("projector file equal to the model's own file: Parse accepted it")
	}
	// the projector is another model's file
	a, b := projModel("a", goodProj()), projModel("b", nil)
	a.Projector.File = b.File
	if err := parseModels(t, a, b); err == nil {
		t.Error("projector file equal to another model's file: Parse accepted it")
	}
	base, err := Parse(mustJSON(t, Catalog{Version: 1, Models: []Model{projModel("b", nil)}}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := base.WithModels([]Model{a}); err == nil {
		t.Error("overlay projector file equal to a catalog model's file: WithModels accepted it")
	}
	// two models sharing one projector file: the same download is fine, a different one is not
	s1, s2 := projModel("s1", goodProj()), projModel("s2", goodProj())
	if err := parseModels(t, s1, s2); err != nil {
		t.Errorf("same projector download shared: %v", err)
	}
	for name, mut := range map[string]func(*Projector){
		"sha":  func(q *Projector) { q.SHA256 = otherSHA2 },
		"size": func(q *Projector) { q.Size++ },
		"url":  func(q *Projector) { q.URL = strings.Replace(q.URL, "mmproj-F16", "mmproj-BF16", 1) },
	} {
		d := projModel("s2", goodProj())
		mut(d.Projector)
		if err := parseModels(t, s1, d); err == nil {
			t.Errorf("shared projector file with a different %s: Parse accepted it", name)
		}
		if _, err := base.WithModels([]Model{s1, d}); err == nil {
			t.Errorf("shared projector file with a different %s: WithModels accepted it", name)
		}
	}
}

func TestEmbeddedCampfireProjectorPinned(t *testing.T) {
	c, err := Parse(embedded)
	if err != nil {
		t.Fatal(err)
	}
	m, ok := c.Model("gemma4-rp-maxxed-q3kxl")
	if !ok || m.Projector == nil {
		t.Fatal("Campfire has no projector in the embedded catalog")
	}
	p := m.Projector
	if p.SHA256 != projSHA || p.Size != projSize {
		t.Errorf("sha %s size %d", p.SHA256, p.Size)
	}
	if !strings.Contains(p.URL, "/resolve/c099eb48e663fd284577b04978a94ffccb261841/") {
		t.Errorf("url %s is not pinned to the commit", p.URL)
	}
	if pm, isProj := c.ProjectorFor(p.File); !isProj || pm.ID != m.ID {
		t.Errorf("ProjectorFor(%q) = %v %v", p.File, pm.ID, isProj)
	}
}

// checkKwargs: at most MaxKwargs keys, each a plain identifier, each value one JSON scalar.
func TestCheckKwargs(t *testing.T) {
	raw := func(s string) json.RawMessage { return json.RawMessage(s) }
	ok := map[string]map[string]json.RawMessage{
		"none":   nil,
		"bool":   {"enable_thinking": raw("false")},
		"true":   {"a": raw("true")},
		"number": {"budget": raw("1024"), "temp": raw("0.5")},
		"short":  {"mode": raw(`"fast"`)},
		"256":    {"s": raw(`"` + strings.Repeat("x", 256) + `"`)},
	}
	for name, kw := range ok {
		if err := checkKwargs("m", kw); err != nil {
			t.Errorf("%s refused: %v", name, err)
		}
	}
	nine := map[string]json.RawMessage{}
	for i := 0; i <= MaxKwargs; i++ {
		nine["k"+strings.Repeat("a", i)] = raw("1")
	}
	eight := map[string]json.RawMessage{}
	for k, v := range nine {
		if len(eight) < MaxKwargs {
			eight[k] = v
		}
	}
	if err := checkKwargs("m", eight); err != nil {
		t.Errorf("%d keys refused: %v", MaxKwargs, err)
	}
	bad := map[string]struct {
		kw   map[string]json.RawMessage
		want string
	}{
		"nine keys":   {nine, "more than"},
		"upper key":   {map[string]json.RawMessage{"Bad": raw("1")}, "plain identifier"},
		"dash key":    {map[string]json.RawMessage{"a-b": raw("1")}, "plain identifier"},
		"digit first": {map[string]json.RawMessage{"1a": raw("1")}, "plain identifier"},
		"empty key":   {map[string]json.RawMessage{"": raw("1")}, "plain identifier"},
		"long key":    {map[string]json.RawMessage{strings.Repeat("a", 65): raw("1")}, "plain identifier"},
		"object":      {map[string]json.RawMessage{"a": raw(`{"x":1}`)}, "must be"},
		"array":       {map[string]json.RawMessage{"a": raw(`[1]`)}, "must be"},
		"null":        {map[string]json.RawMessage{"a": raw(`null`)}, "must be"},
		"long string": {map[string]json.RawMessage{"a": raw(`"` + strings.Repeat("x", 257) + `"`)}, "longer than"},
		"not JSON":    {map[string]json.RawMessage{"a": raw(`{`)}, "not JSON"},
	}
	for name, c := range bad {
		err := checkKwargs("m", c.kw)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v (want an error with %q)", name, err, c.want)
		}
	}
}
