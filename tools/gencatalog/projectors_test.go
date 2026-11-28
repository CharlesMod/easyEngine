package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
)

const (
	campfireID = "gemma4-rp-maxxed-q3kxl"
	pinnedRev  = "c099eb48e663fd284577b04978a94ffccb261841"
	f16OID     = "418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f"
)

// fakeHF serves the real tree metadata of the pinned commit and records each request path.
func fakeHF(t *testing.T, tree string) (*[]string, func()) {
	t.Helper()
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		if r.URL.Path != "/api/models/unsloth/gemma-4-26B-A4B-it-GGUF/tree/"+pinnedRev {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(tree))
	}))
	old := projHFBase
	projHFBase = srv.URL
	return &paths, func() { projHFBase = old; srv.Close() }
}

const realTree = `[
{"type":"file","path":"mmproj-BF16.gguf","size":1194828256,"lfs":{"oid":"41926ed5f1403cf5add23b0684992805ea6f97253096132e769e65646b8cef9d","size":1194828256}},
{"type":"file","path":"mmproj-F16.gguf","size":1193058784,"lfs":{"oid":"418a6d8723067cd712235facbbc5cba6c8fbbd413fc1292d2aace5a027d5a42f","size":1193058784}}
]`

func committed(t *testing.T) catalog.Catalog {
	t.Helper()
	b, err := os.ReadFile("../../catalog/catalog.json")
	if err != nil {
		t.Fatal(err)
	}
	var c catalog.Catalog
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

func TestProjectorMatchesCommittedCatalog(t *testing.T) {
	paths, done := fakeHF(t, realTree)
	defer done()
	c := committed(t)
	var want *catalog.Projector
	for i := range c.Models {
		if c.Models[i].ID == campfireID {
			want = c.Models[i].Projector
			c.Models[i].Projector = nil
		}
	}
	if want == nil {
		t.Fatal("committed catalog has no projector on Campfire")
	}
	if err := applyProjectors(projClient(), &c); err != nil {
		t.Fatal(err)
	}
	if len(*paths) != 1 || !strings.HasPrefix((*paths)[0], "/api/models/unsloth/gemma-4-26B-A4B-it-GGUF/tree/"+pinnedRev+"?") {
		t.Fatalf("the tree must be read at the pinned commit: %v", *paths)
	}
	var got *catalog.Projector
	for _, m := range c.Models {
		if m.ID == campfireID {
			got = m.Projector
		}
	}
	// the url differs only in host (the fake server); the rest is exact
	if got == nil || !strings.HasPrefix(got.URL, projHFBase+"/") {
		t.Fatalf("projector %+v", got)
	}
	g := *got
	g.URL = strings.Replace(g.URL, projHFBase, "https://huggingface.co", 1)
	if !reflect.DeepEqual(&g, want) {
		t.Fatalf("built %+v\ncommitted %+v", g, *want)
	}
}

func TestProjectorSpecErrors(t *testing.T) {
	good := projectorSpecs[campfireID]
	cases := []struct {
		name string
		mut  func(*projSpec)
		tree string
	}{
		{"short rev", func(s *projSpec) { s.rev = "c099eb48" }, realTree},
		{"branch rev", func(s *projSpec) { s.rev = "main" }, realTree},
		{"uppercase rev", func(s *projSpec) { s.rev = strings.ToUpper(s.rev) }, realTree},
		{"missing file", func(s *projSpec) { s.file = "mmproj-Q8_0.gguf" }, realTree},
		{"no lfs", func(*projSpec) {}, `[{"type":"file","path":"mmproj-F16.gguf","size":5}]`},
		{"short oid", func(*projSpec) {}, `[{"type":"file","path":"mmproj-F16.gguf","size":5,"lfs":{"oid":"abc","size":5}}]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, done := fakeHF(t, tc.tree)
			defer done()
			sp := good
			tc.mut(&sp)
			c := committed(t)
			err := applyProjectorSpecs(projClient(), &c, map[string]projSpec{campfireID: sp})
			if err == nil {
				t.Fatal("expected an error")
			}
		})
	}
	t.Run("unknown model", func(t *testing.T) {
		_, done := fakeHF(t, realTree)
		defer done()
		c := committed(t)
		if err := applyProjectorSpecs(projClient(), &c, map[string]projSpec{"nope": good}); err == nil {
			t.Fatal("expected an error")
		}
	})
}

func TestProjectorSpecKeysAreCatalogModels(t *testing.T) {
	c := committed(t)
	ids := map[string]bool{}
	for _, m := range c.Models {
		ids[m.ID] = true
	}
	for id, sp := range projectorSpecs {
		if !ids[id] {
			t.Errorf("projector spec %q is not a model id in the committed catalog", id)
		}
		if !fullRev.MatchString(sp.rev) {
			t.Errorf("projector spec %q: rev %q is not a full commit sha", id, sp.rev)
		}
	}
}

func projClient() *client {
	return &client{http: http.DefaultClient, log: func(string, ...any) {}}
}
