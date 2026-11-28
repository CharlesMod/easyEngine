package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
)

const fakeRev = "0123456789abcdef0123456789abcdef01234567"

// fakeHF serves the Hugging Face API: oids maps a repo to its file's LFS oid, files a repo to the
// file name, sums a repo to its SHA256SUMS body (a repo not in sums has none).
func fakeHF(t *testing.T, oids, sums, files map[string]string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Path
		for repo, oid := range oids {
			switch p {
			case "/api/models/" + repo:
				w.Write([]byte(`{"sha":"` + fakeRev + `"}`))
				return
			case "/api/models/" + repo + "/tree/" + fakeRev:
				tree := `[{"path":"README.md","size":10},{"path":"` + files[repo] + `","size":1234,"lfs":{"oid":"` + oid + `","size":1234}}`
				if _, ok := sums[repo]; ok {
					tree += `,{"path":"SHA256SUMS","size":100}`
				}
				w.Write([]byte(tree + `]`))
				return
			case "/" + repo + "/resolve/" + fakeRev + "/SHA256SUMS":
				if s, ok := sums[repo]; ok {
					w.Write([]byte(s))
					return
				}
			}
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func hfClient(srv *httptest.Server) *client {
	return &client{http: srv.Client(), hf: srv.URL, log: func(string, ...any) {}}
}

func TestHFModelFields(t *testing.T) {
	upper := strings.Repeat("AB", 32)
	for _, useMain := range []bool{true, false} {
		sp := modelSpec{id: "m1", name: "Model One", codename: "Spark", optIn: true, repo: "o/r", file: "m 1.gguf", useMain: useMain, quality: 7}
		srv := fakeHF(t, map[string]string{"o/r": upper}, nil, map[string]string{"o/r": sp.file})
		m, err := hfClient(srv).hfModel(sp)
		if err != nil {
			t.Fatal(err)
		}
		rev := fakeRev
		if useMain {
			rev = "main"
		}
		if m.ID != "m1" || m.Name != "Model One" || m.Codename != "Spark" || !m.OptIn || m.Size != 1234 || m.Quality != 7 {
			t.Errorf("fields: %+v", m)
		}
		if m.SHA256 != strings.ToLower(upper) {
			t.Errorf("sha256 %s", m.SHA256)
		}
		if want := srv.URL + "/o/r/resolve/" + rev + "/m%201.gguf"; m.URL != want {
			t.Errorf("url %s, want %s", m.URL, want)
		}
	}
}

func TestHFSha256Sums(t *testing.T) {
	oid := strings.Repeat("ab", 32)
	other := strings.Repeat("cd", 32)
	sp := modelSpec{id: "m1", repo: "o/r", file: "m.gguf"}
	cases := []struct {
		name, sums string
		wantErr    bool
	}{
		{"match", oid + "  m.gguf\n" + other + "  x.bin\n", false},
		{"match upper star", strings.ToUpper(oid) + " *m.gguf\n", false},
		{"disagree", other + "  m.gguf\n", true},
		{"not named", other + "  other.gguf\n", false},
	}
	for _, tc := range cases {
		srv := fakeHF(t, map[string]string{"o/r": oid}, map[string]string{"o/r": tc.sums}, map[string]string{"o/r": sp.file})
		_, err := hfClient(srv).hfModel(sp)
		if (err != nil) != tc.wantErr {
			t.Errorf("%s: err %v", tc.name, err)
		}
		if tc.wantErr && err != nil && (!strings.Contains(err.Error(), "m.gguf") || !strings.Contains(err.Error(), other) || !strings.Contains(err.Error(), oid)) {
			t.Errorf("%s: error does not name the file and both hashes: %v", tc.name, err)
		}
	}
}

func TestModelsOnlyKeepsTheRest(t *testing.T) {
	sum := strings.Repeat("11", 32)
	old := catalog.Catalog{
		Version: 7,
		Llama: []catalog.LlamaCpp{{Tag: "b1-easy1", Repo: "o/llama.cpp", Patched: true, Assets: []catalog.Asset{
			{ID: "llama-linux", OS: "linux", Arch: "amd64", URL: "https://x/a.tar.gz", SHA256: sum, Size: 10, UnpackedSize: 20}}}},
		UV:           []catalog.Asset{{ID: "uv-linux", OS: "linux", Arch: "amd64", URL: "https://x/uv.tar.gz", SHA256: sum, Size: 5}},
		Models:       []catalog.Model{{ID: "old", Name: "Old", File: "old.gguf", URL: "https://x/old.gguf", SHA256: sum, Size: 3}},
		AgentRuntime: []catalog.RuntimeSize{{OS: "linux", Arch: "amd64", Python: "3.12", PeakBytes: 9, KeepBytes: 4, MeasuredOn: "d", MeasuredWith: "w"}},
	}
	out := filepath.Join(t.TempDir(), "catalog.json")
	if err := writeCatalog(out, &old); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	pb, err := catalog.Parse(before)
	if err != nil {
		t.Fatal(err)
	}

	oids := map[string]string{refSpec.repo: refSHA256}
	files := map[string]string{refSpec.repo: refSpec.file}
	for _, sp := range modelSpecs {
		oids[sp.repo] = strings.Repeat("22", 32)
		files[sp.repo] = sp.file
	}
	srv := fakeHF(t, oids, nil, files)
	if err := refreshModels(hfClient(srv), out, 0); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	pa, err := catalog.Parse(after)
	if err != nil {
		t.Fatal(err)
	}
	if pa.Version != 7 {
		t.Errorf("version %d", pa.Version)
	}
	if !reflect.DeepEqual(pb.Llama, pa.Llama) || !reflect.DeepEqual(pb.UV, pa.UV) || !reflect.DeepEqual(pb.AgentRuntime, pa.AgentRuntime) {
		t.Error("llama, uv or agent_runtime changed")
	}
	want := map[string][2]string{refID: {"Campfire", "Campfire-26B"}, "gemma4-e4b-rp-q40": {"Lantern", "Lantern-4B"}, "gemma4-e2b-q4km": {"Flashlight", "Flashlight-2B"}}
	if len(pa.Models) != len(want) || pa.Models[0].ID != refID {
		t.Fatalf("models %+v", pa.Models)
	}
	for _, m := range pa.Models {
		if w, ok := want[m.ID]; !ok || m.Codename != w[0] || m.CodenameFull != w[1] {
			t.Errorf("model %s codename %q %q", m.ID, m.Codename, m.CodenameFull)
		}
	}

	// unchanged Hugging Face data: a second run is byte-identical
	if err := refreshModels(hfClient(srv), out, 0); err != nil {
		t.Fatal(err)
	}
	again, _ := os.ReadFile(out)
	if string(again) != string(after) {
		t.Error("second models-only run changed the file")
	}
	// an explicit version wins
	if err := refreshModels(hfClient(srv), out, 9); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(out); !strings.Contains(string(b), `"version": 9`) {
		t.Error("version not applied")
	}
	// a reference model that is not on Hugging Face is an error here, never an import-only entry
	bad := fakeHF(t, map[string]string{}, nil, nil)
	if err := refreshModels(hfClient(bad), out, 0); err == nil {
		t.Error("missing reference accepted")
	}
}
