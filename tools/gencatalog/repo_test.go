package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestParseTag(t *testing.T) {
	for _, c := range []struct {
		tag          string
		build, patch int
	}{{"b12651", 12651, 0}, {"b12651-easy1", 12651, 1}, {"b7-easy12", 7, 12}} {
		b, p, err := parseTag(c.tag)
		if err != nil || b != c.build || p != c.patch {
			t.Errorf("%s: got %d %d %v", c.tag, b, p, err)
		}
	}
	for _, bad := range []string{"", "b", "12651", "b12651-easy", "b12651-easy1-x", "b12651-rc1", "v1.0.0", "b12651-easy1\n", " b12651", "b1-easy1-easy2"} {
		if _, _, err := parseTag(bad); err == nil {
			t.Errorf("tag %q must be refused", bad)
		}
	}
}

func TestValidRepo(t *testing.T) {
	for _, ok := range []string{"ggml-org/llama.cpp", "CharlesMod/llama.cpp", "a/b_c-d.e"} {
		if err := validRepo(ok); err != nil {
			t.Errorf("%s: %v", ok, err)
		}
	}
	for _, bad := range []string{"", "llama.cpp", "a/b/c", "a/", "/b", "a b/c", "a/b?x=1", "a/../b", "../b", "a/b#c", "a/b\n", "https://github.com/a/b", "-a/b"} {
		if err := validRepo(bad); err == nil {
			t.Errorf("repo %q must be refused", bad)
		}
	}
}

func TestDefaultRepoUnchanged(t *testing.T) {
	if defaultRepo != "ggml-org/llama.cpp" {
		t.Fatal(defaultRepo)
	}
	if got := releasesURL(defaultAPI, defaultRepo); got != "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100" {
		t.Fatal(got)
	}
	if (&client{}).apiBase() != "https://api.github.com" {
		t.Fatal("default API base changed")
	}
}

// fakeGitHub serves one repo's release list and records the request paths it saw.
func fakeGitHub(t *testing.T, repo string, rs []ghRelease) (*httptest.Server, *[]string) {
	t.Helper()
	var mu sync.Mutex
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.RequestURI())
		mu.Unlock()
		if r.URL.Path != "/repos/"+repo+"/releases" {
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(rs)
	}))
	t.Cleanup(srv.Close)
	return srv, &paths
}

func TestForkReleaseListAndAssets(t *testing.T) {
	now := time.Now()
	const tag = "b12651-easy1"
	fork := rel(tag, false, 2*time.Hour, now, fullNames(tag)...)
	stock := rel("b12651", true, 30*24*time.Hour, now, fullNames("b12651")...)
	srv, paths := fakeGitHub(t, "CharlesMod/llama.cpp", []ghRelease{fork, stock})
	c := &client{http: srv.Client(), api: srv.URL, log: func(string, ...any) {}}

	var rs []ghRelease
	if err := c.getJSON(releasesURL(c.apiBase(), "CharlesMod/llama.cpp"), &rs); err != nil {
		t.Fatal(err)
	}
	if len(*paths) != 1 || (*paths)[0] != "/repos/CharlesMod/llama.cpp/releases?per_page=100" {
		t.Fatalf("paths %v", *paths)
	}
	got, why, err := selectRelease(rs, now, 7*24*time.Hour, tag)
	if err != nil || got.TagName != tag || why != "pinned by flag" {
		t.Fatalf("%v %v %s", got.TagName, err, why)
	}
	as, err := c.llamaAssets(got)
	if err != nil {
		t.Fatal(err)
	}
	if len(as) == 0 {
		t.Fatal("no assets")
	}
	for _, a := range as {
		if strings.HasPrefix(a.URL, "https://x/") && !strings.Contains(a.URL, tag) && !strings.HasPrefix(a.URL, "https://x/cudart-llama-bin-win") {
			t.Errorf("asset %s does not carry the full tag %s", a.URL, tag)
		}
	}
	// the stock tag is selectable from the same list, and the bare tag does not match the fork assets
	got, _, err = selectRelease(rs, now, 7*24*time.Hour, "b12651")
	if err != nil || got.TagName != "b12651" {
		t.Fatalf("stock pin: %v %v", got.TagName, err)
	}
	// a tag-less newest pick still works with both tag shapes in the list
	got, _, err = selectRelease(rs, now, 7*24*time.Hour, "")
	if err != nil || got.TagName != tag {
		t.Fatalf("newest: %v %v", got.TagName, err)
	}
}

func TestForkMissingAssetIsNamed(t *testing.T) {
	now := time.Now()
	const tag = "b12651-easy1"
	var names []string
	for _, n := range fullNames(tag) {
		if !strings.Contains(n, "macos-arm64") {
			names = append(names, n)
		}
	}
	r := rel(tag, false, time.Hour, now, names...)
	c := &client{log: func(string, ...any) {}}
	_, err := c.llamaAssets(r)
	if err == nil || !strings.Contains(err.Error(), "metal-mac-arm64") {
		t.Fatalf("want the missing platform named, got %v", err)
	}
	if _, _, err := selectRelease([]ghRelease{r}, now, time.Hour, tag); err == nil {
		t.Fatal("a release missing a required platform must not be selected")
	}
}

func TestRunRefusesBadRepoAndTagBeforeAnyRequest(t *testing.T) {
	srv, paths := fakeGitHub(t, "x/y", nil)
	c := &client{http: srv.Client(), api: srv.URL, log: func(string, ...any) {}}
	if err := run(c, "not a repo", defaultRepo, "unused.json", "", "", "", time.Hour, "", 1); err == nil {
		t.Fatal("bad repo accepted")
	}
	if len(*paths) != 0 {
		t.Fatalf("a bad repo reached the network: %v", *paths)
	}
	now := time.Now()
	rs := []ghRelease{rel("b1-easy1", false, time.Hour, now, fullNames("b1-easy1")...)}
	for _, bad := range []string{"b1-easy", "b1-rc1", "easy1"} {
		if _, _, err := selectRelease(rs, now, time.Hour, bad); err == nil {
			t.Errorf("pin %q accepted", bad)
		}
	}
	// a release whose tag has the wrong shape is never a candidate
	junk := rel("b1-easyX", false, time.Hour, now, fullNames("b1-easyX")...)
	if _, _, err := selectRelease([]ghRelease{junk}, now, time.Hour, ""); err == nil {
		t.Fatal("junk tag selected")
	}
}
