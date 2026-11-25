// Command gencatalog writes engine/catalog/catalog.json from the public GitHub releases API
// (llama.cpp, uv) and the Hugging Face API (stock models). Hashes are never typed by hand:
// they come from the API's asset digest, the LFS object id, or a download-and-hash.
//
//	go run ./tools/gencatalog -out catalog/catalog.json
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"hermesengine/catalog"
)

// The reference build is published on Hugging Face by the keeper; its sha256 is checked against
// the digest below. If the upload is not visible yet the entry falls back to import-only (url "").
const (
	refID     = "gemma4-rp-maxxed-q3kxl"
	refFile   = "gemma4-rp-maxxed-Q3_K_XL.gguf"
	refSHA256 = "595eddca7b75d56ee4a6fa363deb27f1c70c8792b97508e3336a3a3d3f071b29"
)

type ghAsset struct {
	Name               string `json:"name"`
	Size               int64  `json:"size"`
	Digest             string `json:"digest"`
	BrowserDownloadURL string `json:"browser_download_url"`
}

type ghRelease struct {
	TagName     string    `json:"tag_name"`
	Prerelease  bool      `json:"prerelease"`
	Draft       bool      `json:"draft"`
	PublishedAt time.Time `json:"published_at"`
	Assets      []ghAsset `json:"assets"`
}

type client struct {
	http  *http.Client
	token string
	log   func(string, ...any)
}

func (c *client) get(u string) (*http.Response, error) {
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "hermes-engine-gencatalog")
	req.Header.Set("Accept", "application/json")
	if c.token != "" && strings.HasPrefix(u, "https://api.github.com/") {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, fmt.Errorf("GET %s: %s", u, resp.Status)
	}
	return resp, nil
}

func (c *client) getJSON(u string, v any) error {
	resp, err := c.get(u)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

// hashURL downloads u and returns its sha256 (used only when the API gives no digest).
func (c *client) hashURL(u string) (string, error) {
	c.log("no digest from the API; downloading %s to hash it", u)
	resp, err := c.get(u)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	h := sha256.New()
	if _, err := io.Copy(h, resp.Body); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (c *client) digest(a ghAsset) (string, error) {
	if d, ok := strings.CutPrefix(a.Digest, "sha256:"); ok && len(d) == 64 {
		return strings.ToLower(d), nil
	}
	return c.hashURL(a.BrowserDownloadURL)
}

// ---- llama.cpp ----

type llamaSpec struct {
	id, os, arch, accel string
	re                  string // %s = tag; one capture group = toolkit version when present
	extraID             string
	extraRe             string
	optional            bool
}

var llamaSpecs = []llamaSpec{
	{id: "cuda-linux-x64", os: "linux", arch: "amd64", accel: "cuda", re: `^llama-%s-bin-ubuntu-cuda-(12\.\d+)-x64\.tar\.gz$`,
		extraID: "cudart-linux-x64", extraRe: `^cudart-llama-%s-bin-ubuntu-cuda-(12\.\d+)-x64\.tar\.gz$`},
	{id: "vulkan-linux-x64", os: "linux", arch: "amd64", accel: "vulkan", re: `^llama-%s-bin-ubuntu-vulkan-x64\.tar\.gz$`},
	{id: "cpu-linux-x64", os: "linux", arch: "amd64", accel: "cpu", re: `^llama-%s-bin-ubuntu-x64\.tar\.gz$`},
	{id: "cuda-win-x64", os: "windows", arch: "amd64", accel: "cuda", re: `^llama-%s-bin-win-cuda-(12\.\d+)-x64\.zip$`,
		extraID: "cudart-win-x64", extraRe: `^cudart-llama-bin-win-cuda-(12\.\d+)-x64\.zip$`},
	{id: "cuda13-win-x64", os: "windows", arch: "amd64", accel: "cuda", re: `^llama-%s-bin-win-cuda-(1[3-9]\.\d+)-x64\.zip$`,
		extraID: "cudart13-win-x64", extraRe: `^cudart-llama-bin-win-cuda-(1[3-9]\.\d+)-x64\.zip$`, optional: true},
	{id: "vulkan-win-x64", os: "windows", arch: "amd64", accel: "vulkan", re: `^llama-%s-bin-win-vulkan-x64\.zip$`},
	{id: "cpu-win-x64", os: "windows", arch: "amd64", accel: "cpu", re: `^llama-%s-bin-win-cpu-x64\.zip$`},
	{id: "metal-mac-arm64", os: "darwin", arch: "arm64", accel: "metal", re: `^llama-%s-bin-macos-arm64\.tar\.gz$`},
	{id: "cpu-mac-x64", os: "darwin", arch: "amd64", accel: "cpu", re: `^llama-%s-bin-macos-x64\.tar\.gz$`},
}

// pat fills the tag into a pattern that has a %s slot; the Windows cudart names carry no tag.
func pat(p, tag string) string {
	if strings.Contains(p, "%s") {
		return fmt.Sprintf(p, regexp.QuoteMeta(tag))
	}
	return p
}

var tagRe = regexp.MustCompile(`^b\d+$`)

// verOf turns "12.8" into a comparable number.
func verOf(s string) float64 {
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

// findAsset returns the asset matching pattern, preferring the highest toolkit version when
// the pattern captures one.
func findAsset(assets []ghAsset, pattern string) (ghAsset, bool) {
	re := regexp.MustCompile(pattern)
	var best ghAsset
	bestV := -1.0
	for _, a := range assets {
		m := re.FindStringSubmatch(a.Name)
		if m == nil {
			continue
		}
		v := 0.0
		if len(m) > 1 {
			v = verOf(m[1])
		}
		if v > bestV {
			best, bestV = a, v
		}
	}
	return best, bestV >= 0
}

// complete reports whether the release has every required asset.
func complete(r ghRelease) bool {
	for _, sp := range llamaSpecs {
		if sp.optional {
			continue
		}
		if _, ok := findAsset(r.Assets, pat(sp.re, r.TagName)); !ok {
			return false
		}
		if sp.extraRe != "" {
			if _, ok := findAsset(r.Assets, pat(sp.extraRe, r.TagName)); !ok {
				return false
			}
		}
	}
	return true
}

// selectRelease picks the newest non-prerelease build tag that has every needed asset. When no
// non-prerelease build tag exists (llama.cpp flags all of its releases as prereleases), it takes
// the newest complete build tag at least minAge old, which keeps a half-uploaded release out.
func selectRelease(rs []ghRelease, now time.Time, minAge time.Duration, pin string) (ghRelease, string, error) {
	var cands []ghRelease
	for _, r := range rs {
		if r.Draft || !tagRe.MatchString(r.TagName) {
			continue
		}
		cands = append(cands, r)
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].PublishedAt.After(cands[j].PublishedAt) })
	if pin != "" {
		for _, r := range cands {
			if r.TagName == pin {
				if !complete(r) {
					return r, "", fmt.Errorf("release %s lacks needed assets", pin)
				}
				return r, "pinned by flag", nil
			}
		}
		return ghRelease{}, "", fmt.Errorf("pinned release %s not found in the newest releases", pin)
	}
	for _, r := range cands {
		if !r.Prerelease && complete(r) {
			return r, "newest non-prerelease", nil
		}
	}
	for _, r := range cands {
		if r.Prerelease && now.Sub(r.PublishedAt) >= minAge && complete(r) {
			return r, "no non-prerelease build tag exists; newest complete prerelease at least " + minAge.String() + " old", nil
		}
	}
	return ghRelease{}, "", errors.New("no suitable llama.cpp release found")
}

func (c *client) llamaAssets(r ghRelease) ([]catalog.Asset, error) {
	var out []catalog.Asset
	for _, sp := range llamaSpecs {
		a, ok := findAsset(r.Assets, pat(sp.re, r.TagName))
		if !ok {
			if sp.optional {
				continue
			}
			return nil, fmt.Errorf("missing asset for %s", sp.id)
		}
		sum, err := c.digest(a)
		if err != nil {
			return nil, err
		}
		as := catalog.Asset{ID: sp.id, OS: sp.os, Arch: sp.arch, Accel: sp.accel, URL: a.BrowserDownloadURL, SHA256: sum, Size: a.Size}
		var extra *catalog.Asset
		if sp.extraRe != "" {
			if e, ok := findAsset(r.Assets, pat(sp.extraRe, r.TagName)); ok {
				es, err := c.digest(e)
				if err != nil {
					return nil, err
				}
				extra = &catalog.Asset{ID: sp.extraID, OS: sp.os, Arch: sp.arch, Accel: "cudart", URL: e.BrowserDownloadURL, SHA256: es, Size: e.Size}
				as.Extra = []string{sp.extraID}
			} else if !sp.optional {
				return nil, fmt.Errorf("missing extra asset for %s", sp.id)
			} else {
				continue
			}
		}
		out = append(out, as)
		if extra != nil {
			out = append(out, *extra)
		}
	}
	return out, nil
}

// ---- uv ----

var uvSpecs = []struct{ id, os, arch, name string }{
	{"uv-linux-amd64", "linux", "amd64", "uv-x86_64-unknown-linux-gnu.tar.gz"},
	{"uv-linux-arm64", "linux", "arm64", "uv-aarch64-unknown-linux-gnu.tar.gz"},
	{"uv-windows-amd64", "windows", "amd64", "uv-x86_64-pc-windows-msvc.zip"},
	{"uv-windows-arm64", "windows", "arm64", "uv-aarch64-pc-windows-msvc.zip"},
	{"uv-darwin-arm64", "darwin", "arm64", "uv-aarch64-apple-darwin.tar.gz"},
	{"uv-darwin-amd64", "darwin", "amd64", "uv-x86_64-apple-darwin.tar.gz"},
}

func (c *client) uvAssets(r ghRelease) ([]catalog.Asset, error) {
	var out []catalog.Asset
	for _, sp := range uvSpecs {
		var found *ghAsset
		for i := range r.Assets {
			if r.Assets[i].Name == sp.name {
				found = &r.Assets[i]
			}
		}
		if found == nil {
			return nil, fmt.Errorf("uv %s: asset %s missing", r.TagName, sp.name)
		}
		sum, err := c.digest(*found)
		if err != nil {
			return nil, err
		}
		out = append(out, catalog.Asset{ID: sp.id, OS: sp.os, Arch: sp.arch, URL: found.BrowserDownloadURL, SHA256: sum, Size: found.Size})
	}
	return out, nil
}

// ---- Hugging Face ----

type hfEntry struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	LFS  *struct {
		OID  string `json:"oid"`
		Size int64  `json:"size"`
	} `json:"lfs"`
}

type modelSpec struct {
	id, name, repo, file, chat, notes string
	moe, useMain                      bool // useMain: url says resolve/main (the keeper's own repo)
	quality                           int
}

var modelSpecs = []modelSpec{
	{id: "gemma4-e4b-q4km", name: "Gemma 4 E4B (Q4_K_M)", repo: "unsloth/gemma-4-E4B-it-GGUF", file: "gemma-4-E4B-it-Q4_K_M.gguf", quality: 60, chat: "gemma4"},
	{id: "gemma4-e2b-q4km", name: "Gemma 4 E2B (Q4_K_M)", repo: "unsloth/gemma-4-E2B-it-GGUF", file: "gemma-4-E2B-it-Q4_K_M.gguf", quality: 40, chat: "gemma4"},
}

func (c *client) hfModel(sp modelSpec) (catalog.Model, error) {
	var info struct {
		SHA string `json:"sha"`
	}
	if err := c.getJSON("https://huggingface.co/api/models/"+sp.repo, &info); err != nil {
		return catalog.Model{}, err
	}
	if info.SHA == "" {
		return catalog.Model{}, fmt.Errorf("%s: no commit sha", sp.repo)
	}
	var tree []hfEntry
	// the tree is listed at the pinned commit so the url, size and hash describe one revision
	if err := c.getJSON("https://huggingface.co/api/models/"+sp.repo+"/tree/"+info.SHA+"?limit=1000", &tree); err != nil {
		return catalog.Model{}, err
	}
	for _, e := range tree {
		if e.Path != sp.file {
			continue
		}
		if e.LFS == nil || len(e.LFS.OID) != 64 {
			return catalog.Model{}, fmt.Errorf("%s/%s: no LFS sha256", sp.repo, sp.file)
		}
		size := e.LFS.Size
		if size == 0 {
			size = e.Size
		}
		rev := info.SHA
		if sp.useMain {
			rev = "main"
		}
		return catalog.Model{
			ID: sp.id, Name: sp.name, File: sp.file,
			URL:    "https://huggingface.co/" + sp.repo + "/resolve/" + rev + "/" + url.PathEscape(sp.file),
			SHA256: strings.ToLower(e.LFS.OID), Size: size, MoE: sp.moe, Quality: sp.quality, Chat: sp.chat, Notes: sp.notes,
		}, nil
	}
	return catalog.Model{}, fmt.Errorf("%s: file %s not in the repo", sp.repo, sp.file)
}

var refSpec = modelSpec{id: refID, name: "Gemma 4 26B RP (HermesVoice build)", repo: "cmod/gemma4-rp-maxxed-GGUF", file: refFile,
	moe: true, useMain: true, quality: 100, chat: "gemma4", notes: "reference model"}

func main() {
	out := flag.String("out", "catalog/catalog.json", "output file")
	pin := flag.String("llama-tag", "", "pin a llama.cpp build tag instead of picking the newest")
	minAge := flag.Duration("min-age", 24*time.Hour, "when every release is a prerelease, skip builds newer than this")
	ref := flag.String("ref-file", filepath.Join(os.Getenv("HOME"), ".hive/models/gemma4-rp-maxxed", refFile), "the reference model file; its size is recorded (not copied or loaded)")
	version := flag.Int("version", 1, "catalog version to write")
	flag.Parse()

	logf := func(f string, a ...any) { fmt.Fprintf(os.Stderr, "gencatalog: "+f+"\n", a...) }
	c := &client{http: &http.Client{Timeout: 30 * time.Minute}, token: os.Getenv("GITHUB_TOKEN"), log: logf}
	if err := run(c, *out, *pin, *minAge, *ref, *version); err != nil {
		fmt.Fprintln(os.Stderr, "gencatalog:", err)
		os.Exit(1)
	}
}

func run(c *client, out, pin string, minAge time.Duration, refPath string, version int) error {
	st, err := os.Stat(refPath)
	if err != nil {
		return fmt.Errorf("reference model: %w", err)
	}
	var rs []ghRelease
	if err := c.getJSON("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100", &rs); err != nil {
		return err
	}
	rel, why, err := selectRelease(rs, time.Now(), minAge, pin)
	if err != nil {
		return err
	}
	c.log("llama.cpp %s (%s)", rel.TagName, why)
	la, err := c.llamaAssets(rel)
	if err != nil {
		return err
	}
	var uvRel ghRelease
	if err := c.getJSON("https://api.github.com/repos/astral-sh/uv/releases/latest", &uvRel); err != nil {
		return err
	}
	c.log("uv %s", uvRel.TagName)
	ua, err := c.uvAssets(uvRel)
	if err != nil {
		return err
	}
	refModel, err := c.hfModel(refSpec)
	switch {
	case err != nil:
		c.log("WARNING: reference model not on Hugging Face yet (%v); writing it import-only", err)
		refModel = catalog.Model{ID: refID, Name: refSpec.name, File: refFile, SHA256: refSHA256, Size: st.Size(),
			MoE: true, Quality: 100, Chat: "gemma4", Notes: "reference model; import-only"}
	case refModel.SHA256 != refSHA256:
		return fmt.Errorf("reference model on Hugging Face has sha256 %s, expected %s", refModel.SHA256, refSHA256)
	}
	cat := catalog.Catalog{
		Version:  version,
		LlamaCpp: catalog.LlamaCpp{Tag: rel.TagName, Assets: la},
		UV:       ua,
		Models:   []catalog.Model{refModel},
	}
	for _, sp := range modelSpecs {
		m, err := c.hfModel(sp)
		if err != nil {
			return err
		}
		c.log("model %s %d bytes", m.ID, m.Size)
		cat.Models = append(cat.Models, m)
	}
	b, err := json.MarshalIndent(cat, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if err := os.WriteFile(out, b, 0o644); err != nil {
		return err
	}
	c.log("wrote %s", out)
	return nil
}
