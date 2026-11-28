// Command gencatalog writes engine/catalog/catalog.json from the public GitHub releases API
// (llama.cpp, uv) and the Hugging Face API (stock models). Hashes are never typed by hand:
// they come from the API's asset digest, the LFS object id, or a download-and-hash.
//
//	go run ./tools/gencatalog -out catalog/catalog.json -version 10 -llama-tag b12651 -uv-tag 0.13.10
//	go run ./tools/gencatalog -out catalog/catalog.json -version 10 -repo CharlesMod/llama.cpp -llama-tag b12651-easy1
//
// A patched pin (b<N>-easy<M>) writes two rungs: the pin, then stock upstream (-stock-repo,
// -stock-tag, default b<N>) holding only builds for platforms the pin has none for.
//
// Flags: -llama-tag and -uv-tag pin the llama.cpp build and the uv release (default: newest);
// -cache DIR (default $HOME/.cache/gencatalog) keeps every downloaded archive as <sha256>.<ext>
// and its measured unpacked size as <sha256>.unpacked, so a rerun neither downloads nor unpacks
// again; -measured-only re-attaches measured.json and runtime.json to an existing catalog with no
// network. Each asset gets unpacked_size: the archive is unpacked, every regular file is counted
// rounded up to 4 KiB, plus 4 KiB per directory (the root too) and per symlink.
//
// -models-only touches no GitHub: it re-reads -out, keeps its llama, uv and agent_runtime parts
// exactly, rebuilds the models from Hugging Face and writes it back:
//
//	go run ./tools/gencatalog -models-only -out catalog/catalog.json
package main

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/fetch"
)

// The reference build is published on Hugging Face by the keeper; its sha256 is checked against
// the digest below. If the upload is not visible yet the entry falls back to import-only (url "").
const (
	refID     = "gemma4-rp-maxxed-q3kxl"
	refFile   = "gemma4-rp-maxxed-Q3_K_XL.gguf"
	refSHA256 = "595eddca7b75d56ee4a6fa363deb27f1c70c8792b97508e3336a3a3d3f071b29"
)

// measured.json is the checked-in measured footprint per model id (written by tools/fitcheck from
// the sweep CSV). Every regeneration carries it into the catalog, so a refreshed llama.cpp pin
// or model list never drops the numbers.
//
//go:embed measured.json
var measuredJSON []byte

// runtime.json is the checked-in measured agent setup sizes (a JSON array of catalog.RuntimeSize),
// carried into every catalog like measured.json.
//
//go:embed runtime.json
var runtimeJSON []byte

// loadRuntime parses runtime.json.
func loadRuntime(b []byte) ([]catalog.RuntimeSize, error) {
	var r []catalog.RuntimeSize
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("runtime.json: %w", err)
	}
	return r, nil
}

// loadMeasured parses measured.json.
func loadMeasured(b []byte) (map[string]catalog.Footprint, error) {
	var m map[string]catalog.Footprint
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("measured.json: %w", err)
	}
	return m, nil
}

// applyMeasured attaches the measured footprint to every model that has one and reports the
// ids in measured.json that match no model (a typo or a removed model).
func applyMeasured(models []catalog.Model, measured map[string]catalog.Footprint) (unused []string) {
	used := map[string]bool{}
	for i := range models {
		if fp, ok := measured[models[i].ID]; ok {
			fp := fp
			models[i].Footprint = &fp
			used[models[i].ID] = true
		} else {
			models[i].Footprint = nil
		}
	}
	for id := range measured {
		if !used[id] {
			unused = append(unused, id)
		}
	}
	sort.Strings(unused)
	return unused
}

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
	api   string // GitHub API base; "" = https://api.github.com (tests point it at a fake server)
	cache string // archive and unpacked-size cache dir; "" = no unpacked sizes (offline tests)
	hf    string // Hugging Face base; "" = https://huggingface.co (tests point it at a fake server)
}

const defaultHF = "https://huggingface.co"

func (c *client) hfBase() string {
	if c.hf != "" {
		return c.hf
	}
	return defaultHF
}

const block = 4096

func roundUp(n int64) int64 { return (n + block - 1) / block * block }

// allocSize is the disk a tree takes: every regular file rounded up to 4 KiB, plus 4 KiB for
// every directory (root included) and every symlink.
func allocSize(root string) (int64, error) {
	var total int64
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		switch {
		case d.IsDir(), d.Type()&fs.ModeSymlink != 0:
			total += block
		case d.Type().IsRegular():
			fi, err := d.Info()
			if err != nil {
				return err
			}
			total += roundUp(fi.Size())
		}
		return nil
	})
	return total, err
}

// archiveExt returns .zip or .tar.gz from a URL's path.
func archiveExt(u string) (string, error) {
	p := strings.ToLower(u)
	if i := strings.IndexAny(p, "?#"); i >= 0 {
		p = p[:i]
	}
	switch {
	case strings.HasSuffix(p, ".zip"):
		return ".zip", nil
	case strings.HasSuffix(p, ".tar.gz"):
		return ".tar.gz", nil
	}
	return "", fmt.Errorf("unknown archive type in %s", u)
}

func fileSHA(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// download fetches u into dst (via a temp file in the same dir) and returns its sha256.
func (c *client) download(u, dst string) (string, error) {
	c.log("downloading %s", u)
	resp, err := c.get(u)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	f, err := os.CreateTemp(filepath.Dir(dst), "dl-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(f.Name())
	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(f, h), resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", err
	}
	sum := hex.EncodeToString(h.Sum(nil))
	return sum, os.Rename(f.Name(), dst)
}

// unpackedSize returns the allocated size of the archive at u (sha256 sum, size bytes), using
// the cache dir: download when absent or wrong, unpack into a fresh dir, measure, remove it.
func (c *client) unpackedSize(u, sum string, size int64) (int64, error) {
	if c.cache == "" {
		return 0, nil
	}
	if err := os.MkdirAll(c.cache, 0o755); err != nil {
		return 0, err
	}
	szFile := filepath.Join(c.cache, sum+".unpacked")
	if b, err := os.ReadFile(szFile); err == nil {
		if n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil && n > 0 {
			return n, nil
		}
	}
	ext, err := archiveExt(u)
	if err != nil {
		return 0, err
	}
	arc := filepath.Join(c.cache, sum+ext)
	ok := false
	if fi, err := os.Stat(arc); err == nil && (size == 0 || fi.Size() == size) {
		if got, err := fileSHA(arc); err == nil && got == sum {
			ok = true
		}
	}
	if !ok {
		got, err := c.download(u, arc)
		if err != nil {
			return 0, err
		}
		if got != sum {
			os.Remove(arc)
			return 0, fmt.Errorf("%s: sha256 %s, expected %s", u, got, sum)
		}
	}
	tmp, err := os.MkdirTemp(c.cache, "unpack-")
	if err != nil {
		return 0, err
	}
	defer os.RemoveAll(tmp)
	dest := filepath.Join(tmp, "x")
	if err := fetch.Unpack(arc, dest); err != nil {
		return 0, fmt.Errorf("unpack %s: %w", u, err)
	}
	n, err := allocSize(dest)
	if err != nil {
		return 0, err
	}
	c.log("%s unpacks to %d bytes on disk", filepath.Base(arc), n)
	return n, os.WriteFile(szFile, []byte(strconv.FormatInt(n, 10)+"\n"), 0o644)
}

func (c *client) apiBase() string {
	if c.api != "" {
		return c.api
	}
	return defaultAPI
}

func (c *client) get(u string) (*http.Response, error) {
	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "easy-engine-gencatalog")
	req.Header.Set("Accept", "application/json")
	if c.token != "" && strings.HasPrefix(u, c.apiBase()+"/") {
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
	if c.cache != "" {
		if ext, err := archiveExt(a.BrowserDownloadURL); err == nil {
			if err := os.MkdirAll(c.cache, 0o755); err != nil {
				return "", err
			}
			tmp := filepath.Join(c.cache, "nodigest-"+a.Name)
			sum, err := c.download(a.BrowserDownloadURL, tmp)
			if err != nil {
				return "", err
			}
			// keep the file under its hash so unpackedSize finds it
			return sum, os.Rename(tmp, filepath.Join(c.cache, sum+ext))
		}
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
	{id: "cuda-win-x64", os: "windows", arch: "amd64", accel: "cuda", re: `^llama-(?:%s-)?bin-win-cuda-(12\.\d+)-x64\.zip$`,
		extraID: "cudart-win-x64", extraRe: `^cudart-llama-bin-win-cuda-(12\.\d+)-x64\.zip$`},
	{id: "cuda13-win-x64", os: "windows", arch: "amd64", accel: "cuda", re: `^llama-(?:%s-)?bin-win-cuda-(1[3-9]\.\d+)-x64\.zip$`,
		extraID: "cudart13-win-x64", extraRe: `^cudart-llama-bin-win-cuda-(1[3-9]\.\d+)-x64\.zip$`, optional: true},
	{id: "vulkan-win-x64", os: "windows", arch: "amd64", accel: "vulkan", re: `^llama-(?:%s-)?bin-win-vulkan-x64\.zip$`},
	{id: "cpu-win-x64", os: "windows", arch: "amd64", accel: "cpu", re: `^llama-(?:%s-)?bin-win-cpu-x64\.zip$`},
	{id: "metal-mac-arm64", os: "darwin", arch: "arm64", accel: "metal", re: `^llama-%s-bin-macos-arm64\.tar\.gz$`},
	{id: "cpu-mac-x64", os: "darwin", arch: "amd64", accel: "cpu", re: `^llama-%s-bin-macos-x64\.tar\.gz$`},
	// arm64 builds: the fork builds none, so these come from the stock rung (all optional).
	{id: "cpu-linux-arm64", os: "linux", arch: "arm64", accel: "cpu", re: `^llama-%s-bin-ubuntu-arm64\.tar\.gz$`, optional: true},
	{id: "vulkan-linux-arm64", os: "linux", arch: "arm64", accel: "vulkan", re: `^llama-%s-bin-ubuntu-vulkan-arm64\.tar\.gz$`, optional: true},
	{id: "cuda13-linux-arm64", os: "linux", arch: "arm64", accel: "cuda", re: `^llama-%s-bin-ubuntu-cuda-(1[3-9]\.\d+)-arm64\.tar\.gz$`,
		extraID: "cudart13-linux-arm64", extraRe: `^cudart-llama-%s-bin-ubuntu-cuda-(1[3-9]\.\d+)-arm64\.tar\.gz$`, optional: true},
	{id: "cpu-win-arm64", os: "windows", arch: "arm64", accel: "cpu", re: `^llama-(?:%s-)?bin-win-cpu-arm64\.zip$`, optional: true},
	{id: "vulkan-win-arm64", os: "windows", arch: "arm64", accel: "vulkan", re: `^llama-(?:%s-)?bin-win-vulkan-arm64\.zip$`, optional: true},
}

// pat fills the tag into a pattern that has a %s slot; the Windows cudart names carry no tag.
func pat(p, tag string) string {
	if strings.Contains(p, "%s") {
		return fmt.Sprintf(p, regexp.QuoteMeta(tag))
	}
	return p
}

// tagRe accepts an upstream build tag (b12651) and a patched fork tag (b12651-easy1); nothing else.
var tagRe = regexp.MustCompile(`^b(\d+)(?:-easy(\d+))?$`)

// parseTag returns the llama.cpp build number and the patch revision of a tag: b12651 is (12651, 0),
// b12651-easy1 is (12651, 1). The full tag stays the identity wherever it is written out.
func parseTag(tag string) (build, patch int, err error) {
	m := tagRe.FindStringSubmatch(tag)
	if m == nil {
		return 0, 0, fmt.Errorf("tag %q is neither bNNNN nor bNNNN-easyN", tag)
	}
	if build, err = strconv.Atoi(m[1]); err != nil {
		return 0, 0, fmt.Errorf("tag %q: %w", tag, err)
	}
	if m[2] != "" {
		if patch, err = strconv.Atoi(m[2]); err != nil {
			return 0, 0, fmt.Errorf("tag %q: %w", tag, err)
		}
	}
	return build, patch, nil
}

// repoRe is a GitHub owner/name: plain name characters only, so it is safe inside a URL path.
var repoRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)

const defaultRepo = "ggml-org/llama.cpp"
const defaultAPI = "https://api.github.com"

func validRepo(r string) error {
	if !repoRe.MatchString(r) || strings.Contains(r, "..") {
		return fmt.Errorf("repo %q is not owner/name", r)
	}
	return nil
}

// releasesURL is the API call that lists a repo's newest releases.
func releasesURL(api, repo string) string {
	return api + "/repos/" + repo + "/releases?per_page=100"
}

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
	return c.llamaAssetsFor(r, nil)
}

// platKey is a platform and accelerator: what catalog.offered compares between rungs.
type platKey struct{ os, arch, accel string }

// llamaAssetsFor builds the assets of release r. With skip non-nil it is the stock rung's call:
// specs whose platform and accelerator are in skip (the patched rung has them) are left out, and
// a missing asset is logged and skipped, never an error. With skip nil every required spec must
// be found.
func (c *client) llamaAssetsFor(r ghRelease, skip map[platKey]bool) ([]catalog.Asset, error) {
	var out []catalog.Asset
	for _, sp := range llamaSpecs {
		if skip != nil && skip[platKey{sp.os, sp.arch, sp.accel}] {
			continue
		}
		a, ok := findAsset(r.Assets, pat(sp.re, r.TagName))
		if !ok {
			if sp.optional || skip != nil {
				if skip != nil {
					c.log("stock %s has no build for %s", r.TagName, sp.id)
				}
				continue
			}
			return nil, fmt.Errorf("missing asset for %s", sp.id)
		}
		sum, err := c.digest(a)
		if err != nil {
			return nil, err
		}
		as := catalog.Asset{ID: sp.id, OS: sp.os, Arch: sp.arch, Accel: sp.accel, URL: a.BrowserDownloadURL, SHA256: sum, Size: a.Size}
		if as.UnpackedSize, err = c.unpackedSize(as.URL, sum, a.Size); err != nil {
			return nil, err
		}
		var extra *catalog.Asset
		if sp.extraRe != "" {
			if e, ok := findAsset(r.Assets, pat(sp.extraRe, r.TagName)); ok {
				es, err := c.digest(e)
				if err != nil {
					return nil, err
				}
				extra = &catalog.Asset{ID: sp.extraID, OS: sp.os, Arch: sp.arch, Accel: "cudart", URL: e.BrowserDownloadURL, SHA256: es, Size: e.Size}
				if extra.UnpackedSize, err = c.unpackedSize(extra.URL, es, e.Size); err != nil {
					return nil, err
				}
				as.Extra = []string{sp.extraID}
			} else if !sp.optional && skip == nil {
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
		un, err := c.unpackedSize(found.BrowserDownloadURL, sum, found.Size)
		if err != nil {
			return nil, err
		}
		out = append(out, catalog.Asset{ID: sp.id, OS: sp.os, Arch: sp.arch, URL: found.BrowserDownloadURL, SHA256: sum, Size: found.Size, UnpackedSize: un})
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
	id, name, repo, file, chat, notes, fallbackNote string
	codename, codenameFull                          string // the tier's short and full display names
	optIn                                           bool   // never auto-picked
	moe, useMain, fallback                          bool   // useMain: url says resolve/main (the keeper's own repo)
	quality                                         int
}

var modelSpecs = []modelSpec{
	// The keeper's rp build of E4B (QAT-exact Q4_0, refusals removed, vocabulary pruned); replaced stock E4B Q4_K_M on 2026-11-25.
	{id: "gemma4-e4b-rp-q40", name: "Gemma 4 E4B RP", codename: "Lantern", codenameFull: "Lantern-4B", repo: "cmod/gemma4-e4b-rp-GGUF", file: "gemma-4-E4B-it-qat-rp-Q4_0.gguf",
		useMain: true, quality: 60, chat: "gemma4"},
	{id: "gemma4-e2b-q4km", name: "Gemma 4 E2B (Q4_K_M)", codename: "Flashlight", codenameFull: "Flashlight-2B", repo: "unsloth/gemma-4-E2B-it-GGUF", file: "gemma-4-E2B-it-Q4_K_M.gguf", quality: 40, chat: "gemma4",
		fallback: true, fallbackNote: "Smallest model: expect repeated lines and weaker understanding of what was said."},
}

func (c *client) hfModel(sp modelSpec) (catalog.Model, error) {
	var info struct {
		SHA string `json:"sha"`
	}
	if err := c.getJSON(c.hfBase()+"/api/models/"+sp.repo, &info); err != nil {
		return catalog.Model{}, err
	}
	if info.SHA == "" {
		return catalog.Model{}, fmt.Errorf("%s: no commit sha", sp.repo)
	}
	var tree []hfEntry
	// the tree is listed at the pinned commit so the url, size and hash describe one revision
	if err := c.getJSON(c.hfBase()+"/api/models/"+sp.repo+"/tree/"+info.SHA+"?limit=1000", &tree); err != nil {
		return catalog.Model{}, err
	}
	hasSums := false
	for _, e := range tree {
		if e.Path == "SHA256SUMS" {
			hasSums = true
		}
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
		if hasSums {
			if err := c.checkSums(sp, info.SHA, e.LFS.OID); err != nil {
				return catalog.Model{}, err
			}
		}
		rev := info.SHA
		if sp.useMain {
			rev = "main"
		}
		return catalog.Model{
			ID: sp.id, Name: sp.name, Codename: sp.codename, CodenameFull: sp.codenameFull, OptIn: sp.optIn, File: sp.file,
			URL:    c.hfBase() + "/" + sp.repo + "/resolve/" + rev + "/" + url.PathEscape(sp.file),
			SHA256: strings.ToLower(e.LFS.OID), Size: size, MoE: sp.moe, Quality: sp.quality, Chat: sp.chat, Notes: sp.notes,
			Fallback: sp.fallback, FallbackNote: sp.fallbackNote,
		}, nil
	}
	return catalog.Model{}, fmt.Errorf("%s: file %s not in the repo", sp.repo, sp.file)
}

// checkSums fetches SHA256SUMS at the pinned commit; when it names the model file, the hash there
// must equal the LFS oid. A file it does not name is left to the LFS oid.
func (c *client) checkSums(sp modelSpec, sha, oid string) error {
	resp, err := c.get(c.hfBase() + "/" + sp.repo + "/resolve/" + sha + "/SHA256SUMS")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Fields(strings.TrimSpace(line))
		if len(f) != 2 || len(f[0]) != 64 {
			continue
		}
		if strings.TrimPrefix(f[1], "*") != sp.file {
			continue
		}
		if !strings.EqualFold(f[0], oid) {
			return fmt.Errorf("%s/%s: SHA256SUMS says %s but the LFS oid is %s", sp.repo, sp.file, strings.ToLower(f[0]), strings.ToLower(oid))
		}
		return nil
	}
	return nil
}

var refSpec = modelSpec{id: refID, name: "Gemma 4 26B RP", codename: "Campfire", codenameFull: "Campfire-26B", repo: "cmod/gemma4-rp-maxxed-GGUF", file: refFile,
	moe: true, useMain: true, quality: 100, chat: "gemma4", notes: "reference model"}

func main() {
	out := flag.String("out", "catalog/catalog.json", "output file")
	pin := flag.String("llama-tag", "", "pin a llama.cpp build tag (bNNNN or bNNNN-easyN) instead of picking the newest")
	repo := flag.String("repo", defaultRepo, "GitHub owner/name whose releases carry the llama.cpp builds")
	stockRepo := flag.String("stock-repo", defaultRepo, "GitHub owner/name of the stock upstream release that fills the platforms the pin has no build for")
	stockTag := flag.String("stock-tag", "", "the stock release tag (bNNNN); default: the build number of a patched -llama-tag")
	// A week, not a day: Windows Smart App Control refuses unsigned DLLs with no reputation yet. In the first
	// Windows test it blocked b12680's llama.dll at about a day old, while b12651 (3 days) and older ran.
	minAge := flag.Duration("min-age", 7*24*time.Hour, "when every release is a prerelease, skip builds newer than this (Windows Smart App Control needs a build to have a reputation)")
	ref := flag.String("ref-file", filepath.Join(os.Getenv("HOME"), ".hive/models/gemma4-rp-maxxed", refFile), "the reference model file; its size is recorded (not copied or loaded)")
	uvTag := flag.String("uv-tag", "", "pin a uv release tag instead of the latest")
	cacheDir := flag.String("cache", filepath.Join(os.Getenv("HOME"), ".cache", "gencatalog"), "archive and unpacked-size cache")
	version := flag.Int("version", 1, "catalog version to write")
	modelsOnly := flag.Bool("models-only", false, "no GitHub: re-read -out, keep its llama, uv and agent_runtime parts, rebuild the models from Hugging Face and write it back (-version if given, else the file's)")
	measuredOnly := flag.Bool("measured-only", false, "no network: re-read -out, attach tools/gencatalog/measured.json and write it back with -version")
	flag.Parse()

	logf := func(f string, a ...any) { fmt.Fprintf(os.Stderr, "gencatalog: "+f+"\n", a...) }
	if *measuredOnly {
		if err := refreshMeasured(*out, *version); err != nil {
			fmt.Fprintln(os.Stderr, "gencatalog:", err)
			os.Exit(1)
		}
		return
	}
	if *modelsOnly {
		explicit := false
		flag.Visit(func(f *flag.Flag) {
			if f.Name == "version" {
				explicit = true
			}
		})
		v := 0
		if explicit {
			v = *version
		}
		c := &client{http: &http.Client{Timeout: 5 * time.Minute}, log: logf}
		if err := refreshModels(c, *out, v); err != nil {
			fmt.Fprintln(os.Stderr, "gencatalog:", err)
			os.Exit(1)
		}
		return
	}
	if err := validRepo(*repo); err != nil {
		fmt.Fprintln(os.Stderr, "gencatalog:", err)
		os.Exit(2)
	}
	if err := validRepo(*stockRepo); err != nil {
		fmt.Fprintln(os.Stderr, "gencatalog: -stock-repo:", err)
		os.Exit(2)
	}
	if *pin != "" {
		if _, _, err := parseTag(*pin); err != nil {
			fmt.Fprintln(os.Stderr, "gencatalog: -llama-tag:", err)
			os.Exit(2)
		}
	}
	if *stockTag != "" {
		if _, p, err := parseTag(*stockTag); err != nil || p != 0 {
			fmt.Fprintf(os.Stderr, "gencatalog: -stock-tag %q is not a stock build tag (bNNNN)\n", *stockTag)
			os.Exit(2)
		}
	}
	hc := &http.Client{Timeout: 30 * time.Minute} // network timeout
	c := &client{http: hc, token: os.Getenv("GITHUB_TOKEN"), log: logf, cache: *cacheDir}
	if err := run(c, *repo, *stockRepo, *out, *pin, *stockTag, *uvTag, *minAge, *ref, *version); err != nil {
		fmt.Fprintln(os.Stderr, "gencatalog:", err)
		os.Exit(1)
	}
}

// buildModels builds the model list from Hugging Face (the reference model first, then modelSpecs)
// and attaches measured.json and the samplers. With refFallback, a reference model that is not on
// Hugging Face yet becomes an import-only entry sized from refPath; without it that is an error.
func buildModels(c *client, refPath string, refFallback bool) ([]catalog.Model, error) {
	refModel, err := c.hfModel(refSpec)
	switch {
	case err != nil && !refFallback:
		return nil, fmt.Errorf("reference model: %w", err)
	case err != nil:
		c.log("WARNING: reference model not on Hugging Face yet (%v); writing it import-only", err)
		st, serr := os.Stat(refPath)
		if serr != nil {
			return nil, fmt.Errorf("reference model is not on Hugging Face (%v) and the local file is missing: %w", err, serr)
		}
		refModel = catalog.Model{ID: refID, Name: refSpec.name, Codename: refSpec.codename, CodenameFull: refSpec.codenameFull, File: refFile, SHA256: refSHA256, Size: st.Size(),
			MoE: true, Quality: 100, Chat: "gemma4", Notes: "reference model; import-only"}
	case refModel.SHA256 != refSHA256:
		return nil, fmt.Errorf("reference model on Hugging Face has sha256 %s, expected %s", refModel.SHA256, refSHA256)
	}
	models := []catalog.Model{refModel}
	for _, sp := range modelSpecs {
		m, err := c.hfModel(sp)
		if err != nil {
			return nil, err
		}
		c.log("model %s %d bytes", m.ID, m.Size)
		models = append(models, m)
	}
	measured, err := loadMeasured(measuredJSON)
	if err != nil {
		return nil, err
	}
	if un := applyMeasured(models, measured); len(un) > 0 {
		c.log("WARNING: measured.json has footprints for models not in the catalog: %v", un)
	}
	if un := applySamplers(models); len(un) > 0 {
		return nil, fmt.Errorf("no sampler spec for models %v (add them to samplerSpecs)", un)
	}
	return models, nil
}

// refreshModels rebuilds only the models of an existing catalog file from Hugging Face; llama, uv
// and agent_runtime stay as read. version 0 keeps the file's version.
func refreshModels(c *client, out string, version int) error {
	b, err := os.ReadFile(out)
	if err != nil {
		return err
	}
	cat, err := catalog.Parse(b)
	if err != nil {
		return err
	}
	if cat.Models, err = buildModels(c, "", false); err != nil {
		return err
	}
	if version > 0 {
		cat.Version = version
	}
	if err := writeCatalog(out, cat); err != nil {
		return err
	}
	c.log("wrote %s", out)
	return nil
}

func run(c *client, repo, stockRepo, out, pin, stockTag, uvTag string, minAge time.Duration, refPath string, version int) error {
	if err := validRepo(repo); err != nil {
		return err
	}
	var rs []ghRelease
	if err := c.getJSON(releasesURL(c.apiBase(), repo), &rs); err != nil {
		return err
	}
	rel, why, err := selectRelease(rs, time.Now(), minAge, pin)
	if err != nil {
		return err
	}
	build, patch, _ := parseTag(rel.TagName)
	c.log("llama.cpp %s from %s: build %d, patch revision %d (%s)", rel.TagName, repo, build, patch, why)
	rungs, err := c.llamaRungs(repo, stockRepo, stockTag, rel, patch > 0, build, rs, minAge)
	if err != nil {
		return err
	}
	var uvRel ghRelease
	uvURL := "https://api.github.com/repos/astral-sh/uv/releases/latest"
	if uvTag != "" {
		uvURL = "https://api.github.com/repos/astral-sh/uv/releases/tags/" + url.PathEscape(uvTag)
	}
	if err := c.getJSON(uvURL, &uvRel); err != nil {
		return err
	}
	c.log("uv %s", uvRel.TagName)
	ua, err := c.uvAssets(uvRel)
	if err != nil {
		return err
	}
	models, err := buildModels(c, refPath, true)
	if err != nil {
		return err
	}
	cat := catalog.Catalog{
		Version: version,
		Llama:   rungs,
		UV:      ua,
		Models:  models,
	}
	if cat.AgentRuntime, err = loadRuntime(runtimeJSON); err != nil {
		return err
	}
	if err := writeCatalog(out, &cat); err != nil {
		return err
	}
	c.log("wrote %s", out)
	return nil
}

// llamaRungs builds the ladder: rung 0 is the pin (rel, from repo); when the pin is patched, rung 1
// is the stock release (stockTag, default b<build>) from stockRepo, holding only the builds for a
// platform and accelerator the pin has none for. An empty stock rung is not written.
func (c *client) llamaRungs(repo, stockRepo, stockTag string, rel ghRelease, patched bool, build int, rs []ghRelease, minAge time.Duration) ([]catalog.LlamaCpp, error) {
	la, err := c.llamaAssets(rel)
	if err != nil {
		return nil, err
	}
	rungs := []catalog.LlamaCpp{{Tag: rel.TagName, Repo: repo, Patched: patched, Assets: la}}
	for _, a := range la {
		c.log("build %s from rung 0 (%s, %s)", a.ID, rel.TagName, repo)
	}
	if !patched {
		return rungs, nil
	}
	if err := validRepo(stockRepo); err != nil {
		return nil, err
	}
	if stockTag == "" {
		stockTag = fmt.Sprintf("b%d", build)
	}
	if _, p, err := parseTag(stockTag); err != nil || p != 0 {
		return nil, fmt.Errorf("stock tag %q is not a stock build tag (bNNNN)", stockTag)
	}
	srs := rs
	if stockRepo != repo {
		srs = nil
		if err := c.getJSON(releasesURL(c.apiBase(), stockRepo), &srs); err != nil {
			return nil, err
		}
	}
	srel, _, err := selectRelease(srs, time.Now(), minAge, stockTag)
	if err != nil {
		return nil, fmt.Errorf("stock rung: %w", err)
	}
	covered := map[platKey]bool{}
	for _, a := range la {
		if a.Accel != "cudart" {
			covered[platKey{a.OS, a.Arch, a.Accel}] = true
		}
	}
	sa, err := c.llamaAssetsFor(srel, covered)
	if err != nil {
		return nil, err
	}
	if len(sa) == 0 {
		c.log("stock %s adds no build; writing one rung", stockTag)
		return rungs, nil
	}
	for _, a := range sa {
		c.log("build %s from rung 1 (%s, %s)", a.ID, srel.TagName, stockRepo)
	}
	return append(rungs, catalog.LlamaCpp{Tag: srel.TagName, Repo: stockRepo, Assets: sa}), nil
}

func writeCatalog(out string, cat *catalog.Catalog) error {
	b, err := json.MarshalIndent(cat, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(out, append(b, '\n'), 0o644)
}

// refreshMeasured re-attaches measured.json to an existing catalog file without touching the
// network, at the given version.
func refreshMeasured(out string, version int) error {
	b, err := os.ReadFile(out)
	if err != nil {
		return err
	}
	cat, err := catalog.Parse(b)
	if err != nil {
		return err
	}
	measured, err := loadMeasured(measuredJSON)
	if err != nil {
		return err
	}
	if un := applyMeasured(cat.Models, measured); len(un) > 0 {
		return fmt.Errorf("measured.json has footprints for models not in %s: %v", out, un)
	}
	if un := applySamplers(cat.Models); len(un) > 0 {
		return fmt.Errorf("no sampler spec for models %v (add them to samplerSpecs)", un)
	}
	if cat.AgentRuntime, err = loadRuntime(runtimeJSON); err != nil {
		return err
	}
	cat.Version = version
	return writeCatalog(out, cat)
}
