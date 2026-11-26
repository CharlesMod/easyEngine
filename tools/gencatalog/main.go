// Command gencatalog writes engine/catalog/catalog.json from the public GitHub releases API
// (llama.cpp, uv) and the Hugging Face API (stock models). Hashes are never typed by hand:
// they come from the API's asset digest, the LFS object id, or a download-and-hash.
//
//	go run ./tools/gencatalog -out catalog/catalog.json -version 8 -llama-tag b12651 -uv-tag 0.13.10
//
// Flags: -llama-tag and -uv-tag pin the llama.cpp build and the uv release (default: newest);
// -cache DIR (default $HOME/.cache/gencatalog) keeps every downloaded archive as <sha256>.<ext>
// and its measured unpacked size as <sha256>.unpacked, so a rerun neither downloads nor unpacks
// again; -measured-only re-attaches measured.json and runtime.json to an existing catalog with no
// network. Each asset gets unpacked_size: the archive is unpacked, every regular file is counted
// rounded up to 4 KiB, plus 4 KiB per directory (the root too) and per symlink.
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

	"hermesengine/catalog"
	"hermesengine/fetch"
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
	cache string // archive and unpacked-size cache dir; "" = no unpacked sizes (offline tests)
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
	moe, useMain, fallback                          bool // useMain: url says resolve/main (the keeper's own repo)
	quality                                         int
}

var modelSpecs = []modelSpec{
	// The keeper's rp build of E4B (QAT-exact Q4_0, refusals removed, vocabulary pruned); replaced stock E4B Q4_K_M on 2026-11-25.
	{id: "gemma4-e4b-rp-q40", name: "Gemma 4 E4B RP (HermesVoice build)", repo: "cmod/gemma4-e4b-rp-GGUF", file: "gemma-4-E4B-it-qat-rp-Q4_0.gguf",
		useMain: true, quality: 60, chat: "gemma4"},
	{id: "gemma4-e2b-q4km", name: "Gemma 4 E2B (Q4_K_M)", repo: "unsloth/gemma-4-E2B-it-GGUF", file: "gemma-4-E2B-it-Q4_K_M.gguf", quality: 40, chat: "gemma4",
		fallback: true, fallbackNote: "Smallest model: expect repeated lines and weaker understanding of what was said."},
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
			Fallback: sp.fallback, FallbackNote: sp.fallbackNote,
		}, nil
	}
	return catalog.Model{}, fmt.Errorf("%s: file %s not in the repo", sp.repo, sp.file)
}

var refSpec = modelSpec{id: refID, name: "Gemma 4 26B RP (HermesVoice build)", repo: "cmod/gemma4-rp-maxxed-GGUF", file: refFile,
	moe: true, useMain: true, quality: 100, chat: "gemma4", notes: "reference model"}

func main() {
	out := flag.String("out", "catalog/catalog.json", "output file")
	pin := flag.String("llama-tag", "", "pin a llama.cpp build tag instead of picking the newest")
	// A week, not a day: Windows Smart App Control refuses unsigned DLLs with no reputation yet. In the first
	// Windows test it blocked b12680's llama.dll at about a day old, while b12651 (3 days) and older ran.
	minAge := flag.Duration("min-age", 7*24*time.Hour, "when every release is a prerelease, skip builds newer than this (Windows Smart App Control needs a build to have a reputation)")
	ref := flag.String("ref-file", filepath.Join(os.Getenv("HOME"), ".hive/models/gemma4-rp-maxxed", refFile), "the reference model file; its size is recorded (not copied or loaded)")
	uvTag := flag.String("uv-tag", "", "pin a uv release tag instead of the latest")
	cacheDir := flag.String("cache", filepath.Join(os.Getenv("HOME"), ".cache", "gencatalog"), "archive and unpacked-size cache")
	version := flag.Int("version", 1, "catalog version to write")
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
	c := &client{http: &http.Client{Timeout: 30 * time.Minute}, token: os.Getenv("GITHUB_TOKEN"), log: logf, cache: *cacheDir}
	if err := run(c, *out, *pin, *uvTag, *minAge, *ref, *version); err != nil {
		fmt.Fprintln(os.Stderr, "gencatalog:", err)
		os.Exit(1)
	}
}

func run(c *client, out, pin, uvTag string, minAge time.Duration, refPath string, version int) error {
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
	refModel, err := c.hfModel(refSpec)
	switch {
	case err != nil:
		c.log("WARNING: reference model not on Hugging Face yet (%v); writing it import-only", err)
		st, serr := os.Stat(refPath)
		if serr != nil {
			return fmt.Errorf("reference model is not on Hugging Face (%v) and the local file is missing: %w", err, serr)
		}
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
	measured, err := loadMeasured(measuredJSON)
	if err != nil {
		return err
	}
	if un := applyMeasured(cat.Models, measured); len(un) > 0 {
		c.log("WARNING: measured.json has footprints for models not in the catalog: %v", un)
	}
	if un := applySamplers(cat.Models); len(un) > 0 {
		return fmt.Errorf("no sampler spec for models %v (add them to samplerSpecs)", un)
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
