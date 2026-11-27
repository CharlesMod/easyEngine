package run

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/fetch"
	"github.com/CharlesMod/easyEngine/plan"
)

// Which llama.cpp build runs, and how the engine comes to run the catalog's pin
// (docs/ENGINE.md, "Which llama.cpp build").
//
// A build is installed under backends/<id>@<tag> (cuda-linux-x64@b12651-easy1); a build
// installed before the engine recorded tags sits under backends/<id> and has tag "". state.json
// names the build that runs (Backend, Build) and the one staged beside it for the next load
// (NextBuild).
//
// When the selected build is not the one the catalog gives this machine for that backend (the
// pin, or stock where the pin has no build), the move fetches the pin beside it in the
// background: downloaded, verified by its sha256 and unpacked, never run (a model server may hold
// the card). It is then staged. The next load of the model server, with the card free, smoke-tests
// it and runs it (promoteNext); a smoke test that fails there leaves the engine starting on the
// build it has, says why, and is not tried again for that tag until something changed that could
// change the outcome (a new pin, another driver, a person's or a client's word). A failed fetch is
// tried again only on a word (the network, a person's or a client's word), never on a request at
// the door alone and never on a clock. Once the new build has served ready, the other builds of
// the same backend id are removed (pruneBuilds). The engine never falls back from the patched
// build to stock because the patched one failed: a failure is reported.

// backendDir is where a build is installed.
func backendDir(dataDir, id, tag string) string {
	return filepath.Join(dataDir, "backends", buildDirName(id, tag))
}

// buildDirName is a build's directory name under backends/: <id>@<tag>, or <id> for a build
// installed before tags were recorded.
func buildDirName(id, tag string) string {
	if tag == "" {
		return id
	}
	return id + "@" + tag
}

// BuildInfo is status build and the build fact: the llama.cpp build the engine runs (or would
// run at its next start, with no model server up) and the move to the catalog's pin.
type BuildInfo struct {
	Backend string `json:"backend"` // the catalog's asset id: cuda-linux-x64
	// Tag is the llama.cpp release the build is from (b12651-easy1, b12651); "" for a build
	// installed before the engine recorded it.
	Tag     string `json:"tag"`
	Patched bool   `json:"patched"` // the fork's patched build (catalog.PatchedTag)
	// Rung is the build's place in the catalog's ladder (0 = the pin), -1 when the catalog does
	// not name it any more; Why says why this machine runs a build that is not patched.
	Rung int    `json:"rung"`
	Why  string `json:"why,omitempty"`
	// Pin is the tag the catalog gives this machine for this backend; the move runs while it
	// differs from Tag.
	Pin string `json:"pin,omitempty"`
	// VMM is what the patched server said of its CUDA buffers ("on (1 device)", "off
	// (GGML_CUDA_VMM_BUFFERS=0)"), from its last start; empty before one, and for a build that
	// does not say it.
	VMM  string     `json:"vmm,omitempty"`
	Next *BuildNext `json:"next,omitempty"`
}

// BuildNext is the move to the pin: fetching it beside the build that runs, and running it.
type BuildNext struct {
	Backend string `json:"backend"`
	Tag     string `json:"tag"`
	// State: downloading | installing (unpacking) | waiting (the fetch failed or the disk is
	// short: Error says why; a word tries again) | staged (fetched and verified by its sha256:
	// the next load smoke-tests and runs it) | failed (its smoke test failed at a load: Error says
	// why; the engine runs the build it has, and tries it again once the pin, the driver or a
	// person's or client's word changed).
	State    string    `json:"state"`
	Error    string    `json:"error,omitempty"`
	Attempt  int       `json:"attempt,omitempty"`
	Progress *Progress `json:"progress,omitempty"`
}

// smokeFailure is a staged build whose smoke test failed at a load, with what stood then.
type smokeFailure struct {
	id, tag, driver string
	words           uint64
}

// buildInfoLocked is status build; nil with no build selected. The caller holds s.mu.
func (s *Supervisor) buildInfoLocked() *BuildInfo {
	id, tag := s.st.Backend, s.st.Build
	if id == "" {
		return nil
	}
	b := &BuildInfo{Backend: id, Tag: tag, Patched: catalog.PatchedTag(tag), Rung: -1, VMM: s.vmm}
	if c := s.catLocked(); c != nil {
		if a, ok := c.LlamaAsset(tag, id); ok && tag != "" {
			b.Rung, b.Why = a.Rung, a.RungWhy
		}
		if p, ok := c.Asset(id); ok {
			b.Pin = p.Tag
		}
	}
	if tag == "" {
		b.Why = "installed before the engine recorded its build"
	}
	if s.move != nil {
		n := *s.move
		b.Next = &n
	}
	return b
}

// setMove changes the move under s.mu and says the build fact.
func (s *Supervisor) setMove(f func(n *BuildNext)) {
	s.mu.Lock()
	if s.move != nil {
		f(s.move)
	}
	s.refreshLocked()
	s.mu.Unlock()
}

// driverLocked is the NVIDIA driver the last probe saw ("" with none); the caller holds s.mu.
func (s *Supervisor) driverLocked() string {
	if s.hw != nil {
		for _, g := range s.hw.GPUs {
			if g.Driver != "" {
				return g.Driver
			}
		}
	}
	return ""
}

// smokeBlockedLocked reports whether build id@tag failed its smoke test and nothing that could
// change that has changed since: the same tag, the same driver, no word. The caller holds s.mu.
func (s *Supervisor) smokeBlockedLocked(id, tag string) bool {
	f := s.smokeFail
	return f != nil && f.id == id && f.tag == tag && f.driver == s.driverLocked() && f.words == s.words.Load()
}

// moveBusyLocked: a move may not begin while one runs, an install job runs (it may be writing a
// build), or the downloads are being removed. The caller holds s.mu.
func (s *Supervisor) moveBusyLocked() bool {
	return s.moveCancel != nil || s.jobRunning || s.removing
}

// checkPin starts the move when the selected build is not the one the catalog gives this
// machine for its backend. It is called on the words that could change that: the engine's
// start, an install's end, each load.
func (s *Supervisor) checkPin() {
	s.mu.Lock()
	id, tag, next := s.st.Backend, s.st.Build, s.st.NextBuild
	local := s.st.Mode == "local"
	busy := s.moveBusyLocked()
	s.mu.Unlock()
	if id == "" || !local || busy {
		return
	}
	cat, err := s.catalog()
	if err != nil {
		return
	}
	pin, ok := cat.Asset(id)
	if !ok || pin.URL == "" {
		return // the catalog has no build of this backend any more: nothing to move to
	}
	if pin.Tag == tag || (pin.Tag == next && buildUnpacked(s.dir, id, next, pin.SHA256)) {
		return // the pin runs, or is staged: the next load smoke-tests and runs it
	}
	ctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	if s.moveBusyLocked() || s.st.Backend != id { // looked at again under the lock those are set under
		s.mu.Unlock()
		cancel()
		return
	}
	s.moveCancel = cancel
	gen := s.removeGen
	s.move = &BuildNext{Backend: id, Tag: pin.Tag, State: "downloading"}
	s.refreshLocked()
	s.mu.Unlock()
	s.logf("build: %s@%s runs; moving to the catalog's %s@%s", id, orUnknown(tag), id, pin.Tag)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.runMove(ctx, cat, pin, gen)
		cancel()
		s.mu.Lock()
		s.moveCancel = nil
		if s.move != nil && s.move.State != "staged" && s.move.State != "failed" {
			s.move = nil // cancelled: the next start looks again
		}
		s.refreshLocked()
		s.mu.Unlock()
	}()
}

// cancelMove ends the move's job (a cancel, a remove of the downloads); the next start looks
// again. The partial download stays for resume.
func (s *Supervisor) cancelMove() {
	s.mu.Lock()
	c := s.moveCancel
	s.mu.Unlock()
	if c != nil {
		c()
	}
}

func orUnknown(tag string) string {
	if tag == "" {
		return "(unrecorded)"
	}
	return tag
}

// runMove fetches pin beside the build that runs and stages it. It checks the disk first. A
// failure waits for a word (moveWord) and tries again; a request at the door is not one.
func (s *Supervisor) runMove(ctx context.Context, cat *catalog.Catalog, pin catalog.Asset, gen uint64) {
	for attempt := 1; ; attempt++ {
		seen := s.words.Load() // before the attempt: a word said while it runs is a word for the next
		s.setMove(func(n *BuildNext) { n.Attempt, n.State, n.Error, n.Progress = attempt, "downloading", "", nil })
		err := s.checkNeed(plan.NeedOf(s.buildDiskSteps(cat, pin), s.logRoom(), s.hostFiles()), "llama.cpp "+pin.Tag)
		if err == nil {
			s.buildMu.Lock()
			err = s.fetchBuildLocked(ctx, cat, pin, s.moveSteps())
			s.buildMu.Unlock()
		}
		if err == nil {
			s.mu.Lock()
			still := s.st.Backend == pin.ID && s.removeGen == gen && !s.removing
			if still {
				s.st.NextBuild = pin.Tag
			}
			s.mu.Unlock()
			if !still {
				s.logf("build: %s@%s fetched, but the selection changed or the downloads were removed: not staged", pin.ID, pin.Tag)
				return
			}
			s.save() // next_build stands on disk before the fact says staged
			s.setMove(func(n *BuildNext) { n.State, n.Error, n.Progress = "staged", "", nil })
			s.logf("build: %s@%s staged; the next load smoke-tests and runs it", pin.ID, pin.Tag)
			return
		}
		if ctx.Err() != nil {
			return
		}
		s.addError("fetching llama.cpp %s (%s) failed: %v (still running the build it has; waiting for the network or a word from a client)", pin.Tag, pin.ID, err)
		s.setMove(func(n *BuildNext) { n.State, n.Error, n.Progress = "waiting", err.Error(), nil })
		if s.moveWord(ctx, seen) != nil {
			return
		}
	}
}

// moveWord waits until a word was said after seen: the network came back, or a person or a
// client said one through the API (Supervisor.Nudge, an install, a settings change). A request
// at the door is not one: a failed fetch is not tried again per request.
func (s *Supervisor) moveWord(ctx context.Context, seen uint64) error {
	for {
		w, err := s.hub.Wake("nudge", "network") // before looking, so a word said in between is heard
		if err != nil {
			return err
		}
		if s.words.Load() != seen {
			w.Close()
			return nil
		}
		select {
		case <-ctx.Done():
			w.Close()
			return ctx.Err()
		case <-s.hub.Done():
			w.Close()
			return event.ErrClosed
		case <-w.C():
		}
		w.Close()
	}
}

// buildSteps is how an install of a build shows itself: the install job sets the phase and the
// install progress; the move only its own fact, so what serves is not painted over.
type buildSteps struct {
	download func(ctx context.Context, url, dest, sha string, size int64, step string) error
	phase    func(msg string)
}

func (s *Supervisor) jobSteps() buildSteps {
	return buildSteps{
		download: func(ctx context.Context, url, dest, sha string, size int64, step string) error {
			return s.download(ctx, url, dest, sha, size, s.jobProgress(step))
		},
		phase: func(msg string) { s.setPhase("installing", msg) },
	}
}

func (s *Supervisor) moveSteps() buildSteps {
	return buildSteps{
		download: func(ctx context.Context, url, dest, sha string, size int64, step string) error {
			// one attempt: a download that moves no bytes ends, and the move waits for a word
			return fetch.Download(ctx, url, dest, sha, size, func(p fetch.Progress) {
				s.setMove(func(n *BuildNext) {
					n.Progress = &Progress{DoneBytes: p.Done, TotalBytes: p.Total, RateBps: p.RateBps, ETAS: int(p.ETA.Seconds()), Step: stepLabel(step, p)}
				})
			}, nil)
		},
		phase: func(string) { s.setMove(func(n *BuildNext) { n.State, n.Progress = "installing", nil }) },
	}
}

// promoteNext runs at a load with no model server up (loadContext), opMu held: a staged build
// is smoke-tested here, where the card is free, and selected when it passes. When it fails the
// engine loads the build it has, the move says failed and why, and the same tag is not tried
// again until the pin, the driver or a word changed (smokeBlockedLocked).
func (s *Supervisor) promoteNext() {
	s.mu.Lock()
	id, next := s.st.Backend, s.st.NextBuild
	blocked := s.smokeBlockedLocked(id, next)
	s.mu.Unlock()
	if next == "" {
		return
	}
	if !backendInstalled(s.dir, id, next) {
		if blocked {
			return
		}
		a, ok := s.stagedAsset(id, next)
		if !ok {
			return // nothing unpacked under that name: the move fetches it again
		}
		s.buildMu.Lock()
		err := s.verifyBuild(s.root, a)
		s.buildMu.Unlock()
		if err != nil {
			s.mu.Lock()
			s.smokeFail = &smokeFailure{id: id, tag: next, driver: s.driverLocked(), words: s.words.Load()}
			if s.move == nil || s.move.Tag != next {
				s.move = &BuildNext{Backend: id, Tag: next}
			}
			s.move.State, s.move.Error, s.move.Progress = "failed", err.Error(), nil
			s.refreshLocked()
			s.mu.Unlock()
			s.addError("llama.cpp %s (%s) failed its smoke test: %v; starting on the build the engine has", next, id, err)
			return
		}
	}
	s.mu.Lock()
	old := s.st.Build
	s.st.Build, s.st.NextBuild = next, ""
	s.move, s.smokeFail = nil, nil
	s.mu.Unlock()
	s.logf("build: starting on %s@%s (was %s)", id, next, orUnknown(old))
	s.save()
}

// stagedAsset is the asset of the staged build id@tag with the sha256 it was unpacked from
// (its .unpacked marker), the catalog's accelerator when it still names it.
func (s *Supervisor) stagedAsset(id, tag string) (catalog.Asset, bool) {
	b, err := os.ReadFile(filepath.Join(backendDir(s.dir, id, tag), ".unpacked"))
	sha := strings.TrimSpace(string(b))
	if err != nil || sha == "" || !buildUnpacked(s.dir, id, tag, sha) {
		return catalog.Asset{}, false
	}
	a := catalog.Asset{ID: id, Tag: tag, Patched: catalog.PatchedTag(tag), Accel: plan.AccelOf(id)}
	if cat, err := s.catalog(); err == nil {
		if c, ok := cat.LlamaAsset(tag, id); ok {
			a = c
		}
	}
	a.SHA256 = sha
	return a, true
}

// stagedBuild is the tag staged for the next load that would run there, "" with none (none
// either while its smoke test stands failed and nothing changed).
func (s *Supervisor) stagedBuild() string {
	s.mu.Lock()
	id, next := s.st.Backend, s.st.NextBuild
	blocked := s.smokeBlockedLocked(id, next)
	s.mu.Unlock()
	if next == "" || blocked {
		return ""
	}
	if backendInstalled(s.dir, id, next) {
		return next
	}
	if _, ok := s.stagedAsset(id, next); ok {
		return next
	}
	return ""
}

// pruneBuilds runs once a model server of build in has served ready, in its own goroutine (never
// under opMu), serialized with every install by buildMu: the other builds of the same backend id
// go (pruneBuildDirs). It does nothing while an install job, the move or a remove runs.
func (s *Supervisor) pruneBuilds(in install) {
	s.buildMu.Lock()
	defer s.buildMu.Unlock()
	s.mu.Lock()
	busy := s.jobRunning || s.moveCancel != nil || s.removing
	keep := map[string]bool{buildDirName(in.backend, in.tag): true}
	if s.st.Backend == in.backend {
		keep[buildDirName(s.st.Backend, s.st.Build)] = true
		if s.st.NextBuild != "" {
			keep[buildDirName(s.st.Backend, s.st.NextBuild)] = true
		}
	}
	cat := s.catLocked()
	s.mu.Unlock()
	if busy || in.backend == "" {
		return
	}
	var extras []string
	if cat != nil {
		for _, r := range cat.Rungs() {
			for _, a := range r.Assets {
				if a.ID == in.backend {
					extras = append(extras, a.Extra...)
				}
			}
		}
	}
	removed := pruneBuildDirs(s.dir, in.backend, keep, extras)
	if len(removed) > 0 {
		s.logf("build: removed %v now that %s served ready", removed, buildDirName(in.backend, in.tag))
		s.say("build.pruned", "", map[string]any{"backend": in.backend, "removed": removed})
	}
}

// pruneBuildDirs removes, by exact names only, what other builds of backend id left under the
// data dir: backends/<id> and backends/<id>@<tag> that keep does not name (marked or not: a
// build that failed is dead weight too), and the archives and partials main's old naming left in
// downloads/ (<id> and its extras' ids, .tar.gz or .zip, with or without .part). A symlinked
// backends/ or downloads/, a symlinked entry, a file where a build dir would be, another id's
// dir and anything else are left. It answers the paths removed, relative to the data dir.
func pruneBuildDirs(dataDir, id string, keep map[string]bool, extras []string) []string {
	var removed []string
	root := filepath.Join(dataDir, "backends")
	if realDir(root) {
		ents, _ := os.ReadDir(root)
		for _, e := range ents {
			n := e.Name()
			if keep[n] {
				continue
			}
			tag, isTagged := strings.CutPrefix(n, id+"@")
			if n != id && !(isTagged && plainTag.MatchString(tag)) {
				continue
			}
			if !realDir(filepath.Join(root, n)) {
				continue
			}
			if os.RemoveAll(filepath.Join(root, n)) == nil { // only a path under our own data dir
				removed = append(removed, "backends/"+n)
			}
		}
	}
	dl := filepath.Join(dataDir, "downloads")
	if realDir(dl) {
		for _, x := range append([]string{id}, extras...) {
			for _, name := range []string{x + ".tar.gz", x + ".tar.gz.part", x + ".zip", x + ".zip.part"} {
				p := filepath.Join(dl, name)
				if fi, err := os.Lstat(p); err == nil && fi.Mode().IsRegular() && os.Remove(p) == nil {
					removed = append(removed, "downloads/"+name)
				}
			}
		}
	}
	return removed
}

// realDir reports whether path is a directory itself, not a symlink or junction to one (Lstat):
// emptyDir (storage.go) and pruneBuildDirs enter nothing else.
func realDir(path string) bool {
	fi, err := os.Lstat(path)
	return err == nil && fi.IsDir()
}

var plainTag = regexp.MustCompile(`^[A-Za-z0-9_-][A-Za-z0-9._-]*$`)

// vmmRe is the patched server's word on its CUDA buffers: "easy: CUDA VMM buffers on (1 device)",
// "... off (GGML_CUDA_VMM_BUFFERS=0)", "... off (2 devices)".
var vmmRe = regexp.MustCompile(`easy: CUDA VMM buffers (on|off)\b\s*(.*)$`)

// vmmLine reads that line: "on (1 device)", or "" when l is not it.
func vmmLine(l string) string {
	m := vmmRe.FindStringSubmatch(strings.TrimRight(l, "\r"))
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1] + " " + strings.TrimSpace(m[2]))
}

func (s *Supervisor) setVMM(v string) {
	s.mu.Lock()
	if s.vmm != v {
		s.vmm = v
		s.refreshLocked()
	}
	s.mu.Unlock()
}

// errNoBuild is ensureBackend's answer for a machine the catalog has no build for.
var errNoBuild = errors.New("no llama.cpp build in the catalog for this machine")

// selectBuildLocked makes the build an install put in place the selected one. A build staged
// for the next start of another backend, or of this same build, is moot. The caller holds s.mu.
func (s *Supervisor) selectBuildLocked(a catalog.Asset) {
	if a.ID != s.st.Backend || a.Tag == s.st.NextBuild {
		s.st.NextBuild = ""
		if s.move != nil && (s.move.State == "staged" || s.move.State == "failed") {
			s.move = nil
		}
	}
	s.st.Backend, s.st.Build = a.ID, a.Tag
}

// buildKey is the build a measurement is made with, for the keys of what the engine remembers
// per build (the context key, the cap's build, the book): the backend id, the release's full tag
// and the archive's sha256 (its .installed marker). A build installed before tags were recorded
// keeps its old form (id@sha), so what was measured with it still matches it.
func buildKey(id, tag, sha string) string {
	k := id
	if tag != "" {
		k += "@" + tag
	}
	if sha != "" {
		k += "@" + sha
	}
	return k
}

// bookBackend is the backend part of the book's key (fit.BookKey): the id and the release's tag,
// so a fixed part measured with one build (stock b12651) is not taken for another
// (b12651-easy1, whose CUDA buffers are laid out otherwise).
func bookBackend(id, tag string) string {
	if tag == "" {
		return id
	}
	return fmt.Sprintf("%s@%s", id, tag)
}
