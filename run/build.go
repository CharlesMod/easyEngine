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
	"github.com/CharlesMod/easyEngine/fetch"
)

// Which llama.cpp build runs, and how the engine comes to run the catalog's pin
// (docs/ENGINE.md, "Which llama.cpp build").
//
// A build is installed under backends/<id>@<tag> (cuda-linux-x64@b12651-easy1); a build
// installed before the engine recorded tags sits under backends/<id> and has tag "". state.json
// names the build that runs (Backend, Build) and the one installed beside it for the next start
// (NextBuild). When the selected build is not the one the catalog gives this machine for that
// backend (the pin, or stock where the pin has no build), the move installs the pin beside it in
// the background; the next start of the model server runs it (promoteNext), and once it has
// served ready every other complete build is removed (pruneBuilds). A failed download or smoke
// test leaves the engine on what it has, says why in the build fact, and is tried again on the
// next word (a nudge, the network, a start), never on a clock. The engine never falls back from
// the patched build to stock because the patched one failed: a failed start is reported.

// backendDir is where a build is installed.
func backendDir(dataDir, id, tag string) string {
	if tag == "" {
		return filepath.Join(dataDir, "backends", id)
	}
	return filepath.Join(dataDir, "backends", id+"@"+tag)
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

// BuildNext is the move to the pin: installing it beside the build that runs.
type BuildNext struct {
	Backend string `json:"backend"`
	Tag     string `json:"tag"`
	// State: downloading | installing | waiting (a download or a smoke test failed: Error says
	// why; the next word tries again) | ready (installed: the next start of the model server
	// runs it).
	State    string    `json:"state"`
	Error    string    `json:"error,omitempty"`
	Attempt  int       `json:"attempt,omitempty"`
	Progress *Progress `json:"progress,omitempty"`
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

// checkPin starts the move when the selected build is not the one the catalog gives this
// machine for its backend. It is called on the words that could change that: the engine's
// start, an install's end, a nudge while a move waits (the move's own loop listens for those).
func (s *Supervisor) checkPin() {
	s.mu.Lock()
	id, tag, next := s.st.Backend, s.st.Build, s.st.NextBuild
	local := s.st.Mode == "local"
	running := s.moveCancel != nil
	s.mu.Unlock()
	if id == "" || !local || running {
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
	if pin.Tag == tag || (pin.Tag == next && backendInstalled(s.dir, id, next)) {
		return
	}
	ctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	if s.moveCancel != nil {
		s.mu.Unlock()
		cancel()
		return
	}
	s.moveCancel = cancel
	s.move = &BuildNext{Backend: id, Tag: pin.Tag, State: "downloading"}
	s.refreshLocked()
	s.mu.Unlock()
	s.logf("build: %s@%s runs; moving to the catalog's %s@%s", id, orUnknown(tag), id, pin.Tag)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.runMove(ctx, cat, pin)
		cancel()
		s.mu.Lock()
		s.moveCancel = nil
		if s.move != nil && s.move.State != "ready" {
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

// runMove installs pin beside the build that runs. A failure waits for a word (a nudge, the
// network) and tries again; there is no cadence.
func (s *Supervisor) runMove(ctx context.Context, cat *catalog.Catalog, pin catalog.Asset) {
	for attempt := 1; ; attempt++ {
		s.setMove(func(n *BuildNext) { n.Attempt, n.State, n.Progress = attempt, "downloading", nil })
		w, err := s.listenQuiet() // listened before the attempt: a word said while it runs is heard
		if err != nil {
			return
		}
		err = s.installBuild(ctx, cat, pin, s.moveSteps())
		if err == nil {
			w.Close()
			s.mu.Lock()
			s.st.NextBuild = pin.Tag
			s.mu.Unlock()
			s.setMove(func(n *BuildNext) { n.State, n.Error, n.Progress = "ready", "", nil })
			s.save()
			s.logf("build: %s@%s installed; the next start of the model server runs it", pin.ID, pin.Tag)
			return
		}
		if ctx.Err() != nil {
			w.Close()
			return
		}
		s.addError("installing llama.cpp %s (%s) failed: %v (still running the build it has; waiting for the network or a word from a client)", pin.Tag, pin.ID, err)
		s.setMove(func(n *BuildNext) { n.State, n.Error, n.Progress = "waiting", err.Error(), nil })
		w.count()
		if s.awaitWord(ctx, w) != nil {
			return
		}
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

// promoteNext makes the build installed for the next start the selected one; it is called by a
// start of the model server, before the install is resolved. opMu is held.
func (s *Supervisor) promoteNext() {
	s.mu.Lock()
	id, next := s.st.Backend, s.st.NextBuild
	if next == "" || !backendInstalled(s.dir, id, next) {
		s.mu.Unlock()
		return
	}
	old := s.st.Build
	s.st.Build, s.st.NextBuild = next, ""
	s.move = nil
	s.mu.Unlock()
	s.logf("build: starting on %s@%s (was %s)", id, next, orUnknown(old))
	s.save()
}

// stagedBuild is the tag installed for the next start, "" with none.
func (s *Supervisor) stagedBuild() string {
	s.mu.Lock()
	id, next := s.st.Backend, s.st.NextBuild
	s.mu.Unlock()
	if next != "" && backendInstalled(s.dir, id, next) {
		return next
	}
	return ""
}

// pruneBuilds removes every complete build but the one that served ready and one installed for
// the next start, once a model server of in's build is ready, and only while no install job or
// move runs (one of them may be writing a build). A build whose install never completed (no
// marker) is left: an install resumes it or a remove of the downloads takes it.
func (s *Supervisor) pruneBuilds(in install) {
	s.mu.Lock()
	busy := s.jobRunning || s.moveCancel != nil || s.removing
	keep := map[string]bool{backendDir(s.dir, in.backend, in.tag): true}
	if s.st.NextBuild != "" {
		keep[backendDir(s.dir, s.st.Backend, s.st.NextBuild)] = true
	}
	s.mu.Unlock()
	if busy {
		return
	}
	root := filepath.Join(s.dir, "backends")
	ents, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, e := range ents {
		d := filepath.Join(root, e.Name())
		if !e.IsDir() || keep[d] || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if _, err := os.Stat(filepath.Join(d, ".installed")); err != nil {
			continue
		}
		if err := os.RemoveAll(d); err != nil { // only a path under our own data dir
			s.logf("build: %s kept: %v", e.Name(), err)
			continue
		}
		s.logf("build: removed %s now that %s@%s served ready", e.Name(), in.backend, orUnknown(in.tag))
	}
}

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
		if s.move != nil && s.move.State == "ready" {
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
