package run

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/fetch"
	"github.com/CharlesMod/easyEngine/plan"
)

// Vision (docs/ENGINE.md, "Vision"): the served model's projector (catalog.Projector) is loaded
// beside it (--mmproj) when vision is wanted, the file is present and verified, the build knows
// the flag, and the model fits with it. Vision is wanted by the engine setting (settings.json
// "vision"), by serve --vision for one run, or by any attachment whose profile says vision: true.
// Text never waits on vision: a projector that is missing, downloading, failed or too big for
// the card leaves the model serving text, and vision_note says why.

// visionState is under Supervisor.mu.
type visionState struct {
	setting bool // settings.json "vision"
	wanted  bool // as last worked out (visionWantChanged, planVision): read by Status
	// the start being made, or the one that runs: tryLoad says --mmproj is (to be) given
	tryLoad bool
	proj    *catalog.Projector
	path    string
	loaded  bool   // the running server was started with the projector (with s.shape != nil)
	note    string // why vision is wanted and not loaded; "" otherwise
	dl      *visionDL
	failed  map[string]string // projector sha256 -> why its download failed; asked again on a new want
	roomErr string            // the last "no room" detail put in status errors (said once per kind)
	// bad: projector files (by identity) that failed to load or did not match their sha256; each is
	// left alone until the file changes, the pinned build changes, or a person asks again (the
	// setting turned on, serve restarted: this is memory only)
	bad map[projIdent]string
	// pendingBad: a start with the projector failed and the text-only retry is under way; it is
	// marked bad only if that retry comes up (so a model that fails for its own sake blames nobody)
	pendingBad string
}

// ProjRecord is what the engine verified of one projector file: its size and mtime when its sha256
// was checked (by the download, or by hashing the file itself). The file is trusted only while it
// still has that size and mtime. It is in state.json ("projectors", by file name).
type ProjRecord struct {
	Size    int64  `json:"size"`
	MTimeNS int64  `json:"mtime_ns"`
	SHA256  string `json:"sha256"`
}

// projIdent is the identity of a projector file as the not-loadable record knows it: the file,
// its size and mtime, the sha256 the catalog names for it, and the llama-server build it was tried
// with.
type projIdent struct {
	file         string
	size, mtime  int64
	sha, backend string
}

type visionDL struct {
	file     string
	progress *Progress
	cancel   context.CancelFunc
	done     chan struct{} // closed when the download goroutine has stopped writing and said so in vis
	verify   bool          // hashing a file already there, not downloading
}

// VisionInfo is the hub fact "vision".
type VisionInfo struct {
	Vision bool   `json:"vision"`
	Note   string `json:"note"`
}

// visionInfoLocked is what status and the fact say. Caller holds s.mu.
func (s *Supervisor) visionInfoLocked() VisionInfo {
	on := s.vis.loaded && s.shape != nil
	note := ""
	if s.vis.wanted && !on {
		note = s.vis.note
		if note == "" {
			note = "the model server is not running"
		}
	}
	return VisionInfo{Vision: on, Note: note}
}

// visionProgressLocked is the projector download's progress, nil when none runs.
func (s *Supervisor) visionProgressLocked() *Progress {
	if s.vis.dl == nil || s.vis.dl.progress == nil {
		return nil
	}
	p := *s.vis.dl.progress
	return &p
}

// visionByClients says whether any attachment's profile wants vision.
func (r *clientReg) visionLocked() bool {
	for _, a := range r.list {
		if a.prof.Vision {
			return true
		}
	}
	return false
}

// visionWanted works out whether vision is wanted now and records it. It takes the clients' lock
// and then s.mu, never both at once.
func (s *Supervisor) visionWanted() bool {
	r := &s.clients
	r.mu.Lock()
	byClients := r.visionLocked()
	r.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.vis.wanted = s.cfg.Vision || s.vis.setting || byClients
	return s.vis.wanted
}

// projectorPath is where a projector lives: beside the models.
func (s *Supervisor) projectorPath(p *catalog.Projector) string {
	return filepath.Join(s.dir, "models", p.File)
}

// projectorStat is the projector file's stat when it is a regular, non-empty file of the catalog's size.
func (s *Supervisor) projectorStat(p *catalog.Projector) (os.FileInfo, bool) {
	st, err := os.Lstat(s.projectorPath(p))
	if err != nil || !st.Mode().IsRegular() || st.Size() <= 0 || (p.Size != 0 && st.Size() != p.Size) {
		return nil, false
	}
	return st, true
}

// projectorPresent: the file is there and the engine verified its sha256 (it downloaded it, which
// checks the hash before the file has its name, or it hashed the file itself): the record in
// state.json has the catalog's sha256 and the file's size and mtime. A file with no matching
// record is never trusted, whatever its size.
func (s *Supervisor) projectorPresent(p *catalog.Projector) bool {
	st, ok := s.projectorStat(p)
	if !ok || p.SHA256 == "" {
		return false
	}
	s.mu.Lock()
	r, have := s.st.Projectors[p.File]
	s.mu.Unlock()
	return have && r.SHA256 == p.SHA256 && r.Size == st.Size() && r.MTimeNS == st.ModTime().UnixNano()
}

// recordProjector writes down that the file p names, as it is on disk now, has the sha256 p names
// (state first; the caller says the fact after).
func (s *Supervisor) recordProjector(p *catalog.Projector, st os.FileInfo) {
	s.mu.Lock()
	if s.st.Projectors == nil {
		s.st.Projectors = map[string]ProjRecord{}
	}
	s.st.Projectors[p.File] = ProjRecord{Size: st.Size(), MTimeNS: st.ModTime().UnixNano(), SHA256: p.SHA256}
	s.mu.Unlock()
	s.save()
}

// projectorOwned says the engine put (downloaded) or verified the file p names, for p's sha256.
func (s *Supervisor) projectorOwned(p *catalog.Projector) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.st.Projectors[p.File]
	return ok && p.SHA256 != "" && r.SHA256 == p.SHA256
}

// forgetProjector drops the record of a file the engine deleted.
func (s *Supervisor) forgetProjector(file string) {
	s.mu.Lock()
	_, had := s.st.Projectors[file]
	delete(s.st.Projectors, file)
	s.mu.Unlock()
	if had {
		s.save()
	}
}

// projIdentOf is p's file as it is on disk with the build bin; ok false when it is not a file.
func (s *Supervisor) projIdentOf(p *catalog.Projector, bin string) (projIdent, bool) {
	st, err := os.Lstat(s.projectorPath(p))
	if err != nil || !st.Mode().IsRegular() {
		return projIdent{}, false
	}
	return projIdent{file: p.File, size: st.Size(), mtime: st.ModTime().UnixNano(), sha: p.SHA256, backend: bin}, true
}

// projectorBad says whether p's file, as it is now and with this build, is known not to load (or
// not to match its sha256), and the note that says so.
func (s *Supervisor) projectorBad(p *catalog.Projector, bin string) (string, bool) {
	id, ok := s.projIdentOf(p, bin)
	if !ok {
		return "", false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	note, bad := s.vis.bad[id]
	return note, bad
}

// markProjectorBad records that p's file, as it is now, is not loadable with bin, and why (note).
func (s *Supervisor) markProjectorBad(p *catalog.Projector, bin, note string) {
	id, ok := s.projIdentOf(p, bin)
	if !ok {
		return
	}
	s.mu.Lock()
	if s.vis.bad == nil {
		s.vis.bad = map[projIdent]string{}
	}
	s.vis.bad[id] = note
	s.mu.Unlock()
}

// VisionAskedAgain is a person asking for vision again (the setting turned on): a projector that
// failed to load or did not match is tried once more. A client's attach is not that.
func (s *Supervisor) visionAskedAgainLocked() {
	clear(s.vis.bad)
}

// projectorFailedToLoad marks the projector of the served model as not loadable (by its file's
// identity and the build), after a start with it failed and the same start without it came up.
// It is not tried again until the file changes, the build changes, or a person asks again.
func (s *Supervisor) projectorFailedToLoad(in install, why string) {
	p := in.model.Projector
	if p == nil {
		return
	}
	note := "the vision projector failed to load: " + why
	s.markProjectorBad(p, in.bin, note)
	s.mu.Lock()
	s.vis.note = note
	s.refreshLocked()
	s.mu.Unlock()
	s.logf("vision: %s; not tried again until the file or the llama.cpp build changes or vision is asked for again", note)
}

// lastWords is the last line of a server's complaint with every path in it left out.
func lastWords(msg string) string {
	if i := strings.LastIndex(msg, "exited while loading: "); i >= 0 {
		msg = msg[i+len("exited while loading: "):]
	}
	f := strings.Fields(msg)
	for i, w := range f {
		if strings.ContainsAny(w, `/\`) {
			f[i] = "<path>"
		}
	}
	out := strings.Join(f, " ")
	if len(out) > 200 {
		out = out[:200]
	}
	return out
}

// servingModel is the catalog model the next start loads (and the projector it carries).
func (s *Supervisor) servingModel() (install, bool) { return s.resolveInstall() }

// visionSnap is what an attach or leave needs to judge a change of the clients' vision want
// without holding s.mu under the clients' lock.
type visionSnap struct {
	base    bool // wanted by the setting or the flag
	ready   bool // the served model has a projector on disk the build can load
	loaded  bool // the running server has it
	running bool
}

func (s *Supervisor) visionSnapNow() visionSnap {
	in, ok := s.servingModel()
	ready := false
	if ok && in.model.Projector != nil && s.projectorPresent(in.model.Projector) {
		if _, bad := s.projectorBad(in.model.Projector, in.bin); !bad {
			h := helpOf(in.bin, s.cfg.ExtraEnv)
			ready = h == nil || h.Has("--mmproj")
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return visionSnap{base: s.cfg.Vision || s.vis.setting, ready: ready, loaded: s.vis.loaded && s.shape != nil, running: s.shape != nil}
}

// restartFor says whether a change of the want to wanted asks for a restart of the running server.
func (v visionSnap) restartFor(wanted bool) bool {
	if !v.running {
		return false
	}
	return wanted && v.ready && !v.loaded || !wanted && v.loaded
}

// visionWantChanged follows a change of the want (the setting, an attach, a leave): it starts the
// projector's download when it is wanted and missing, lets go of a download no longer wanted,
// and with restart asks the running server to reload (when idle) if it should load or drop the
// projector. An attach's or leave's restart goes through the clients' plan instead (charged to
// the client that asked), so they pass restart false.
func (s *Supervisor) visionWantChanged(why string, restart bool) {
	wanted := s.visionWanted()
	in, ok := s.servingModel()
	s.mu.Lock()
	if !wanted {
		if s.vis.dl != nil {
			s.vis.dl.cancel()
		}
	}
	snapLoaded := s.vis.loaded && s.shape != nil
	running := s.shape != nil
	s.mu.Unlock()
	if wanted && ok && in.model.Projector != nil && !s.projectorPresent(in.model.Projector) {
		s.mu.Lock()
		delete(s.vis.failed, in.model.Projector.SHA256) // a new want asks again
		s.mu.Unlock()
	}
	if ok {
		// the running server's note: why the projector is not (yet) loaded, said at once and not
		// when the next start plans it
		try, note := s.visionPlan(in, wanted)
		if try && !snapLoaded {
			note = "the vision projector is loaded at the model server's next restart (when it is idle)"
		}
		s.mu.Lock()
		if !snapLoaded && running {
			s.vis.note = note
		}
		s.mu.Unlock()
	}
	if !restart || !running {
		s.refresh()
		return
	}
	need := !wanted && snapLoaded
	if wanted && !snapLoaded && ok && in.model.Projector != nil && s.projectorPresent(in.model.Projector) {
		if _, bad := s.projectorBad(in.model.Projector, in.bin); !bad {
			h := helpOf(in.bin, s.cfg.ExtraEnv)
			need = h == nil || h.Has("--mmproj")
		}
	}
	if need {
		s.mu.Lock()
		s.reshape = true
		s.reshapeWhy = why
		s.mu.Unlock()
		s.Nudge("vision") // after reshape is set: the reactor sees it when it wakes
	}
	s.refresh()
}

// visionPlan says whether a start loads the projector (try) and, when vision is wanted and it
// does not, why (note). A missing projector starts its download.
func (s *Supervisor) visionPlan(in install, wanted bool) (try bool, note string) {
	p := in.model.Projector
	switch {
	case !wanted:
	case p == nil:
		note = fmt.Sprintf("the model %s has no vision projector", in.model.ID)
	default:
		h := helpOf(in.bin, s.cfg.ExtraEnv)
		switch {
		case h != nil && !h.Has("--mmproj"):
			note = "this llama-server build does not know --mmproj"
		case !s.projectorPresent(p):
			note = s.prepareProjector(in)
		default:
			if bad, isBad := s.projectorBad(p, in.bin); isBad {
				note = bad
			} else {
				try = true
			}
		}
	}
	return try, note
}

// planVision decides, before a start, whether it loads the projector, and says why not.
func (s *Supervisor) planVision(in install) {
	wanted := s.visionWanted()
	p := in.model.Projector
	try, note := s.visionPlan(in, wanted)
	s.mu.Lock()
	s.vis.tryLoad, s.vis.note, s.vis.loaded = try, note, false
	s.vis.proj, s.vis.path = nil, ""
	if try {
		s.vis.proj, s.vis.path = p, s.projectorPath(p)
	}
	s.refreshLocked()
	s.mu.Unlock()
}

// visionTrying says the start being made gives the projector.
func (s *Supervisor) visionTrying() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.vis.tryLoad
}

// dropVision makes the start being made a text-only one, and says why.
func (s *Supervisor) dropVision(why string) {
	s.mu.Lock()
	s.vis.tryLoad, s.vis.proj, s.vis.path, s.vis.note = false, nil, "", why
	s.refreshLocked()
	s.mu.Unlock()
	s.logf("vision: %s", why)
}

// visionRoomMiB is what the fit gives up on the card for the projector of the start being made.
func (s *Supervisor) visionRoomMiB() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.vis.tryLoad {
		return 0
	}
	return s.vis.proj.RoomMiB()
}

// withProjectorRoom is r with mib more margin on the card: the fit's budget (core/fit: card total
// less others, the game's reserve and the margin) is that much smaller, so the model's weights,
// the experts kept on the card and the cache are sized around the projector. Margin 0 means the
// default margin, which is made explicit first.
func withProjectorRoom(r fit.Room, mib int) fit.Room {
	if mib <= 0 || r.NoGPU {
		return r
	}
	if r.Margin == 0 {
		r.Margin = fit.DefaultMargin(r.VRAMTotal)
	}
	if r.Margin < 0 {
		r.Margin = 0
	}
	r.Margin += mib
	return r
}

// withoutProjectorRoom undoes withProjectorRoom for a start that drops vision.
func withoutProjectorRoom(r fit.Room, mib int) fit.Room {
	if mib <= 0 || r.NoGPU {
		return r
	}
	r.Margin -= mib
	return r
}

// visionArgs are the launch's projector arguments: --mmproj <path> when the start gives it.
func (s *Supervisor) visionArgs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.vis.tryLoad || s.vis.path == "" {
		return nil
	}
	return []string{"--mmproj", s.vis.path}
}

// visionUp records that a server came up, with or without the projector; it returns the
// projector's sha256 for the prompt-cache key and the MiB its load adds to the server's reading.
func (s *Supervisor) visionUpLocked() (sha string, loadMiB int) {
	s.vis.loaded = s.vis.tryLoad
	if !s.vis.loaded || s.vis.proj == nil {
		return "", 0
	}
	s.vis.note = ""
	return s.vis.proj.SHA256, s.vis.proj.LoadMiB
}

// fetchProjector starts the download of m's projector if none runs and none failed for it, and
// returns the note status says meanwhile.
func (s *Supervisor) fetchProjector(m catalog.Model) string {
	p := m.Projector
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.vis.failed == nil {
		s.vis.failed = map[string]string{}
	}
	if _, ok := s.vis.failed[p.SHA256]; ok {
		return projectorFailedNote
	}
	if p.URL == "" {
		return fmt.Sprintf("the vision projector %s is not in the models folder (this catalog entry has no download)", p.File)
	}
	if s.vis.dl != nil {
		return "the vision projector is downloading"
	}
	s.mu.Unlock()
	short, err := s.projectorRoom(p)
	s.mu.Lock()
	if err != nil {
		// the detail (sizes, no path) is in status errors; the note is plain
		detail := err.Error()
		say := s.vis.roomErr != detail
		s.vis.roomErr = detail
		if say {
			s.mu.Unlock()
			s.addError("vision projector %s: %v", p.File, err)
			s.mu.Lock()
		}
		return fmt.Sprintf("no room on disk for the vision projector (needs %d MiB more)", short)
	}
	s.vis.roomErr = ""
	if s.vis.dl != nil { // another want started it while the disk was read
		return "the vision projector is downloading"
	}
	if _, ok := s.vis.failed[p.SHA256]; ok {
		return projectorFailedNote
	}
	ctx, cancel := context.WithCancel(s.root)
	dl := &visionDL{file: p.File, cancel: cancel, done: make(chan struct{})}
	s.vis.dl = dl
	path := s.projectorPath(p)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer cancel()
		err := s.downloadQuiet(ctx, p.URL, path, p.SHA256, p.Size, func(fp fetch.Progress) {
			s.mu.Lock()
			dl.progress = &Progress{Step: stepLabel("vision projector "+p.File, fp), DoneBytes: fp.Done, TotalBytes: fp.Total, RateBps: fp.RateBps, ETAS: int(fp.ETA.Seconds())}
			sayPtr(s, "vision.progress", "", dl.progress) // said per step of bytes, as install.progress is
			s.mu.Unlock()
		})
		letGo := err != nil && ctx.Err() != nil // let go: no longer wanted, or the engine stops
		if err == nil {
			// fetch.Download renamed the file into place only after its sha256 matched: that is
			// the verification the engine records (state first, the fact after)
			if st, e := os.Lstat(path); e == nil && st.Mode().IsRegular() {
				s.recordProjector(p, st)
			} else {
				err = errors.New("the downloaded projector is not on disk")
			}
		}
		s.mu.Lock()
		s.vis.dl = nil
		s.stoCache = nil
		s.sayGone("vision.progress", "")
		switch {
		case err == nil, letGo:
		default:
			s.vis.failed[p.SHA256] = err.Error()
			s.vis.note = projectorFailedNote
		}
		s.refreshLocked()
		s.mu.Unlock()
		close(dl.done) // after the state is written: RemoveDownloads waits on it before it sweeps the .part
		if letGo {
			return
		}
		if err != nil {
			s.addError("vision projector %s: %v", p.File, err)
			return
		}
		s.logf("vision projector %s downloaded and verified", p.File)
		s.projectorReady("the vision projector is ready")
	}()
	return "the vision projector is downloading"
}

// projectorMismatchNote is the note of a projector file in the models folder that is not the one
// its catalog entry names.
const projectorMismatchNote = "the vision projector file in the models folder did not match its sha256: it is not loaded"

// prepareProjector is the plan for a wanted projector that is not (yet) verified on disk: no file
// starts its download, an older pin's own file is downloaded over, and any other file in its place
// is hashed once in the background (never trusted on its size, never overwritten).
func (s *Supervisor) prepareProjector(in install) string {
	p := in.model.Projector
	st, err := os.Lstat(s.projectorPath(p))
	if err != nil || !st.Mode().IsRegular() {
		return s.fetchProjector(in.model)
	}
	if note, bad := s.projectorBad(p, in.bin); bad {
		return note
	}
	s.mu.Lock()
	r, have := s.st.Projectors[p.File]
	s.mu.Unlock()
	if have && r.SHA256 != p.SHA256 && r.Size == st.Size() && r.MTimeNS == st.ModTime().UnixNano() {
		return s.fetchProjector(in.model) // the engine's own file of an earlier pin: replaced by the download
	}
	if p.SHA256 == "" || p.Size != 0 && st.Size() != p.Size {
		s.markProjectorBad(p, in.bin, projectorMismatchNote)
		return projectorMismatchNote
	}
	return s.verifyProjector(in)
}

// verifyProjector hashes the file at the projector's name in the background (progress in
// vision_progress), and records it when its sha256 is the catalog's.
func (s *Supervisor) verifyProjector(in install) string {
	p, bin := in.model.Projector, in.bin
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.vis.dl != nil {
		if s.vis.dl.verify {
			return "the vision projector is being verified"
		}
		return "the vision projector is downloading"
	}
	ctx, cancel := context.WithCancel(s.root)
	dl := &visionDL{file: p.File, cancel: cancel, done: make(chan struct{}), verify: true,
		progress: &Progress{Step: "verifying vision projector"}}
	s.vis.dl = dl
	path := s.projectorPath(p)
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer cancel()
		st0, e0 := os.Lstat(path)
		sum, err := hashFile(ctx, path, func(done, total int64) {
			s.mu.Lock()
			dl.progress = &Progress{Step: "verifying vision projector", DoneBytes: done, TotalBytes: total}
			sayPtr(s, "vision.progress", "", dl.progress)
			s.mu.Unlock()
		})
		letGo := err != nil && ctx.Err() != nil
		st1, e1 := os.Lstat(path)
		changed := e0 != nil || e1 != nil || st0.Size() != st1.Size() || !st0.ModTime().Equal(st1.ModTime())
		match := err == nil && !changed && sum == p.SHA256
		note := ""
		switch {
		case match:
			s.recordProjector(p, st1)
		case letGo || changed:
		case err != nil:
			note = "the vision projector file could not be read"
			s.markProjectorBad(p, bin, note)
		default:
			note = projectorMismatchNote
			s.markProjectorBad(p, bin, note)
		}
		s.mu.Lock()
		s.vis.dl = nil
		s.stoCache = nil
		s.sayGone("vision.progress", "")
		if note != "" {
			s.vis.note = note
		}
		s.refreshLocked()
		s.mu.Unlock()
		close(dl.done)
		switch {
		case letGo:
		case match:
			s.logf("vision projector %s verified (sha256 matches)", p.File)
			s.projectorReady("the vision projector is verified")
		case changed:
			s.visionWantChanged("the vision projector file changed while it was verified", false)
		default:
			s.addError("vision projector %s: %s", p.File, note)
		}
	}()
	return "the vision projector is being verified"
}

// hashFile is the sha256 of the file at path, read in chunks; done is called as it goes, and ctx
// ends it.
func hashFile(ctx context.Context, path string, progress func(done, total int64)) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	buf := make([]byte, 1<<20)
	var done, since int64
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := f.Read(buf)
		h.Write(buf[:n])
		done += int64(n)
		since += int64(n)
		if since >= 32<<20 {
			since = 0
			progress(done, st.Size())
		}
		if err == io.EOF {
			return hex.EncodeToString(h.Sum(nil)), nil
		}
		if err != nil {
			return "", err
		}
	}
}

// projectorReady follows a projector that is now on disk and verified (downloaded, or hashed).
// A local want (the setting, serve --vision, a local attachment) reloads the running server when
// idle, as a setting change does, and nobody is charged. With remote wants only, the restart goes
// through the clients' plan charged to one of the attachments that want vision and has restarts
// left, as an attach's own restart would; with none, nothing restarts: the projector is loaded at
// the next start made for another reason, and the note says so. No remote act alone restarts a
// running server uncharged.
func (s *Supervisor) projectorReady(why string) {
	wanted := s.visionWanted()
	vs := s.visionSnapNow()
	r := &s.clients
	r.mu.Lock()
	local := vs.base
	for _, a := range r.list {
		if a.prof.Vision && a.local {
			local = true
		}
	}
	r.mu.Unlock() // s.mu is never taken under the clients' lock
	if wanted && local && !vs.running {
		// a start is being made without the projector (it was still being verified when the start
		// was planned): the server reloads with it once that start is up
		s.mu.Lock()
		if (s.phase == "starting" || s.phase == "loading") && !s.vis.tryLoad {
			s.reshape = true
			s.reshapeWhy = why
		}
		s.mu.Unlock()
		s.Nudge("vision") // after reshape is set
	}
	if !wanted || !vs.restartFor(true) {
		s.visionWantChanged(why, false)
		return
	}
	if local {
		s.visionWantChanged(why, true)
		return
	}
	r.mu.Lock()
	planned := false
	for _, a := range r.list {
		if !a.prof.Vision || a.local || a.base {
			continue
		}
		who := budgetKey(a.identity, false)
		if who != "" && r.restarts[who] >= maxRemoteRestarts {
			continue
		}
		if r.planLocked(2, a.key, who, false, a) {
			planned = true
			break
		}
	}
	r.mu.Unlock()
	s.visionWantChanged(why, false)
	if planned {
		s.sayClients()
		s.wakeMind(func(c *causes) { c.clients = true })
		return
	}
	s.mu.Lock()
	s.vis.note = "the vision projector is ready and loads at the model server's next restart (no attached client has a restart left to spend on it)"
	s.refreshLocked()
	s.mu.Unlock()
}

// projectorRoom checks the free disk for the projector's download as a model download does
// (checkNeed: the file less the .part already there, logs, the host's files, headroom). It returns
// the MiB short and the error (wrapping ErrNoSpace) when there is no room. Caller does not hold s.mu.
func (s *Supervisor) projectorRoom(p *catalog.Projector) (shortMiB int, err error) {
	var part int64
	if st, e := os.Stat(s.projectorPath(p) + ".part"); e == nil {
		part = st.Size()
	}
	need := plan.NeedOf([]plan.DiskStep{plan.ModelStep("vision projector", p.Size, part)}, s.logRoom(), s.hostFiles())
	err = s.checkNeed(need, "the vision projector")
	if !errors.Is(err, ErrNoSpace) {
		return 0, nil
	}
	if free, e := s.cfg.DiskFree(s.dir); e == nil {
		shortMiB = int((need.Total - int64(free)<<20 + (1 << 20) - 1) >> 20)
	}
	return max(shortMiB, 1), err
}

// projectorFailedNote is the note of a projector download that failed: the error itself (which
// can name paths) is in status errors, never in the note a paired client sees. A new want of
// vision (the setting, an attach) asks again.
const projectorFailedNote = "the vision projector could not be downloaded or did not match its sha256 (status errors says why); it is tried again when vision is next asked for"

// downloadQuiet is fetch.Download for a background file: an attempt that moved no bytes waits for
// a word (the network, a client) like the install job's, without painting over the phase.
func (s *Supervisor) downloadQuiet(ctx context.Context, url, dest, sha string, size int64, onProgress func(fetch.Progress)) error {
	a := s.newWordAgain()
	defer a.close()
	again := a.again(func(err error) {
		s.mu.Lock()
		s.vis.note = "the vision projector's download made no progress: waiting for the network or a word from a client"
		s.refreshLocked()
		s.mu.Unlock()
	}, func() {
		s.mu.Lock()
		s.vis.note = "the vision projector is downloading"
		s.refreshLocked()
		s.mu.Unlock()
	})
	err := fetch.Download(ctx, url, dest, sha, size, onProgress, again)
	if errors.Is(err, fetch.ErrHashMismatch) {
		return fmt.Errorf("the file did not match its sha256: %w", err)
	}
	return err
}

// visionFiles are the projector files storage must not touch: the served model's while vision is
// wanted or loaded, and the one downloading (its .part).
func (s *Supervisor) visionFilesLocked(cat *catalog.Catalog, servedID string) []string {
	var out []string
	if s.vis.loaded && s.shape != nil && s.vis.proj != nil {
		out = append(out, s.vis.proj.File)
	}
	if s.vis.wanted && cat != nil {
		if m, ok := cat.Model(servedID); ok && m.Projector != nil {
			out = append(out, m.Projector.File)
		}
	}
	if s.vis.dl != nil {
		out = append(out, s.vis.dl.file, s.vis.dl.file+".part")
	}
	return out
}

// VisionLoaded says the running model server has the vision projector loaded (status vision).
func (s *Supervisor) VisionLoaded() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.vis.loaded && s.shape != nil
}

// LookSpec is what the look call needs of the projector it will run with: the loaded one, or,
// while no server runs (asleep, loading), the served model's, which the start the look wakes
// would load. ok is false when no projector can serve a look: none loaded on a running server,
// or the served model has none.
func (s *Supervisor) LookSpec() (p catalog.Projector, ok bool) {
	s.mu.Lock()
	if s.shape != nil {
		defer s.mu.Unlock()
		if !s.vis.loaded || s.vis.proj == nil {
			return catalog.Projector{}, false
		}
		return *s.vis.proj, true
	}
	s.mu.Unlock()
	in, found := s.servingModel()
	if !found || in.model.Projector == nil {
		return catalog.Projector{}, false
	}
	return *in.model.Projector, true
}
