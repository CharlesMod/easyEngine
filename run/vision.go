package run

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

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
}

type visionDL struct {
	file     string
	progress *Progress
	cancel   context.CancelFunc
	done     chan struct{} // closed when the download goroutine has stopped writing and said so in vis
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

// projectorPresent: the file is there at its catalog size. The file has that name only after its
// sha256 was verified (fetch.Download renames the .part once the hash matches), as for a model.
func (s *Supervisor) projectorPresent(p *catalog.Projector) bool {
	st, err := os.Stat(s.projectorPath(p))
	return err == nil && st.Mode().IsRegular() && st.Size() > 0 && (p.Size == 0 || st.Size() == p.Size)
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
		h := helpOf(in.bin, s.cfg.ExtraEnv)
		ready = h == nil || h.Has("--mmproj")
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
		h := helpOf(in.bin, s.cfg.ExtraEnv)
		need = h == nil || h.Has("--mmproj")
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
			note = s.fetchProjector(in.model)
		default:
			try = true
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
		s.visionWantChanged("the vision projector is ready", true)
	}()
	return "the vision projector is downloading"
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
