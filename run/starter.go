package run

import (
	"context"
	"errors"
	"os"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/fetch"
	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
)

// First-run staging: when the chosen model (the target) is not on disk, the engine first
// downloads and serves the smallest catalog model that fits (the starter), downloads the target
// in the background, and swaps to it once it is verified and nothing is in flight.
// Design and status fields: docs/ENGINE.md section 7a.

const (
	// starterRatio: a starter is only worth it when the target is at least this many times its size.
	starterRatio = 2
)

// Upgrade states (UpgradeInfo.State).
const (
	upQueued      = "queued"      // the starter is still downloading
	upDownloading = "downloading" // the target is downloading
	upRetrying    = "retrying"    // the last attempt failed; waiting to retry
	upNoSpace     = "no_space"    // the target cannot fit on disk; staying on the starter
	upSwapping    = "swapping"
	upDone        = "done"
	upCancelled   = "cancelled"
	upFailed      = "failed" // the target downloaded but would not load; back on the starter
)

type upgrade struct {
	target   catalog.Model
	state    string
	progress *Progress
	err      string
	attempt  int
	// switching: the user chose another model while this one serves; it is not a first-run starter.
	switching bool
}

func (u *upgrade) activeState() bool {
	switch u.state {
	case upQueued, upDownloading, upRetrying, upNoSpace, upSwapping:
		return true
	}
	return false
}

// UpgradeInfo is status.upgrade.
type UpgradeInfo struct {
	Target     string `json:"target"`
	TargetName string `json:"target_name"`
	// TargetCodename is the target tier's code name; "" when it has none.
	TargetCodename string    `json:"target_codename"`
	State          string    `json:"state"`
	Progress       *Progress `json:"progress"`
	Error          string    `json:"error,omitempty"`
	Attempt        int       `json:"attempt"`
	RetryInS       int       `json:"retry_in_s,omitempty"` // always 0 (omitted): there is no retry cadence
	// Switch is true when the user switched models (the current one is not a starter).
	Switch bool `json:"switch,omitempty"`
}

const starterGenericNote = "A smaller model is serving in place of the full one: expect simpler speech."

// stagingLocked fills model_tier, model_stage, upgrade and the fallback flag. Caller holds s.mu.
func (s *Supervisor) stagingLocked(st *Status) {
	if s.st.Mode == "external" || s.st.Model == "" && s.upg == nil {
		return
	}
	st.ModelTier = "full"
	u := s.upg
	if u == nil {
		return
	}
	info := &UpgradeInfo{Switch: u.switching, Target: u.target.ID, TargetName: u.target.Name, TargetCodename: u.target.Codename, State: u.state, Error: u.err, Attempt: u.attempt}
	if u.progress != nil {
		p := *u.progress
		info.Progress = &p
	}
	st.Upgrade = info
	switch u.state {
	case upQueued:
		st.ModelTier, st.ModelStage = "starter", "starter_downloading"
	case upDownloading:
		st.ModelTier, st.ModelStage = "starter", "target_downloading"
	case upRetrying:
		st.ModelTier, st.ModelStage = "starter", "target_retrying"
	case upNoSpace:
		st.ModelTier, st.ModelStage = "starter", "target_no_space"
	case upSwapping:
		st.ModelTier, st.ModelStage = "starter", "swapping"
	case upCancelled, upFailed:
		st.ModelTier, st.ModelStage = "starter", "starter_serving"
	case upDone:
		st.ModelStage = "target_serving"
	}
	if u.switching {
		st.ModelTier = "full"
		return
	}
	if st.ModelTier == "starter" && !st.ModelFallback {
		st.ModelFallback, st.ModelNote = true, starterGenericNote
	}
}

// pickStarter chooses the smallest installable catalog model that fits this machine, is at most
// 1/starterRatio the size of the target, and is not a fallback model (a fallback is only there for
// machines that can run nothing else; its speech is too degraded to be anyone's first impression). nil: no starter (the target is already small, or it
// is the smallest that fits, as on a CPU-only machine, or the starter would not fit the disk).
func (s *Supervisor) pickStarter(ctx context.Context, cat *catalog.Catalog, hw probe.Hardware, target catalog.Model) *catalog.Model {
	need := func(m catalog.Model) plan.DiskNeed { return s.setupNeed(cat, hw, "", m) }
	res := plan.PlanSensedDisk(cat, s.liveDisk(hw), s.wants(), s.measure(ctx, hw, s.ours.pids()).sum, s.modelInstalled, nil, need)
	var best *catalog.Model
	for _, v := range res.Models {
		m, ok := cat.Model(v.ID)
		if !ok || m.ID == target.ID || m.URL == "" || v.Fits == "no" || m.Fallback || m.OptIn {
			continue
		}
		if m.Size*starterRatio > target.Size {
			continue
		}
		if best == nil || m.Size < best.Size {
			mm := m
			best = &mm
		}
	}
	if best == nil {
		return nil
	}
	// The starter stays on disk while the target downloads: both must fit, with the rest.
	if err := s.checkNeed(s.setupNeed(cat, hw, "", *best, target), best.Label()+" first, then "+target.Label()); err != nil {
		s.logf("no starter: %v", err)
		return nil
	}
	return best
}

// diskNeed is what setting up m still needs from the current state (setupNeed with the
// catalog and hardware at hand).
func (s *Supervisor) diskNeed(m catalog.Model) plan.DiskNeed {
	cat, err := s.catalog()
	if err != nil {
		return plan.NeedOf([]plan.DiskStep{plan.ModelStep(m.Name, m.Size, 0)}, s.logRoom(), s.hostFiles())
	}
	hw, _ := s.Hardware(s.root, false)
	return s.setupNeed(cat, hw, "", m)
}

// checkDisk errors (ErrNoSpace) when setting up m does not fit in the free disk space.
func (s *Supervisor) checkDisk(m catalog.Model) error { return s.checkNeed(s.diskNeed(m), m.Name) }

// ---- upgrade bookkeeping ------------------------------------------------------------------

func (s *Supervisor) beginQueued(target catalog.Model) {
	s.mu.Lock()
	s.upg = &upgrade{target: target, state: upQueued}
	s.refreshLocked()
	s.mu.Unlock()
}

// dropQueued clears a queued upgrade whose install job failed or was cancelled.
func (s *Supervisor) dropQueued() {
	s.mu.Lock()
	if s.upg != nil && s.upg.state == upQueued {
		s.upg = nil
		s.refreshLocked()
	}
	s.mu.Unlock()
}

// prepareInstall runs before an install job: the same target already upgrading is busy; any other
// install replaces a running upgrade.
func (s *Supervisor) prepareInstall(model string) error {
	s.mu.Lock()
	u := s.upg
	active := u != nil && u.activeState()
	same := active && u.target.ID == model
	s.mu.Unlock()
	if same {
		return ErrBusy
	}
	if active {
		s.cancelUpgrade()
		s.mu.Lock()
		s.st.Target = ""
		s.upg = nil
		s.mu.Unlock()
		s.save()
	}
	return nil
}

// cancelUpgrade stops the background download; the .part file stays for resume. The target stays
// in state.json, so the next engine start (or an install of it) resumes.
func (s *Supervisor) cancelUpgrade() {
	s.mu.Lock()
	c := s.upgCancel
	s.mu.Unlock()
	if c != nil {
		c()
	}
}

func (s *Supervisor) setUpgrade(f func(u *upgrade)) {
	s.mu.Lock()
	if s.upg != nil {
		f(s.upg)
	}
	s.refreshLocked()
	s.mu.Unlock()
}

// resumeUpgrade restarts an unfinished upgrade after an engine restart.
func (s *Supervisor) resumeUpgrade() {
	s.mu.Lock()
	id, cur, local := s.st.Target, s.st.Model, s.st.Mode == "local"
	s.mu.Unlock()
	if id == "" || id == cur || !local {
		return
	}
	cat, err := s.catalog()
	if err != nil {
		return
	}
	if m, ok := cat.Model(id); ok && m.URL != "" {
		s.beginUpgrade(m)
	}
}

// beginUpgrade starts the background job that fetches the target and swaps to it.
func (s *Supervisor) beginUpgrade(target catalog.Model) {
	ctx, cancel := context.WithCancel(s.root)
	s.mu.Lock()
	if s.upgCancel != nil {
		s.upgCancel()
	}
	s.upgCancel = cancel
	s.upg = &upgrade{target: target, state: upDownloading}
	s.refreshLocked()
	s.mu.Unlock()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer cancel()
		s.runUpgrade(ctx, target)
	}()
}

// runUpgrade downloads the target (a failure never touches what serves), then swaps. There is no
// retry cadence: a download that moves bytes resumes at once; one that moves none, or a disk
// with no room, waits for a word that could change the outcome (the network coming back, a
// client's install or nudge, the engine's next start).
func (s *Supervisor) runUpgrade(ctx context.Context, target catalog.Model) {
	for attempt := 1; ; attempt++ {
		if s.modelInstalled(target) {
			break
		}
		s.setUpgrade(func(u *upgrade) { u.attempt = attempt })
		// listened before the attempt, so a word said while it still runs is heard; counted as
		// waiting (a request is a word for it) only once it waits
		w, err := s.listenQuiet()
		if err != nil {
			return
		}
		wa := s.newWordAgain()
		again := wa.again(func(err error) {
			s.addError("download of %s made no progress: %v (waiting for the network or a word from a client)", target.Name, err)
			s.setUpgrade(func(u *upgrade) { u.state, u.err = upRetrying, err.Error() })
		}, func() {
			s.setUpgrade(func(u *upgrade) { u.state, u.err, u.attempt = upDownloading, "", u.attempt+1 })
		})
		if err = s.checkDisk(target); err != nil {
			w.count() // it waits now: before the status says so, so a request that sees it is heard
			s.setUpgrade(func(u *upgrade) { u.state, u.err, u.progress = upNoSpace, err.Error(), nil })
			s.addError("%v; staying on the smaller model", err)
		} else {
			s.setUpgrade(func(u *upgrade) { u.state, u.err = upDownloading, "" })
			err = fetch.Download(ctx, target.URL, s.modelPath(target), target.SHA256, target.Size, func(p fetch.Progress) {
				s.setUpgrade(func(u *upgrade) {
					u.progress = &Progress{DoneBytes: p.Done, TotalBytes: p.Total, RateBps: p.RateBps, ETAS: int(p.ETA.Seconds()), Step: stepLabel("model "+target.ID, p)}
				})
			}, again)
			wa.close()
			if err == nil {
				w.Close()
				break
			}
			if ctx.Err() != nil {
				w.Close()
				s.setUpgrade(func(u *upgrade) { u.state, u.progress = upCancelled, nil })
				s.logf("upgrade to %s cancelled", target.ID)
				return
			}
			w.count() // it waits now: before the status says so, so a request that sees it is heard
			s.addError("download of %s failed: %v (waiting for a word from a client)", target.Name, err)
			s.setUpgrade(func(u *upgrade) { u.state, u.err = upRetrying, err.Error() })
		}
		wa.close()
		w.count()
		if s.awaitWord(ctx, w) != nil {
			s.setUpgrade(func(u *upgrade) { u.state, u.progress = upCancelled, nil })
			return
		}
	}
	s.setUpgrade(func(u *upgrade) { u.progress = nil })
	s.swapTo(ctx, target)
}

// busyPhase is a phase in which a swap waits: a load, a resize or an install under way.
func busyPhase(p string) bool {
	switch p {
	case "starting", "loading", "resizing", "downloading", "installing":
		return true
	}
	return false
}

// swapTo switches the served model to target at a safe moment. It waits (on the phase fact) until
// no load or resize is under way and the card is not yielded, then answers new requests with
// "Switching to the full model", waits for the ones in flight (up to the named moment SwapDrain)
// without holding the operation lock, takes it once the count is zero, stops the old server and
// starts the target with a fresh fit. If the card is yielded when the target would load, the
// target loads when the yield ends. If the target does not load, the starter is started again
// and the upgrade is marked failed.
func (s *Supervisor) swapTo(ctx context.Context, target catalog.Model) {
	free := func() bool { return !busyPhase(s.Phase()) && s.yielded() == nil }
	if !s.waitFact(ctx, free, "phase", "yield") {
		s.setUpgrade(func(u *upgrade) { u.state = upCancelled })
		return
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	if ctx.Err() != nil {
		s.setUpgrade(func(u *upgrade) { u.state = upCancelled })
		return
	}
	s.mu.Lock()
	starterID := s.st.Model
	stopped := s.userStopped
	s.mu.Unlock()
	if s.srv != nil { // new requests are refused before the stage says swapping
		s.mu.Lock()
		s.swapping = true
		s.mu.Unlock()
	}
	s.setUpgrade(func(u *upgrade) { u.state = upSwapping })
	s.logf("swapping %s -> %s", starterID, target.ID)
	if s.srv != nil {
		s.setPhase("resizing", "Switching to the full model")
		// wait for the requests in flight without the operation lock, then take it and look again
		for s.inflightNow() != 0 {
			s.opMu.Unlock()
			dctx, cancel := context.WithCancel(ctx)
			// named moment: SwapDrain, the longest a swap waits for the requests in flight
			stop := host.Moment(s.cfg.SwapDrain, cancel)
			s.waitIdle(dctx)
			late := dctx.Err() != nil && ctx.Err() == nil
			stop()
			cancel()
			s.opMu.Lock()
			if ctx.Err() != nil {
				s.mu.Lock()
				s.swapping = false
				s.mu.Unlock()
				if s.srv != nil && s.Phase() == "resizing" {
					s.setPhase("ready", "Ready")
				}
				s.setUpgrade(func(u *upgrade) { u.state = upCancelled })
				return
			}
			if late {
				s.addError("requests still running after %s: switching to the full model now", s.cfg.SwapDrain)
				break
			}
		}
		if s.srv != nil {
			s.countRestart()
			s.stopServerLocked()
		}
		// the old server is gone: from here the phase says what the load does (the ready of the
		// target must route to it)
		s.mu.Lock()
		s.swapping = false
		s.mu.Unlock()
	}
	s.mu.Lock()
	s.st.Model, s.st.Target = target.ID, ""
	s.mu.Unlock()
	s.save()
	if stopped || s.Phase() == "sleeping" {
		s.dropStarter(starterID, target.ID)
		s.recordPrevious(starterID, target.ID)
		s.finishSwap(upDone, "") // last: the done fact means the starter is let go too
		s.setPhase("sleeping", "Stopped")
		return
	}
	err := s.startLocked()
	s.mu.Lock()
	if t := s.steppedLocked(); t != nil && err == nil {
		// the card cannot hold the target whole: a lower tier serves for it, and the starter (which
		// that tier may be) is kept; the swap has failed
		err = errors.New(t.note)
	}
	s.mu.Unlock()
	if err == nil && s.yielded() != nil {
		// the card was yielded before the target could load: the yield's end starts the selected
		// model (resumeYield), and the swap is judged by how that start ends
		s.logf("swap to %s waits for the card: it is yielded", target.ID)
		s.opMu.Unlock()
		settled := func() bool {
			if s.yielded() != nil {
				return false
			}
			switch s.Phase() {
			case "ready", "error", "sleeping", "not_installed":
				return true
			}
			return false
		}
		ok := s.waitFact(ctx, settled, "phase", "yield")
		s.opMu.Lock()
		if !ok {
			s.setUpgrade(func(u *upgrade) { u.state = upCancelled })
			return
		}
		if p := s.Phase(); p == "sleeping" {
			s.dropStarter(starterID, target.ID)
			s.recordPrevious(starterID, target.ID)
			s.finishSwap(upDone, "") // last: the done fact means the starter is let go too
			return
		}
	}
	if err == nil && s.Phase() == "ready" {
		s.dropStarter(starterID, target.ID)
		s.recordPrevious(starterID, target.ID)
		s.finishSwap(upDone, "") // last: the done fact means the starter is let go too
		return
	}
	msg := "the full model would not load"
	if err != nil {
		msg = err.Error()
	}
	s.addError("swap to %s failed (%s): back on the starter", target.Label(), msg)
	s.mu.Lock()
	s.st.Model, s.st.Target = starterID, target.ID
	s.mu.Unlock()
	s.save()
	s.stopServerLocked()
	s.finishSwap(upFailed, msg)
	s.startLocked()
}

// dropStarter deletes the starter's model file once the target serves, but only when this engine
// downloaded it for the first run (state.StarterOwned) and nothing else uses it.
func (s *Supervisor) dropStarter(starterID, targetID string) {
	s.mu.Lock()
	owned := s.st.StarterOwned
	s.st.StarterOwned = ""
	s.mu.Unlock()
	s.save()
	if owned == "" || owned != starterID || owned == targetID {
		return
	}
	cat, err := s.catalog()
	if err != nil {
		s.logf("starter %s kept: %v", owned, err)
		return
	}
	m, ok := cat.Model(owned)
	if !ok {
		return
	}
	p := s.modelPath(m)
	if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
		s.logf("starter %s kept: %v", owned, err)
		return
	}
	os.Remove(p + ".part")
	s.logf("removed starter %s (%d MiB) now that %s serves", owned, m.Size>>20, targetID)
}

func (s *Supervisor) finishSwap(state, errMsg string) {
	s.mu.Lock()
	s.swapping = false
	if s.upg != nil {
		s.upg.state, s.upg.err = state, errMsg
	}
	s.refreshLocked()
	s.mu.Unlock()
}

// starterFor is pickStarter for an install job that stays alive after it (the CLI install exits
// with the job, so it never stages).
func (s *Supervisor) starterFor(ctx context.Context, cat *catalog.Catalog, hw probe.Hardware, target catalog.Model) *catalog.Model {
	if s.cfg.NoAutoStart {
		return nil
	}
	return s.pickStarter(ctx, cat, hw, target)
}
