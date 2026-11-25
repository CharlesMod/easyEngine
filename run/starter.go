package run

import (
	"context"
	"fmt"
	"os"
	"time"

	"hermesengine/catalog"
	"hermesengine/fetch"
	"hermesengine/fit"
	"hermesengine/probe"
)

// First-run staging: when the chosen model (the target) is not on disk, the engine first
// downloads and serves the smallest catalog model that fits (the starter), downloads the target
// in the background, and swaps to it once it is verified and nothing is in flight.
// Design and status fields: docs/ENGINE.md section 7a.

const (
	// starterRatio: a starter is only worth it when the target is at least this many times its size.
	starterRatio = 2
	// diskHeadroomMiB stays free after a download.
	diskHeadroomMiB = 512
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
	retryAt  time.Time
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
	Target     string    `json:"target"`
	TargetName string    `json:"target_name"`
	State      string    `json:"state"`
	Progress   *Progress `json:"progress"`
	Error      string    `json:"error,omitempty"`
	Attempt    int       `json:"attempt"`
	RetryInS   int       `json:"retry_in_s,omitempty"`
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
	info := &UpgradeInfo{Target: u.target.ID, TargetName: u.target.Name, State: u.state, Error: u.err, Attempt: u.attempt}
	if u.progress != nil {
		p := *u.progress
		info.Progress = &p
	}
	if !u.retryAt.IsZero() && (u.state == upRetrying || u.state == upNoSpace) {
		if d := u.retryAt.Sub(s.cfg.Now()); d > 0 {
			info.RetryInS = int(d.Seconds()) + 1
		}
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
	if st.ModelTier == "starter" && !st.ModelFallback {
		st.ModelFallback, st.ModelNote = true, starterGenericNote
	}
}

// pickStarter chooses the smallest installable catalog model that fits this machine and is at
// most 1/starterRatio the size of the target. nil: no starter (the target is already small, or it
// is the smallest that fits, as on a CPU-only machine, or the starter would not fit the disk).
func (s *Supervisor) pickStarter(ctx context.Context, cat *catalog.Catalog, hw probe.Hardware, target catalog.Model) *catalog.Model {
	res := fit.Plan(cat, hw, s.wants(), s.cfg.Profile.GameReserveMiB, s.modelInstalled)
	var best *catalog.Model
	for _, v := range res.Models {
		m, ok := cat.Model(v.ID)
		if !ok || m.ID == target.ID || m.URL == "" || v.Fits == "no" {
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
	if !s.modelInstalled(*best) {
		if err := s.checkDisk(*best); err != nil {
			s.logf("no starter: %v", err)
			return nil
		}
	}
	return best
}

// diskNeed is the free MiB a model still needs: its size minus what a partial download holds,
// plus headroom.
func (s *Supervisor) diskNeed(m catalog.Model) int {
	left := m.Size
	if st, err := os.Stat(s.modelPath(m) + ".part"); err == nil && st.Size() < left {
		left -= st.Size()
	}
	return int(left>>20) + diskHeadroomMiB
}

// checkDisk errors when the model does not fit in the free disk space.
func (s *Supervisor) checkDisk(m catalog.Model) error {
	free, err := s.cfg.DiskFree(s.dir)
	if err != nil {
		return nil // unknown: let the download find out
	}
	if need := s.diskNeed(m); free < need {
		return fmt.Errorf("not enough disk space for %s: needs %d MiB free, %d MiB available", m.Name, need, free)
	}
	return nil
}

// ---- upgrade bookkeeping ------------------------------------------------------------------

func (s *Supervisor) beginQueued(target catalog.Model) {
	s.mu.Lock()
	s.upg = &upgrade{target: target, state: upQueued}
	s.mu.Unlock()
}

// dropQueued clears a queued upgrade whose install job failed or was cancelled.
func (s *Supervisor) dropQueued() {
	s.mu.Lock()
	if s.upg != nil && s.upg.state == upQueued {
		s.upg = nil
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
	s.mu.Unlock()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer cancel()
		s.runUpgrade(ctx, target)
	}()
}

func (s *Supervisor) sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// runUpgrade downloads the target (retrying with backoff; a failure never touches what serves),
// then swaps.
func (s *Supervisor) runUpgrade(ctx context.Context, target catalog.Model) {
	backoff := s.cfg.UpgradeBackoff
	for attempt := 1; ; attempt++ {
		if s.modelInstalled(target) {
			break
		}
		s.setUpgrade(func(u *upgrade) { u.attempt = attempt })
		var err error
		if err = s.checkDisk(target); err != nil {
			s.setUpgrade(func(u *upgrade) { u.state, u.err, u.progress = upNoSpace, err.Error(), nil })
			s.addError("%v; staying on the smaller model", err)
		} else {
			s.setUpgrade(func(u *upgrade) { u.state, u.err, u.retryAt = upDownloading, "", time.Time{} })
			err = fetch.Download(ctx, target.URL, s.modelPath(target), target.SHA256, target.Size, func(p fetch.Progress) {
				s.setUpgrade(func(u *upgrade) {
					u.progress = &Progress{DoneBytes: p.Done, TotalBytes: p.Total, RateBps: p.RateBps, ETAS: int(p.ETA.Seconds()), Step: "model " + target.ID}
				})
			})
			if err == nil {
				break
			}
			if ctx.Err() != nil {
				s.setUpgrade(func(u *upgrade) { u.state, u.progress = upCancelled, nil })
				s.logf("upgrade to %s cancelled", target.ID)
				return
			}
			s.addError("download of %s failed: %v (retrying)", target.Name, err)
			s.setUpgrade(func(u *upgrade) { u.state, u.err = upRetrying, err.Error() })
		}
		s.setUpgrade(func(u *upgrade) { u.retryAt = s.cfg.Now().Add(backoff) })
		if !s.sleepCtx(ctx, backoff) {
			s.setUpgrade(func(u *upgrade) { u.state, u.progress = upCancelled, nil })
			return
		}
		if backoff *= 2; backoff > s.cfg.UpgradeBackoffMax {
			backoff = s.cfg.UpgradeBackoffMax
		}
	}
	s.setUpgrade(func(u *upgrade) { u.progress = nil })
	s.swapTo(ctx, target)
}

// swapTo switches the served model to target at a safe moment. It waits until no load or resize
// is under way, then (holding the operation lock, so the watchdog stays out) answers new
// requests with 503 and Retry-After, waits for the ones in flight (up to SwapDrain), stops the
// old server, and starts the target with a fresh fit. If the target does not load, the starter
// is started again and the upgrade is marked failed.
func (s *Supervisor) swapTo(ctx context.Context, target catalog.Model) {
	for {
		switch s.Phase() {
		case "starting", "loading", "resizing", "downloading", "installing":
			if !s.sleepCtx(ctx, s.cfg.HealthPoll) {
				s.setUpgrade(func(u *upgrade) { u.state = upCancelled })
				return
			}
			continue
		}
		break
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
	s.setUpgrade(func(u *upgrade) { u.state = upSwapping })
	s.logf("swapping %s -> %s", starterID, target.ID)
	if srv := s.srv; srv != nil {
		s.mu.Lock()
		s.swapping = true
		s.mu.Unlock()
		s.setPhase("resizing", "Switching to the full model")
		base := fmt.Sprintf("http://127.0.0.1:%d", srv.port)
		deadline := time.Now().Add(s.cfg.SwapDrain)
		for !s.slotsIdle(base) && time.Now().Before(deadline) && !srv.proc.Exited() {
			time.Sleep(10 * time.Millisecond)
		}
		s.countRestart()
		s.stopServerLocked()
	}
	s.mu.Lock()
	s.st.Model, s.st.Target = target.ID, ""
	s.mu.Unlock()
	s.save()
	if stopped || s.Phase() == "sleeping" {
		s.finishSwap(upDone, "")
		s.setPhase("sleeping", "Stopped")
		return
	}
	err := s.startLocked()
	if err == nil && s.Phase() == "ready" {
		s.finishSwap(upDone, "")
		return
	}
	msg := "the full model would not load"
	if err != nil {
		msg = err.Error()
	}
	s.addError("swap to %s failed (%s): back on the starter", target.Name, msg)
	s.mu.Lock()
	s.st.Model, s.st.Target = starterID, target.ID
	s.mu.Unlock()
	s.save()
	s.stopServerLocked()
	s.finishSwap(upFailed, msg)
	s.startLocked()
}

func (s *Supervisor) finishSwap(state, errMsg string) {
	s.mu.Lock()
	s.swapping = false
	if s.upg != nil {
		s.upg.state, s.upg.err = state, errMsg
	}
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
