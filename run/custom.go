package run

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/core/gguf"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
)

// customID is state.Model while the user's own GGUF file serves.
const customID = "custom"

// CustomFile is state.json's record of the user's own GGUF file. The file stays where it is.
type CustomFile struct {
	Path  string `json:"path"`
	Name  string `json:"name"`
	Size  int64  `json:"size"`
	MTime int64  `json:"mtime"` // unix seconds
}

// CustomInfo is the status view of it.
type CustomInfo struct {
	Path      string `json:"path"`
	Name      string `json:"name"`
	Size      int64  `json:"size"`
	Missing   bool   `json:"missing"`
	Params    int64  `json:"params"`
	SizeClass string `json:"size_class"`
}

// genericSampler is the profile of a custom file the catalog has no profile for: the rp values.
func genericSampler() catalog.SamplerProfile {
	t, p, mp := 1.0, 0.95, 0.0
	k := 64
	return catalog.SamplerProfile{Temperature: &t, TopK: &k, TopP: &p, MinP: &mp}
}

// customModel is the synthetic catalog entry of the custom file: its alias is custom:<name>, its
// file is the user's path, its sha stands in as a cheap key. A catalog model of exactly the same
// size lends its footprint and sampler profiles; a catalog model its name matches lends profiles.
func (s *Supervisor) customModel(cat *catalog.Catalog) (catalog.Model, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.customModelLocked(cat)
}

func (s *Supervisor) customModelLocked(cat *catalog.Catalog) (catalog.Model, bool) {
	c := s.st.Custom
	if c == nil {
		return catalog.Model{}, false
	}
	m := catalog.Model{ID: customID + ":" + c.Name, Name: c.Name, File: c.Path, Size: c.Size,
		SHA256: plan.CustomKey(c.Path, c.Size, c.MTime)}
	g := genericSampler()
	m.Sampler = map[string]catalog.SamplerProfile{catalog.ProfileOfficial: g, catalog.ProfileRP: g}
	m.SamplerSource = "generic rp values (no catalog profile for this file)"
	if k, ok := plan.MatchCatalog(cat, c.Size); ok {
		m.Footprint, m.MoE = k.Footprint, k.MoE
		if len(k.Sampler) > 0 {
			m.Sampler, m.SamplerSource = k.Sampler, k.SamplerSource
		}
	} else if k, ok := cat.ModelByName(c.Name); ok && len(k.Sampler) > 0 {
		m.Sampler, m.SamplerSource = k.Sampler, k.SamplerSource
	}
	return m, true
}

// modelFor is the catalog entry of id, or the synthetic one for the custom file.
func (s *Supervisor) modelFor(cat *catalog.Catalog, id string) (catalog.Model, bool) {
	if id == customID {
		return s.customModel(cat)
	}
	return cat.Model(id)
}

// customEstimated is true while the custom file runs on a header guess (no catalog match).
func (s *Supervisor) customEstimated() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.st.Model != customID || s.st.Custom == nil {
		return false
	}
	return plan.MatchCatalogOrNil(s.catLocked(), s.st.Custom.Size) == nil
}

// customMissingPhase puts the engine in phase "error" when the custom file is selected and gone,
// and reports whether it did. No retry is armed: custom/use or custom/clear recovers.
func (s *Supervisor) customMissingPhase() bool {
	s.mu.Lock()
	c := s.st.Custom
	on := s.st.Model == customID && c != nil && s.st.Mode != "external"
	s.mu.Unlock()
	if !on {
		return false
	}
	if st, err := os.Stat(c.Path); err == nil && !st.IsDir() {
		return false
	}
	s.mu.Lock()
	s.retryable = false
	s.mu.Unlock()
	s.setPhase("error", "Model file not found: "+c.Path)
	return true
}

func (s *Supervisor) customStatusLocked(st *Status) {
	if s.st.Model == customID && s.st.Custom != nil {
		c := s.st.Custom
		st.Model, st.ModelName = customID+":"+c.Name, c.Name
	} else if m, ok := s.selectedModelLocked(); ok {
		st.ModelName = m.Name
	}
	if c := s.st.Custom; c != nil {
		fi, err := os.Stat(c.Path)
		st.Custom = &CustomInfo{Path: c.Path, Name: c.Name, Size: c.Size, Missing: err != nil || fi.IsDir()}
		if !st.Custom.Missing {
			st.Custom.Params = customParams(c.Path, fi.Size(), fi.ModTime().UnixNano())
			st.Custom.SizeClass = gguf.SizeClass(st.Custom.Params)
			st.ModelParams, st.ModelSizeClass = st.Custom.Params, st.Custom.SizeClass
		}
	}
}

// customRoom is the planning room for a custom check: the probed hardware, nothing of ours.
func (s *Supervisor) planRoom(hw probe.Hardware, sn sensed) fit.Room {
	return plan.RoomForSensed(hw, sn.sum, s.backendNow())
}

func (s *Supervisor) planCustom(res *plan.PlanResult, cat *catalog.Catalog, hw probe.Hardware, sn sensed) {
	s.mu.Lock()
	c := s.st.Custom
	s.mu.Unlock()
	if c == nil {
		return
	}
	ck := plan.CheckCustom(c.Path, cat, s.planRoom(hw, sn), s.wants())
	res.Custom = &ck
}

// CheckCustom reads the header of a local GGUF file and judges it. Nothing is stored.
func (s *Supervisor) CheckCustom(ctx context.Context, path string) (plan.CustomCheck, error) {
	hw, err := s.Hardware(ctx, false)
	if err != nil {
		return plan.CustomCheck{}, err
	}
	cat, _ := s.catalog()
	sn := s.measure(ctx, hw, s.ours.pids())
	return plan.CheckCustom(path, cat, s.planRoom(hw, sn), s.wants()), nil
}

// UseCustom selects the file at path and loads it in the background. The file is validated first
// (an error is the plain-words reason); it is never copied, moved or deleted.
func (s *Supervisor) UseCustom(path string) error {
	if path == "" {
		return errors.New("no path given")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	cat, _ := s.catalog()
	ck := plan.CheckCustom(abs, cat, fit.Room{NoGPU: true}, s.wants())
	if !ck.OK {
		return errors.New(ck.Error)
	}
	fi, err := os.Stat(abs)
	if err != nil {
		return errors.New("file not found")
	}
	// A background download (first-run upgrade or switch) would swap away from the file later.
	if err := s.prepareInstall(customID); err != nil {
		return err
	}
	s.mu.Lock()
	if s.jobRunning {
		s.mu.Unlock()
		return ErrBusy
	}
	needBackend := s.st.Backend == ""
	if s.st.Model != customID && s.st.Model != "" {
		s.st.CustomPrev = s.st.Model
	}
	s.st.Custom = &CustomFile{Path: abs, Name: plan.CustomName(abs), Size: fi.Size(), MTime: fi.ModTime().Unix()}
	s.st.Model, s.st.Target = customID, ""
	s.userStopped = false
	ext := s.st.Mode == "external"
	lc := s.loadCancel
	var jctx context.Context
	if needBackend {
		// A fresh machine: install the backend and the agent only (no catalog model), then serve.
		var cancel context.CancelFunc
		jctx, cancel = context.WithCancel(s.root)
		s.jobRunning, s.jobCancel = true, cancel
	}
	s.mu.Unlock()
	s.save()
	if lc != nil {
		lc()
	}
	if needBackend {
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			err := s.installBackendOnly(jctx)
			s.endJob(jctx, err, "Install")
		}()
		return nil
	}
	if !ext {
		// Say so at once: the old model's "ready" must not stand for the new file.
		s.setPhase("starting", "Switching to "+plan.CustomName(abs))
		go s.reloadSelected()
	}
	return nil
}

// installBackendOnly is runInstall without a model: llama.cpp and the agent. endJob then settles
// and starts the selected (custom) file.
func (s *Supervisor) installBackendOnly(ctx context.Context) error {
	cat, err := s.catalog()
	if err != nil {
		return err
	}
	hw, err := s.Hardware(ctx, false)
	if err != nil {
		return err
	}
	if err := s.checkNeed(s.setupNeed(cat, hw, ""), "the model server"); err != nil {
		return err
	}
	s.setPhase("downloading", "Installing the model server")
	be, err := s.ensureBackend(ctx, cat, hw, "")
	if err != nil {
		return err
	}
	if s.agentSpec() != nil && s.agentSpec().Wheel != "" {
		if err := s.ensureAgent(ctx, cat); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.selectBuildLocked(be)
	s.mu.Unlock()
	s.save()
	return nil
}

// ClearCustom forgets the custom file (the file itself is left alone). The last catalog model
// serves again when it is installed.
func (s *Supervisor) ClearCustom() {
	s.mu.Lock()
	had := s.st.Custom != nil
	wasOn := s.st.Model == customID
	s.st.Custom = nil
	if wasOn {
		s.st.Model = ""
		if s.st.CustomPrev != "" {
			if cat := s.catLocked(); cat != nil {
				if m, ok := cat.Model(s.st.CustomPrev); ok && s.modelInstalledLocked(m) {
					s.st.Model = m.ID
				}
			}
		}
	}
	s.st.CustomPrev = ""
	ext := s.st.Mode == "external"
	lc := s.loadCancel
	s.mu.Unlock()
	if !had && !wasOn {
		return
	}
	s.save()
	if lc != nil {
		lc()
	}
	if wasOn && !ext {
		s.setPhase("starting", "Stopping your own file")
		go s.reloadSelected()
	}
}

func (s *Supervisor) modelInstalledLocked(m catalog.Model) bool {
	st, err := os.Stat(s.modelPath(m))
	return err == nil && st.Size() > 0 && (m.Size == 0 || st.Size() == m.Size)
}

// reloadSelected stops whatever serves and loads the selected model (or settles to not_installed).
func (s *Supervisor) reloadSelected() {
	s.opMu.Lock()
	defer s.opMu.Unlock()
	s.stopServerLocked()
	if _, ok := s.resolveInstall(); !ok {
		if !s.customMissingPhase() {
			s.setPhase("not_installed", "No model installed")
		}
		return
	}
	if err := s.startLocked(); err != nil {
		s.logf("custom: start failed: %v", err)
	}
}

var paramCache struct {
	sync.Mutex
	path        string
	size, mtime int64
	params      int64
}

// customParams reads the header's parameter count, cached by path, size and mtime (status is polled).
func customParams(path string, size, mtime int64) int64 {
	paramCache.Lock()
	defer paramCache.Unlock()
	if paramCache.path == path && paramCache.size == size && paramCache.mtime == mtime {
		return paramCache.params
	}
	var n int64
	if fh, err := os.Open(path); err == nil {
		if f, err := gguf.Parse(fh, size); err == nil {
			n = f.ParamCount()
		}
		fh.Close()
	}
	paramCache.path, paramCache.size, paramCache.mtime, paramCache.params = path, size, mtime, n
	return n
}
