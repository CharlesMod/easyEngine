package run

import (
	"fmt"
	"os"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/plan"
)

// Switching models while one serves: the current model keeps serving, the new one downloads in
// the background through the upgrade path (starter.go) and is swapped in when ready. The old
// file is kept; status.previous names it until it is deleted or another switch happens.

// PreviousInfo is status.previous.
type PreviousInfo struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	File  string `json:"file"`
	Bytes int64  `json:"bytes"`
}

func gb(mib int) float64 { return float64(mib) / 1024 }

// trySwitch handles an install of a different catalog model while one is serving. handled is
// false when the normal install path applies (nothing serves, same model, external mode).
func (s *Supervisor) trySwitch(model, backend string) (handled bool, err error) {
	if s.selectServed(model) { // the asked model already serves for a step (tiers.go)
		return true, nil
	}
	s.mu.Lock()
	cur, be := s.st.Model, s.st.Backend
	local := s.st.Mode == "local"
	ready := s.phase == "ready"
	busy := s.jobRunning || s.removing
	s.mu.Unlock()
	if !local || !ready || cur == "" || cur == model || be == "" {
		return false, nil
	}
	if busy {
		return true, ErrBusy
	}
	cat, cerr := s.catalog()
	if cerr != nil {
		return true, cerr
	}
	m, ok := cat.Model(model)
	if !ok {
		return true, fmt.Errorf("unknown model %q", model)
	}
	if m.URL == "" && !s.modelInstalled(m) {
		return true, fmt.Errorf("%s is import-only: run `easy-engine import <path>` (or POST /engine/v1/import)", m.Name)
	}
	if !s.modelInstalled(m) {
		if err := s.checkDisk(m); err != nil {
			return true, err
		}
	}
	if err := s.prepareInstall(model); err != nil {
		return true, err
	}
	s.mu.Lock()
	s.st.Target = m.ID
	s.mu.Unlock()
	s.save()
	s.beginUpgrade(m)
	s.setUpgrade(func(u *upgrade) { u.switching = true })
	s.logf("switching %s -> %s (the current model keeps serving)", cur, m.ID)
	return true, nil
}

// forgetSwitch is called on cancel: a cancelled switch must not resume at the next start.
func (s *Supervisor) forgetSwitch() {
	s.mu.Lock()
	u := s.upg
	sw := u != nil && u.switching && u.activeState()
	s.mu.Unlock()
	if !sw {
		return
	}
	s.mu.Lock()
	s.st.Target = ""
	s.mu.Unlock()
	s.save()
}

// recordPrevious notes the model a swap replaced, when its file is still on disk.
func (s *Supervisor) recordPrevious(oldID, newID string) {
	prev := ""
	if oldID != "" && oldID != newID {
		if cat, err := s.catalog(); err == nil {
			if m, ok := cat.Model(oldID); ok && s.modelInstalled(m) {
				prev = oldID
			}
		}
	}
	s.mu.Lock()
	s.st.Previous = prev
	s.mu.Unlock()
	s.save()
}

// previousLocked is status.previous. Caller holds s.mu.
func (s *Supervisor) previousLocked() *PreviousInfo {
	if s.st.Previous == "" {
		return nil
	}
	c := s.cat
	if c == nil {
		c = s.cfg.Catalog
	}
	if c == nil {
		var err error
		if c, err = catalog.Load(s.dir); err != nil {
			return nil
		}
		s.cat = c
	}
	m, ok := c.Model(s.st.Previous)
	if !ok {
		return nil
	}
	fi, err := os.Stat(s.modelPath(m))
	if err != nil {
		return nil
	}
	return &PreviousInfo{ID: m.ID, Name: m.Name, File: m.File, Bytes: fi.Size()}
}

// fillDisk adds the disk numbers to a plan.
func (s *Supervisor) fillDisk(res *plan.PlanResult, cat *catalog.Catalog) {
	res.DiskFreeBytes = -1
	if free, err := s.cfg.DiskFree(s.dir); err == nil {
		res.DiskFreeBytes = int64(free) << 20
	}
	for i := range res.Models {
		v := &res.Models[i]
		if v.Installed {
			continue
		}
		if m, ok := cat.Model(v.ID); ok {
			n := s.diskNeed(m)
			v.DiskNeedBytes, v.DiskNeed = n.Total, &n
		}
	}
}
