package run

import (
	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
)

// SetAgentNoVenv switches the agent layout (no venv: base Python + <env>/site) for a test; it returns the undo.
func SetAgentNoVenv(on bool) func() {
	old := agentNoVenv
	agentNoVenv = on
	return func() { agentNoVenv = old }
}

// Disk-need seams for the external tests.

func (s *Supervisor) SetupNeed(cat *catalog.Catalog, hw probe.Hardware, backendID string, models ...catalog.Model) plan.DiskNeed {
	return s.setupNeed(cat, hw, backendID, models...)
}
func (s *Supervisor) LogRoom() int64                               { return s.logRoom() }
func (s *Supervisor) CheckNeed(n plan.DiskNeed, what string) error { return s.checkNeed(n, what) }
func LogCeiling() int64                                            { return logCeiling() }

// React runs one pass of the mind's reactor now, in the caller: a test that changed something
// and wants to know the engine has looked at it calls it once the change holds (HeldReserve).
func (s *Supervisor) React() { s.react() }

// HeldReserve is the room's last held reserve (others plus the host's share, MiB), while a model
// is loaded.
func (s *Supervisor) HeldReserve() (int, bool) {
	r, ok := s.card.held()
	return r.total, ok
}
