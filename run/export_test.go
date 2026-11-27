package run

import (
	"hermesengine/catalog"
	"hermesengine/plan"
	"hermesengine/probe"
)

// SetAgentNoVenv switches the agent layout (no venv: base Python + <env>/site) for a test; it returns the undo.
func SetAgentNoVenv(on bool) func() {
	old := agentNoVenv
	agentNoVenv = on
	return func() { agentNoVenv = old }
}

// YieldTick is how often the yield loop looks for launcher games.
const YieldTick = yieldTick

// Disk-need seams for the external tests.

func (s *Supervisor) SetupNeed(cat *catalog.Catalog, hw probe.Hardware, backendID string, models ...catalog.Model) plan.DiskNeed {
	return s.setupNeed(cat, hw, backendID, models...)
}
func (s *Supervisor) LogRoom() int64                               { return s.logRoom() }
func (s *Supervisor) CheckNeed(n plan.DiskNeed, what string) error { return s.checkNeed(n, what) }
func LogCeiling() int64                                            { return logCeiling() }
