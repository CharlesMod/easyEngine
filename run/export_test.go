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

// RoomMoved is what a stranger's open of the card and then its end would say (two words): the
// room is read twice, so a new level holds without the level's own period.
func (s *Supervisor) RoomMoved() {
	s.card.event()
	s.card.event()
}

// Network says the network happening, as the system's word that a link or an address changed
// would (watchNetwork).
func (s *Supervisor) Network() { s.say("network", "", nil) }

// HeldReserve is the room's last held reserve (others plus the host's share, MiB), while a model
// is loaded.
func (s *Supervisor) HeldReserve() (int, bool) {
	r, ok := s.card.held()
	return r.total, ok
}

// Listeners is how many listeners a request is said to as a word.
func (s *Supervisor) Listeners() int { return int(s.listeners.Load()) }

// CatalogNow is the catalog the engine uses now (overlays included).
func (s *Supervisor) CatalogNow() *catalog.Catalog { c, _ := s.catalog(); return c }

// PlansNow is how many reshapes are asked for the clients now, and the slot map's generation.
func (s *Supervisor) PlansNow() (int, uint64) {
	s.clients.mu.Lock()
	defer s.clients.mu.Unlock()
	return len(s.clients.plans), s.clients.gen
}

// RestartsOf is how many restarts a budget key has been charged.
func (s *Supervisor) RestartsOf(who string) int {
	s.clients.mu.Lock()
	defer s.clients.mu.Unlock()
	return s.clients.restarts[who]
}
