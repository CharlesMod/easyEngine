package yield

import (
	"fmt"
	"time"
)

// Config is the yield policy.
type Config struct {
	Grace  time.Duration // a game keeps the engine away until it has been gone this long
	Steady time.Duration // a non-game newcomer whose use holds this long lets the engine back beside it
	Floor  int           // MiB of change that still counts as steady
	Strict bool          // stay away until every newcomer has gone
}

// Action is what the supervisor should do.
type Action int

const (
	None Action = iota
	Yield
	Resume
)

// Decision is one step's outcome.
type Decision struct {
	Action Action
	Reason string // for Yield: what the card is yielded to
	Kind   string // for Yield: game | gpu
	Beside bool   // for Resume: newcomers still run; load only into the room they leave
}

// Input is one observation.
type Input struct {
	Now     time.Time
	Games   []Launch // unsupported games running now
	Changes []Change // from Tracker.Changes
}

// Machine is the yield state machine. It is not safe for concurrent use.
type Machine struct {
	cfg         Config
	yielded     bool
	lastGame    time.Time
	changed     bool // newcomers were in the last observation
	hadGame     bool // a game was in the last observation: its going is a sighting at that step
	steadySince time.Time
	snap        map[int]int
}

// NewMachine starts in the active (not yielded) state.
func NewMachine(c Config) *Machine { return &Machine{cfg: c} }

// Yielded says whether the card is yielded.
func (m *Machine) Yielded() bool { return m.yielded }

// Reset returns to the active state (after a resume, or when a person stops the engine).
func (m *Machine) Reset() { *m = Machine{cfg: m.cfg} }

// Hold restarts the steady wait: a resume beside newcomers found no shape that fits.
func (m *Machine) Hold(now time.Time) { m.steadySince = now }

// Step takes one observation.
func (m *Machine) Step(in Input) Decision {
	// Observations come when something changed, not on a beat, so the last sighting of a game is
	// the step that finds it gone.
	if len(in.Games) > 0 || m.hadGame {
		m.lastGame = in.Now
	}
	m.hadGame = len(in.Games) > 0
	m.changed = len(in.Changes) > 0
	if !m.yielded {
		switch {
		case len(in.Games) > 0:
			m.enter(in)
			return Decision{Action: Yield, Kind: "game", Reason: in.Games[0].Label()}
		case len(in.Changes) > 0:
			m.enter(in)
			return Decision{Action: Yield, Kind: "gpu", Reason: describe(in.Changes[0])}
		}
		return Decision{}
	}
	if !m.lastGame.IsZero() && in.Now.Sub(m.lastGame) < m.cfg.Grace {
		return Decision{}
	}
	if len(in.Changes) == 0 {
		return Decision{Action: Resume}
	}
	if m.cfg.Strict {
		return Decision{}
	}
	cur := map[int]int{}
	for _, c := range in.Changes {
		cur[c.PID] = c.MiB
	}
	if !m.same(cur) {
		m.snap, m.steadySince = cur, in.Now
	}
	if in.Now.Sub(m.steadySince) >= m.cfg.Steady {
		return Decision{Action: Resume, Beside: true}
	}
	return Decision{}
}

// Due says how long until Step, given the observation it last had, could answer other than
// nothing for no reason but time: the rest of the grace after a game, or of the steady wait
// beside newcomers. Zero: only a new observation can change the answer. The caller arms a
// named moment (host.Moment) for it; the machine itself reads no clock.
func (m *Machine) Due(now time.Time) time.Duration {
	if !m.yielded || m.hadGame {
		return 0
	}
	if !m.lastGame.IsZero() {
		if left := m.cfg.Grace - now.Sub(m.lastGame); left > 0 {
			return left
		}
	}
	if !m.changed || m.cfg.Strict { // no newcomers: Resume is due now; strict: they must go
		return 0
	}
	if left := m.cfg.Steady - now.Sub(m.steadySince); left > 0 {
		return left
	}
	return 0
}

func (m *Machine) enter(in Input) {
	m.yielded = true
	m.snap, m.steadySince = nil, in.Now
}

func (m *Machine) same(cur map[int]int) bool {
	if m.snap == nil || len(cur) != len(m.snap) {
		return false
	}
	for pid, mib := range cur {
		old, ok := m.snap[pid]
		if !ok || mib-old > m.cfg.Floor || old-mib > m.cfg.Floor {
			return false
		}
	}
	return true
}

func describe(c Change) string {
	name := c.Name
	if name == "" {
		name = fmt.Sprintf("pid %d", c.PID)
	}
	switch c.Why {
	case "opened":
		return name + " (opened the GPU)"
	case "grew":
		return fmt.Sprintf("%s (grew to %d MiB)", name, c.MiB)
	}
	return fmt.Sprintf("%s (%d MiB)", name, c.MiB)
}
