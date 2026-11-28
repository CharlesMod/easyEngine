package run

import (
	"context"
	"fmt"
	"sort"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
)

// tierStep is a selected catalog model the card cannot hold whole, served by a lower tier
// instead. It lives for one start (startLocked drops it and tries the selected model again), is
// never saved, and holds only while the selection is still from.
type tierStep struct {
	from, to catalog.Model
	note     string
}

// servedIDLocked is the catalog id that serves: the selected model, or the tier it stepped down
// to. Caller holds s.mu.
func (s *Supervisor) servedIDLocked() string {
	if t := s.steppedLocked(); t != nil {
		return t.to.ID
	}
	return s.st.Model
}

// steppedLocked is the step in force, if any. Caller holds s.mu.
func (s *Supervisor) steppedLocked() *tierStep {
	if t := s.step; t != nil && t.from.ID == s.st.Model && s.st.Mode != "external" {
		return t
	}
	return nil
}

// dropStep ends a step at the start of a load (every start tries the selection first) and says
// the model fact when one was in force: the state first, then the fact.
func (s *Supervisor) dropStep() {
	s.mu.Lock()
	was := s.steppedLocked() != nil
	s.step = nil
	s.mu.Unlock()
	if was {
		s.sayModel()
	}
}

// cardOnlyRoom is the room as the card alone would leave it: nothing else on it, no game's
// share, no block bound, and the machine's whole memory for whatever a model may lay in RAM.
// What passes (other programs, today's free memory) is left out, so a model that does not fit
// it is too big for this card and this machine, not for this moment.
func cardOnlyRoom(room fit.Room, ramTotal int) fit.Room {
	r := room
	r.Others, r.GameReserve, r.LargestBlock, r.CardHostFree, r.Unread = 0, 0, 0, 0, nil
	if ramTotal > r.RAMAvail {
		r.RAMAvail = ramTotal
	}
	return r
}

// stepTier is the load's answer to a selected catalog model that does not fit. When the model
// would not fit this card even with nothing else on it (cardOnlyRoom), it is too big for the
// card (a dense model is never split between card and RAM, so it never runs part on the card):
// the best installed catalog model below it that nothing marks opt-in, and that fits the room
// now, serves instead, and the status says so. When it would fit the card alone, others or
// today's free memory stand in the way, and the caller waits as before (ok false). A swap's
// target never steps (noStepFor): the swap fails at the fit and the starter comes back. opMu is
// held.
func (s *Supervisor) stepTier(ctx context.Context, lc loadCtx, room fit.Room, w fit.Wants) (loadCtx, fit.Room, fit.Wants, fit.Shape, bool) {
	sel := lc.in.model
	s.mu.Lock()
	selected, noStep := s.st.Model, s.noStepFor
	s.mu.Unlock()
	if sel.ID != selected || selected == customID || selected == noStep || room.NoGPU {
		return lc, room, w, fit.Shape{}, false
	}
	hw, err := s.cfg.Probe(ctx)
	if err != nil {
		return lc, room, w, fit.Shape{}, false
	}
	if _, err := fit.Choose(lc.fp, cardOnlyRoom(room, hw.RAMTotalMiB), w); err == nil {
		return lc, room, w, fit.Shape{}, false // it fits this card: the card or the memory is only busy
	}
	cat, err := s.catalog()
	if err != nil {
		return lc, room, w, fit.Shape{}, false
	}
	var below []catalog.Model
	for _, m := range cat.Models {
		if m.ID != sel.ID && !m.OptIn && m.Quality < sel.Quality && s.modelInstalled(m) {
			below = append(below, m)
		}
	}
	sort.SliceStable(below, func(i, j int) bool { return below[i].Quality > below[j].Quality })
	why := "it does not fit this card"
	if !sel.MoE {
		why = "it does not fit this card whole, and a dense model is never split between the card and RAM"
	}
	for _, m := range below {
		step := &tierStep{from: sel, to: m, note: fmt.Sprintf("%s is selected, but %s: %s serves instead.", sel.Label(), why, m.Label())}
		s.mu.Lock()
		s.step = step
		s.mu.Unlock()
		if in, ok := s.resolveInstall(); ok && in.model.ID == m.ID {
			s.planVision(in) // the projector is the served tier's (run/vision.go), planned before its room is read
		}
		lc2, room2, w2, err := s.loadContext(ctx)
		if err != nil || lc2.in.model.ID != m.ID {
			continue
		}
		shape, err := fit.Choose(lc2.fp, room2, w2)
		if err != nil && s.visionTrying() { // text first: this tier without its projector before the next tier
			room2 = withoutProjectorRoom(room2, s.visionRoomMiB())
			s.dropVision("the model does not fit this card with its vision projector: serving text only")
			shape, err = fit.Choose(lc2.fp, room2, w2)
		}
		if err != nil {
			continue
		}
		s.logf("tier step: %s", step.note)
		s.sayModel() // the step stands (state first, then the fact)
		return lc2, room2, w2, shape, true
	}
	s.mu.Lock()
	s.step = nil
	s.mu.Unlock()
	return lc, room, w, fit.Shape{}, false
}

// selectServed answers a person who selects the tier that already serves under a step: the
// selection moves to it, the step ends, and nothing restarts. ok false: no such step.
func (s *Supervisor) selectServed(model string) bool {
	s.mu.Lock()
	t := s.steppedLocked()
	if t == nil || t.to.ID != model || s.st.Mode != "local" {
		s.mu.Unlock()
		return false
	}
	s.st.Model, s.step = model, nil
	s.mu.Unlock()
	s.logf("selected %s, which already serves: the step from %s ends, nothing restarts", model, t.from.ID)
	s.save() // saves state.json and says the model fact
	return true
}

// tierStatusLocked lays a step over the status: the model, its names and the note are the
// served tier's, flagged as a fallback. Caller holds s.mu.
func (s *Supervisor) tierStatusLocked(st *Status) {
	t := s.steppedLocked()
	if t == nil {
		return
	}
	st.Model, st.ModelName, st.ModelCodename, st.ModelCodenameFull = t.to.ID, t.to.Name, t.to.Codename, t.to.CodenameFull
	st.ModelFallback, st.ModelNote = true, t.note
}
