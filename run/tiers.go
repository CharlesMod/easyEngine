package run

import (
	"context"
	"fmt"
	"sort"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
)

// tierStep is a selected catalog model the card cannot hold whole, served by a lower tier
// instead. It lives for one start (startLocked clears it and tries the selected model again), is
// never saved, and holds only while the selection is still from.
type tierStep struct {
	from, to catalog.Model
	note     string
}

// servedIDLocked is the catalog id that serves: the selected model, or the tier it stepped down
// to. Caller holds s.mu.
func (s *Supervisor) servedIDLocked() string {
	if t := s.step; t != nil && t.from.ID == s.st.Model && s.st.Mode != "external" {
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

// stepTier is the load's answer to a selected catalog model that does not fit. When the model
// would not fit this card even with nothing else on it, it is too big for the card (a dense
// model is never split between card and RAM, so it never runs part on the card): the best
// installed catalog model below it that nothing marks opt-in, and that fits the room now,
// serves instead, and the status says so. When it would fit an empty card, others hold the card,
// and the caller waits for the card as before (ok false). opMu is held.
func (s *Supervisor) stepTier(ctx context.Context, lc loadCtx, room fit.Room, w fit.Wants) (loadCtx, fit.Room, fit.Wants, fit.Shape, bool) {
	sel := lc.in.model
	s.mu.Lock()
	selected := s.st.Model
	s.mu.Unlock()
	if sel.ID != selected || selected == customID || room.NoGPU {
		return lc, room, w, fit.Shape{}, false
	}
	empty := room
	empty.Others, empty.GameReserve, empty.LargestBlock = 0, 0, 0
	if _, err := fit.Choose(lc.fp, empty, w); err == nil {
		return lc, room, w, fit.Shape{}, false // it fits the card: the card is only busy
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
		lc2, room2, w2, err := s.loadContext(ctx)
		if err != nil || lc2.in.model.ID != m.ID {
			continue
		}
		shape, err := fit.Choose(lc2.fp, room2, w2)
		if err != nil {
			continue
		}
		s.logf("tier step: %s", step.note)
		return lc2, room2, w2, shape, true
	}
	s.mu.Lock()
	s.step = nil
	s.mu.Unlock()
	return lc, room, w, fit.Shape{}, false
}

// tierStatusLocked lays a step over the status: the model, its names and the note are the
// served tier's, flagged as a fallback. Caller holds s.mu.
func (s *Supervisor) tierStatusLocked(st *Status) {
	t := s.steppedLocked()
	if t == nil {
		return
	}
	st.Model, st.ModelName, st.ModelCodename = t.to.ID, t.to.Name, t.to.Codename
	st.ModelFallback, st.ModelNote = true, t.note
}
