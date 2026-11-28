package run

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/CharlesMod/easyEngine/core/fit"
)

// The door's rules and the registry's bounds, on a registry built by hand (no server runs).

func regSup(base bool) *Supervisor {
	s := &Supervisor{cfg: Config{ProfileGiven: base, Profile: Profile{Slots: 2}}}
	s.cfg.Profile.CtxPerSlot.Max, s.cfg.Profile.CtxPerSlot.Floor = 8192, 4096
	s.clientsInit()
	return s
}

// add registers an attachment with slots given in the running map.
func (s *Supervisor) addAtt(id, identity string, local bool, want int, slots ...int) *attachment {
	r := &s.clients
	r.seq++
	a := &attachment{id: id, name: id, identity: identity, key: identity + "/" + id, local: local, wanting: want > 0,
		seq: r.seq, notify: make(chan struct{}, 1), slots: slots}
	a.prof.Slots, a.prof.MinSlots = want, min(want, 1)
	a.prof.CtxPerSlot.Max, a.prof.CtxPerSlot.Floor = 8192, 4096
	r.list = append(r.list, a)
	if r.run != nil {
		for _, e := range slots {
			r.run.owner[e], r.run.keys[e] = a.id, a.key
		}
	}
	return a
}

func (s *Supervisor) setMap(n int, baseSlots ...int) {
	r := &s.clients
	r.run = &runMap{owner: make([]string, n), keys: make([]string, n), want: make([]int, n), ctx: 8192, opts: map[string]string{}}
	for _, a := range r.list {
		if a.base {
			a.slots = baseSlots
			for _, e := range baseSlots {
				r.run.owner[e], r.run.keys[e] = a.id, a.key
			}
			r.run.base = len(baseSlots) > 0
		}
	}
}

func doorOf(s *Supervisor, q DoorAsk) (ClientDoor, bool) { return s.clientDoorLocked(q) }

func TestDoorRules(t *testing.T) {
	// a base (0,1), remote client-1 with A (2,3), remote client-2 with B (4)
	s := regSup(true)
	s.setMap(6, 0, 1)
	s.addAtt("A", "client-1", false, 2, 2, 3)
	s.addAtt("B", "client-2", false, 1, 4)
	slot := func(q DoorAsk) DoorAsk { q.Touch = TouchSlot; return q }

	// 1. the header names an attachment of the caller: its own range, its numbering
	if d, _ := doorOf(s, slot(DoorAsk{Att: "A", Identity: "client-1"})); d.Err != nil || !d.Own || len(d.Slots) != 2 || d.Slots[0] != 2 {
		t.Fatalf("own header: %+v", d)
	}
	// ... of another identity: not_yours, also for a local caller naming a client's attachment
	for _, q := range []DoorAsk{{Att: "A", Identity: "client-2"}, {Att: "A", Identity: "loopback", Local: true}} {
		if d, _ := doorOf(s, slot(q)); d.Err == nil || d.Err.Code != "not_yours" || d.Err.Status != 403 {
			t.Fatalf("%+v: %+v", q, d)
		}
	}
	// a local caller may name the base
	if d, _ := doorOf(s, slot(DoorAsk{Att: baseID, Identity: "loopback", Local: true})); d.Err != nil {
		t.Fatalf("local naming the base: %+v", d.Err)
	}
	// 2. no header, one attachment with slots: that one
	if d, _ := doorOf(s, slot(DoorAsk{Identity: "client-2"})); d.Err != nil || !d.Own || len(d.Slots) != 1 || d.Slots[0] != 4 {
		t.Fatalf("no header, one own: %+v", d)
	}
	// ... two: client_required
	s.addAtt("A2", "client-1", false, 1)
	if d, _ := doorOf(s, slot(DoorAsk{Identity: "client-1"})); d.Err == nil || d.Err.Code != "client_required" || d.Err.Status != 400 {
		t.Fatalf("no header, two own: %+v", d)
	}
	// 3. none of its own with slots (a pid-only attachment is none): the commons, engine numbers
	s.addAtt("P", "client-3", false, 0)
	for _, q := range []DoorAsk{{Identity: "client-3"}, {Att: "P", Identity: "client-3"}, {Identity: "loopback", Local: true}} {
		if d, _ := doorOf(s, slot(q)); d.Err != nil || d.Own || len(d.Slots) != 2 || d.Slots[0] != 0 {
			t.Fatalf("commons for %+v: %+v", q, d)
		}
	}
	// a path that touches no slot is only attributed
	if d, _ := doorOf(s, DoorAsk{Identity: "client-3"}); d.Err != nil || d.Slots != nil {
		t.Fatalf("free path: %+v", d)
	}
	// an unconfinable path while attachments hold ranges: refused to every caller (the invariant:
	// no caller is run unconfined), this machine's and the owner of a range alike
	for _, q := range []DoorAsk{{Identity: "client-3"}, {Identity: "client-1"}, {Identity: "loopback", Local: true}, {Identity: "master@10.0.0.2"}} {
		q.Touch = TouchUnconfinable
		if d, _ := doorOf(s, q); d.Err == nil || d.Err.Code != "path_not_offered" || d.Err.Status != 403 {
			t.Fatalf("unconfinable %+v: %+v", q, d)
		}
	}
}

func TestDoorRulesWithNoBase(t *testing.T) {
	s := regSup(false)
	s.setMap(4)
	s.addAtt("A", "client-1", false, 2, 0, 1)
	slot := DoorAsk{Identity: "client-3", Touch: TouchSlot}
	// 4. no base: the slots no attachment owns
	if d, _ := doorOf(s, slot); d.Err != nil || d.Own || len(d.Slots) != 2 || d.Slots[0] != 2 {
		t.Fatalf("unowned: %+v", d)
	}
	// 5. every slot owned: a remote caller must attach, a local one is held while the engine
	// reshapes to give the commons a range
	s.addAtt("B", "client-2", false, 2, 2, 3)
	if d, _ := doorOf(s, slot); d.Err == nil || d.Err.Code != "attach_required" || d.Err.Status != 409 {
		t.Fatalf("all owned, remote: %+v", d)
	}
	if d, p := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchSlot}); d.Err != nil || d.Slots != nil || !p || !s.clients.commons {
		t.Fatalf("all owned, local: %+v pending %v", d, p)
	}
}

func TestDoorWithOnlyTheCommons(t *testing.T) {
	// the base alone holds slots (0..2), P holds none: a trusted caller is not confined; a remote
	// one is confined to the commons under its own tag, and refused what cannot be confined
	s := regSup(true)
	s.setMap(3, 0, 1, 2)
	s.addAtt("P", "client-1", false, 0) // holds no slots
	if d, p := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchSlot}); d.Err != nil || d.Slots != nil || p {
		t.Fatalf("local: %+v", d)
	}
	if d, p := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchUnconfinable}); d.Err != nil || p {
		t.Fatalf("local unconfinable: %+v", d)
	}
	d, _ := doorOf(s, DoorAsk{Identity: "client-2", Touch: TouchSlot})
	if d.Err != nil || len(d.Slots) != 3 || d.Own || d.Tag != (SlotTag{Who: "anon:client-2"}) {
		t.Fatalf("remote: %+v", d)
	}
	if d, _ := doorOf(s, DoorAsk{Identity: "client-2", Touch: TouchUnconfinable}); d.Err == nil || d.Err.Code != "path_not_offered" {
		t.Fatalf("remote unconfinable: %+v", d)
	}
	// with the RAM prompt cache on, an untrusted caller is never served in the commons
	s.clients.run.ramCache = true
	if d, _ := doorOf(s, DoorAsk{Identity: "client-2", Touch: TouchSlot}); d.Err == nil || d.Err.Code != "attach_required" {
		t.Fatalf("remote, RAM cache on: %+v", d)
	}
}

func TestTakeSlotLeastRecentlyUsedAndAtomic(t *testing.T) {
	s := regSup(false)
	slots := []int{3, 4, 5}
	e1, r1 := s.TakeSlot(slots, -1)
	e2, r2 := s.TakeSlot(slots, -1)
	if e1 != 3 || e2 != 4 {
		t.Fatalf("first picks %d %d", e1, e2)
	}
	r1()
	r1()                                       // a release is once
	if e, r := s.TakeSlot(slots, -1); e != 5 { // free and never used
		t.Fatalf("third pick %d", e)
	} else {
		r()
	}
	if e, r := s.TakeSlot(slots, -1); e != 3 { // free and least recently used (4 is busy)
		t.Fatalf("fourth pick %d", e)
	} else {
		r()
	}
	r2()
	// a named slot is marked too, and moves to most recently used
	_, rn := s.TakeSlot(slots, 4)
	if e, r := s.TakeSlot(slots, -1); e != 5 {
		t.Fatalf("after a named take: %d", e)
	} else {
		r()
	}
	rn()
	// many at once over as many free slots never share one
	many := []int{0, 1, 2, 3, 4, 5, 6, 7}
	var mu sync.Mutex
	got := map[int]int{}
	var wg sync.WaitGroup
	var rels []func()
	for range many {
		wg.Add(1)
		go func() {
			defer wg.Done()
			e, r := s.TakeSlot(many, -1)
			mu.Lock()
			got[e]++
			rels = append(rels, r)
			mu.Unlock()
		}()
	}
	wg.Wait()
	for e, n := range got {
		if n != 1 {
			t.Fatalf("slot %d taken %d times at once: %v", e, n, got)
		}
	}
	for _, r := range rels {
		r()
	}
}

func TestRemoteFloorIsClamped(t *testing.T) {
	s := regSup(true) // the base's floor 4096
	r := s.addAtt("R", "client-1", false, 1)
	r.prof.CtxPerSlot.Max, r.prof.CtxPerSlot.Floor = 32768, 16384
	w, _ := s.mergedWants()
	if w.CtxFloor != 4096 || w.CtxMax != 8192 { // both clamped to the owner's
		t.Fatalf("merged: floor %d max %d", w.CtxFloor, w.CtxMax)
	}
	l := s.addAtt("L", "loopback", true, 1)
	l.prof.CtxPerSlot.Max, l.prof.CtxPerSlot.Floor = 12288, 12288
	if w, _ = s.mergedWants(); w.CtxFloor != 12288 {
		t.Fatalf("a local floor raises it: %d", w.CtxFloor)
	}
	// a shape below the remote's own floor leaves it unserved, said why, and never a reshape's cause
	if !s.floorShort(12288) {
		t.Fatal("the remote below its floor was not left out")
	}
	s.startMap(fit.Shape{Slots: 4, CtxPerSlot: 12288}, launchSet{seq: s.clients.seq})
	if len(r.slots) != 0 || !r.short || !strings.Contains(r.unserved, "below its floor (16384)") {
		t.Fatalf("remote: slots %v short %v unserved %q", r.slots, r.short, r.unserved)
	}
	if len(l.slots) != 1 {
		t.Fatalf("local slots %v", l.slots)
	}
	if s.clients.givesLocked() {
		t.Fatal("the short remote is a cause of a reshape")
	}
}

func TestDropOrderRemoteThenLaterNeverBase(t *testing.T) {
	s := regSup(true)
	s.addAtt("L1", "loopback", true, 1)
	s.addAtt("R1", "client-1", false, 1)
	s.addAtt("L2", "loopback", true, 1)
	s.addAtt("R2", "client-2", false, 1)
	var order []string
	for {
		id, ok := s.dropLatest()
		if !ok {
			break
		}
		order = append(order, id)
	}
	if strings.Join(order, ",") != "R2,R1,L2,L1" {
		t.Fatalf("drop order %v", order)
	}
}

func TestRestartBudget(t *testing.T) {
	s := regSup(true)
	s.setMap(2, 0, 1)
	a := s.addAtt("R", "client-1", false, 1)
	r := &s.clients
	for i := 0; i < maxRemoteRestarts; i++ {
		if !r.planLocked(2, a.key, "client-1", false, a) || a.why != whyIdle || r.reLocal {
			t.Fatalf("restart %d: asked false, why %q, local %v", i+1, a.why, r.reLocal)
		}
		s.countRestartCauses()
	}
	if r.planLocked(2, a.key, "client-1", false, a) || a.why != whyBudget || r.reclient != 0 {
		t.Fatalf("over the budget: why %q reclient %d", a.why, r.reclient)
	}
	// the budget-spent client's requests are answered at once (503 no_room with the why), not held
	d, pending := doorOf(s, DoorAsk{Att: "R", Identity: "client-1", Touch: TouchSlot})
	if pending || d.Err == nil || d.Err.Code != "no_room" || !strings.Contains(d.Err.Msg, whyBudget) {
		t.Fatalf("budget spent at the door: pending %v %+v", pending, d.Err)
	}
	// a return (uncharged) still plans, and waits for idle; a local caller's grow drains
	if !r.planLocked(2, a.key, "", false, a) || r.reLocal || r.reclient != 2 {
		t.Fatalf("return: local %v reclient %d", r.reLocal, r.reclient)
	}
	if d, pending := doorOf(s, DoorAsk{Att: "R", Identity: "client-1", Touch: TouchSlot}); !pending || d.Err != nil {
		t.Fatalf("with a reshape coming it is held: %v %+v", pending, d.Err)
	}
	r.unplanLocked(a.key) // its key came back to its range
	if r.reclient != 0 {
		t.Fatalf("a cancelled plan stands: %d", r.reclient)
	}
	if !r.planLocked(2, commonsKey, "", true, nil) || !r.reLocal {
		t.Fatal("a local grow does not drain")
	}
	s.countRestartCauses() // an uncharged restart
	if r.restarts["client-1"] != maxRemoteRestarts || len(r.restarts) != 1 {
		t.Fatalf("counted %v", r.restarts)
	}
}

func TestBudgetAndCapKeys(t *testing.T) {
	for _, c := range []struct {
		id           string
		local        bool
		budget, caps string
	}{
		{"loopback", true, "", ""},
		{"client-1", false, "client-1", "client-1"},
		{"open@10.0.0.1", false, "open", "open"},
		{"open@10.0.0.2", false, "open", "open"}, // every --open address shares one
		{"master@10.0.0.1", false, "", "master@10.0.0.1"},
		{"engine", true, "", ""},
	} {
		if b, k := budgetKey(c.id, c.local), capKey(c.id, c.local); b != c.budget || k != c.caps {
			t.Errorf("%s: budget %q caps %q", c.id, b, k)
		}
	}
}

func TestRemoteCtxMaxIsClamped(t *testing.T) {
	s := regSup(true) // the base's max 8192 (also the default's)
	r := s.addAtt("R", "client-1", false, 1)
	r.prof.CtxPerSlot.Max = 131072
	if w, _ := s.mergedWants(); w.CtxMax != 8192 {
		t.Fatalf("a remote max raised the merge: %d", w.CtxMax)
	}
	l := s.addAtt("L", "loopback", true, 1)
	l.prof.CtxPerSlot.Max = 32768
	if w, _ := s.mergedWants(); w.CtxMax != 32768 {
		t.Fatalf("a local max is the ceiling: %d", w.CtxMax)
	}
}

func TestLastClientLeavingReshapesNothing(t *testing.T) {
	s := regSup(true)
	s.setMap(4, 0, 1) // the base wants 2; a client made it 4
	s.cfg.Profile.CtxPerSlot.Max = 8192
	r := &s.clients
	r.run.ctx = 4096
	r.stale = true
	// a local anonymous request the stale shape under-serves (4 slots of 4096 for wants of 2 x
	// 8192) is held and asks for a reshape that drains as a local grow
	if d, pending := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchSlot}); !pending || d.Err != nil || r.reclient != 2 || !r.reLocal {
		t.Fatalf("under-served: pending %v reclient %d local %v", pending, r.reclient, r.reLocal)
	}
	// a remote one is not (a remote anonymous caller never restarts the engine)
	r.unplanLocked(commonsKey)
	if _, pending := doorOf(s, DoorAsk{Identity: "client-9", Touch: TouchSlot}); pending || r.reclient != 0 {
		t.Fatal("a remote anonymous request reshaped")
	}
	// a shape that serves the base fully is not under-served
	r.run.ctx = 8192
	if _, pending := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchSlot}); pending {
		t.Fatal("held with the base served")
	}
}

func TestSlotMarkOfAnEarlierMap(t *testing.T) {
	s := regSup(false)
	s.setMap(2)
	_, rel := s.TakeSlot([]int{0, 1}, 0)
	s.clients.gen++ // a restart: a new map (startMap clears the marks)
	clear(s.clients.busy)
	_, rel2 := s.TakeSlot([]int{0, 1}, 0)
	rel() // the old map's release
	if s.clients.busy[0] != 1 {
		t.Fatalf("an earlier map's release lowered the new mark: %d", s.clients.busy[0])
	}
	rel2()
	if s.clients.busy[0] != 0 {
		t.Fatalf("busy %d", s.clients.busy[0])
	}
}

func TestFlagSpellings(t *testing.T) {
	for in, want := range map[string]string{"--Ctx_Size": "--ctx-size", "--lookup_cache_dynamic": "--lookup-cache-dynamic", "-lcd": "-lcd", "-C": "-C"} {
		if got := normFlag(in); got != want {
			t.Errorf("normFlag(%s) = %s", in, got)
		}
	}
	_, h := fixtureHelp(t)
	for _, args := range [][]string{{"-lcd", "f"}, {"-lcs", "f"}, {"--lookup-cache-dynamic", "f"}, {"--lookup_cache_dynamic", "f"},
		{"--slot_save_path", "/x"}, {"--log_file", "/x"}, {"--CTX_SIZE", "9"}} {
		if _, err := CheckExtraArgs(args, h); err == nil || err.Code != "flag_owned" {
			t.Errorf("%v: %v", args, err)
		}
		if _, err := CheckExtraArgs(args, nil); err == nil || err.Code != "flag_owned" {
			t.Errorf("%v with no help: %v", args, err)
		}
	}
}

func TestNoHelpNoExtraArgs(t *testing.T) {
	a := &attachment{id: "x", name: "x"}
	a.prof.Server = &ServerOpts{Ubatch: 256, ExtraArgs: []string{"--top-n-sigma", "1.5"}}
	_, _, reps := mergeOptions([]*attachment{a}, nil)
	rep := reps["x"]
	if len(rep.NotTaken) != 1 || rep.NotTaken[0].Option != "--top-n-sigma" || rep.NotTaken[0].Why != whyNoHelp {
		t.Fatalf("not taken: %+v", rep.NotTaken)
	}
	args, _, _ := launchOpts([]*attachment{a}, nil)
	for _, g := range args {
		if strings.HasPrefix(g[0], "--top") {
			t.Fatalf("launched with %v", args)
		}
	}
	// a help that could not be read is not kept: read again at the next launch
	bin := t.TempDir() + "/no-such-llama-server"
	if helpOf(bin, nil) != nil {
		t.Fatal("a help from nothing")
	}
	if _, kept := helpCache.Load(bin); kept {
		t.Fatal("a nil help was cached")
	}
}

// F2, the engine's order of a handover (TakeSlotFor): the newcomer waits while a request the door
// sent to the slot runs, the erase is sent only then, the label changes only on its success, a
// waiter whose client goes leaves the line, a held refusal is answered, and an idle slot of the
// range is taken over a busy one.
func TestTakeSlotForOrdersTheHandover(t *testing.T) {
	s := regSup(true)
	s.setMap(3, 0, 1, 2)
	m := s.clients.run
	m.tags = []SlotTag{{Who: "commons", Trusted: true}, {Who: "commons", Trusted: true}, {Who: "commons", Trusted: true}}
	all := []int{0, 1, 2}
	commons, anon := SlotTag{Who: "commons", Trusted: true}, SlotTag{Who: "anon:client-2"}
	_, relY, ce := s.TakeSlotFor(context.Background(), all, 1, commons, nil, nil)
	if ce != nil {
		t.Fatal(ce)
	}
	type res struct {
		e  int
		ce *ClientError
		r  func()
	}
	waiting := make(chan struct{}, 4)
	held := func() (func(), *ClientError) { waiting <- struct{}{}; return func() {}, nil }
	erased := make(chan int, 4)
	erase := func(e int) bool { erased <- e; return true }
	// a waiter whose client goes leaves the line
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan res, 1)
	go func() { e, r, ce := s.TakeSlotFor(ctx, all, 1, anon, erase, held); out <- res{e, ce, r} }()
	<-waiting
	if n := s.SlotsWaiting(); n != 1 {
		t.Fatalf("waiting %d", n)
	}
	cancel()
	if r := <-out; r.ce == nil || r.ce.Code != "client_gone" || s.SlotsWaiting() != 0 || m.tags[1] != commons {
		t.Fatalf("cancelled: %+v, waiting %d, tag %+v", r.ce, s.SlotsWaiting(), m.tags[1])
	}
	// X1 then X2 for busy slot 1: both wait; nothing is erased while Y runs
	x1, x2 := make(chan res, 1), make(chan res, 1)
	go func() {
		e, r, ce := s.TakeSlotFor(context.Background(), all, 1, anon, erase, held)
		x1 <- res{e, ce, r}
	}()
	<-waiting
	go func() {
		e, r, ce := s.TakeSlotFor(context.Background(), all, 1, anon, erase, held)
		x2 <- res{e, ce, r}
	}()
	<-waiting
	select {
	case e := <-erased:
		t.Fatalf("slot %d erased while Y runs on it", e)
	default:
	}
	if m.tags[1] != commons {
		t.Fatalf("labelled before the erase: %+v", m.tags[1])
	}
	relY()
	if e := <-erased; e != 1 {
		t.Fatalf("erased %d", e)
	}
	r1, r2 := <-x1, <-x2
	if r1.ce != nil || r2.ce != nil || r1.e != 1 || r2.e != 1 || m.tags[1] != anon {
		t.Fatalf("X1 %+v X2 %+v tag %+v", r1, r2, m.tags[1])
	}
	if len(erased) != 0 {
		t.Fatal("X2, the same owner after the handover, erased again")
	}
	r1.r()
	r2.r()
	// an idle slot of the range is taken over a busy one (0 and 2 idle, the commons'; 1 is X's)
	_, relB, _ := s.TakeSlotFor(context.Background(), all, 1, anon, erase, held)
	e, rel, ce := s.TakeSlotFor(context.Background(), all, -1, SlotTag{Who: "anon:client-3"}, erase, held)
	if ce != nil || e == 1 {
		t.Fatalf("idle pick: slot %d, %+v", e, ce)
	}
	<-erased
	rel()
	relB()
	// a held refusal is answered and leaves no waiter
	refuse := func() (func(), *ClientError) { return nil, cerr(429, "busy", "x") }
	_, relC, _ := s.TakeSlotFor(context.Background(), all, 1, anon, erase, nil)
	if _, _, ce := s.TakeSlotFor(context.Background(), all, 1, commons, erase, refuse); ce == nil || ce.Code != "busy" || s.SlotsWaiting() != 0 {
		t.Fatalf("held refusal: %+v, waiting %d", ce, s.SlotsWaiting())
	}
	relC()
	// a failed erase: answered 502, the slot labelled anyone's
	_, _, ce = s.TakeSlotFor(context.Background(), all, 1, commons, func(int) bool { return false }, nil)
	if ce == nil || ce.Code != "slot_not_cleared" || m.tags[1].Who != "?" {
		t.Fatalf("failed erase: %+v, tag %+v", ce, m.tags[1])
	}
}

// A local caller's ask for the commons that comes after a start's merge was taken is asked again
// at that start (a grow), never answered 503 no_room as if the card had no room for it.
func TestCommonsAskedAfterTheMergeIsAskedAgain(t *testing.T) {
	s := regSup(false)
	s.addAtt("A", "client-1", false, 2)
	s.mergedWants() // the start's merge: the commons not asked yet
	s.clients.commons = true
	s.startMap(fit.Shape{Slots: 2, CtxPerSlot: 8192}, launchSet{seq: 1 << 60})
	r := &s.clients
	if r.commonsShort {
		t.Fatal("the commons said short by a start whose merge did not ask for it")
	}
	if _, ok := r.plans[commonsKey]; !ok {
		t.Fatalf("the commons not asked again: plans %v", r.plans)
	}
	// a start whose merge did ask, and still gave it nothing: short
	s.mergedWants()
	s.startMap(fit.Shape{Slots: 2, CtxPerSlot: 8192}, launchSet{seq: 1 << 60})
	if !r.commonsShort {
		t.Fatal("a start that merged the commons and gave it nothing is not short")
	}
}

// A lost port is read only from a line that names this launch's port.
func TestPortLostReadsOnlyThisPort(t *testing.T) {
	for _, c := range []struct {
		tail string
		want bool
	}{
		{"srv start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 4242\n", true},
		{"listen tcp 127.0.0.1:4242: bind: address already in use\n", true},
		{"srv start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 42421\n", false},
		{"rpc: listen tcp 0.0.0.0:50052: bind: address already in use\nmain: model load failed\n", false},
		{"error: out of memory\n", false},
	} {
		if got := portLostIn(c.tail, 4242); got != c.want {
			t.Errorf("%q: %v", c.tail, got)
		}
	}
}
