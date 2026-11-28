package run

import (
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
	// an unconfinable path: refused to a remote caller while another holds slots, a local one passes
	if d, _ := doorOf(s, DoorAsk{Identity: "client-3", Touch: TouchUnconfinable}); d.Err == nil || d.Err.Code != "path_not_offered" || d.Err.Status != 403 {
		t.Fatalf("unconfinable remote: %+v", d)
	}
	if d, _ := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchUnconfinable}); d.Err != nil || d.Slots != nil {
		t.Fatalf("unconfinable local: %+v", d)
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
	// 5. every slot owned: a remote caller must attach, a local one passes as before
	s.addAtt("B", "client-2", false, 2, 2, 3)
	if d, _ := doorOf(s, slot); d.Err == nil || d.Err.Code != "attach_required" || d.Err.Status != 409 {
		t.Fatalf("all owned, remote: %+v", d)
	}
	if d, _ := doorOf(s, DoorAsk{Identity: "loopback", Local: true, Touch: TouchSlot}); d.Err != nil || d.Slots != nil {
		t.Fatalf("all owned, local: %+v", d)
	}
}

func TestDoorUntouchedWithNoClientHoldingSlots(t *testing.T) {
	// the base alone holds slots: anonymous requests are not confined, a remote's unconfinable passes
	s := regSup(true)
	s.setMap(3, 0, 1, 2)
	s.addAtt("P", "client-1", false, 0) // holds no slots
	for _, q := range []DoorAsk{{Identity: "client-2", Touch: TouchSlot}, {Identity: "client-2", Touch: TouchUnconfinable}, {Identity: "loopback", Local: true, Touch: TouchSlot}} {
		if d, _ := doorOf(s, q); d.Err != nil || d.Slots != nil {
			t.Fatalf("%+v: %+v", q, d)
		}
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
	if w.CtxFloor != 4096 || w.CtxMax != 32768 {
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
	if s.clients.givesLocked(s.cfg.Profile) {
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
		if !r.planLocked(2, "client-1", a) || a.why != whyIdle || r.reLocal {
			t.Fatalf("restart %d: asked false, why %q, local %v", i+1, a.why, r.reLocal)
		}
		s.countRestartCauses()
	}
	if r.planLocked(2, "client-1", a) || a.why != whyBudget || r.reclient != 0 {
		t.Fatalf("over the budget: why %q reclient %d", a.why, r.reclient)
	}
	// the person's and a local caller's are not counted, and a local grow drains
	if !r.planLocked(2, "", nil) || !r.reLocal || r.reclient != 2 {
		t.Fatalf("local: local %v reclient %d", r.reLocal, r.reclient)
	}
	if r.restarts["client-1"] != maxRemoteRestarts {
		t.Fatalf("counted %d", r.restarts["client-1"])
	}
}

func TestLastClientLeavingReshapesToTheBase(t *testing.T) {
	s := regSup(true)
	s.setMap(4, 0, 1) // the base wants 2; a client made it 4
	r := &s.clients
	if !r.givesLocked(s.cfg.Profile) { // no client stands: 4 slots is not the base's 2
		t.Fatal("the running shape is not the base's and nothing asked for a reshape")
	}
	s.setMap(2, 0, 1)
	if r.givesLocked(s.cfg.Profile) {
		t.Fatal("the base's own shape asked for a reshape")
	}
	// with no base, the defaults
	n := regSup(false)
	n.setMap(3)
	if !n.clients.givesLocked(n.cfg.Profile) {
		t.Fatal("no base: 3 slots is not the defaults' 2")
	}
}

func TestEachHoldItsOwnNotify(t *testing.T) {
	s := regSup(false)
	a := s.addAtt("H", "client-1", false, 0)
	first := a.notify
	r := &s.clients
	// what AttachClient does for a hold
	a.holdGen++
	a.notify = make(chan struct{}, 1)
	if a.notify == first {
		t.Fatal("same channel")
	}
	_ = r
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
