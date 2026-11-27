//go:build !windows

package run_test

import (
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// Clients (run/clients.go): attachments with profiles merge into one running mind; the slot map
// changes only at a start; a leaving attachment's slots stay vacant. Presence of a pid attachment
// is a child process (cat on a pipe the test holds): closing its stdin ends it, and that end is
// the event the engine watches.

// clKid is a child process standing for a client's pid.
type clKid struct {
	cmd  *exec.Cmd
	in   io.WriteCloser
	once sync.Once
}

func newKid(t *testing.T) *clKid {
	t.Helper()
	cmd := exec.Command("cat")
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	k := &clKid{cmd: cmd, in: in}
	t.Cleanup(k.end)
	return k
}

func (k *clKid) pid() int { return k.cmd.Process.Pid }

// end closes the child's stdin (cat ends) and reaps it.
func (k *clKid) end() {
	k.once.Do(func() {
		k.in.Close()
		k.cmd.Wait()
	})
}

// clProf is a profile that wants n slots at a context of at most max and at least floor.
func clProf(n, max, floor int) *run.Profile {
	p := &run.Profile{Slots: n, MinSlots: 1}
	p.CtxPerSlot.Max, p.CtxPerSlot.Floor = max, floor
	return p
}

func clReq(name string, k *clKid, hold bool, p *run.Profile) run.AttachRequest {
	r := run.AttachRequest{Name: name, Hold: hold, Profile: p, Identity: "loopback", Local: true}
	if k != nil {
		r.Pid = k.pid()
	}
	return r
}

func clAttach(t *testing.T, s *run.Supervisor, req run.AttachRequest) (run.AttachView, *run.AttachHold) {
	t.Helper()
	v, h, e := s.AttachClient(req)
	if e != nil {
		t.Fatalf("attach %q refused: %d %s %s", req.Name, e.Status, e.Code, e.Msg)
	}
	return v, h
}

// clRefused attaches and wants the refusal.
func clRefused(t *testing.T, s *run.Supervisor, req run.AttachRequest) *run.ClientError {
	t.Helper()
	_, h, e := s.AttachClient(req)
	if e == nil {
		t.Fatalf("attach %q was taken (hold %v), want a refusal", req.Name, h != nil)
	}
	return e
}

func clView(t *testing.T, s *run.Supervisor, id string) run.AttachView {
	t.Helper()
	v, ok := s.ClientView(id)
	if !ok {
		t.Fatalf("no attachment %s", id)
	}
	return v
}

func clFirst(v run.AttachView) int {
	if v.Slots.First == nil {
		return -1
	}
	return *v.Slots.First
}

// clFactGone: the attachment's fact no longer stands.
func clFactGone(s *run.Supervisor, id string) bool {
	_, ok := s.Hub().Fact("attachment", id)
	return !ok
}

// clStarted waits for the fake to have been started n times and the mind ready again.
func clStarted(t *testing.T, e *testkit.Env, s *run.Supervisor, n int) {
	t.Helper()
	testkit.WaitFor(t, 20*time.Second, "the start that makes the new shape", func() bool {
		return len(e.Starts()) >= n && s.Phase() == "ready"
	})
}

// clQuiet is what a test of "nothing happened" can honestly say: it makes the reactor look once
// now (React) after the change, then a marker comes after that look.
func clQuiet(t *testing.T, e *testkit.Env, s *run.Supervisor, starts int) {
	t.Helper()
	s.React()
	testkit.Mark(t, s, "after the reactor's look")
	if n := len(e.Starts()); n != starts {
		t.Fatalf("starts = %d, want %d: %v", n, starts, e.Starts())
	}
	if s.Phase() != "ready" {
		t.Fatalf("phase = %s", s.Phase())
	}
}

func clWantsFact(t *testing.T, s *run.Supervisor) run.WantsInfo {
	t.Helper()
	f, ok := s.Hub().Fact("wants", "")
	if !ok {
		t.Fatal("no wants fact")
	}
	var w run.WantsInfo
	if err := json.Unmarshal(f.Body, &w); err != nil {
		t.Fatal(err)
	}
	return w
}

// clTwo attaches A (pid, 2 slots) then B (hold, 3 slots) at the given context max and waits for
// the reshape to five slots. It answers A, B, A's child and B's hold.
func clTwo(t *testing.T, e *testkit.Env, s *run.Supervisor, max int) (a, b run.AttachView, ka *clKid, hb *run.AttachHold) {
	t.Helper()
	ka = newKid(t)
	a, _ = clAttach(t, s, clReq("a", ka, false, clProf(2, max, 4096)))
	clQuiet(t, e, s, 1) // served from the running mind's free anonymous slots
	b, hb = clAttach(t, s, clReq("b", nil, true, clProf(3, max, 4096)))
	clStarted(t, e, s, 2)
	testkit.WaitFor(t, 15*time.Second, "b served", func() bool { return clView(t, s, b.Attachment).Slots.Given == 3 })
	return a, b, ka, hb
}

// 1. Two attachments merge: A is served from the free slots, B makes the mind reshape to 5; A's
// leave leaves its range vacant.
func TestClientMergeReshapeAndPidLeave(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	ka := newKid(t)
	a, _ := clAttach(t, s, clReq("a", ka, false, clProf(2, 8192, 4096)))
	if a.Slots.Given != 2 || clFirst(a) != 0 || a.Slots.Pending || a.Key != "loopback/a" || a.Presence != "pid" {
		t.Fatalf("a = %+v", a)
	}
	clQuiet(t, e, s, 1)
	b, hb := clAttach(t, s, clReq("b", nil, true, clProf(3, 8192, 4096)))
	if hb == nil || b.Presence != "hold" || b.Slots.Given != 0 || !b.Slots.Pending {
		t.Fatalf("b at attach = %+v", b)
	}
	clStarted(t, e, s, 2)
	testkit.WaitFor(t, 15*time.Second, "b served", func() bool { return clView(t, s, b.Attachment).Slots.Given == 3 })
	starts := e.Starts()
	if len(starts) != 2 || testkit.Field(starts[1], "np") != "5" {
		t.Fatalf("starts = %v", starts)
	}
	if w := clWantsFact(t, s); w.Slots != 5 || len(w.From) != 2 || w.From[0] != a.Attachment || w.From[1] != b.Attachment {
		t.Fatalf("wants fact = %+v", w)
	}
	av, bv := clView(t, s, a.Attachment), clView(t, s, b.Attachment)
	if clFirst(av) != 0 || av.Slots.Given != 2 || clFirst(bv) != 2 || bv.Slots.Given != 3 {
		t.Fatalf("a = %+v b = %+v", av.Slots, bv.Slots)
	}
	if len(s.Status().Attachments) != 2 {
		t.Fatalf("status attachments = %+v", s.Status().Attachments)
	}
	// A's process ends
	ka.end()
	testkit.WaitFor(t, 15*time.Second, "a gone", func() bool { return clFactGone(s, a.Attachment) })
	if _, ok := s.Hub().Fact("attachment", b.Attachment); !ok {
		t.Fatal("b's fact went with a's")
	}
	if s.Attached() != 1 {
		t.Fatalf("attached = %d", s.Attached())
	}
	w := s.Status().Wants
	if w.Slots != 3 || len(w.Vacant) != 1 || w.Vacant[0] != (run.VacantInfo{First: 0, Count: 2, Key: "loopback/a"}) {
		t.Fatalf("wants = %+v", w)
	}
}

// 2a. A leave that gives nobody anything makes no reshape.
func TestClientLeaveGivingNothingMakesNoReshape(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	a, _, ka, _ := clTwo(t, e, s, 8192)
	ctx := s.Status().Shape.CtxPerSlot
	if ctx != 8192 {
		t.Skipf("the fit gives %d per slot, not the max 8192 this test needs", ctx)
	}
	ka.end()
	testkit.WaitFor(t, 15*time.Second, "a gone", func() bool { return clFactGone(s, a.Attachment) })
	clQuiet(t, e, s, 2)
	if w := s.Status().Wants; len(w.Vacant) != 1 || w.Vacant[0].Count != 2 {
		t.Fatalf("wants = %+v", w)
	}
}

// 2b. A leave that gives someone more context reshapes when idle: B's slots become 0..2.
func TestClientLeaveGivingSomethingReshapes(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	a, b, ka, _ := clTwo(t, e, s, 65536)
	if c := s.Status().Shape.CtxPerSlot; c >= 65536 {
		t.Skipf("five slots already get %d per slot, nothing for a leave to give", c)
	}
	before := s.Status().Shape.CtxPerSlot
	ka.end()
	testkit.WaitFor(t, 15*time.Second, "a gone", func() bool { return clFactGone(s, a.Attachment) })
	clStarted(t, e, s, 3)
	testkit.WaitFor(t, 15*time.Second, "b renumbered", func() bool { return clFirst(clView(t, s, b.Attachment)) == 0 })
	bv := clView(t, s, b.Attachment)
	if bv.Slots.Given != 3 || bv.Ctx.Given <= before {
		t.Fatalf("b = %+v (ctx before %d)", bv, before)
	}
	if n := testkit.Field(e.Starts()[2], "np"); n != "3" {
		t.Fatalf("third start np = %s", n)
	}
	if w := s.Status().Wants; len(w.Vacant) != 0 {
		t.Fatalf("vacant after the reshape: %+v", w.Vacant)
	}
}

// clCtxSaid is the context happening with the client fields.
type clCtxSaid struct {
	What       string `json:"what"`
	Slot       int    `json:"slot"`
	Tokens     int    `json:"tokens"`
	Why        string `json:"why"`
	Client     string `json:"client"`
	ClientSlot *int   `json:"client_slot"`
}

// 3. Contexts follow their client across a reshape that renumbers the slots.
func TestClientContextsAcrossRenumberingReshape(t *testing.T) {
	e := testkit.New(t)
	e.WatchFile(filepath.Join(e.Ctl, "slots.log"))
	s := ready(t, e)
	ka := newKid(t)
	a, _ := clAttach(t, s, clReq("a", ka, false, clProf(3, 65536, 4096)))
	b, _ := clAttach(t, s, clReq("b", nil, true, clProf(2, 65536, 4096)))
	clStarted(t, e, s, 2)
	testkit.WaitFor(t, 15*time.Second, "b served", func() bool { return clView(t, s, b.Attachment).Slots.Given == 2 })
	if c := s.Status().Shape.CtxPerSlot; c >= 65536 {
		t.Skipf("five slots already get %d per slot", c)
	}
	if f := clFirst(clView(t, s, b.Attachment)); f != 3 {
		t.Fatalf("b first = %d, want 3", f)
	}
	ask(t, s, 3, 400) // b's slot 0
	ask(t, s, 0, 200) // a's slot 0
	from := testkit.Mark(t, s, "before a leaves")
	ka.end()
	testkit.WaitFor(t, 15*time.Second, "a gone", func() bool { return clFactGone(s, a.Attachment) })
	clStarted(t, e, s, 3)
	testkit.WaitFor(t, 15*time.Second, "b renumbered", func() bool { return clFirst(clView(t, s, b.Attachment)) == 0 })
	var l []clCtxSaid
	for _, ev := range testkit.Said(t, s, from, 0, "context") {
		var c clCtxSaid
		if err := json.Unmarshal(ev.Body, &c); err != nil {
			t.Fatal(err)
		}
		l = append(l, c)
	}
	var saved3, back0, goneA bool
	for _, c := range l {
		switch {
		case c.What == "saved" && c.Slot == 3 && c.Client == "loopback/b" && c.ClientSlot != nil && *c.ClientSlot == 0:
			saved3 = true
		case c.What == "laid_back" && c.Slot == 0 && c.Client == "loopback/b" && c.ClientSlot != nil && *c.ClientSlot == 0:
			back0 = true
			if c.Tokens != savedTokens(400) {
				t.Fatalf("laid back %d tokens, want %d", c.Tokens, savedTokens(400))
			}
		case c.What == "let_go" && c.Client == "loopback/a" && strings.Contains(c.Why, "the client has left"):
			goneA = true
		}
	}
	if !saved3 || !back0 || !goneA {
		t.Fatalf("saved(3,b,0)=%v laid_back(0,b,0)=%v let_go(a)=%v; said %+v", saved3, back0, goneA, l)
	}
	// a prompt that extends the saved one (the fake counts tokens: ceil(n/4)) finds it in the cache
	if c, _ := ask(t, s, 0, 440); c != savedTokens(400) {
		t.Fatalf("b's slot 0 cache_n = %d, want %d", c, savedTokens(400))
	}
}

// 4. A held attachment that leaves keeps its slots vacant, and the same key takes them back; the
// same key while it stands replaces its presence.
func TestClientResumeAndReplace(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	req := clReq("b", nil, true, clProf(2, 8192, 4096))
	b, hb := clAttach(t, s, req)
	if clFirst(b) != 0 || b.Slots.Given != 2 || b.Resumed || b.Replaced {
		t.Fatalf("b = %+v", b)
	}
	first := clFirst(b)
	ask(t, s, first, 400)
	s.LeaveHold(hb.ID, hb.Gen)
	if !clFactGone(s, b.Attachment) {
		t.Fatal("b's fact stands after its hold ended")
	}
	if w := s.Status().Wants; len(w.Vacant) != 1 || w.Vacant[0] != (run.VacantInfo{First: 0, Count: 2, Key: "loopback/b"}) {
		t.Fatalf("wants = %+v", w)
	}
	clQuiet(t, e, s, 1)
	b2, hb2 := clAttach(t, s, req)
	if !b2.Resumed || clFirst(b2) != first || b2.Slots.Given != 2 {
		t.Fatalf("b2 = %+v", b2)
	}
	clQuiet(t, e, s, 1)
	if c, _ := ask(t, s, first, 440); c != savedTokens(400) {
		t.Fatalf("cache_n = %d after the resume, want %d", c, savedTokens(400))
	}
	// the same key while it stands
	b3, hb3 := clAttach(t, s, req)
	if !b3.Replaced || b3.Attachment != b2.Attachment || clFirst(b3) != first {
		t.Fatalf("b3 = %+v", b3)
	}
	select {
	case w := <-hb2.End:
		if w != "replaced" {
			t.Fatalf("old hold ended with %q", w)
		}
	default:
		t.Fatal("the old held stream was not told it was replaced")
	}
	// the old hold's connection ending later changes nothing: its gen is stale
	s.LeaveHold(hb2.ID, hb2.Gen)
	if clFactGone(s, b3.Attachment) || s.Attached() != 1 {
		t.Fatal("the stale hold's end took the replacing attachment away")
	}
	clQuiet(t, e, s, 1)
	_ = hb3
}

// 5. A new key is served from another's vacant range without a start.
func TestClientServedFromVacantRange(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	ka := newKid(t)
	a, _ := clAttach(t, s, clReq("a", ka, false, clProf(2, 8192, 4096)))
	if clFirst(a) != 0 {
		t.Fatalf("a = %+v", a)
	}
	ka.end()
	testkit.WaitFor(t, 15*time.Second, "a gone", func() bool { return clFactGone(s, a.Attachment) })
	clQuiet(t, e, s, 1)
	kc := newKid(t)
	c, _ := clAttach(t, s, clReq("c", kc, false, clProf(2, 8192, 4096)))
	if c.Resumed || c.Slots.Given != 2 || clFirst(c) != 0 {
		t.Fatalf("c = %+v", c)
	}
	clQuiet(t, e, s, 1)
	if w := s.Status().Wants; len(w.Vacant) != 0 {
		t.Fatalf("vacant = %+v", w.Vacant)
	}
}

// 6. An attachment whose floor does not fit the card is dropped from the merge: unserved.
func TestClientUnservedWhenItDoesNotFit(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	ka := newKid(t)
	a, _ := clAttach(t, s, clReq("a", ka, false, clProf(1, 8192, 4096)))
	b, _ := clAttach(t, s, clReq("b", nil, true, clProf(1, 1048576, 1048576)))
	clStarted(t, e, s, 2)
	testkit.WaitFor(t, 15*time.Second, "b said unserved", func() bool { return clView(t, s, b.Attachment).Slots.Unserved != "" })
	bv, av := clView(t, s, b.Attachment), clView(t, s, a.Attachment)
	if bv.Slots.Given != 0 || bv.Slots.Unserved == "" {
		t.Fatalf("b = %+v", bv.Slots)
	}
	if av.Slots.Given != 1 || av.Slots.Unserved != "" {
		t.Fatalf("a = %+v", av.Slots)
	}
	w := s.Status().Wants
	if len(w.Unserved) != 1 || w.Unserved[0].Attachment != b.Attachment || w.Unserved[0].Why == "" {
		t.Fatalf("wants.unserved = %+v", w.Unserved)
	}
}

// clCountFacts is the attachment facts standing.
func clCountFacts(s *run.Supervisor) int { return len(s.Hub().Facts("attachment")) }

// 7. Launch options, conflicts and the build's flags.
func TestClientServerOptions(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	ka := newKid(t)
	pa := clProf(1, 8192, 4096)
	pa.Server = &run.ServerOpts{Reasoning: "on", Ubatch: 256, ExtraArgs: []string{"--top-n-sigma", "1.5"}}
	a, _ := clAttach(t, s, clReq("a", ka, false, pa))
	if len(a.Server.NotTaken) != 0 {
		t.Fatalf("a not taken: %+v", a.Server.NotTaken)
	}
	for _, o := range []string{"reasoning", "ubatch", "--top-n-sigma"} {
		found := false
		for _, x := range a.Server.Taken {
			found = found || x == o
		}
		if !found {
			t.Fatalf("a taken = %v, lacks %s", a.Server.Taken, o)
		}
	}
	clStarted(t, e, s, 2)
	line := e.Starts()[1]
	for _, w := range []string{"--reasoning on", "-ub 256", "--top-n-sigma 1.5"} {
		if !strings.Contains(line, w) {
			t.Fatalf("start args lack %q: %s", w, line)
		}
	}
	if strings.Contains(line, "--reasoning off") {
		t.Fatalf("the engine's own reasoning off stayed: %s", line)
	}
	// B: reasoning off, no slots: told it conflicts, held by a
	pb := &run.Profile{Server: &run.ServerOpts{Reasoning: "off"}}
	b, _ := clAttach(t, s, clReq("b", nil, true, pb))
	if len(b.Server.NotTaken) != 1 || b.Server.NotTaken[0] != (run.NotTaken{Option: "reasoning", Asked: "off", Standing: "on", HeldBy: "a", Why: "conflict"}) {
		t.Fatalf("b not taken = %+v", b.Server.NotTaken)
	}
	// B2: another value of the same extra flag
	pb2 := &run.Profile{Server: &run.ServerOpts{ExtraArgs: []string{"--top-n-sigma", "2"}}}
	b2, _ := clAttach(t, s, clReq("b2", nil, true, pb2))
	if len(b2.Server.NotTaken) != 1 || b2.Server.NotTaken[0].Option != "--top-n-sigma" || b2.Server.NotTaken[0].Why != "conflict" || b2.Server.NotTaken[0].HeldBy != "a" {
		t.Fatalf("b2 not taken = %+v", b2.Server.NotTaken)
	}
	// the same flag and value is shared
	pb3 := &run.Profile{Server: &run.ServerOpts{ExtraArgs: []string{"--top-n-sigma", "1.5"}}}
	b3, _ := clAttach(t, s, clReq("b3", nil, true, pb3))
	if len(b3.Server.NotTaken) != 0 || len(b3.Server.Taken) != 1 {
		t.Fatalf("b3 = %+v", b3.Server)
	}
	// C: required, refused with nothing registered
	n, facts := s.Attached(), clCountFacts(s)
	pc := &run.Profile{Server: &run.ServerOpts{Reasoning: "off", Required: []string{"reasoning"}}}
	ce := clRefused(t, s, clReq("c", nil, true, pc))
	if ce.Status != 409 || ce.Code != "conflict" || ce.Extra["option"] != "reasoning" || ce.Extra["held_by"] != "a" {
		t.Fatalf("c refused with %+v", ce)
	}
	if s.Attached() != n || clCountFacts(s) != facts {
		t.Fatalf("a refused attach registered something: attached %d->%d facts %d->%d", n, s.Attached(), facts, clCountFacts(s))
	}
	// a required extra flag with another value
	pc2 := &run.Profile{Server: &run.ServerOpts{ExtraArgs: []string{"--top-n-sigma", "2"}, Required: []string{"--top-n-sigma"}}}
	if ce := clRefused(t, s, clReq("c2", nil, true, pc2)); ce.Code != "conflict" {
		t.Fatalf("c2 refused with %+v", ce)
	}
	// a server block asking for a projector is refused plainly (not offered in this version)
	pm := clProf(1, 8192, 4096)
	pm.Server = &run.ServerOpts{MMProj: json.RawMessage(`"/x/p.gguf"`)}
	if ce := clRefused(t, s, clReq("m", nil, true, pm)); ce.Code != "not_offered" {
		t.Fatalf("mmproj refused with %+v", ce)
	}
}

// 8. Refusals at attach: each says its code, and nothing is registered.
func TestClientRefusedAtAttach(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	loc := func(sv *run.ServerOpts) run.AttachRequest {
		p := &run.Profile{Server: sv}
		return clReq("x", nil, true, p)
	}
	rem := func(p *run.Profile) run.AttachRequest {
		return run.AttachRequest{Name: "r", Hold: true, Profile: p, Identity: "client-1", Local: false}
	}
	gg := filepath.Join(t.TempDir(), "p.gguf")
	if err := os.WriteFile(gg, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	many := map[string]catalog.SamplerProfile{}
	for i := 0; i < 17; i++ {
		many[string(rune('a'+i))+"s"] = catalog.SamplerProfile{}
	}
	cases := []struct {
		name string
		req  run.AttachRequest
		code string
	}{
		{"-c", loc(&run.ServerOpts{ExtraArgs: []string{"-c", "4096"}}), "flag_owned"},
		{"--ctx-size=", loc(&run.ServerOpts{ExtraArgs: []string{"--ctx-size=4096"}}), "flag_owned"},
		{"--SLOT-SAVE-PATH", loc(&run.ServerOpts{ExtraArgs: []string{"--SLOT-SAVE-PATH", "/x"}}), "flag_owned"},
		{"--slot-save-path=", loc(&run.ServerOpts{ExtraArgs: []string{"--slot-save-path=/x"}}), "flag_owned"},
		{"-np", loc(&run.ServerOpts{ExtraArgs: []string{"-np", "2"}}), "flag_owned"},
		{"--port", loc(&run.ServerOpts{ExtraArgs: []string{"--port", "1"}}), "flag_owned"},
		{"--host", loc(&run.ServerOpts{ExtraArgs: []string{"--host", "0.0.0.0"}}), "flag_owned"},
		{"--api-key", loc(&run.ServerOpts{ExtraArgs: []string{"--api-key", "k"}}), "flag_owned"},
		{"-m", loc(&run.ServerOpts{ExtraArgs: []string{"-m", "x"}}), "flag_owned"},
		{"unknown flag", loc(&run.ServerOpts{ExtraArgs: []string{"--no-such-flag"}}), "flag_unknown"},
		{"remote extra_args", rem(&run.Profile{Server: &run.ServerOpts{ExtraArgs: []string{"--top-n-sigma", "1"}}}), "extra_args_not_allowed"},
		{"remote mmproj", rem(&run.Profile{Server: &run.ServerOpts{MMProj: json.RawMessage(`"` + gg + `"`)}}), "not_offered"},
		{"remote speculative", rem(&run.Profile{Server: &run.ServerOpts{Speculative: json.RawMessage(`{"draft":"` + gg + `"}`)}}), "not_offered"},
		{"local speculative", loc(&run.ServerOpts{Speculative: json.RawMessage(`{"draft":"` + gg + `"}`)}), "not_offered"},
		{"--mmproj in extra_args", loc(&run.ServerOpts{ExtraArgs: []string{"--mmproj", "x.gguf"}}), "flag_owned"},
		{"--no-mmproj in extra_args", loc(&run.ServerOpts{ExtraArgs: []string{"--no-mmproj"}}), "flag_owned"},
		{"-md in extra_args", loc(&run.ServerOpts{ExtraArgs: []string{"-md", "x.gguf"}}), "flag_owned"},
		{"--spec-draft-n-max in extra_args", loc(&run.ServerOpts{ExtraArgs: []string{"--spec-draft-n-max", "4"}}), "flag_owned"},
		{"remote pid", run.AttachRequest{Pid: os.Getpid(), Name: "r", Identity: "client-1", Local: false}, "pid_remote"},
		{"remote agent", rem(&run.Profile{Agent: &run.AgentSpec{ID: "ag", Wheel: "/abs/a.whl"}}), "agent_not_allowed"},
		{"remote host_pid", rem(&run.Profile{HostPID: os.Getpid()}), "pid_remote"},
		{"slots 17", loc(nil), "bad_request"}, // replaced below
		{"ctx max 100", clReq("x", nil, true, clProf(1, 100, 0)), "bad_request"},
		{"17 samplers", clReq("x", nil, true, &run.Profile{Samplers: many}), "bad_request"},
		{"bad sampler name", clReq("x", nil, true, &run.Profile{Samplers: map[string]catalog.SamplerProfile{"Bad Name!": {}}}), "bad_request"},
		{"can_use ../x", clReq("x", nil, true, &run.Profile{Models: &run.ModelPrefs{CanUse: []string{"../x"}}}), "bad_request"},
		{"required nonsense", loc(&run.ServerOpts{Required: []string{"nonsense"}}), "bad_request"},
		{"no pid and no hold", run.AttachRequest{Name: "x", Identity: "loopback", Local: true, Profile: clProf(1, 8192, 4096)}, "bad_request"},
	}
	for i := range cases {
		if cases[i].name == "slots 17" {
			cases[i].req = clReq("x", nil, true, clProf(17, 8192, 4096))
		}
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			n, facts := s.Attached(), clCountFacts(s)
			e := clRefused(t, s, c.req)
			if e.Code != c.code || e.Status != 400 {
				t.Fatalf("refused with %d %s (%s), want 400 %s", e.Status, e.Code, e.Msg, c.code)
			}
			if s.Attached() != n || clCountFacts(s) != facts {
				t.Fatalf("a refused attach registered something")
			}
		})
	}
	t.Run("5th of one identity", func(t *testing.T) {
		for i := 0; i < 4; i++ {
			rq := run.AttachRequest{Name: string(rune('a' + i)), Hold: true, Identity: "client-2", Local: false}
			clAttach(t, s, rq)
		}
		n, facts := s.Attached(), clCountFacts(s)
		e := clRefused(t, s, run.AttachRequest{Name: "e", Hold: true, Identity: "client-2", Local: false})
		if e.Status != 429 || e.Code != "too_many_attachments" {
			t.Fatalf("refused with %d %s", e.Status, e.Code)
		}
		if s.Attached() != n || clCountFacts(s) != facts {
			t.Fatal("the refused 5th registered something")
		}
		// another identity is not held back by it
		clAttach(t, s, run.AttachRequest{Name: "e", Hold: true, Identity: "client-3", Local: false})
	})
	t.Run("sampler names are lower-cased", func(t *testing.T) {
		temp := 0.2
		p := &run.Profile{Samplers: map[string]catalog.SamplerProfile{"Cold": {Temperature: &temp}}}
		v, _ := clAttach(t, s, clReq("sam", nil, true, p))
		info := s.SamplerFor(v.Attachment)
		if _, ok := info.Profiles["cold"]; !ok {
			t.Fatalf("profiles = %v, want cold", info.Profiles)
		}
		if _, ok := info.Profiles["Cold"]; ok {
			t.Fatalf("profiles keep the capitalised name: %v", info.Profiles)
		}
	})
}

// 9. Advice on the model: told, never swapped; required makes it a refusal.
func TestClientModelAdvice(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	p := &run.Profile{Models: &run.ModelPrefs{CanUse: []string{"other"}, Prefer: []string{"other"}}}
	v, _ := clAttach(t, s, clReq("m", nil, true, p))
	if v.Model == nil || v.Model.Usable || v.Model.Selected != "m1" || v.Model.Advice == "" {
		t.Fatalf("model view = %+v", v.Model)
	}
	p2 := &run.Profile{Models: &run.ModelPrefs{CanUse: []string{"m1"}}}
	if v2, _ := clAttach(t, s, clReq("m2", nil, true, p2)); v2.Model == nil || !v2.Model.Usable {
		t.Fatalf("m2 = %+v", v2.Model)
	}
	n, facts := s.Attached(), clCountFacts(s)
	p3 := &run.Profile{Models: &run.ModelPrefs{CanUse: []string{"other"}}, Server: &run.ServerOpts{Required: []string{"model"}}}
	ce := clRefused(t, s, clReq("m3", nil, true, p3))
	if ce.Status != 409 || ce.Code != "model_unsupported" {
		t.Fatalf("refused with %d %s", ce.Status, ce.Code)
	}
	if s.Attached() != n || clCountFacts(s) != facts {
		t.Fatal("the refused attach registered something")
	}
	if len(e.Starts()) != 1 || s.Status().Model != "m1" {
		t.Fatalf("the model moved: starts %v", e.Starts())
	}
}

// 10. The old wire at run level: a pid alone adds presence, no wants, no start.
func TestClientOldWirePidAlone(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	k := newKid(t)
	v, h := clAttach(t, s, clReq("", k, false, nil))
	if h != nil || v.Slots.Want != 0 || v.Slots.Given != 0 || v.Status != "attached" || v.Presence != "pid" {
		t.Fatalf("view = %+v", v)
	}
	if w := s.Status().Wants; len(w.From) != 0 || len(w.Unserved) != 0 || len(w.Vacant) != 0 {
		t.Fatalf("wants = %+v", w)
	}
	clQuiet(t, e, s, 1)
	if s.Attached() != 1 {
		t.Fatalf("attached = %d", s.Attached())
	}
	k.end()
	testkit.WaitFor(t, 15*time.Second, "the pid attachment gone", func() bool { return clFactGone(s, v.Attachment) })
	clQuiet(t, e, s, 1)
}
