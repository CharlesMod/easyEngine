//go:build !windows

package run_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/yield"
)

// The prompt cache across a reshape (run/context.go): llama-server's slot save and restore keep
// each slot's prompt across a planned restart. The fake llama-server holds a token count per
// slot (see internal/fakellama), writes slot files into --slot-save-path and logs each save and
// restore in <ctl>/slots.log.

// ctxSaid is one "context" happening.
type ctxSaid struct {
	What   string `json:"what"`
	Slot   int    `json:"slot"`
	Tokens int    `json:"tokens"`
	Bytes  int64  `json:"bytes"`
	Why    string `json:"why"`
}

// contextsSince is the context happenings said after seq from, oldest first.
func contextsSince(t *testing.T, s *run.Supervisor, from uint64) []ctxSaid {
	t.Helper()
	var out []ctxSaid
	for _, ev := range testkit.Said(t, s, from, 0, "context") {
		var c ctxSaid
		if err := json.Unmarshal(ev.Body, &c); err != nil {
			t.Fatal(err)
		}
		out = append(out, c)
	}
	return out
}

// whatsOf picks the happenings of one kind (slot -1: any slot).
func whatsOf(l []ctxSaid, what string, slot int) []ctxSaid {
	var out []ctxSaid
	for _, c := range l {
		if c.What == what && (slot < 0 || c.Slot == slot) {
			out = append(out, c)
		}
	}
	return out
}

// ask sends one completion with a prompt of n bytes to slot id of the running server, and answers
// the cache_n and prompt_n of its timings.
func ask(t *testing.T, s *run.Supervisor, id, n int) (cacheN, promptN int) {
	t.Helper()
	r := s.Route()
	if r.Kind != "local" {
		t.Fatalf("route = %+v", r)
	}
	body, _ := json.Marshal(map[string]any{"id_slot": id, "messages": []map[string]string{{"role": "user", "content": strings.Repeat("x", n)}}})
	resp, err := http.Post(r.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Timings struct {
			CacheN  int `json:"cache_n"`
			PromptN int `json:"prompt_n"`
		} `json:"timings"`
	}
	if resp.StatusCode != 200 || json.NewDecoder(resp.Body).Decode(&out) != nil {
		t.Fatalf("completion answered %d", resp.StatusCode)
	}
	return out.Timings.CacheN, out.Timings.PromptN
}

// savedTokens is the tokens a slot holds after a prompt of n bytes (the fake: ceil(n/4) + 5).
func savedTokens(n int) int { return (n+3)/4 + 5 }

// slotsLog is <ctl>/slots.log, one line per save and restore the fake served.
func slotsLog(e *testkit.Env) []string {
	b, err := os.ReadFile(filepath.Join(e.Ctl, "slots.log"))
	if err != nil || len(bytes.TrimSpace(b)) == 0 {
		return nil
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

// logCount counts the slots.log lines that begin with prefix.
func logCount(e *testkit.Env, prefix string) int {
	n := 0
	for _, l := range slotsLog(e) {
		if strings.HasPrefix(l, prefix+" ") {
			n++
		}
	}
	return n
}

// contextFiles are the slot files and the index in <data>/contexts.
func contextFiles(t *testing.T, e *testkit.Env) []string {
	t.Helper()
	ents, err := os.ReadDir(filepath.Join(e.Data, "contexts"))
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, en := range ents {
		out = append(out, en.Name())
	}
	return out
}

// noContextFiles fails when contexts/ holds a slot file or the index.
func noContextFiles(t *testing.T, e *testkit.Env) {
	t.Helper()
	if l := contextFiles(t, e); len(l) != 0 {
		t.Fatalf("contexts/ holds %v", l)
	}
}

// ctxSup is a supervisor ready at the fake's usual shape, its config tweaked first.
func ctxSup(t *testing.T, e *testkit.Env, tweak func(*run.Config)) *run.Supervisor {
	t.Helper()
	e.WatchFile(filepath.Join(e.Ctl, "slots.log"))
	e.InstallFake()
	cfg := e.Config()
	if tweak != nil {
		tweak(&cfg)
	}
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return s
}

// shrinkNow takes the room away (the player's heavy save) and waits for the smaller server to
// be ready: starts goes from n to n+1.
func shrinkNow(t *testing.T, e *testkit.Env, s *run.Supervisor) {
	t.Helper()
	n := len(e.Starts())
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitFor(t, 15*time.Second, "the shrunk server", func() bool { return len(e.Starts()) > n && s.Phase() == "ready" })
	e.SetUsed(0)
}

// A shrink keeps what each slot holds: saved before the stop, laid back before ready, the files
// gone after, and the next request on the slot finds its prompt in the cache.
func TestContextShrinkKeepsSlots(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	ask(t, s, 1, 2000)
	want0, want1 := savedTokens(4000), savedTokens(2000)
	from := testkit.Mark(t, s, "before the shrink")
	shrinkNow(t, e, s)
	l := contextsSince(t, s, from)
	for slot, tok := range map[int]int{0: want0, 1: want1} {
		sv, lb := whatsOf(l, "saved", slot), whatsOf(l, "laid_back", slot)
		if len(sv) != 1 || len(lb) != 1 || sv[0].Tokens != tok || lb[0].Tokens != tok {
			t.Fatalf("slot %d: saved %+v laid back %+v, want %d tokens\nsaid %+v", slot, sv, lb, tok, l)
		}
		if sv[0].Bytes <= 0 || sv[0].Why == "" {
			t.Fatalf("saved says too little: %+v", sv[0])
		}
	}
	if len(whatsOf(l, "saved", 2)) != 0 {
		t.Fatalf("a slot that held nothing was saved: %+v", l)
	}
	// every saved happening comes before every laid-back one (the stop sits between)
	seenBack := false
	for _, c := range l {
		if c.What == "laid_back" {
			seenBack = true
		} else if c.What == "saved" && seenBack {
			t.Fatalf("saved after a lay back: %+v", l)
		}
	}
	if logCount(e, "save") != 2 || logCount(e, "restore") != 2 || strings.Contains(strings.Join(slotsLog(e), "\n"), "fail") {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
	if c := s.Status().Contexts; c != nil {
		t.Fatalf("status still lists kept contexts: %+v", c)
	}
	// the continuation finds the saved prompt in the new server's slot
	if c, _ := ask(t, s, 0, 4100); c != want0 {
		t.Fatalf("slot 0 cache_n = %d, want %d", c, want0)
	}
	if c, _ := ask(t, s, 1, 2100); c != want1 {
		t.Fatalf("slot 1 cache_n = %d, want %d", c, want1)
	}
}

// A request held at the door while the shape is being changed is answered by the new server, from
// the kept prompt. (The holding itself is api's; here the door's two words, Enter and Hold, are
// used the way api uses them.)
func TestContextHeldRequestAnsweredFromKeptPrompt(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	want := savedTokens(4000)
	done := s.Begin() // a line in flight keeps the model in "resizing" until it ends
	e.SetUsed(16000)
	e.SetOthers(8000)
	testkit.WaitPhase(t, s, "resizing", 10*time.Second)
	rt, end := s.Enter("held-1")
	if !rt.Hold || end != nil {
		t.Fatalf("a request during the reshape is not held: %+v", rt)
	}
	release := s.Hold()
	if s.Held() != 1 {
		t.Fatalf("held = %d", s.Held())
	}
	done()
	testkit.WaitFor(t, 15*time.Second, "the shrunk server", func() bool { return len(e.Starts()) >= 2 && s.Phase() == "ready" })
	e.SetUsed(0)
	rt, end = s.Enter("held-1") // the door enters again when the phase changes
	release()
	if rt.Hold || rt.Kind != "local" || end == nil {
		t.Fatalf("not let in after the reshape: %+v", rt)
	}
	defer end(nil)
	if c, _ := ask(t, s, 0, 4100); c != want {
		t.Fatalf("the held request found cache_n %d, want %d", c, want)
	}
	noContextFiles(t, e)
}

// yieldSupWith is yieldSup with a config the test made.
func yieldSupWith(t *testing.T, e *testkit.Env, f *fakeYieldOS, cfg run.Config) *run.Supervisor {
	t.Helper()
	cfg.Sense = f.sense
	y := &run.YieldConfig{Except: []string{"294"}, Every: time.Hour, Drain: 2 * time.Second,
		OS: func() yield.OS { return f }, Procs: f.memProcs, Looked: f.looked}
	y.Policy = yield.Config{Grace: 400 * time.Millisecond, Steady: 600 * time.Millisecond, Floor: 128}
	cfg.Yield = y
	return e.StartSup(cfg)
}

// A yield saves under keep_context "on" when the save is small (the estimate stays under
// YieldSaveMaxMiB) and the return lays it back.
func TestContextYieldAndReturnKeepsSmallContext(t *testing.T) {
	e := testkit.New(t)
	f := newFakeYieldOS()
	e.WatchFile(filepath.Join(e.Ctl, "slots.log"))
	e.InstallFake()
	s := yieldSupWith(t, e, f, e.Config())
	settled(t, s, f)
	if err := s.SetSettings(run.Settings{Mode: "local", KVType: "q4_0", KeepContext: "on"}); err != nil {
		t.Fatal(err)
	}
	ask(t, s, 0, 400)
	want := savedTokens(400)
	from := testkit.Mark(t, s, "before the yield")
	f.setLaunches(game)
	testkit.WaitFor(t, 15*time.Second, "yielded and unloaded", func() bool { st := s.Status(); return st.Phase == "yielded" && st.Shape == nil })
	if sv := whatsOf(contextsSince(t, s, from), "saved", 0); len(sv) != 1 || sv[0].Tokens != want {
		t.Fatalf("a small context was not kept at the yield: %+v", contextsSince(t, s, from))
	}
	if c := s.Status().Contexts; c == nil || c.Slots != 1 || c.Tokens != want || len(c.Kept) != 1 || c.Kept[0].Slot != 0 || c.Bytes <= 0 {
		t.Fatalf("status contexts while yielded = %+v", c)
	}
	f.setLaunches()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if lb := whatsOf(contextsSince(t, s, from), "laid_back", 0); len(lb) != 1 || lb[0].Tokens != want {
		t.Fatalf("not laid back after the return: %+v", contextsSince(t, s, from))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 500); c != want {
		t.Fatalf("cache_n after the return = %d, want %d", c, want)
	}
}

// A yield with keep_context "on" waits for no large write: over YieldSaveMaxMiB nothing is
// saved and not_saved says so; "always" saves it all the same and lays it back.
func TestContextYieldCapOnAndAlways(t *testing.T) {
	for _, mode := range []string{"on", "always"} {
		t.Run(mode, func(t *testing.T) {
			e := testkit.New(t)
			f := newFakeYieldOS()
			e.WatchFile(filepath.Join(e.Ctl, "slots.log"))
			e.InstallFake()
			cfg := e.Config()
			cfg.YieldSaveMaxMiB = 1 // far under what a slot of 20000 tokens is priced at
			s := yieldSupWith(t, e, f, cfg)
			settled(t, s, f)
			if err := s.SetSettings(run.Settings{Mode: "local", KVType: "q4_0", KeepContext: mode}); err != nil {
				t.Fatal(err)
			}
			ask(t, s, 0, 80000)
			want := savedTokens(80000)
			from := testkit.Mark(t, s, "before the yield")
			f.setLaunches(game)
			testkit.WaitFor(t, 15*time.Second, "yielded and unloaded", func() bool { st := s.Status(); return st.Phase == "yielded" && st.Shape == nil })
			l := contextsSince(t, s, from)
			if mode == "on" {
				if len(whatsOf(l, "not_saved", -1)) != 1 || len(whatsOf(l, "saved", -1)) != 0 || logCount(e, "save") != 0 {
					t.Fatalf("a yield over the cap: %+v, slots.log %v", l, slotsLog(e))
				}
				if c := s.Status().Contexts; c != nil {
					t.Fatalf("status lists contexts that were not kept: %+v", c)
				}
			} else if sv := whatsOf(l, "saved", 0); len(sv) != 1 || sv[0].Tokens != want {
				t.Fatalf("always did not save: %+v", l)
			}
			f.setLaunches()
			testkit.WaitPhase(t, s, "ready", 15*time.Second)
			l = contextsSince(t, s, from)
			if mode == "on" {
				if len(whatsOf(l, "laid_back", -1)) != 0 || logCount(e, "restore") != 0 {
					t.Fatalf("something was laid back that was not saved: %+v", l)
				}
			} else if lb := whatsOf(l, "laid_back", 0); len(lb) != 1 || lb[0].Tokens != want {
				t.Fatalf("always: not laid back: %+v", l)
			}
			noContextFiles(t, e)
		})
	}
}

// Sleep and Wake keep the slots.
func TestContextSleepWakeKeeps(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	want := savedTokens(4000)
	from := testkit.Mark(t, s, "before the sleep")
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	if sv := whatsOf(contextsSince(t, s, from), "saved", 0); len(sv) != 1 || sv[0].Tokens != want || !strings.Contains(sv[0].Why, "sleep") {
		t.Fatalf("not saved at the sleep: %+v", contextsSince(t, s, from))
	}
	if c := s.Status().Contexts; c == nil || c.Slots != 1 {
		t.Fatalf("status contexts asleep = %+v", c)
	}
	if err := s.Wake(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if lb := whatsOf(contextsSince(t, s, from), "laid_back", 0); len(lb) != 1 || lb[0].Tokens != want {
		t.Fatalf("not laid back at the wake: %+v", contextsSince(t, s, from))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 4100); c != want {
		t.Fatalf("cache_n after the wake = %d, want %d", c, want)
	}
}

// keep_context off saves nothing at a sleep.
func TestContextOffSavesNothingAtSleep(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	if err := s.SetSettings(run.Settings{Mode: "local", KVType: "q4_0", KeepContext: "off"}); err != nil {
		t.Fatal(err)
	}
	ask(t, s, 0, 4000)
	from := testkit.Mark(t, s, "before the sleep")
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	if err := s.Wake(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if l := contextsSince(t, s, from); len(l) != 0 {
		t.Fatalf("keep_context off said %+v", l)
	}
	if len(slotsLog(e)) != 0 {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 4100); c != 0 {
		t.Fatalf("a prompt came back with keep_context off: cache_n %d", c)
	}
}

// A cache type change lets the kept slots go: the new server would not take them.
func TestContextKVTypeChangeLetsGo(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	from := testkit.Mark(t, s, "before the change")
	n := len(e.Starts())
	if err := s.SetSettings(run.Settings{Mode: "local", KVType: "q8_0"}); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 15*time.Second, "the server restarted at the new cache type", func() bool { return len(e.Starts()) > n && s.Phase() == "ready" })
	l := contextsSince(t, s, from)
	if len(whatsOf(l, "laid_back", -1)) != 0 || logCount(e, "restore") != 0 {
		t.Fatalf("a slot was laid back across a cache type change: %+v, slots.log %v", l, slotsLog(e))
	}
	if len(whatsOf(l, "saved", -1)) > 0 && len(whatsOf(l, "let_go", 0)) != 1 {
		t.Fatalf("a saved slot was not let go: %+v", l)
	}
	noContextFiles(t, e)
	if c := s.Status().Contexts; c != nil {
		t.Fatalf("status lists contexts: %+v", c)
	}
	if c, _ := ask(t, s, 0, 4100); c != 0 {
		t.Fatalf("cache_n = %d after a cache type change", c)
	}
}

// reshapeTo takes the room down to what others leaves (MiB used by other programs) and waits for
// the smaller server to be ready.
func reshapeTo(t *testing.T, e *testkit.Env, s *run.Supervisor, others int) {
	t.Helper()
	n := len(e.Starts())
	e.SetUsed(16000)
	e.SetOthers(others)
	testkit.WaitFor(t, 15*time.Second, "the reshaped server", func() bool { return len(e.Starts()) > n && s.Phase() == "ready" })
	e.SetUsed(0)
}

// A slot the new server does not have (the room left room for one slot) is let go, its file
// removed; the slot that is still there is laid back.
func TestContextSlotBeyondNewSlotCountLetGo(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	if sh := s.Status().Shape; sh == nil || sh.Slots != 3 {
		t.Fatalf("shape = %+v", sh)
	}
	ask(t, s, 0, 4000)
	ask(t, s, 2, 4000)
	from := testkit.Mark(t, s, "before the shrink")
	reshapeTo(t, e, s, 11000)
	if sh := s.Status().Shape; sh == nil || sh.Slots != 1 {
		t.Fatalf("the shrunk shape was meant to have one slot: %+v", sh)
	}
	l := contextsSince(t, s, from)
	if len(whatsOf(l, "saved", 2)) != 1 || len(whatsOf(l, "let_go", 2)) != 1 || len(whatsOf(l, "laid_back", 2)) != 0 {
		t.Fatalf("slot 2: %+v", l)
	}
	if len(whatsOf(l, "laid_back", 0)) != 1 {
		t.Fatalf("slot 0 not laid back: %+v", l)
	}
	if logCount(e, "restore 2") != 0 || logCount(e, "restore 0") != 1 {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
}

// A slot whose tokens do not fit the new context per slot is let go with no restore tried; a
// slot that fits is laid back.
func TestContextTooManyTokensForNewContextLetGo(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 80000) // 20005 tokens
	ask(t, s, 1, 4000)
	from := testkit.Mark(t, s, "before the shrink")
	reshapeTo(t, e, s, 8000) // three slots of 14336
	if sh := s.Status().Shape; sh == nil || sh.Slots != 3 || sh.CtxPerSlot >= savedTokens(80000) {
		t.Fatalf("the shrunk shape was meant to hold fewer tokens per slot than the saved one: %+v", sh)
	}
	l := contextsSince(t, s, from)
	if lg := whatsOf(l, "let_go", 0); len(lg) != 1 || !strings.Contains(lg[0].Why, "do not fit") {
		t.Fatalf("slot 0: %+v", l)
	}
	if len(whatsOf(l, "laid_back", 0)) != 0 || logCount(e, "restore 0") != 0 {
		t.Fatalf("a restore was tried for a slot that does not fit: %+v %v", l, slotsLog(e))
	}
	if lb := whatsOf(l, "laid_back", 1); len(lb) != 1 || lb[0].Tokens != savedTokens(4000) {
		t.Fatalf("slot 1 not laid back: %+v", l)
	}
	noContextFiles(t, e)
}

// A save the server refuses is said and not kept, and the shrink goes on.
func TestContextFailedSaveDoesNotBlockTheShrink(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	e.Touch("savefail")
	from := testkit.Mark(t, s, "before the shrink")
	shrinkNow(t, e, s)
	if s.Phase() != "ready" || len(e.Starts()) != 2 {
		t.Fatalf("the shrink did not complete: %s %v", s.Phase(), e.Starts())
	}
	l := contextsSince(t, s, from)
	if len(whatsOf(l, "saved", -1)) != 0 || len(whatsOf(l, "laid_back", -1)) != 0 || len(whatsOf(l, "not_saved", 0)) != 1 {
		t.Fatalf("said %+v", l)
	}
	found := false
	for _, m := range s.Status().Errors {
		found = found || strings.Contains(m, "saving slot 0")
	}
	if !found {
		t.Fatalf("no error said: %v", s.Status().Errors)
	}
	if logCount(e, "restore") != 0 {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 4100); c != 0 {
		t.Fatalf("cache_n = %d with nothing kept", c)
	}
}

// A restore the server refuses is let go, its file removed, and the start reaches ready.
func TestContextFailedRestoreDoesNotFailTheStart(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	e.Touch("restorefail")
	from := testkit.Mark(t, s, "before the shrink")
	shrinkNow(t, e, s)
	if s.Phase() != "ready" {
		t.Fatalf("the start failed: %s", s.Phase())
	}
	l := contextsSince(t, s, from)
	if lg := whatsOf(l, "let_go", 0); len(lg) != 1 || !strings.Contains(lg[0].Why, "restore failed") || len(whatsOf(l, "laid_back", -1)) != 0 {
		t.Fatalf("said %+v", l)
	}
	if logCount(e, "save") != 1 || logCount(e, "restore 0") != 1 || !strings.Contains(strings.Join(slotsLog(e), "\n"), " fail") {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 4100); c != 0 {
		t.Fatalf("cache_n = %d after a failed restore", c)
	}
}

// A crash is no planned stop: nothing is saved (the server is gone), and the restart has
// nothing to lay back.
func TestContextCrashSavesNothing(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	ask(t, s, 0, 4000)
	from := testkit.Mark(t, s, "before the crash")
	e.Touch("crash")
	testkit.WaitFor(t, 15*time.Second, "restart after the crash", func() bool { return len(e.Starts()) >= 2 && s.Phase() == "ready" })
	if l := contextsSince(t, s, from); len(whatsOf(l, "saved", -1)) != 0 || len(whatsOf(l, "laid_back", -1)) != 0 {
		t.Fatalf("a crash kept something: %+v", l)
	}
	if len(slotsLog(e)) != 0 {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
	if c, _ := ask(t, s, 0, 4100); c != 0 {
		t.Fatalf("cache_n = %d after a crash", c)
	}
}

// Too little free disk for the files: nothing is saved, not_saved says why, the shrink goes on.
func TestContextTooLittleDiskSavesNothing(t *testing.T) {
	e := testkit.New(t)
	var free atomic.Int32
	free.Store(1 << 20) // MiB
	s := ctxSup(t, e, func(c *run.Config) {
		c.DiskFree = func(string) (int, error) { return int(free.Load()), nil }
	})
	ask(t, s, 0, 4000)
	free.Store(1)
	from := testkit.Mark(t, s, "before the shrink")
	shrinkNow(t, e, s)
	l := contextsSince(t, s, from)
	if ns := whatsOf(l, "not_saved", -1); len(ns) != 1 || !strings.Contains(ns[0].Why, "disk") || len(whatsOf(l, "saved", -1)) != 0 {
		t.Fatalf("said %+v", l)
	}
	if logCount(e, "save") != 0 {
		t.Fatalf("slots.log = %v", slotsLog(e))
	}
	noContextFiles(t, e)
}

// An engine that starts finds the files a dead one's index names and removes them, only by the
// names the engine gives a slot, and the index with them; nothing else in the directory, and
// nothing outside it, is touched.
func TestContextEngineStartRemovesStrays(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	dir := filepath.Join(e.Data, "contexts")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	stray := filepath.Join(dir, "slot3-12345-7.bin")
	other := filepath.Join(dir, "notes.txt")
	outside := filepath.Join(e.Data, "state.json") // InstallFake wrote it; an index naming it must not remove it
	for _, p := range []string{stray, other} {
		if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	idx := `[{"slot":3,"tokens":10,"bytes":1,"file":"slot3-12345-7.bin","key":{}},` +
		`{"slot":0,"tokens":10,"bytes":1,"file":"../state.json","key":{}},` +
		`{"slot":1,"tokens":10,"bytes":1,"file":"notes.txt","key":{}}]`
	if err := os.WriteFile(filepath.Join(dir, "index.json"), []byte(idx), 0o600); err != nil {
		t.Fatal(err)
	}
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if _, err := os.Stat(stray); !os.IsNotExist(err) {
		t.Fatalf("the stray slot file stays: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "index.json")); !os.IsNotExist(err) {
		t.Fatalf("the index stays: %v", err)
	}
	for _, p := range []string{other, outside} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("%s was removed: %v", p, err)
		}
	}
	if c := s.Status().Contexts; c != nil {
		t.Fatalf("an earlier run's slots are listed as kept: %+v", c)
	}
	if logCount(e, "restore") != 0 {
		t.Fatalf("an earlier run's slot was laid back: %v", slotsLog(e))
	}
}

// Every launch passes --slot-save-path <data>/contexts, and the directory is private.
func TestContextLaunchArgsAndDirectory(t *testing.T) {
	e := testkit.New(t)
	s := ctxSup(t, e, nil)
	_ = s
	lines := e.Starts()
	if len(lines) != 1 {
		t.Fatalf("starts = %v", lines)
	}
	want := "--slot-save-path " + filepath.Join(e.Data, "contexts")
	if !strings.Contains(lines[0], want) {
		t.Fatalf("launch line lacks %q:\n%s", want, lines[0])
	}
	st, err := os.Stat(filepath.Join(e.Data, "contexts"))
	if err != nil || !st.IsDir() {
		t.Fatalf("contexts: %v %v", st, err)
	}
	if st.Mode().Perm() != 0o700 {
		t.Fatalf("contexts mode = %v, want 0700", st.Mode().Perm())
	}
}
