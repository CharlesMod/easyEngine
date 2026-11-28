//go:build !windows

package run_test

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// remoteReq is a remote identity's held attach (the hold is never read: it stands until detached).
func remoteReq(who, name string, slots int) run.AttachRequest {
	return run.AttachRequest{Name: name, Profile: clProf(slots, 8192, 4096), Identity: who, Hold: true}
}

// settle lets the reactor act on every reshape asked, each to its start, with no traffic. The
// mind may take a plan itself (a client's word wakes it) before the test's pass: a start under way
// is waited for too (its phase), so the next act never lands in its loading window by chance.
func settle(t *testing.T, s *run.Supervisor) {
	t.Helper()
	moving := func() bool {
		switch s.Phase() {
		case "resizing", "starting", "loading":
			return true
		}
		return false
	}
	for i := 0; i < 8; i++ {
		testkit.WaitFor(t, 15*time.Second, "no start under way", func() bool { return !moving() })
		n, g := s.PlansNow()
		if n == 0 {
			return
		}
		s.React()
		testkit.WaitFor(t, 15*time.Second, "the start the plans asked for", func() bool {
			_, g2 := s.PlansNow()
			return g2 > g && s.Phase() == "ready"
		})
	}
	t.Fatal("plans still asked after 8 starts")
}

// bindFails is how many starts could not bind their port and never listened (the fake's
// bindfails.log): the engine starts again after one, as the same start.
func bindFails(e *testkit.Env) int {
	b, err := os.ReadFile(filepath.Join(e.Ctl, "bindfails.log"))
	if err != nil {
		return 0
	}
	return strings.Count(string(b), "\n")
}

// The walk's intermittent (6 restarts, 5 charged, under load): a reshape's start lost its port to
// another process between the engine's pick (freePort) and llama-server's bind; the server
// exited while loading and the mind started it again as an error's retry. One restart, charged
// once, two lines in starts.log; and the lost port counted as a failure, so a second one before
// the run was stable left the engine waiting in error. Held here at that point by the fake's
// bindfail: the launch starts again at once on a new port, counts no failure, the charge is
// one, and the walk's count of restarts takes the start that never listened out.
func TestStartThatNeverListenedIsOneRestart(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	clAttach(t, s, remoteReq("client-1", "x", 1))
	settle(t, s)
	n, f, c := len(e.Starts()), bindFails(e), s.RestartsOf("client-1")
	if err := os.WriteFile(filepath.Join(e.Ctl, "bindfail"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	clAttach(t, s, remoteReq("client-1", "x", 4)) // a change of wants: one charged restart
	settle(t, s)
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("phase %s, starts %d (from %d), bind failures %d (from %d), charged %d", s.Phase(), len(e.Starts()), n, bindFails(e), f, s.RestartsOf("client-1"))
		}
	})
	// (under load another process may take a port too: each lost one is a start that never listened)
	testkit.WaitFor(t, 15*time.Second, "ready after the retry", func() bool {
		return s.Phase() == "ready" && len(e.Starts()) == n+1+bindFails(e)-f
	})
	if bindFails(e) < f+1 || s.RestartsOf("client-1") != c+1 {
		t.Fatalf("bind failures %d (from %d), charged %d (from %d)", bindFails(e), f, s.RestartsOf("client-1"), c)
	}
	if r := s.Status().Recovery; r != nil {
		t.Fatalf("a lost port was counted as a failure: %+v", r)
	}
	if restarts := len(e.Starts()) - n - (bindFails(e) - f); restarts != 1 {
		t.Fatalf("the walk's count: %d restarts for one change", restarts)
	}
}

// spend charges who's whole budget by changes of its wants on one attachment.
func spend(t *testing.T, s *run.Supervisor, who, name string) {
	t.Helper()
	for i := 1; i <= 8 && s.RestartsOf(who) < 4; i++ {
		clAttach(t, s, remoteReq(who, name, i)) // each a change of wants
		settle(t, s)
	}
	if got := s.RestartsOf(who); got != 4 {
		t.Fatalf("%s charged %d, want its 4", who, got)
	}
}

// S2 and rules 1-2: within one waking period (between two stops the engine or the person makes),
// restarts of a running model server caused by remote identities alone never exceed what was
// charged, and the charge never exceeds their budgets. A seeded walk of attach, detach, return and
// change of wants by two remote identities, with unloads (the linger's) and returns while down,
// no local traffic. The unload makes every budget whole; the wake's start from down is the
// period's own and never charged. (An --open caller from another machine cannot sleep the model:
// api.TestOpenRemoteCallerIsScopedLikeAPairedToken; a sleep from another machine never refills:
// TestRemoteCausedStopsNeverRefill.)
func TestRemoteWalkNeverRestartsBeyondTheBudgets(t *testing.T) {
	for _, seed := range []uint64{1, 2, 3, 4, 5, 6} {
		t.Run(fmt.Sprint(seed), func(t *testing.T) {
			e := testkit.New(t)
			s := ready(t, e)
			rng := rand.New(rand.NewPCG(seed, 99))
			whos := []string{"client-1", "client-2"}
			names := []string{"x", "y"}
			type state struct {
				id    string
				slots int
			}
			on := map[string]*state{} // who/name -> standing attachment
			last := map[string]int{}  // who/name -> the slots it last attached with
			charged := func() int { return s.RestartsOf("client-1") + s.RestartsOf("client-2") }
			from, failed, woke, periods, down := len(e.Starts()), bindFails(e), 0, 0, false
			endPeriod := func(step int) {
				// a start that never listened (its port taken between the engine's pick and its bind,
				// as under load) is retried: the retry is the same restart, not another
				restarts := len(e.Starts()) - from - woke - (bindFails(e) - failed)
				if c := charged(); restarts > c || c > 8 {
					t.Fatalf("step %d: a waking period had %d restarts by remote acts, %d charged (budgets 8)", step, restarts, c)
				}
				periods++
			}
			for step := 0; step < 80; step++ {
				switch {
				case !down && rng.IntN(8) == 0: // the linger unloads the model: the period ends
					endPeriod(step)
					if err := s.Sleep(); err != nil {
						t.Fatal(err)
					}
					testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
					if charged() != 0 {
						t.Fatalf("step %d: the unload left %d charged", step, charged())
					}
					down = true
					continue
				case down && rng.IntN(4) == 0: // a first request loads it: a new period
					from, failed, woke = len(e.Starts()), bindFails(e), 1
					go s.Wake()
					testkit.WaitPhase(t, s, "ready", 15*time.Second)
					down = false
					continue
				}
				who, name := whos[rng.IntN(2)], names[rng.IntN(2)]
				k := who + "/" + name
				switch st := on[k]; {
				case st != nil && rng.IntN(2) == 0: // leave
					if err := s.Detach(st.id, who); err != nil {
						t.Fatalf("step %d: detach %s: %v", step, k, err)
					}
					delete(on, k)
				case st != nil: // change of wants, or the same again
					n := 1 + rng.IntN(6)
					v, _ := clAttach(t, s, remoteReq(who, name, n))
					on[k], last[k] = &state{v.Attachment, n}, n
				default: // a return (its last profile) or a new attach
					n := last[k]
					if n == 0 || rng.IntN(3) == 0 {
						n = 1 + rng.IntN(6)
					}
					v, _ := clAttach(t, s, remoteReq(who, name, n))
					on[k], last[k] = &state{v.Attachment, n}, n
				}
				if down && charged() != 0 && len(e.Starts()) == from {
					// nothing runs: an attach, return or change while down is never charged
					t.Fatalf("step %d: charged %d while the model was down", step, charged())
				}
				settle(t, s)
			}
			if !down {
				endPeriod(80)
			}
			t.Logf("seed %d: %d waking periods", seed, periods)
		})
	}
}

// S2, the traced bypass: P has spent its budget and holds X and Y; X leaves (Y would gain) and
// comes back with the same profile, again and again. No restart.
func TestSpentClientLeaveAndReturnRestartsNothing(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	y := remoteReq("client-1", "y", 1)
	y.Profile = clProf(1, 1<<20, 4096) // more context than any shape gives: every leave would give Y some
	clAttach(t, s, y)
	settle(t, s)
	spend(t, s, "client-1", "x")
	n := len(e.Starts())
	for i := 0; i < 3; i++ {
		v, _ := s.ClientView(attID(t, s, "client-1", "x"))
		if err := s.Detach(v.Attachment, "client-1"); err != nil {
			t.Fatal(err)
		}
		settle(t, s)
		clAttach(t, s, remoteReq("client-1", "x", v.Slots.Want))
		settle(t, s)
	}
	if got := len(e.Starts()); got != n {
		t.Fatalf("a spent client's leave and return restarted the model: %d -> %d", n, got)
	}
}

// S2, the variant: P (spent) leaves, a local caller's grow reshapes its range away, P returns: the
// grow it would need is charged, so a spent P waits for the next start made for another reason.
func TestSpentClientReturnAfterALocalReshape(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	spend(t, s, "client-1", "x")
	v, _ := s.ClientView(attID(t, s, "client-1", "x"))
	if err := s.Detach(v.Attachment, "client-1"); err != nil {
		t.Fatal(err)
	}
	settle(t, s)
	clAttach(t, s, clReq("l", nil, true, clProf(3, 8192, 4096))) // a local caller's want: its restart is its own
	settle(t, s)
	n := len(e.Starts())
	r, _ := clAttach(t, s, remoteReq("client-1", "x", v.Slots.Want))
	settle(t, s)
	if got := len(e.Starts()); got != n {
		t.Fatalf("a spent client's return restarted the model: %d -> %d", n, got)
	}
	if r.Slots.Why != "restart_budget_spent" || r.Slots.Given != 0 {
		t.Fatalf("the return: %+v", r.Slots)
	}
}

// attID is the id of who's attachment called name.
func attID(t *testing.T, s *run.Supervisor, who, name string) string {
	t.Helper()
	for _, id := range s.ClientAttachments(who) {
		if v, ok := s.ClientView(id); ok && v.Name == name {
			return id
		}
	}
	t.Fatalf("%s has no attachment %q", who, name)
	return ""
}

// A remote client's grow waits for the door's empty moment and never drains: proved by the
// reactor's own pass (React runs it in the test), not by a clock. With a reply in flight the pass
// leaves the phase ready, starts nothing and keeps the plan; a local caller's grow in the same
// place begins a drain at once. Once the reply ends, the next pass grows for the remote client.
func TestRemoteGrowWaitsForIdleByTheReactor(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	end := s.Begin() // a reply in flight
	v, _ := clAttach(t, s, remoteReq("client-1", "r", 6))
	if v.Slots.Why != "waiting_for_idle" || !v.Slots.Pending {
		t.Fatalf("not waiting for idle: %+v", v.Slots)
	}
	n := len(e.Starts())
	_, g := s.PlansNow()
	s.React()
	if p, g2 := s.PlansNow(); s.Phase() != "ready" || len(e.Starts()) != n || p != 1 || g2 != g {
		t.Fatalf("the pass with a reply in flight: phase %s, starts %d (from %d), plans %d", s.Phase(), len(e.Starts()), n, p)
	}
	end() // the door's empty moment
	settle(t, s)
	if got, _ := s.ClientView(v.Attachment); got.Slots.Given != 6 || len(e.Starts()) != n+1 {
		t.Fatalf("after the reply: %+v, starts %d (from %d)", got.Slots, len(e.Starts()), n)
	}
	// the contrast: a local caller's grow with a reply in flight drains at once
	end = s.Begin()
	defer end()
	clAttach(t, s, clReq("l", nil, true, clProf(6, 8192, 4096)))
	s.React()
	if s.Phase() != "resizing" {
		t.Fatalf("a local grow with a reply in flight: phase %s, want a drain (resizing)", s.Phase())
	}
}

// left is the restarts_left an attachment's view says.
func left(t *testing.T, s *run.Supervisor, id string) int {
	t.Helper()
	v, ok := s.ClientView(id)
	if !ok || v.RestartsLeft == nil {
		t.Fatalf("no restarts_left for %s: %+v", id, v)
	}
	return *v.RestartsLeft
}

// sleepThenWake unloads the model as the linger does and loads it again as a first request does,
// between them running between (attaches, returns, changes while down).
func sleepThenWake(t *testing.T, s *run.Supervisor, between func()) {
	t.Helper()
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
	between()
	go s.Wake()
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
}

// Rules 1 and 2, the real use: one remote client (a game on a laptop) attaches, the laptop
// sleeps (its attachment leaves), the engine's linger unloads the model, the laptop comes back
// (a return while the model is down) and its first request loads it. Ten times over: never
// charged, restarts_left 4 each time, exactly one model start per wake.
func TestLoneRemoteUnloadAndReturnNeverCharged(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	v, _ := clAttach(t, s, remoteReq("client-1", "game", 6))
	settle(t, s) // a grow of a running server: charged once
	if l := left(t, s, v.Attachment); l != 3 {
		t.Fatalf("the first grow: restarts left %d", l)
	}
	for i := 0; i < 10; i++ {
		if err := s.Detach(v.Attachment, "client-1"); err != nil {
			t.Fatal(err)
		}
		n := len(e.Starts())
		sleepThenWake(t, s, func() {
			v, _ = clAttach(t, s, remoteReq("client-1", "game", 6)) // the return, while down
		})
		settle(t, s)
		if got := len(e.Starts()); got != n+1 {
			t.Fatalf("cycle %d: %d starts (from %d), want the wake's one", i, got, n)
		}
		if l := left(t, s, v.Attachment); l != 4 || s.RestartsOf("client-1") != 0 {
			t.Fatalf("cycle %d: restarts left %d, charged %d", i, l, s.RestartsOf("client-1"))
		}
		if got, _ := s.ClientView(v.Attachment); got.Slots.Given != 6 {
			t.Fatalf("cycle %d: given %+v", i, got.Slots)
		}
	}
}

// Rule 2: four changes of wants while the model runs are charged (four restarts), a fifth is
// refused with the why; a linger unload makes the budget whole, and a change reshapes again.
func TestRemoteBudgetWholeAgainAfterAnUnload(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	spend(t, s, "client-1", "x")
	id := attID(t, s, "client-1", "x")
	n := len(e.Starts())
	v, _ := clAttach(t, s, remoteReq("client-1", "x", 8))
	settle(t, s)
	if v.Slots.Why != "restart_budget_spent" || len(e.Starts()) != n || left(t, s, id) != 0 {
		t.Fatalf("the fifth change: %+v, starts %d (from %d)", v.Slots, len(e.Starts()), n)
	}
	sleepThenWake(t, s, func() {})
	if l := left(t, s, id); l != 4 {
		t.Fatalf("after the unload: restarts left %d", l)
	}
	n = len(e.Starts())
	clAttach(t, s, remoteReq("client-1", "x", 3))
	settle(t, s)
	if len(e.Starts()) != n+1 || left(t, s, id) != 3 {
		t.Fatalf("a change after the refill: starts %d (from %d), left %d", len(e.Starts()), n, left(t, s, id))
	}
}

// Rule 2: no stop a remote client causes refills: a sleep or stop a remote caller asked, a failed
// start after its own change, an out-of-memory walk after its own grow.
func TestRemoteCausedStopsNeverRefill(t *testing.T) {
	t.Run("a sleep or stop asked from another machine", func(t *testing.T) {
		e := testkit.New(t)
		s := ready(t, e)
		clAttach(t, s, remoteReq("client-1", "x", 6))
		settle(t, s)
		if err := s.SleepBy(false); err != nil {
			t.Fatal(err)
		}
		testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
		if err := s.StopBy(false); err != nil {
			t.Fatal(err)
		}
		if s.RestartsOf("client-1") != 1 {
			t.Fatalf("refilled by a remote caller's sleep/stop: charged %d", s.RestartsOf("client-1"))
		}
	})
	t.Run("a failed start after its change", func(t *testing.T) {
		e := testkit.New(t)
		s := ready(t, e)
		clAttach(t, s, remoteReq("client-1", "x", 6))
		e.Touch("loadfail")
		s.React()
		testkit.WaitFor(t, 15*time.Second, "the start after its change failed", func() bool { return s.Phase() == "error" })
		e.Remove("loadfail")
		s.Route() // a request at the door is the word
		testkit.WaitPhase(t, s, "ready", 15*time.Second)
		if s.RestartsOf("client-1") != 1 {
			t.Fatalf("charged %d after a failed start of its own change, want 1", s.RestartsOf("client-1"))
		}
	})
	t.Run("an out-of-memory walk after its grow", func(t *testing.T) {
		e := testkit.New(t)
		s := ready(t, e)
		c0 := ctxOf(t, e.Starts()[0])
		writeCtl(t, e, "oom_above", strconv.Itoa(c0/3)) // the grow's shape runs out of memory: a walk down
		clAttach(t, s, remoteReq("client-1", "x", 12))
		n := len(e.Starts())
		s.React()
		testkit.WaitFor(t, 20*time.Second, "ready after the walk", func() bool {
			l := e.Starts()
			return len(l) >= n+2 && ctxOf(t, l[len(l)-1]) <= c0/3 && s.Phase() == "ready"
		})
		if s.RestartsOf("client-1") != 1 {
			t.Fatalf("charged %d after the walk its grow led to, want 1", s.RestartsOf("client-1"))
		}
	})
}

// D3: one restart costs a budget key one, however many of its attachments asked for it.
func TestOneRestartChargesAnIdentityOnce(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	clAttach(t, s, remoteReq("client-1", "x", 1))
	clAttach(t, s, remoteReq("client-1", "y", 1))
	settle(t, s)
	r0, n0 := s.RestartsOf("client-1"), len(e.Starts())
	clAttach(t, s, remoteReq("client-1", "x", 2)) // both change before the reactor acts
	clAttach(t, s, remoteReq("client-1", "y", 2))
	settle(t, s)
	if charged, starts := s.RestartsOf("client-1")-r0, len(e.Starts())-n0; starts != 1 || charged != 1 {
		t.Fatalf("one restart for two plans of one identity: %d starts, charged %d", starts, charged)
	}
}

// D6: a change of wants by a standing key while a start is loading is not lost: it stands, and
// once the model is up it gets its own reshape, charged as an attach in a loading window is.
func TestChangeWhileLoadingIsNotLost(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	v, _ := clAttach(t, s, remoteReq("client-1", "x", 6))
	settle(t, s)
	gate := filepath.Join(e.Ctl, "loadgate")
	if err := syscall.Mkfifo(gate, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := s.Sleep(); err != nil { // the budget is whole again
		t.Fatal(err)
	}
	testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
	go s.Wake()
	testkit.WaitPhase(t, s, "loading", 15*time.Second)              // its merge is taken; the server loads
	clAttach(t, s, remoteReq("client-1", "x", 8))                   // the change, while the start loads
	if err := os.WriteFile(gate, []byte("go"), 0o600); err != nil { // the load goes on
		t.Fatal(err)
	}
	os.Remove(gate)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	settle(t, s)
	if got, _ := s.ClientView(v.Attachment); got.Slots.Given != 8 || s.RestartsOf("client-1") != 1 {
		t.Fatalf("the change made while loading: %+v, charged %d", got.Slots, s.RestartsOf("client-1"))
	}
}
