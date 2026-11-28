//go:build !windows

package run_test

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// remoteReq is a remote identity's held attach (the hold is never read: it stands until detached).
func remoteReq(who, name string, slots int) run.AttachRequest {
	return run.AttachRequest{Name: name, Profile: clProf(slots, 8192, 4096), Identity: who, Hold: true}
}

// settle lets the reactor act on every reshape asked, each to its start, with no traffic.
func settle(t *testing.T, s *run.Supervisor) {
	t.Helper()
	for i := 0; i < 8; i++ {
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

// S2: every restart is a local caller's, the engine's own, or charged to the remote identity that
// caused it. A seeded walk of attach, detach, return and change of wants by two remote identities
// with no local traffic never restarts the model more than their two budgets.
func TestRemoteWalkNeverRestartsBeyondTheBudgets(t *testing.T) {
	for _, seed := range []uint64{1, 2, 3, 4} {
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
			n0 := len(e.Starts())
			for step := 0; step < 60; step++ {
				who, name := whos[rng.IntN(2)], names[rng.IntN(2)]
				k := who + "/" + name
				switch st := on[k]; {
				case st != nil && rng.IntN(2) == 0: // leave
					if err := s.Detach(st.id, who); err != nil {
						t.Fatalf("step %d: detach %s: %v", step, k, err)
					}
					delete(on, k)
				case st != nil: // change of wants, or the same again
					n := 1 + rng.IntN(2)
					v, _ := clAttach(t, s, remoteReq(who, name, n))
					on[k], last[k] = &state{v.Attachment, n}, n
				default: // a return (its last profile) or a new attach
					n := last[k]
					if n == 0 || rng.IntN(3) == 0 {
						n = 1 + rng.IntN(2)
					}
					v, _ := clAttach(t, s, remoteReq(who, name, n))
					on[k], last[k] = &state{v.Attachment, n}, n
				}
				settle(t, s)
			}
			starts := len(e.Starts()) - n0
			charged := s.RestartsOf("client-1") + s.RestartsOf("client-2")
			if starts > charged || charged > 8 {
				t.Fatalf("%d restarts by remote acts alone, %d charged (budgets 8)", starts, charged)
			}
			t.Logf("seed %d: %d restarts, %d charged", seed, starts, charged)
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
