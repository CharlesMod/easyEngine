package api_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// F1: a slot is kept across a restart only when it holds an attachment's prompt or the trusted
// commons', and laid back only into a slot the new map gives the same owner, labelled so. An
// untrusted caller's prompt in the commons ("anon:") and a slot whose erase failed ("?") are
// never kept: after the restart the slot is empty.
func TestKeptContextsCarryTheirOwner(t *testing.T) {
	sleepWake := func(t *testing.T, s interface {
		Sleep() error
	}) {
		t.Helper()
		if err := s.Sleep(); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("anon prompt across a sleep", func(t *testing.T) { // the reviewer's proof
		e, s, srv := clientsEnv(t, true, nil) // the base: the commons
		_, rs, _ := remoteSrv(t, s)
		chat(t, rs.URL, secret("two"), 1, as(2)) // tag anon:client-2
		sleepWake(t, s)
		testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
		if strings.Contains(erases(e), "save 1 ") {
			t.Fatalf("client-2's prompt was kept: %q", erases(e))
		}
		if hit, _ := chat(t, srv.URL, secret("two")+" and then?", 1, jsonCT); hit != 0 {
			t.Fatalf("the local probe after the wake: cache_n %d", hit)
		}
	})
	t.Run("unclean slot across a sleep", func(t *testing.T) {
		e, s, srv := clientsEnv(t, true, nil)
		_, rs, _ := remoteSrv(t, s)
		chat(t, rs.URL, secret("two"), 1, as(2)) // tag anon:client-2
		fail := filepath.Join(e.Ctl, "erasefail")
		if err := os.WriteFile(fail, nil, 0o644); err != nil {
			t.Fatal(err)
		}
		c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[{"role":"user","content":"x"}]}`, jsonCT)
		if c != 502 || codeOf(b) != "slot_not_cleared" {
			t.Fatalf("the local request on a slot it could not clear: %d %s", c, b)
		}
		os.Remove(fail)
		sleepWake(t, s)
		testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
		if strings.Contains(erases(e), "save 1 ") {
			t.Fatalf("the unclean slot was kept: %q", erases(e))
		}
		if hit, _ := chat(t, srv.URL, secret("two")+" and then?", 1, jsonCT); hit != 0 {
			t.Fatalf("the local probe after the wake: cache_n %d", hit)
		}
	})
	t.Run("a reshape that renumbers", func(t *testing.T) {
		e, s, srv := clientsEnv(t, false, nil) // no base: the commons is the slots no one owns
		_, rs, _ := remoteSrv(t, s)
		mine := secret("Kay")
		chat(t, srv.URL, mine, 2, jsonCT)        // nothing claimed: a trusted caller is not confined
		chat(t, rs.URL, secret("two"), 1, as(2)) // client-2 in the commons: anon, confined
		n := len(e.Starts())
		A := growRemote(t, e, s, rs, 1, "a", 4) // more than the 3 slots: a reshape makes the map again
		if len(e.Starts()) == n {
			t.Fatal("no reshape")
		}
		if strings.Contains(erases(e), "save 1 ") || !strings.Contains(erases(e), "save 2 ") {
			t.Fatalf("kept: %q (want slot 2, the commons', not slot 1, client-2's)", erases(e))
		}
		v, _ := s.ClientView(A.id)
		for i := 0; i < 4; i++ {
			if hit, _ := chat(t, rs.URL, mine+" and then?", i, as(1)); hit != 0 {
				t.Fatalf("A's probe in its slot %d (engine %d+): cache_n %d", i, *v.Slots.First, hit)
			}
			if hit, _ := chat(t, rs.URL, secret("two")+" and then?", i, as(1)); hit != 0 {
				t.Fatalf("A's probe of client-2's prompt in its slot %d: cache_n %d", i, hit)
			}
		}
	})
}
