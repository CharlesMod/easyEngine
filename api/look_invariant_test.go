package api_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// lookInvEnv is a look engine whose fake says the slot of each reply and its cache_n; base says
// whether a profile is given (the base client and its commons, slots 0..2).
func lookInvEnv(t *testing.T, base bool) (*testkit.Env, *run.Supervisor, string) {
	t.Helper()
	return lookEnvE(t, func(e *testkit.Env) {
		e.Llama["FAKELLAMA_ID_SLOT"] = "1"
		e.Llama["FAKELLAMA_TIMINGS"] = "1"
		e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms" // named moment: the fake's pace between tokens
		e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	}, func(c *run.Config) { c.ProfileGiven = base }, true)
}

// A look is confined as its caller's chat request is (the slot invariant): a look by one remote
// owner in a commons slot, then a chat by another owner in that slot, is erased between them, and
// the second owner's probe that begins with the look's prompt (its label, its image) hits nothing;
// a second owner's look in that slot is erased before too.
func TestLookCommonsErasedForNextOwner(t *testing.T) {
	e, s, _ := lookInvEnv(t, true)
	_, rs, _ := remoteSrv(t, s)
	img := pngB64(t, 40, 30)
	label := "two's secret: the vault opens with 4417, and no one else may know it"
	look := fmt.Sprintf(`{"images":[{"data":%q,"label":%q}],"question":"what is here?","id_slot":1}`, img, label)
	if code, b := postLook(t, rs.URL, look, as(2)); code != 200 || upstreamSlot(t, e) != float64(1) {
		t.Fatalf("client-2's look: %d %s slot %v", code, b, upstreamSlot(t, e))
	}
	n0 := strings.Count(erases(e), "erase 1\n")
	if hit, _ := chat(t, rs.URL, label+" and then?", 1, as(3)); hit != 0 {
		t.Fatalf("client-3's probe of client-2's look prompt in the commons: cache_n %d", hit)
	}
	if strings.Count(erases(e), "erase 1\n") != n0+1 {
		t.Fatalf("no erase between client-2's look and client-3's chat: %q", erases(e))
	}
	// client-2's look takes the slot back from client-3: erased first
	n1 := strings.Count(erases(e), "erase 1\n")
	if code, b := postLook(t, rs.URL, look, as(2)); code != 200 {
		t.Fatalf("client-2's second look: %d %s", code, b)
	}
	if strings.Count(erases(e), "erase 1\n") != n1+1 {
		t.Fatalf("no erase before client-2's look took the slot back from client-3: %q", erases(e))
	}
	n2 := strings.Count(erases(e), "erase 1\n")
	if code, b := postLook(t, rs.URL, look, as(3)); code != 200 {
		t.Fatalf("client-3's look: %d %s", code, b)
	}
	if strings.Count(erases(e), "erase 1\n") != n2+1 {
		t.Fatalf("no erase between two owners' looks: %q", erases(e))
	}
}
