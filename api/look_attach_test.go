package api_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// A remote caller with no attachment of its own: its look meets the same door as its chat
// request (the commons, attach_required, the id_slot confinement), nothing more and nothing less.

func lookClientsEnv(t *testing.T, base bool) (*testkit.Env, *run.Supervisor) {
	e, s, _ := lookEnvE(t, func(e *testkit.Env) {
		e.Llama["FAKELLAMA_ID_SLOT"] = "1"
		e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms" // named moment: the fake's pace between tokens
		e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	}, func(c *run.Config) { c.ProfileGiven = base }, true)
	return e, s
}

// 5a. Served from the commons (200) exactly when its chat request is.
func TestLookUnattachedRemoteServedFromTheCommons(t *testing.T) {
	e, s := lookClientsEnv(t, true) // the base: slots 0..2
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 2) // client-1: engine slots 3, 4
	testkit.WaitFor(t, 20*time.Second, "vision loaded after the grow", func() bool { return s.VisionLoaded() })
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(3))
	if c != 200 {
		t.Fatalf("chat of an unattached caller: %d %s", c, b)
	}
	code, body := postLook(t, rs.URL, oneLook(pngB64(t, 20, 20), ""), as(3))
	if code != 200 || upstreamSlot(t, e) == nil {
		t.Fatalf("look of an unattached caller: %d %s", code, body)
	}
	if sl, ok := upstreamSlot(t, e).(float64); !ok || sl > 2 {
		t.Fatalf("the look ran in slot %v, want the commons (0..2)", upstreamSlot(t, e))
	}
}

// 5b. Where its chat request is 409 attach_required, so is its look, and nothing is forwarded.
func TestLookUnattachedRemoteAttachRequired(t *testing.T) {
	e, s := lookClientsEnv(t, false) // no base: every slot is a client's
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 2)
	growRemote(t, e, s, rs, 1, "a2", 1)
	testkit.WaitFor(t, 20*time.Second, "vision loaded after the grows", func() bool { return s.VisionLoaded() })
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(3))
	if c != 409 || codeOf(b) != "attach_required" {
		t.Fatalf("chat: %d %s", c, b)
	}
	before := upstreamBytes(t, e)
	img := pngB64(t, 20, 20)
	for _, extra := range []string{"", `,"id_slot":0`} {
		code, body := postLook(t, rs.URL, oneLook(img, extra), as(3))
		if code != 409 || codeOf(body) != "attach_required" {
			t.Fatalf("look %q: %d %s", extra, code, body)
		}
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("a refused look reached the model server")
	}
}

// 5c. An id_slot from an unattached remote caller is confined as its chat request's is: the same
// status and code for the same id_slot.
func TestLookUnattachedRemoteIDSlotMatchesChat(t *testing.T) {
	e, s := lookClientsEnv(t, true)
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 2)
	growRemote(t, e, s, rs, 2, "b", 1)
	testkit.WaitFor(t, 20*time.Second, "vision loaded after the grows", func() bool { return s.VisionLoaded() })
	img := pngB64(t, 20, 20)
	var ok, refused int
	for _, slot := range []int{0, 1, 2, 3, 4, 5, 6, 1023} {
		cc, cb, _ := do(t, "POST", rs.URL+"/v1/chat/completions", fmt.Sprintf(`{"id_slot":%d,"messages":[]}`, slot), as(3))
		lc, lb := postLook(t, rs.URL, oneLook(img, fmt.Sprintf(`,"id_slot":%d`, slot)), as(3))
		if cc != lc || codeOf(cb) != codeOf(lb) {
			t.Fatalf("id_slot %d: chat %d %s, look %d %s", slot, cc, cb, lc, lb)
		}
		if cc == 200 {
			ok++
		} else {
			refused++
		}
	}
	if ok == 0 || refused == 0 {
		t.Fatalf("the sweep proved nothing: %d served, %d refused", ok, refused)
	}
}
