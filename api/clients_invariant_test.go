package api_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// The invariant (docs/ENGINE.md §7g): a request runs only in slots of its own range, its
// attachment's or the commons; while the map holds any attachment's range no caller is run
// unconfined; a slot changes owner only through a restart or an erase; the RAM prompt cache is on
// only with no listener but loopback and one owner. The fake reports cache hits by content
// (cache_n is the common prefix with what the slot holds, or, with its RAM cache on, with any
// prompt any slot let go of), so a probe that starts with another owner's prompt shows a leak.

// invEnv is clientsEnv with the RAM prompt cache in the fake's build and remote saying whether
// the engine is told it has a non-loopback listener.
func invEnv(t *testing.T, base, remote bool) (*testkit.Env, *run.Supervisor, string) {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_ID_SLOT"] = "1"
	e.Llama["FAKELLAMA_TIMINGS"] = "1"
	e.Llama["FAKELLAMA_CACHE_RAM"] = "1"
	e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms" // named moment: the fake's pace between tokens
	e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	cfg := e.Config()
	cfg.ProfileGiven = base
	cfg.RemoteListener = remote
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv.URL
}

// secret is a prompt long enough that a hit on it is plain in cache_n.
func secret(who string) string {
	return who + "'s secret: " + strings.Repeat(who+" knows a thing no one else may read. ", 8)
}

// chat posts a chat completion of prompt and answers cache_n and the slot the reply names.
func chat(t *testing.T, url, prompt string, idSlot int, hdr map[string]string) (cacheN, slot int) {
	t.Helper()
	body := fmt.Sprintf(`{"messages":[{"role":"user","content":%q}]}`, prompt)
	if idSlot >= 0 {
		body = fmt.Sprintf(`{"id_slot":%d,"messages":[{"role":"user","content":%q}]}`, idSlot, prompt)
	}
	c, b, _ := do(t, "POST", url+"/v1/chat/completions", body, hdr)
	if c != 200 {
		t.Fatalf("ask %.30q in %d: %d %s", prompt, idSlot, c, b)
	}
	var rep struct {
		IDSlot  int `json:"id_slot"`
		Timings struct {
			CacheN int `json:"cache_n"`
		} `json:"timings"`
	}
	json.Unmarshal([]byte(b), &rep)
	return rep.Timings.CacheN, rep.IDSlot
}

func ramCacheFact(t *testing.T, s *run.Supervisor) bool {
	f, _ := s.Hub().Fact("wants", "")
	return strings.Contains(string(f.Body), `"ram_cache":true`)
}

func erases(e *testkit.Env) string {
	b, _ := os.ReadFile(filepath.Join(e.Ctl, "slots.log"))
	return string(b)
}

// D2, the remote first: a remote client is the one owner; a local caller then never runs in its
// slots (held while the engine reshapes to give the commons a range), and the remote's probe of
// the local's prompt is never a hit, in its slots or through the RAM cache.
func TestLocalPromptsNeverReachARemoteOwner(t *testing.T) {
	for _, remote := range []bool{true, false} {
		t.Run(fmt.Sprintf("remote listener %v", remote), func(t *testing.T) {
			e, s, local := invEnv(t, false, remote)
			_, rs, _ := remoteSrv(t, s)
			A := growRemote(t, e, s, rs, 1, "a", 2)
			if ramCacheFact(t, s) == remote {
				t.Fatalf("RAM cache with one owner and remote listener %v: %v", remote, ramCacheFact(t, s))
			}
			chat(t, rs.URL, secret("A"), 0, as(1))
			mine := secret("the keeper")
			_, ls := chat(t, local, mine, -1, jsonCT) // the commons: spare slots, or a reshape gives it some
			v, _ := s.ClientView(A.id)
			if v.Slots.First == nil || (ls >= *v.Slots.First && ls < *v.Slots.First+v.Slots.Given) || ramCacheFact(t, s) {
				t.Fatalf("the local caller ran in slot %d, A's %+v, RAM cache %v", ls, v.Slots, ramCacheFact(t, s))
			}
			for i := 0; i < 2; i++ {
				if hit, _ := chat(t, rs.URL, mine+" and then?", i, as(1)); hit != 0 {
					t.Fatalf("A's probe of the local prompt in its slot %d: cache_n %d", i, hit)
				}
			}
			A.cancel()
		})
	}
}

// D2, the local first: this machine's prompts are made before the remote attaches, and again
// while it is away; its probes never hit them, on its first attach nor on its return.
func TestRemoteOwnerNeverReadsLocalPrompts(t *testing.T) {
	for _, remote := range []bool{true, false} {
		t.Run(fmt.Sprintf("remote listener %v", remote), func(t *testing.T) {
			e, s, local := invEnv(t, false, remote)
			_, rs, _ := remoteSrv(t, s)
			before := secret("Kay")
			for i := 0; i < 2; i++ {
				chat(t, local, before, i, jsonCT) // no attachment: not confined
			}
			A := growRemote(t, e, s, rs, 1, "a", 2)
			for i := 0; i < 2; i++ {
				if hit, _ := chat(t, rs.URL, before+" and then?", i, as(1)); hit != 0 {
					t.Fatalf("A's probe of the earlier local prompt in its slot %d: cache_n %d", i, hit)
				}
			}
			A.cancel()
			attachmentGone(t, s, A.id)
			away := secret("Wren")
			for i := 0; i < 3; i++ {
				chat(t, local, away, -1, jsonCT)
			}
			R := openHold(t, rs, `{"hold":true,"name":"a","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(1))
			if R.status != 200 {
				t.Fatalf("A's return: %d %v", R.status, R.first)
			}
			R.untilGiven(2)
			testkit.WaitPhase(t, s, "ready", 15*time.Second)
			for i := 0; i < 2; i++ {
				if hit, _ := chat(t, rs.URL, away+" and then?", i, as(1)); hit != 0 {
					t.Fatalf("A's probe of the local prompt made while it was away, in its slot %d: cache_n %d", i, hit)
				}
			}
		})
	}
}

// A slot changes owner by an erase: D is given A's vacant range without a restart, and its first
// request in each slot erases what A left there (the fake logs it); its own next one is a hit.
func TestVacantRangeIsErasedForTheNextOwner(t *testing.T) {
	e, s, _ := clientsEnv(t, false, nil)
	_, rs, _ := remoteSrv(t, s)
	A := growRemote(t, e, s, rs, 1, "a", 2)
	growRemote(t, e, s, rs, 3, "c", 1)
	_, _, first := slotsOf(A.first) // A's slot 0 in the engine's numbering
	chat(t, rs.URL, secret("A"), 0, as(1))
	A.cancel()
	attachmentGone(t, s, A.id)
	n := len(e.Starts())
	D := openHold(t, rs, `{"hold":true,"name":"d","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(4))
	if D.status != 200 {
		t.Fatalf("D: %d %v", D.status, D.first)
	}
	D.untilGiven(2)
	if len(e.Starts()) != n {
		t.Fatalf("D was given A's vacant range by a restart: starts %d (from %d)", len(e.Starts()), n)
	}
	mark := fmt.Sprintf("erase %d\n", first)
	n0 := strings.Count(erases(e), mark)
	if hit, _ := chat(t, rs.URL, secret("A")+" and then?", 0, as(4)); hit != 0 {
		t.Fatalf("D's probe of A's prompt: cache_n %d", hit)
	}
	if strings.Count(erases(e), mark) != n0+1 {
		t.Fatalf("slot %d was not erased for D: %q", first, erases(e))
	}
	chat(t, rs.URL, secret("D"), 0, as(4))
	if hit, _ := chat(t, rs.URL, secret("D")+" and then?", 0, as(4)); hit == 0 {
		t.Fatal("D's own prompt is no hit in its own slot")
	}
	if strings.Count(erases(e), mark) != n0+1 {
		t.Fatalf("D's own slot erased again: %q", erases(e))
	}
}

// Two remote callers with no attachment share the commons, each under its own name: the second
// in a slot erases the first's prompt; a caller of this machine erases an untrusted one's too.
func TestCommonsSlotErasedBetweenCallers(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil) // the base: the commons, slots 0..2
	_, rs, _ := remoteSrv(t, s)
	chat(t, rs.URL, secret("two"), 1, as(2))
	n0 := strings.Count(erases(e), "erase 1\n")
	if hit, _ := chat(t, rs.URL, secret("two")+" and then?", 1, as(3)); hit != 0 {
		t.Fatalf("client-3's probe of client-2's prompt in the commons: cache_n %d", hit)
	}
	if strings.Count(erases(e), "erase 1\n") != n0+1 {
		t.Fatalf("no erase between two remote callers: %q", erases(e))
	}
	n := strings.Count(erases(e), "erase 1\n")
	if hit, _ := chat(t, srv.URL, secret("three")+" and then?", 1, jsonCT); hit != 0 {
		t.Fatalf("the local probe of client-3's prompt: cache_n %d", hit)
	}
	if strings.Count(erases(e), "erase 1\n") != n+1 {
		t.Fatalf("no erase before the local caller: %q", erases(e))
	}
	// trusted callers share the commons: no erase among them
	if hit, _ := chat(t, srv.URL, secret("three")+" and then? and then?", 1, jsonCT); hit == 0 {
		t.Fatal("the local caller's own prompt is no hit")
	}
}

// D5: a lone remote client that detaches leaves its range vacant and no commons: another remote
// caller is told the truth (attach to be given slots), and is served once it attaches.
func TestLoneRemoteDetachTellsTheTruth(t *testing.T) {
	e, s, _ := clientsEnv(t, false, nil)
	_, rs, _ := remoteSrv(t, s)
	A := growRemote(t, e, s, rs, 1, "a", 3) // every slot: no commons
	if v, _ := s.ClientView(A.id); s.Status().Shape.Slots != 3 || v.Slots.Given != 3 {
		t.Fatalf("A is not the one owner: %d slots, %+v", s.Status().Shape.Slots, v.Slots)
	}
	A.cancel()
	attachmentGone(t, s, A.id)
	for _, n := range []int{1, 2} { // the one that left, and another
		c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(n))
		var er struct {
			Error string `json:"error"`
		}
		json.Unmarshal([]byte(b), &er)
		if c != 409 || codeOf(b) != "attach_required" || er.Error != "this caller has no slots of its own in the running model: attach to be given slots" {
			t.Fatalf("client-%d after the lone client left: %d %s", n, c, b)
		}
	}
	B := openHold(t, rs, `{"hold":true,"name":"b","profile":{"slots":1}}`, tok(2))
	if B.status != 200 {
		t.Fatalf("B: %d %v", B.status, B.first)
	}
	B.untilGiven(1)
	chat(t, rs.URL, "hello", -1, as(2))
}
