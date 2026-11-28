package api_test

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// The review's cases (docs/ENGINE.md section 7g, "Whose slots"): two remote clients over paired
// tokens, the commons, the caps, forwarding headers, revocation, a body that stalls, a grow that
// drains (local) or waits (remote), and a server that will not start with a client's extra_args.

// revocable is a token checker whose tokens can be revoked (Valid is what ForgetRevoked asks).
type revocable struct {
	mu   sync.Mutex
	toks map[string]string // bearer -> client id
	gone map[string]bool   // client id -> revoked
}

func (c *revocable) Check(b string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	id, ok := c.toks[b]
	return id, ok && !c.gone[id]
}

func (c *revocable) Valid(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.gone[id]
}

func (c *revocable) revoke(id string) {
	c.mu.Lock()
	c.gone[id] = true
	c.mu.Unlock()
}

func newRevocable() *revocable {
	return &revocable{toks: map[string]string{"tok-1": "client-1", "tok-2": "client-2", "tok-3": "client-3"}, gone: map[string]bool{}}
}

// remoteSrv is a tokened listener on the same engine: every caller is a remote client.
func remoteSrv(t *testing.T, s *run.Supervisor) (*api.API, *httptest.Server, *revocable) {
	t.Helper()
	tk := newRevocable()
	a := &api.API{S: s, AnyHost: true, Tokens: tk}
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	return a, srv, tk
}

func as(n int, extra ...map[string]string) map[string]string {
	h := map[string]string{"Content-Type": "application/json", "X-Easy-Engine": "1", "Authorization": "Bearer tok-" + strconv.Itoa(n)}
	for _, x := range extra {
		for k, v := range x {
			h[k] = v
		}
	}
	return h
}

func tok(n int) map[string]string {
	return map[string]string{"Authorization": "Bearer tok-" + strconv.Itoa(n)}
}

// growRemote attaches a held remote client and waits until the engine gives it n slots.
func growRemote(t *testing.T, e *testkit.Env, s *run.Supervisor, rs *httptest.Server, n int, name string, slots int) *held {
	t.Helper()
	h := openHold(t, rs, fmt.Sprintf(`{"hold":true,"name":%q,"profile":{"slots":%d,"ctx_per_slot":{"max":8192,"floor":4096}}}`, name, slots), tok(n))
	if h.status != 200 {
		t.Fatalf("attach %s: %d %v", name, h.status, h.first)
	}
	h.first = h.untilGiven(slots) // the line that says it, read from here on
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return h
}

func slotActs(e *testkit.Env) string {
	b, _ := os.ReadFile(filepath.Join(e.Ctl, "slotacts.log"))
	return string(b)
}

func upstreamSlot(t *testing.T, e *testkit.Env) any {
	t.Helper()
	var up map[string]any
	json.Unmarshal([]byte(upstreamBytes(t, e)), &up)
	return up["id_slot"]
}

func TestTwoRemoteClientsIsolation(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil) // the base: slots 0..2
	_, rs, _ := remoteSrv(t, s)
	A := growRemote(t, e, s, rs, 1, "a", 2) // client-1: engine 3, 4
	B := growRemote(t, e, s, rs, 2, "b", 1) // client-2: engine 5
	_, _, firstA := slotsOf(A.untilGiven(2))
	_, _, firstB := slotsOf(B.untilGiven(1))
	if firstA != 3 || firstB != 5 {
		t.Fatalf("ranges: a from %d, b from %d", firstA, firstB)
	}
	post := func(n int, path, body string, extra ...map[string]string) (int, string) {
		c, b, _ := do(t, "POST", rs.URL+path, body, as(n, extra...))
		return c, b
	}
	refused := func(what string, c int, body, code string, wantStatus int) {
		t.Helper()
		if c != wantStatus || codeOf(body) != code {
			t.Fatalf("%s: %d %s (want %d %s)", what, c, body, wantStatus, code)
		}
	}
	// client-2 sends no header: its own one attachment, never a's slots
	if c, b := post(2, "/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"x"}]}`); c != 200 || upstreamSlot(t, e) != float64(5) {
		t.Fatalf("id_slot 0: %d %s upstream %v", c, b, upstreamSlot(t, e))
	}
	before := upstreamBytes(t, e)
	c, b := post(2, "/v1/chat/completions", `{"id_slot":1,"messages":[]}`)
	refused("id_slot 1", c, b, "slot_out_of_range", 400)
	c, b = post(2, "/v1/chat/completions", "\xEF\xBB\xBF"+`{"id_slot":0,"messages":[]}`)
	refused("a BOM body", c, b, "bad_request", 400)
	deep := strings.Repeat(`{"a":`, 200) + `1` + strings.Repeat(`}`, 200)
	c, b = post(2, "/v1/chat/completions", `{"id_slot":0,"x":`+deep+`}`)
	refused("a deep body", c, b, "bad_request", 400)
	c, b = post(2, "/v1/chat/completions", `[{"id_slot":3}]`)
	refused("an array body", c, b, "bad_request", 400)
	c, b = post(2, "/v1/chat/completions", `{"id_slot":"3","messages":[]}`)
	refused("a string id_slot", c, b, "bad_request", 400)
	if upstreamBytes(t, e) != before {
		t.Fatal("a refused body reached the model server")
	}
	// the paths beyond the model's completions never reach the door from a paired token: the
	// gate's scope refuses them (api/scope.go)
	for _, p := range []string{"/slots/0?action=erase", "/infill", "/v1/embeddings", "/v1/some/new/path"} {
		c, b = post(2, p, `{}`)
		refused("paired "+p, c, b, "not_allowed", 403)
	}
	// an --open listener's caller from another machine (here: through a forwarder that says so)
	// is not scoped by the gate: the door alone keeps it to its own slots. It attaches C (engine 6).
	openSrv := httptest.NewServer((&api.API{S: s, AnyHost: true, Open: true}).Handler())
	t.Cleanup(openSrv.Close)
	fwd := map[string]string{"X-Forwarded-For": "10.9.8.7"}
	C := openHold(t, openSrv, `{"hold":true,"name":"c","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, fwd)
	if C.status != 200 || C.first["client_id"] != "open@127.0.0.1" || C.first["local"] != false {
		t.Fatalf("open attach: %d %v", C.status, C.first)
	}
	C.untilGiven(1)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	opost := func(path, body string) (int, string) {
		c, b, _ := do(t, "POST", openSrv.URL+path, body, hdr(fwd))
		return c, b
	}
	acts := slotActs(e)
	c, b = opost("/slots/1x?action=erase", ``)
	refused("/slots/1x", c, b, "bad_request", 400)
	c, b = opost("/slots/%201?action=erase", ``)
	refused("/slots/ 1", c, b, "bad_request", 400)
	c, b = opost("/slots/%2B0?action=erase", ``)
	refused("/slots/+0", c, b, "bad_request", 400)
	c, b = opost("/slots/1?action=erase", ``)
	refused("/slots/1", c, b, "slot_out_of_range", 400)
	if slotActs(e) != acts {
		t.Fatalf("a refused slot action reached the model server: %q", slotActs(e))
	}
	opost("/slots/0?action=erase", ``)
	if got := strings.TrimPrefix(slotActs(e), acts); got != "/slots/6?action=erase\n" {
		t.Fatalf("erase of its slot 0 went to %q", got)
	}
	// /infill honours id_slot: confined as a completion, a bad body fails closed
	if c, b := opost("/infill", `{"id_slot":0,"input_prefix":"a","input_suffix":"b"}`); c != 200 || upstreamSlot(t, e) != float64(6) {
		t.Fatalf("/infill: %d %s upstream %v", c, b, upstreamSlot(t, e))
	}
	c, b = opost("/infill", "\xEF\xBB\xBF"+`{"id_slot":0}`)
	refused("/infill BOM", c, b, "bad_request", 400)
	// a path that cannot be confined: refused to a remote caller, a local one passes the door
	c, b = opost("/v1/embeddings", `{"input":"x"}`)
	refused("embeddings", c, b, "path_not_offered", 403)
	c, b = opost("/v1/some/new/path", `{}`)
	refused("an unknown POST", c, b, "path_not_offered", 403)
	if c, b, _ := do(t, "POST", srv.URL+"/v1/embeddings", `{"input":"x"}`, jsonCT); c == 403 {
		t.Fatalf("a local caller's embeddings: %d %s", c, b)
	}
	C.cancel()
	attachmentGone(t, s, C.id)
	// naming a's attachment: not_yours at the door and on DELETE
	before = upstreamBytes(t, e)
	c, b = post(2, "/v1/chat/completions", `{"id_slot":0,"messages":[]}`, asClient(A.id))
	refused("the door, a's id", c, b, "not_yours", 403)
	c, b, _ = do(t, "DELETE", rs.URL+"/engine/v1/attach/"+A.id, "", as(2))
	refused("DELETE a's id", c, b, "not_yours", 403)
	if upstreamBytes(t, e) != before {
		t.Fatal("not_yours reached the model server")
	}
	// client-3 holds nothing: the commons (the base's 0..2, engine numbers), never a client's
	c, b = post(3, "/v1/chat/completions", `{"id_slot":3,"messages":[]}`)
	refused("client-3 id_slot 3", c, b, "slot_not_offered", 400)
	if c, b := post(3, "/v1/chat/completions", `{"id_slot":2,"messages":[]}`); c != 200 || upstreamSlot(t, e) != float64(2) {
		t.Fatalf("client-3 id_slot 2: %d %s upstream %v", c, b, upstreamSlot(t, e))
	}
	if c, b := post(3, "/v1/chat/completions", `{"messages":[]}`); c != 200 {
		t.Fatalf("client-3 no id_slot: %d %s", c, b)
	} else if v, _ := upstreamSlot(t, e).(float64); v > 2 {
		t.Fatalf("client-3's pick went to %v", v)
	}
	// the open caller, gone, now holds nothing: the commons for it too, by the slot path as well
	c, b = opost("/slots/4?action=erase", ``)
	refused("open erase 4", c, b, "slot_not_offered", 400)
	c, b = opost("/infill", `{"id_slot":3}`)
	refused("open infill 3", c, b, "slot_not_offered", 400)
}

func TestClientRequiredAndAttachRequired(t *testing.T) {
	e, s, srv := clientsEnv(t, false, nil) // no base
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 2)
	growRemote(t, e, s, rs, 1, "a2", 1)
	// every slot is a client's: client-1 has two attachments, client-3 none
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(1))
	if c != 400 || codeOf(b) != "client_required" {
		t.Fatalf("two of its own, no header: %d %s", c, b)
	}
	c, b, _ = do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(3))
	if c != 409 || codeOf(b) != "attach_required" {
		t.Fatalf("none of its own, all owned: %d %s", c, b)
	}
	// a local caller passes as before
	if c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[]}`, jsonCT); c != 200 || upstreamSlot(t, e) != float64(1) {
		t.Fatalf("local: %d %s", c, b)
	}
}

func TestRemoteCapsAndPid(t *testing.T) {
	e, s, srv := clientsEnv(t, false, nil)
	_ = e
	_, rs, _ := remoteSrv(t, s)
	// a pid names a process on this machine
	c, b, _ := do(t, "POST", rs.URL+"/engine/v1/attach", fmt.Sprintf(`{"pid":%d,"profile":{}}`, os.Getpid()), as(1))
	if c != 400 || codeOf(b) != "pid_remote" {
		t.Fatalf("remote pid: %d %s", c, b)
	}
	// a remote attach body is at most 64 KiB
	big := `{"hold":true,"name":"x","profile":{"game":"` + strings.Repeat("x", 70<<10) + `"}}`
	c, b, _ = do(t, "POST", rs.URL+"/engine/v1/attach", big, as(1))
	if c != 413 {
		t.Fatalf("70 KiB remote body: %d %.200s", c, b)
	}
	// four attachments per remote identity
	for i := 0; i < 4; i++ {
		if h := openHold(t, rs, fmt.Sprintf(`{"hold":true,"name":"h%d","profile":{}}`, i), tok(1)); h.status != 200 {
			t.Fatalf("hold %d: %d %v", i, h.status, h.first)
		}
	}
	h := openHold(t, rs, `{"hold":true,"name":"h4","profile":{}}`, tok(1))
	if h.status != 429 || h.first["code"] != "too_many_attachments" {
		t.Fatalf("the fifth: %d %v", h.status, h.first)
	}
	// 32 in all: local pid attachments fill the rest
	for i := 0; i < 28; i++ {
		c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", fmt.Sprintf(`{"pid":%d,"name":"p%d","profile":{}}`, os.Getpid(), i), csrfHdr)
		if c != 200 {
			t.Fatalf("local %d: %d %s", i, c, b)
		}
	}
	c, b, _ = do(t, "POST", srv.URL+"/engine/v1/attach", fmt.Sprintf(`{"pid":%d,"name":"p-last","profile":{}}`, os.Getpid()), csrfHdr)
	if c != 429 || codeOf(b) != "too_many_attachments" {
		t.Fatalf("the 33rd: %d %s", c, b)
	}
}

func TestForwardingHeaderIsNeverLocal(t *testing.T) {
	_, _, srv := clientsEnv(t, false, nil)
	body := fmt.Sprintf(`{"pid":%d,"profile":{}}`, os.Getpid())
	for _, h := range []string{"X-Forwarded-For", "Forwarded", "X-Real-IP", "Via", "X-Forwarded-Host", "Tailscale-User-Login"} {
		c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", body, hdr(map[string]string{"X-Easy-Engine": "1", h: "10.1.2.3"}))
		if c != 400 || codeOf(b) != "pid_remote" {
			t.Fatalf("%s: %d %s", h, c, b)
		}
	}
	if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", body, csrfHdr); c != 200 {
		t.Fatalf("no forwarding header: %d %s", c, b)
	}
}

func TestRevokeEndsAttachmentsAtOnce(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	a, rs, tk := remoteSrv(t, s)
	A := growRemote(t, e, s, rs, 1, "a", 2)
	tk.revoke("client-1")
	a.ForgetRevoked()
	for {
		m, ok := A.next("the revoked end")
		if !ok {
			t.Fatal("the stream ended with no end line")
		}
		if m["end"] != nil {
			if m["end"] != "revoked" {
				t.Fatalf("end %v", m["end"])
			}
			break
		}
	}
	attachmentGone(t, s, A.id)
}

func TestAttachParentFailureKeepsTheStanding(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	s := e.StartSup(cfg)
	srv := httptest.NewServer((&api.API{S: s, Attach: func(int) error { return errors.New("not a parent") }}).Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	x := openHold(t, srv, `{"hold":true,"name":"x","profile":{}}`, nil)
	if x.status != 200 {
		t.Fatalf("x: %d %v", x.status, x.first)
	}
	c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", fmt.Sprintf(`{"pid":%d,"name":"x","profile":{}}`, os.Getpid()), csrfHdr)
	if c != 400 {
		t.Fatalf("the failed parent: %d %s", c, b)
	}
	if _, ok := s.Hub().Fact("attachment", x.id); !ok {
		t.Fatal("the standing attachment left when a replace's parent failed")
	}
	select {
	case m, ok := <-x.lines:
		if ok && m["end"] != nil {
			t.Fatalf("the standing hold ended: %v", m)
		}
	default:
	}
}

func TestBodySilenceCutsBeforeTheDoor(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	srv := httptest.NewServer((&api.API{S: s, BodySilence: 200 * time.Millisecond}).Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	mark := testkit.Mark(t, s, "before the slow body")
	conn, err := net.Dial("tcp", srv.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"messages\"")
	// network timeout: the engine must answer long before this
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		t.Fatalf("no answer to a stalled body: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("stalled body: %d", resp.StatusCode)
	}
	// it never entered the door: no request was begun
	evs, _ := s.Hub().Since(mark, "request")
	for _, ev := range evs {
		if ev.Name == "request" {
			t.Fatalf("a request was counted: %+v", ev)
		}
	}
}

// streamGated starts a streamed anonymous completion that the fake holds after its first chunk
// (FAKELLAMA_STREAM_GATE); it answers the response once its first chunk is read.
func streamGated(t *testing.T, base string) (*http.Response, *bufio.Reader) {
	t.Helper()
	req, _ := http.NewRequest("POST", base+"/v1/chat/completions", strings.NewReader(`{"stream":true,"messages":[{"role":"user","content":"x"}]}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	br := bufio.NewReader(resp.Body)
	if _, err := br.ReadString('\n'); err != nil {
		t.Fatalf("first chunk: %v", err)
	}
	return resp, br
}

func gatedEnv(t *testing.T, drainMax time.Duration) (*testkit.Env, *run.Supervisor, *httptest.Server) {
	t.Helper()
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_STREAM_GATE"] = "1"
	e.Llama["FAKELLAMA_STREAM_DELAY"] = "1ms" // named moment: the fake's pace between tokens
	e.Prof.CtxPerSlot.Max, e.Prof.CtxPerSlot.Floor = 8192, 4096
	cfg := e.Config()
	cfg.ProfileGiven = true
	cfg.DrainMax = drainMax // named moment: how long a local grow waits for the replies in flight
	s, srv := serve(t, e, cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return e, s, srv
}

func TestLocalGrowDrainsWithinDrainMax(t *testing.T) {
	e, _, srv := gatedEnv(t, 300*time.Millisecond)
	n0 := len(e.Starts())
	_, br := streamGated(t, srv.URL)
	b := openHold(t, srv, `{"hold":true,"name":"b","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	if b.status != 200 {
		t.Fatalf("attach: %d %v", b.status, b.first)
	}
	// the stream in flight is cut once DrainMax comes: the grow does not wait for it
	rest := ""
	for {
		l, err := br.ReadString('\n')
		rest += l
		if err != nil {
			break
		}
	}
	b.untilGiven(2)
	if len(e.Starts()) <= n0 {
		t.Fatal("no start for the grow")
	}
	if !strings.Contains(rest, "error") {
		t.Fatalf("the cut stream had no terminal event: %q", rest)
	}
}

func TestRemoteGrowWaitsForIdle(t *testing.T) {
	e, s, srv := gatedEnv(t, 300*time.Millisecond)
	_, rs, _ := remoteSrv(t, s)
	n0 := len(e.Starts())
	_, br := streamGated(t, srv.URL)
	h := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(1))
	if h.status != 200 {
		t.Fatalf("attach: %d %v", h.status, h.first)
	}
	sl, _ := h.first["slots"].(map[string]any)
	if sl["pending"] != true || sl["why"] != "waiting_for_idle" {
		t.Fatalf("not told it waits for idle: %v", sl)
	}
	// well past DrainMax the stream still runs: a remote grow never drains
	mark := testkit.Mark(t, s, "the stream still runs")
	_ = mark
	for i := 1; i <= 4; i++ {
		if len(e.Starts()) != n0 || s.Phase() != "ready" {
			t.Fatalf("the engine was restarted or drained for a remote client with a reply in flight (phase %s)", s.Phase())
		}
		if err := os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := br.ReadString('\n'); err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
	}
	// the door's next empty moment: the grow
	h.untilGiven(2)
	testkit.WaitFor(t, 15*time.Second, "the start for r", func() bool { return len(e.Starts()) > n0 })
}

func TestServerThatWillNotStartWithExtraArgs(t *testing.T) {
	e, s, srv := clientsEnv(t, true, func(e *testkit.Env) { e.Llama["FAKELLAMA_DIE_ON"] = "--top-n-sigma" })
	n0 := len(e.Starts())
	h := openHold(t, srv, `{"hold":true,"name":"x","profile":{"slots":1,"server":{"extra_args":["--top-n-sigma","1.5"]}}}`, nil)
	if h.status != 200 {
		t.Fatalf("attach: %d %v", h.status, h.first)
	}
	m := h.untilGiven(1)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	for {
		sv, _ := m["server"].(map[string]any)
		nt, _ := sv["not_taken"].([]any)
		if len(nt) == 1 {
			x := nt[0].(map[string]any)
			if x["option"] != "--top-n-sigma" || x["why"] != "the model server did not start with them" {
				t.Fatalf("not taken: %v", x)
			}
			break
		}
		var ok bool
		if m, ok = h.next("the not_taken report"); !ok {
			t.Fatal("stream ended")
		}
	}
	starts := e.Starts()[n0:]
	if len(starts) != 2 || !strings.Contains(starts[0], "--top-n-sigma") || strings.Contains(starts[1], "--top-n-sigma") {
		t.Fatalf("starts: %q", starts)
	}
	f, _ := s.Hub().Fact("attachment", h.id)
	if !strings.Contains(string(f.Body), "the model server did not start with them") {
		t.Fatalf("the fact does not say it: %s", f.Body)
	}
}
