package api_test

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
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
	toks := map[string]string{}
	for i := 1; i <= 6; i++ {
		toks["tok-"+strconv.Itoa(i)] = "client-" + strconv.Itoa(i)
	}
	return &revocable{toks: toks, gone: map[string]bool{}}
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
	// is held to the paired allow list on the proxy: the same lists (api/scope.go)
	openSrv := httptest.NewServer((&api.API{S: s, AnyHost: true, Open: true}).Handler())
	t.Cleanup(openSrv.Close)
	fwd := map[string]string{"X-Forwarded-For": "10.9.8.7"}
	opost := func(path, body string) (int, string) {
		c, b, _ := do(t, "POST", openSrv.URL+path, body, hdr(fwd))
		return c, b
	}
	for _, p := range []string{"/slots/0?action=erase", "/infill", "/v1/embeddings", "/v1/some/new/path", "/v1/responses", "/v1/streams/lookup"} {
		c, b = opost(p, `{}`)
		refused("open "+p, c, b, "not_allowed", 403)
	}
	for _, p := range []string{"/slots", "/props", "/metrics", "/lora-adapters", "/v1/stream?conv_id=x"} {
		c, b, _ := do(t, "GET", openSrv.URL+p, "", fwd)
		refused("open GET "+p, c, b, "not_allowed", 403)
	}
	C := openHold(t, openSrv, `{"hold":true,"name":"c","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, fwd)
	if C.status != 200 || C.first["client_id"] != "open@127.0.0.1" || C.first["local"] != false {
		t.Fatalf("open attach: %d %v", C.status, C.first)
	}
	C.untilGiven(1) // engine 6
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if c, b := opost("/v1/chat/completions", `{"id_slot":0,"messages":[]}`); c != 200 || upstreamSlot(t, e) != float64(6) {
		t.Fatalf("open completion: %d %s upstream %v", c, b, upstreamSlot(t, e))
	}
	// the master key from another machine is the owner: not held to the list, but the door still
	// refuses it what cannot be confined while clients hold slots, and the stream paths
	msrv := httptest.NewServer((&api.API{S: s, AnyHost: true, Tokens: testChecker{"tok-m": "master"}}).Handler())
	t.Cleanup(msrv.Close)
	mh := map[string]string{"Authorization": "Bearer tok-m", "X-Forwarded-For": "10.9.8.6", "Content-Type": "application/json"}
	for _, p := range []string{"/v1/embeddings", "/v1/some/new/path", "/v1/responses", "/v1/streams/lookup"} {
		c, b, _ := do(t, "POST", msrv.URL+p, `{}`, mh)
		refused("master "+p, c, b, "path_not_offered", 403)
	}
	for _, m := range []string{"GET", "DELETE"} {
		c, b, _ := do(t, m, msrv.URL+"/v1/stream?conv_id=x", "", mh)
		refused("master "+m+" /v1/stream", c, b, "path_not_offered", 403)
	}
	// a local attachment L (engine 7): the slot path takes digits only, its own numbers
	L := openHold(t, srv, `{"hold":true,"name":"l","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	L.untilGiven(1)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	lpost := func(path, body string) (int, string) {
		c, b, _ := do(t, "POST", srv.URL+path, body, asClient(L.id))
		return c, b
	}
	acts := slotActs(e)
	c, b = lpost("/slots/1x?action=erase", ``)
	refused("/slots/1x", c, b, "bad_request", 400)
	c, b = lpost("/slots/%201?action=erase", ``)
	refused("/slots/ 1", c, b, "bad_request", 400)
	c, b = lpost("/slots/%2B0?action=erase", ``)
	refused("/slots/+0", c, b, "bad_request", 400)
	c, b = lpost("/slots/1?action=erase", ``)
	refused("/slots/1", c, b, "slot_out_of_range", 400)
	if slotActs(e) != acts {
		t.Fatalf("a refused slot action reached the model server: %q", slotActs(e))
	}
	lpost("/slots/0?action=erase", ``)
	if got := strings.TrimPrefix(slotActs(e), acts); got != "/slots/7?action=erase\n" {
		t.Fatalf("erase of its slot 0 went to %q", got)
	}
	// /infill honours id_slot: confined as a completion, a bad body fails closed
	if c, b := lpost("/infill", `{"id_slot":0,"input_prefix":"a","input_suffix":"b"}`); c != 200 || upstreamSlot(t, e) != float64(7) {
		t.Fatalf("/infill: %d %s upstream %v", c, b, upstreamSlot(t, e))
	}
	c, b = lpost("/infill", "\xEF\xBB\xBF"+`{"id_slot":0}`)
	refused("/infill BOM", c, b, "bad_request", 400)
	// a local caller passes the door on a path that cannot be confined
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
	// the open caller, gone, now holds nothing: the commons for it too
	c, b = opost("/v1/chat/completions", `{"id_slot":3,"messages":[]}`)
	refused("open id_slot 3", c, b, "slot_not_offered", 400)
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
	// four attachments per remote identity; sixteen remote in all; neither counts a local one
	for n := 1; n <= 4; n++ {
		for i := 0; i < 4; i++ {
			if h := openHold(t, rs, fmt.Sprintf(`{"hold":true,"name":"h%d","profile":{}}`, i), tok(n)); h.status != 200 {
				t.Fatalf("client-%d hold %d: %d %v", n, i, h.status, h.first)
			}
		}
	}
	h := openHold(t, rs, `{"hold":true,"name":"h4","profile":{}}`, tok(1))
	if h.status != 429 || h.first["code"] != "too_many_attachments" {
		t.Fatalf("the fifth of one client: %d %v", h.status, h.first)
	}
	h = openHold(t, rs, `{"hold":true,"name":"h0","profile":{}}`, tok(5))
	if h.status != 429 || h.first["code"] != "too_many_attachments" {
		t.Fatalf("the seventeenth remote: %d %v", h.status, h.first)
	}
	// 32 local attachments, whatever the remote ones hold
	for i := 0; i < 32; i++ {
		c, b, _ := do(t, "POST", srv.URL+"/engine/v1/attach", fmt.Sprintf(`{"pid":%d,"name":"p%d","profile":{}}`, os.Getpid(), i), csrfHdr)
		if c != 200 {
			t.Fatalf("local %d: %d %s", i, c, b)
		}
	}
	c, b, _ = do(t, "POST", srv.URL+"/engine/v1/attach", fmt.Sprintf(`{"pid":%d,"name":"p-last","profile":{}}`, os.Getpid()), csrfHdr)
	if c != 429 || codeOf(b) != "too_many_attachments" {
		t.Fatalf("the 33rd local: %d %s", c, b)
	}
}

func TestRemoteSlotCaps(t *testing.T) {
	e, s, srv := clientsEnv(t, false, nil)
	_, rs, _ := remoteSrv(t, s)
	want := func(n int, name string, slots int) *held {
		return openHold(t, rs, fmt.Sprintf(`{"hold":true,"name":%q,"profile":{"slots":%d,"ctx_per_slot":{"max":8192,"floor":1024}}}`, name, slots), tok(n))
	}
	if h := want(1, "a", 10); h.status != 200 {
		t.Fatalf("10 slots: %d %v", h.status, h.first)
	}
	if h := want(1, "b", 7); h.status != 429 || h.first["code"] != "too_many_slots" {
		t.Fatalf("17 for one client: %d %v", h.status, h.first)
	}
	if h := want(2, "a", 16); h.status != 200 {
		t.Fatalf("16 for another: %d %v", h.status, h.first)
	}
	if h := want(3, "a", 7); h.status != 429 || h.first["code"] != "too_many_slots" {
		t.Fatalf("33 remote in all: %d %v", h.status, h.first)
	}
	// a local caller's slots are counted over local attachments only
	if h := openHold(t, srv, `{"hold":true,"name":"l","profile":{"slots":16,"ctx_per_slot":{"max":8192,"floor":1024}}}`, nil); h.status != 200 {
		t.Fatalf("local 16 beside 26 remote: %d %v", h.status, h.first)
	}
	_ = e
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
	time.Sleep(3 * 300 * time.Millisecond) // named moment: three DrainMax pass before the gate opens
	s.Nudge("test")                        // the reactor looks again
	testkit.Mark(t, s, "past DrainMax")
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

// restartsLeft is a held line's restarts_left (-1: absent).
func restartsLeft(m map[string]any) int {
	v, ok := m["restarts_left"].(float64)
	if !ok {
		return -1
	}
	return int(v)
}

const loneR = `{"hold":true,"name":"r","profile":{"slots":2,"ctx_per_slot":{"max":8192,"floor":4096}}}`

// A lone remote client that drops and returns, idle: no restart, no charge, its cache.
func TestLoneClientDropAndReturnIdle(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	n0 := len(e.Starts())
	R := growRemote(t, e, s, rs, 1, "r", 2)
	if got := len(e.Starts()); got != n0+1 || restartsLeft(R.first) != 3 {
		t.Fatalf("the grow: starts %d (from %d), restarts left %d", got, n0, restartsLeft(R.first))
	}
	if c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"one two three"}]}`, as(1)); c != 200 {
		t.Fatalf("warm: %d %s", c, b)
	}
	R.cancel()
	attachmentGone(t, s, R.id)
	s.Nudge("test") // the reactor looks once after the leave: nothing to do
	testkit.Mark(t, s, "after the leave")
	R2 := openHold(t, rs, loneR, tok(1))
	if R2.status != 200 || R2.first["resumed"] != true || restartsLeft(R2.first) != 3 {
		t.Fatalf("return: %d %v", R2.status, R2.first)
	}
	if got := len(e.Starts()); got != n0+1 {
		t.Fatalf("a drop and return restarted the model: %d", got)
	}
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"id_slot":0,"messages":[{"role":"user","content":"one two three four five six seven eight nine ten eleven"}]}`, as(1))
	var rep struct {
		Timings struct {
			CacheN int `json:"cache_n"`
		}
	}
	json.Unmarshal([]byte(b), &rep)
	if c != 200 || rep.Timings.CacheN <= 0 {
		t.Fatalf("after the return: %d %s", c, b)
	}
}

// Drop and return while another caller's reply is in flight: still no restart, no charge.
func TestLoneClientDropAndReturnBusy(t *testing.T) {
	e, s, srv := gatedEnv(t, 300*time.Millisecond)
	_, rs, _ := remoteSrv(t, s)
	R := growRemote(t, e, s, rs, 1, "r", 2)
	n1 := len(e.Starts())
	_, br := streamGated(t, srv.URL)
	R.cancel()
	attachmentGone(t, s, R.id)
	R2 := openHold(t, rs, loneR, tok(1))
	if R2.status != 200 || R2.first["resumed"] != true || restartsLeft(R2.first) != 3 {
		t.Fatalf("return: %d %v", R2.status, R2.first)
	}
	for i := 1; i <= 4; i++ {
		os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		br.ReadString('\n')
	}
	s.Nudge("test") // the reactor looks again
	testkit.Mark(t, s, "after the stream")
	if got := len(e.Starts()); got != n1 {
		t.Fatalf("restarted: %d -> %d", n1, got)
	}
}

// The model stops for another reason (an unload): the next start uses the wants that stand (no
// client), and the client's return grows it again: a restart its own attach causes, charged as an
// attach is (a free return is only one served from its standing range).
func TestLoneClientReturnAfterAnUnload(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	R := growRemote(t, e, s, rs, 1, "r", 2)
	R.cancel()
	attachmentGone(t, s, R.id)
	if err := s.Stop(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 15*time.Second, "stopped", func() bool { return s.Phase() != "ready" })
	n := len(e.Starts())
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 15*time.Second, "started with the standing wants", func() bool { return len(e.Starts()) > n && s.Phase() == "ready" })
	if sl := s.Status().Shape.Slots; sl != 3 {
		t.Fatalf("the start after the unload has %d slots, want the base's 3", sl)
	}
	n = len(e.Starts())
	R2 := openHold(t, rs, loneR, tok(1))
	m := R2.untilGiven(2)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if got := len(e.Starts()); got != n+1 || restartsLeft(m) != 2 {
		t.Fatalf("return after an unload: starts %d (from %d), restarts left %d", got, n, restartsLeft(m))
	}
}

// Another caller reshaped the client's range away (a local client that wanted more); the
// client's return grows the model again, charged (S2: else leave and return would restart the
// model for nothing, forever).
func TestLoneClientReturnAfterAnotherReshaped(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	R := growRemote(t, e, s, rs, 1, "r", 2)
	R.cancel()
	attachmentGone(t, s, R.id)
	n := len(e.Starts())
	L := openHold(t, srv, `{"hold":true,"name":"l","profile":{"slots":3,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	L.untilGiven(3)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if got := len(e.Starts()); got != n+1 {
		t.Fatalf("the local grow: %d starts (from %d)", got, n)
	}
	R2 := openHold(t, rs, loneR, tok(1))
	if R2.first["resumed"] == true {
		t.Fatalf("resumed a range that is gone: %v", R2.first)
	}
	m := R2.untilGiven(2)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if got := len(e.Starts()); got != n+2 || restartsLeft(m) != 2 {
		t.Fatalf("return: starts %d (from %d), restarts left %d", got, n+1, restartsLeft(m))
	}
	// a change of its wants is charged
	R3 := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":3,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(1))
	m = R3.untilGiven(3)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if restartsLeft(m) != 1 {
		t.Fatalf("a change of wants: restarts left %d", restartsLeft(m))
	}
}

// A remote grow with no other traffic happens at once, and the client's own held request does
// not block the idle it waits for.
func TestRemoteGrowAtOnceWhenIdle(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	n := len(e.Starts())
	R := openHold(t, rs, loneR, tok(1))
	if R.status != 200 {
		t.Fatalf("attach: %d %v", R.status, R.first)
	}
	// its request, sent before the grow is done, is held at the door (not in flight) and served
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"id_slot":1,"messages":[]}`, as(1))
	if c != 200 || upstreamSlot(t, e) != float64(4) {
		t.Fatalf("its request: %d %s upstream %v", c, b, upstreamSlot(t, e))
	}
	if got := len(e.Starts()); got != n+1 {
		t.Fatalf("starts %d (from %d)", got, n)
	}
}

func convLog(e *testkit.Env) []string {
	b, _ := os.ReadFile(filepath.Join(e.Ctl, "convs.log"))
	return strings.Fields(strings.ReplaceAll(string(b), "\n", " "))
}

// Conversation ids are the engine's name space: B cannot replay or cancel A's stream with A's id.
func TestConversationIDsCarryTheirOwner(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	A := openHold(t, srv, `{"hold":true,"name":"a","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	A.untilGiven(1)
	B := openHold(t, srv, `{"hold":true,"name":"b","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	B.untilGiven(1)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	withConv := func(att string) map[string]string {
		h := asClient(att)
		h["X-Conversation-Id"] = "abc"
		return h
	}
	if c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"messages":[]}`, withConv(A.id)); c != 200 {
		t.Fatalf("A: %d %s", c, b)
	}
	do(t, "GET", srv.URL+"/v1/stream?conv_id=abc", "", asClient(B.id))
	do(t, "DELETE", srv.URL+"/v1/stream?conv_id=abc", "", asClient(B.id))
	do(t, "GET", srv.URL+"/v1/stream?conv_id=abc&from=0", "", asClient(A.id))
	do(t, "GET", srv.URL+"/v1/stream?conv_id=abc", "", nil) // the commons
	l := convLog(e)
	if len(l) != 10 || l[0] != "HEADER" || l[2] != "GET" || l[4] != "DELETE" || l[6] != "GET" || l[8] != "GET" {
		t.Fatalf("log %v", l)
	}
	idA, idB, idA2, idC := l[1], l[3], l[7], l[9]
	if idA == "abc" || idB == idA || l[5] != idB || idA2 != idA || idC == idA || idC == idB {
		t.Fatalf("ids: A %s, B %s/%s, A again %s, commons %s", idA, idB, l[5], idA2, idC)
	}
	// the lookup's ids go in owned and come back as the client sent them
	c, b, _ := do(t, "POST", srv.URL+"/v1/streams/lookup", `{"conversation_ids":["abc"]}`, hdr(asClient(A.id)))
	if c != 200 || !strings.Contains(b, `"conversation_id":"abc"`) || strings.Contains(b, idA) {
		t.Fatalf("lookup: %d %s", c, b)
	}
	if l := convLog(e); l[len(l)-1] != idA {
		t.Fatalf("lookup sent %s, want %s", l[len(l)-1], idA)
	}
	// a remote caller never reaches the stream paths
	_, rs, _ := remoteSrv(t, s)
	for _, m := range []string{"GET", "DELETE"} {
		if c, b, _ := do(t, m, rs.URL+"/v1/stream?conv_id=abc", "", as(1)); c != 403 {
			t.Fatalf("remote %s /v1/stream: %d %s", m, c, b)
		}
	}
}

// n / n_cmpl above 1 is refused in a confined request; one completion passes; unconfined, as before.
func TestNAboveOneNotOffered(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil)
	if c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", `{"n":2,"messages":[]}`, jsonCT); c != 200 {
		t.Fatalf("unconfined n 2: %d %s", c, b)
	}
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 2)
	before := upstreamBytes(t, e)
	// every value that is not the literal 1: the model server casts to int32 (-4294967294 and
	// 4294967297 wrap to 2 and 1, -4294967292 to 4), so the token is judged, not a number
	for _, v := range []string{`2`, `-4294967294`, `-4294967292`, `4294967297`, `1e0`, `1.0`, `1e400`, `"1"`, `true`, `[1]`, `1.5`, `0`, `-0`, `{}`} {
		for _, k := range []string{"n", "n_cmpl"} {
			body := `{"` + k + `":` + v + `,"messages":[]}`
			c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", body, as(1))
			if c != 400 || codeOf(b) != "n_not_offered" {
				t.Fatalf("%s: %d %s", body, c, b)
			}
		}
	}
	// a key given twice is refused whichever value comes last (two readers may keep different ones)
	for _, body := range []string{`{"n":1,"n":2,"messages":[]}`, `{"n":2,"n":1,"messages":[]}`, `{"n_cmpl":1,"n_cmpl":-4294967294,"messages":[]}`, `{"n":1,"\u006e":2,"messages":[]}`, `{"id_slot":0,"id_slot":1,"messages":[]}`} {
		c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", body, as(1))
		if c != 400 || !strings.Contains(b, "twice") {
			t.Fatalf("%s: %d %s", body, c, b)
		}
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("a refused n reached the model server")
	}
	for _, body := range []string{`{"n":1,"messages":[]}`, `{"n_cmpl":null,"messages":[]}`, `{"n": 1 ,"messages":[]}`} {
		if c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", body, as(1)); c != 200 {
			t.Fatalf("%s: %d %s", body, c, b)
		}
	}
}

// A client whose budget is spent is answered at once (503 no_room with the why), never held.
func TestBudgetSpentIsAnsweredAtOnce(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	var h *held
	for i := 1; i <= 4; i++ { // four changes of wants: four charged restarts
		h = openHold(t, rs, fmt.Sprintf(`{"hold":true,"name":"r","profile":{"slots":%d,"ctx_per_slot":{"max":8192,"floor":4096}}}`, i), tok(1))
		m := h.untilGiven(i)
		testkit.WaitPhase(t, s, "ready", 15*time.Second)
		if restartsLeft(m) != 4-i {
			t.Fatalf("change %d: restarts left %d", i, restartsLeft(m))
		}
	}
	n := len(e.Starts())
	h = openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":5,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(1))
	sl, _ := h.first["slots"].(map[string]any)
	if h.status != 200 || sl["pending"] != false || sl["why"] != "restart_budget_spent" || sl["given"] != float64(4) {
		t.Fatalf("over budget: %d %v", h.status, h.first) // it keeps what it had
	}
	// a new attachment of the spent client: nothing coming for it, answered at once
	h2 := openHold(t, rs, `{"hold":true,"name":"r2","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, tok(1))
	if sl, _ := h2.first["slots"].(map[string]any); h2.status != 200 || sl["pending"] != false {
		t.Fatalf("r2: %d %v", h2.status, h2.first)
	}
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(1, asClient(h2.id)))
	if c != 503 || codeOf(b) != "no_room" || !strings.Contains(b, "restart_budget_spent") {
		t.Fatalf("its request: %d %s", c, b)
	}
	if len(e.Starts()) != n {
		t.Fatal("a spent budget restarted the model")
	}
}

// Every reason a request is held, and the event that ends it; nothing else holds.
func TestEveryHoldHasItsEvent(t *testing.T) {
	// 1. the model is on its way up: the phase's word (ready) ends it: TestHeldUntilReady and the
	//    door's own tests (api/hold_test.go).
	// 2. its attachment's reshape is asked (a local grow drains; a remote one waits for idle): the
	//    start that serves it (the attachment and wants facts) ends it: TestLocalGrowDrainsWithinDrainMax,
	//    TestRemoteGrowWaitsForIdle, TestRemoteGrowAtOnceWhenIdle.
	// 3. a local caller under-served by a stale shape: the reshape it asked for ends it
	//    (run: TestLastClientLeavingReshapesNothing).
	// Not held: a client with no restart coming (budget spent, floor short, no room): 503 at once
	// (TestBudgetSpentIsAnsweredAtOnce, run: TestRestartBudget); here, a client left out for room.
	// here: a remote client whose floor the owner's ceiling cannot reach (unserved, short)
	_, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	h := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":1,"ctx_per_slot":{"max":65536,"floor":16384}}}`, tok(1))
	if h.status != 200 {
		t.Fatalf("attach: %d %v", h.status, h.first)
	}
	testkit.WaitFor(t, 20*time.Second, "unserved", func() bool {
		v, _ := s.ClientView(h.id)
		return v.Slots.Unserved != ""
	})
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[]}`, as(1))
	if c != 503 || codeOf(b) != "no_room" || !strings.Contains(b, "below its floor") {
		t.Fatalf("a client with nothing coming: %d %s", c, b)
	}
}

// The RAM prompt cache is off when the slots have two owners, on with one; a second owner is
// never served from vacant slots while it is on.
func TestRAMCacheOffWithTwoOwners(t *testing.T) {
	// no base: the commons own the slots; a client that would be served from them is not while
	// the cache is on: it restarts the server (where it is then the one owner, the cache on)
	e, s, srv := clientsEnv(t, false, func(e *testkit.Env) { e.Llama["FAKELLAMA_CACHE_RAM"] = "1" })
	if st := e.Starts(); strings.Contains(st[len(st)-1], "--cache-ram") {
		t.Fatalf("one owner: %s", st[len(st)-1])
	}
	if w := s.Status().Wants; !w.RAMCache {
		t.Fatal("wants does not say the RAM cache is on")
	}
	n := len(e.Starts())
	A := openHold(t, srv, `{"hold":true,"name":"a","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	A.untilGiven(1)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if st := e.Starts(); len(st) != n+1 || strings.Contains(st[len(st)-1], "--cache-ram") {
		t.Fatalf("a second owner was served from vacant slots, or a lone owner lost the cache: %q", st[n:])
	}
	// the base and a client are two owners: the cache is off
	e2, s2, srv2 := clientsEnv(t, true, func(e *testkit.Env) { e.Llama["FAKELLAMA_CACHE_RAM"] = "1" })
	n = len(e2.Starts())
	B := openHold(t, srv2, `{"hold":true,"name":"b","profile":{"slots":1,"ctx_per_slot":{"max":8192,"floor":4096}}}`, nil)
	B.untilGiven(1)
	testkit.WaitPhase(t, s2, "ready", 15*time.Second)
	if st := e2.Starts(); len(st) != n+1 || !strings.Contains(st[len(st)-1], "--cache-ram 0") {
		t.Fatalf("two owners: %q", st[n:])
	}
	if w := s2.Status().Wants; w.RAMCache {
		t.Fatal("wants says the RAM cache is on")
	}
}

// A help is read only for an attach that brings extra_args.
func TestAttachReadsNoHelpWithoutExtraArgs(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.StartSleeping = true // no launch: no help read yet
	s, srv := serve(t, e, cfg)
	helps := func() int {
		b, _ := os.ReadFile(filepath.Join(e.Ctl, "help.log"))
		return strings.Count(string(b), "help")
	}
	_, rs, _ := remoteSrv(t, s)
	if h := openHold(t, rs, `{"hold":true,"name":"r","profile":{"slots":1}}`, tok(1)); h.status != 200 {
		t.Fatalf("remote: %d %v", h.status, h.first)
	}
	if n := helps(); n != 0 {
		t.Fatalf("an attach with no extra_args read the help %d times", n)
	}
	if h := openHold(t, srv, `{"hold":true,"name":"l","profile":{"server":{"extra_args":["--top-n-sigma","1.5"]}}}`, nil); h.status != 200 {
		t.Fatalf("local: %d %v", h.status, h.first)
	}
	if n := helps(); n != 1 {
		t.Fatalf("an attach with extra_args read the help %d times", n)
	}
}

// masterStore is a token checker with a master token that can be replaced.
type masterStore struct {
	mu  sync.Mutex
	tok string
}

func (m *masterStore) Check(b string) (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return "master", b == m.tok
}

func (m *masterStore) MasterIs(sha string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	sum := sha256.Sum256([]byte(m.tok))
	return hex.EncodeToString(sum[:]) == sha
}

// A new master token ends every master@<ip> attachment; a token revoked before the attach
// registers does not stand.
func TestNewMasterEndsItsRemoteAttachments(t *testing.T) {
	_, s, _ := clientsEnv(t, true, nil)
	ms := &masterStore{tok: "m1"}
	a := &api.API{S: s, AnyHost: true, Tokens: ms}
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	h := openHold(t, srv, `{"hold":true,"name":"m","profile":{}}`, map[string]string{"Authorization": "Bearer m1", "X-Forwarded-For": "10.1.1.1"})
	if h.status != 200 || h.first["client_id"] != "master@127.0.0.1" {
		t.Fatalf("master attach: %d %v", h.status, h.first)
	}
	if _, ok := h.first["restarts_left"]; ok {
		t.Fatal("the master key from another machine is budgeted")
	}
	ms.mu.Lock()
	ms.tok = "m2"
	ms.mu.Unlock()
	a.ForgetRevoked()
	for {
		m, ok := h.next("the revoked end")
		if !ok {
			t.Fatal("no end line")
		}
		if m["end"] != nil {
			if m["end"] != "revoked" {
				t.Fatalf("end %v", m["end"])
			}
			break
		}
	}
	attachmentGone(t, s, h.id)
	// revoked between the gate and the registration
	_, _, ce := s.AttachClient(run.AttachRequest{Hold: true, Name: "late", Profile: &run.Profile{}, Identity: "client-1",
		Valid: func() bool { return false }})
	if ce == nil || ce.Code != "token_revoked" || s.Attached() != 0 {
		t.Fatalf("a revoked token's attach: %+v (attached %d)", ce, s.Attached())
	}
}

// At most eight remote bodies are read at once; a drip is cut by the total bound.
func TestRemoteBodyReadsBounded(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	a := &api.API{S: s, AnyHost: true, Tokens: newRevocable(), BodySilence: 10 * time.Second, BodyTotal: 2 * time.Second}
	eight := make(chan struct{})
	var once sync.Once
	a.SayReads(func(n int32) {
		if n == 8 {
			once.Do(func() { close(eight) })
		}
	})
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	open := func() net.Conn {
		conn, err := net.Dial("tcp", srv.Listener.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { conn.Close() })
		fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer tok-1\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{")
		return conn
	}
	answer := func(conn net.Conn) int {
		// network timeout: the engine must answer long before this
		conn.SetReadDeadline(time.Now().Add(10 * time.Second))
		resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
		if err != nil {
			t.Fatalf("no answer: %v", err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	var conns []net.Conn
	for i := 0; i < 8; i++ {
		conns = append(conns, open())
	}
	// the count says nothing to any hub, so the test waits on the API's own word of each count
	select {
	case <-eight:
	case <-time.After(10 * time.Second): // network timeout: the eight reads begin long before this
		t.Fatalf("timed out waiting for eight bodies being read (now %d)", a.RemoteReadsNow())
	}
	ninth := open()
	if c := answer(ninth); c != 429 {
		t.Fatalf("the ninth: %d", c)
	}
	// each of the eight is cut by the total bound (it never finished)
	for _, c := range conns {
		if code := answer(c); code != 400 {
			t.Fatalf("a drip: %d", code)
		}
	}
}
