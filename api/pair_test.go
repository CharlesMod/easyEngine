package api_test

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/pair"
	"github.com/CharlesMod/easyEngine/run"
)

type pairEnv struct {
	st     *tokens.Store
	master string
	sup    *run.Supervisor
	srv    *httptest.Server
	pr     *pair.Pairer
}

// pairServer is a tokened engine on a non-loopback listener with pairing offered.
func pairServer(t *testing.T) pairEnv {
	t.Helper()
	st := tokens.Open(t.TempDir())
	master, err := st.Master()
	if err != nil {
		t.Fatal(err)
	}
	var pe pairEnv
	_, s, srv := lanServer(t, func(a *api.API) {
		a.Tokens = st
		a.Pair = pair.New(a.S.Hub(), st, 2)
		a.EngineID = "0123456789abcdef0123456789abcdef"
		a.Name = func() string { return "test engine" }
		pe.pr = a.Pair
	})
	pe.st, pe.master, pe.sup, pe.srv = st, master, s, srv
	return pe
}

func with(h ...map[string]string) map[string]string {
	out := map[string]string{}
	for _, m := range h {
		for k, v := range m {
			out[k] = v
		}
	}
	return out
}

func decodeJSON(t *testing.T, s string, v any) {
	t.Helper()
	if err := json.Unmarshal([]byte(s), v); err != nil {
		t.Fatalf("not JSON: %q: %v", s, err)
	}
}

// ask opens a pairing request and returns its secret id and its code.
func ask(t *testing.T, pe pairEnv, name string) (string, string) {
	t.Helper()
	c, body, _ := do(t, "POST", pe.srv.URL+"/engine/v1/pair", `{"name":"`+name+`"}`, csrf)
	if c != 200 {
		t.Fatalf("POST pair: %d %s", c, body)
	}
	var r struct{ ID, Code string }
	decodeJSON(t, body, &r)
	if len(r.ID) != 32 || len(r.Code) != 6 {
		t.Fatalf("pair answer: %s", body)
	}
	return r.ID, r.Code
}

type collected struct {
	code int
	body string
}

func collect(t *testing.T, pe pairEnv, id string) <-chan collected {
	ch := make(chan collected, 1)
	go func() {
		c, b, _ := do(t, "GET", pe.srv.URL+"/engine/v1/pair/"+id, "", nil)
		ch <- collected{c, b}
	}()
	return ch
}

func answer(t *testing.T, pe pairEnv, hdr map[string]string, body string) (int, string) {
	t.Helper()
	c, b, _ := do(t, "POST", pe.srv.URL+"/engine/v1/pairing/answer", body, with(csrf, hdr))
	return c, b
}

func waitCollected(t *testing.T, ch <-chan collected) collected {
	t.Helper()
	select {
	case r := <-ch:
		return r
	case <-time.After(20 * time.Second): // watchdog for silence: a held GET that is never answered fails the test
		t.Fatal("the held GET pair/{id} was never answered")
	}
	return collected{}
}

func TestHelloAnswersEveryoneAndNever401s(t *testing.T) {
	pe := pairServer(t)
	hello := func(hdr map[string]string) api.Hello {
		t.Helper()
		c, body, _ := do(t, "GET", pe.srv.URL+"/engine/v1/hello", "", hdr)
		if c != 200 {
			t.Fatalf("hello: %d %s", c, body)
		}
		for _, bad := range []string{pe.master, "tokens", pe.st.Path()} {
			if strings.Contains(body, bad) {
				t.Fatalf("hello leaks %q: %s", bad, body)
			}
		}
		var h api.Hello
		decodeJSON(t, body, &h)
		return h
	}
	h := hello(nil)
	if h.Engine != "easy-engine" || h.Name != "test engine" || h.EngineID != "0123456789abcdef0123456789abcdef" ||
		h.API != run.APIVersion || !h.Pairing || !h.TokenRequired || h.Paired != nil || h.Phase != "ready" {
		t.Fatalf("hello with no token: %+v", h)
	}
	if want := pe.sup.Status().Model; want == "" || h.ModelID != want || strings.HasPrefix(h.ModelID, "custom:") {
		t.Fatalf("hello model_id %q, status model %q", h.ModelID, want)
	}
	if h := hello(bearer(pe.master)); h.Paired == nil || !*h.Paired || h.ClientID != tokens.MasterID {
		t.Fatalf("hello with the master: %+v", h)
	}
	if h := hello(bearer("not-a-token")); h.Paired == nil || *h.Paired || h.Revoked {
		t.Fatalf("hello with an unknown token: %+v", h)
	}
	id, tok, err := pe.st.Issue("a mod")
	if err != nil {
		t.Fatal(err)
	}
	if h := hello(bearer(tok)); h.Paired == nil || !*h.Paired || h.ClientID != id {
		t.Fatalf("hello with an issued token: %+v", h)
	}
	if err := pe.st.Revoke(id); err != nil {
		t.Fatal(err)
	}
	if h := hello(bearer(tok)); h.Paired == nil || *h.Paired || !h.Revoked || h.ClientID != "" {
		t.Fatalf("hello with a revoked token: %+v", h)
	}
	// the gate's other rules still bind it
	if c, _, _ := do(t, "GET", pe.srv.URL+"/engine/v1/hello", "", map[string]string{"Origin": "http://evil.example"}); c != 403 {
		t.Fatalf("hello from another origin: %d", c)
	}
	if c, _, _ := do(t, "OPTIONS", pe.srv.URL+"/engine/v1/hello", "", nil); c != 403 {
		t.Fatalf("hello preflight: %d", c)
	}
}

func Test401BodiesCarryACode(t *testing.T) {
	pe := pairServer(t)
	id, tok, _ := pe.st.Issue("a mod")
	if c, _, _ := do(t, "GET", pe.srv.URL+"/engine/v1/status", "", bearer(tok)); c != 200 {
		t.Fatalf("issued token: %d", c)
	}
	pe.st.Revoke(id)
	for _, tc := range []struct {
		hdr  map[string]string
		code string
	}{{nil, "token_required"}, {bearer("nope"), "token_invalid"}, {bearer(tok), "token_revoked"}} {
		c, body, h := do(t, "GET", pe.srv.URL+"/engine/v1/status", "", tc.hdr)
		var b struct{ Error, Code string }
		decodeJSON(t, body, &b)
		if c != 401 || b.Code != tc.code || b.Error != "bearer token required (easy-engine token prints it)" || h.Get("WWW-Authenticate") != "Bearer" {
			t.Fatalf("want 401 %s: %d %s", tc.code, c, body)
		}
	}
}

func TestPairAllowGivesATokenOnce(t *testing.T) {
	pe := pairServer(t)
	id, code := ask(t, pe, `Kitchen\u0007   PC`)
	got := collect(t, pe, id)

	// the person's side sees the name and the code, never a token or the request id
	c, body, _ := do(t, "GET", pe.srv.URL+"/engine/v1/pairing", "", bearer(pe.master))
	if c != 200 || !strings.Contains(body, `"code":"`+code+`"`) || !strings.Contains(body, `"name":"Kitchen PC"`) || strings.Contains(body, id) {
		t.Fatalf("pairing list: %d %s", c, body)
	}
	c, facts, _ := do(t, "GET", pe.srv.URL+"/engine/v1/facts", "", bearer(pe.master))
	if c != 200 || !strings.Contains(facts, `"name":"pairing"`) || strings.Contains(facts, id) || strings.Contains(facts, "token") {
		t.Fatalf("facts: %d %s", c, facts)
	}
	select {
	case r := <-got:
		t.Fatalf("GET pair/{id} answered before the person did: %+v", r)
	default:
	}
	if c, b := answer(t, pe, bearer(pe.master), `{"code":"`+code+`","allow":true}`); c != 200 {
		t.Fatalf("allow: %d %s", c, b)
	}
	r := waitCollected(t, got)
	var res struct {
		Token    string `json:"token"`
		ClientID string `json:"client_id"`
		EngineID string `json:"engine_id"`
	}
	decodeJSON(t, r.body, &res)
	if r.code != 200 || res.Token == "" || res.ClientID == "" || res.EngineID != "0123456789abcdef0123456789abcdef" {
		t.Fatalf("allowed: %d %s", r.code, r.body)
	}
	if c, _, _ := do(t, "GET", pe.srv.URL+"/engine/v1/status", "", bearer(res.Token)); c != 200 {
		t.Fatalf("the issued token at the gate: %d", c)
	}
	if again := waitCollected(t, collect(t, pe, id)); again.code != 410 || !strings.Contains(again.body, "pair_gone") {
		t.Fatalf("second GET: %+v", again)
	}
	// the paired client is said once as a client fact, with its name and no token
	cl, ok := pe.sup.Hub().Fact("client", res.ClientID)
	if !ok || !strings.Contains(string(cl.Body), "Kitchen PC") || strings.Contains(string(cl.Body), res.Token) {
		t.Fatalf("client fact: %v %s", ok, cl.Body)
	}
	if _, ok := pe.sup.Hub().Fact("pairing", ""); ok {
		t.Fatal("pairing fact stands after the answer")
	}
	// a paired client's token answers nothing
	id2, code2 := ask(t, pe, "other")
	if c, b := answer(t, pe, bearer(res.Token), `{"code":"`+code2+`","allow":true}`); c != 403 || !strings.Contains(b, "not_allowed") {
		t.Fatalf("a paired client answering: %d %s", c, b)
	}
	if c, _, _ := do(t, "GET", pe.srv.URL+"/engine/v1/pairing", "", bearer(res.Token)); c != 403 {
		t.Fatalf("a paired client listing: %d", c)
	}
	// nor does the network without a token
	if c, b := answer(t, pe, nil, `{"code":"`+code2+`","allow":true}`); c != 401 {
		t.Fatalf("no token answering: %d %s", c, b)
	}
	// revoke turns the working token into token_revoked
	if err := pe.st.Revoke(res.ClientID); err != nil {
		t.Fatal(err)
	}
	if c, b, _ := do(t, "GET", pe.srv.URL+"/engine/v1/status", "", bearer(res.Token)); c != 401 || !strings.Contains(b, `"code":"token_revoked"`) {
		t.Fatalf("revoked: %d %s", c, b)
	}
	if !pe.pr.Withdraw(id2) {
		t.Fatal("withdraw")
	}
}

func TestPairDenyAndWithdraw(t *testing.T) {
	pe := pairServer(t)
	id, code := ask(t, pe, "stranger")
	got := collect(t, pe, id)
	if c, b := answer(t, pe, bearer(pe.master), `{"code":"`+code+`","allow":false}`); c != 200 {
		t.Fatalf("deny: %d %s", c, b)
	}
	if r := waitCollected(t, got); r.code != 403 || !strings.Contains(r.body, "pair_denied") {
		t.Fatalf("denied: %+v", r)
	}
	if r := waitCollected(t, collect(t, pe, id)); r.code != 410 {
		t.Fatalf("after deny: %+v", r)
	}
	id, _ = ask(t, pe, "leaver")
	if c, _, _ := do(t, "DELETE", pe.srv.URL+"/engine/v1/pair/"+id, "", csrf); c != 204 {
		t.Fatalf("withdraw: %d", c)
	}
	if r := waitCollected(t, collect(t, pe, id)); r.code != 410 {
		t.Fatalf("after withdraw: %+v", r)
	}
	if r := waitCollected(t, collect(t, pe, "ffffffffffffffffffffffffffffffff")); r.code != 410 {
		t.Fatalf("unknown id: %+v", r)
	}
}

// askFrom opens a pairing request from the local source address ip (127.0.0.x on Linux).
func askFrom(t *testing.T, pe pairEnv, ip, name string) (string, string) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("needs 127.0.0.x source addresses")
	}
	cl := &http.Client{Transport: &http.Transport{DialContext: (&net.Dialer{LocalAddr: &net.TCPAddr{IP: net.ParseIP(ip)}}).DialContext}}
	req, _ := http.NewRequest("POST", pe.srv.URL+"/engine/v1/pair", strings.NewReader(`{"name":"`+name+`"}`))
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := cl.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var r struct{ ID, Code string }
	json.NewDecoder(resp.Body).Decode(&r)
	if resp.StatusCode != 200 {
		t.Fatalf("POST pair from %s: %d", ip, resp.StatusCode)
	}
	return r.ID, r.Code
}

func TestPairCapPushesTheOldestOut(t *testing.T) {
	pe := pairServer(t) // cap 2
	first, _ := askFrom(t, pe, "127.0.0.11", "one")
	got := collect(t, pe, first)
	askFrom(t, pe, "127.0.0.12", "two")
	askFrom(t, pe, "127.0.0.13", "three")
	if r := waitCollected(t, got); r.code != 410 {
		t.Fatalf("the oldest over the cap: %+v", r)
	}
	if n := len(pe.pr.Waiting()); n != 2 {
		t.Fatalf("waiting: %d", n)
	}
}

// One source has one waiting request: a new ask replaces it under the same public key.
func TestPairOneWaitingRequestPerSource(t *testing.T) {
	pe := pairServer(t)
	first, _ := askFrom(t, pe, "127.0.0.21", "mod")
	got := collect(t, pe, first)
	key := pe.pr.Waiting()[0].Key
	var last, code string
	for i := 0; i < 100; i++ {
		last, code = askFrom(t, pe, "127.0.0.21", "mod")
	}
	if r := waitCollected(t, got); r.code != 410 {
		t.Fatalf("the replaced request: %+v", r)
	}
	w := pe.pr.Waiting()
	if len(w) != 1 || w[0].Key != key || w[0].Code != code || w[0].From != "127.0.0.21" {
		t.Fatalf("waiting after 100 asks: %+v", w)
	}
	askFrom(t, pe, "127.0.0.22", "other")
	if len(pe.pr.Waiting()) != 2 {
		t.Fatal("a second source did not get its own request")
	}
	// the person's side must name the code with the key: without it 400
	if c, b := answer(t, pe, bearer(pe.master), `{"key":"`+key+`","allow":false}`); c != 400 {
		t.Fatalf("key without code: %d %s", c, b)
	}
	// after a Deny, the source's next request is listed but quiet
	if c, b := answer(t, pe, bearer(pe.master), `{"key":"`+key+`","code":"`+code+`","allow":false}`); c != 200 {
		t.Fatalf("deny: %d %s", c, b)
	}
	if r := waitCollected(t, collect(t, pe, last)); r.code != 403 {
		t.Fatalf("denied: %+v", r)
	}
	askFrom(t, pe, "127.0.0.21", "mod again")
	quiet := 0
	for _, r := range pe.pr.Waiting() {
		if r.From == "127.0.0.21" && r.Quiet {
			quiet++
		}
		if r.From == "127.0.0.22" && r.Quiet {
			t.Fatal("another source went quiet")
		}
	}
	if quiet != 1 {
		t.Fatalf("the denied source's new request: %+v", pe.pr.Waiting())
	}
}

func TestPairingNotOfferedOnLoopback(t *testing.T) {
	testkit.SkipWindows(t)
	_, _, srv := readyEnv(t) // loopback, no tokens, no pairer
	c, body, _ := do(t, "GET", srv.URL+"/engine/v1/hello", "", nil)
	var h api.Hello
	decodeJSON(t, body, &h)
	if c != 200 || h.Pairing || h.TokenRequired {
		t.Fatalf("loopback hello: %d %s", c, body)
	}
	if c, b, _ := do(t, "POST", srv.URL+"/engine/v1/pair", `{"name":"x"}`, csrf); c != 404 || !strings.Contains(b, "pairing_off") {
		t.Fatalf("loopback pair: %d %s", c, b)
	}
}
