package api_test

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
)

// F3: GET /slots returns each slot's prompt and generated text (llama-server's to_json,
// server-context.cpp:835-866). The door filters it for every caller: a slot whose owner is not
// the caller keeps only id, is_processing and n_ctx. With one owner nothing is hidden.
func TestSlotsReadShowsOnlyOwnSlots(t *testing.T) {
	e, s, srv := clientsEnv(t, true, nil) // the base: the commons, slots 0..2
	_, rs, _ := remoteSrv(t, s)
	read := func(url string, hdr map[string]string) map[int]map[string]any {
		t.Helper()
		c, b, _ := do(t, "GET", url+"/slots", "", hdr)
		if c != 200 {
			t.Fatalf("GET /slots: %d %s", c, b)
		}
		var l []map[string]any
		if err := json.Unmarshal([]byte(b), &l); err != nil {
			t.Fatalf("%v: %s", err, b)
		}
		out := map[int]map[string]any{}
		for _, sl := range l {
			out[int(sl["id"].(float64))] = sl
		}
		return out
	}
	bare := func(sl map[string]any) bool {
		_, p := sl["prompt"]
		_, n := sl["n_prompt_tokens"]
		return len(sl) == 3 && !p && !n
	}
	chat(t, srv.URL, secret("Kay"), 0, jsonCT)
	if v := read(srv.URL, nil); v[0]["prompt"] == nil {
		t.Fatalf("one owner: the local caller's own slot hidden: %v", v[0])
	}
	chat(t, rs.URL, secret("two"), 1, as(2)) // client-2, untrusted, in the commons: anon
	v := read(srv.URL, nil)
	if v[0]["prompt"] == nil || !bare(v[1]) {
		t.Fatalf("local: slot 0 %v, slot 1 %v", v[0], v[1])
	}
	// the master key from another machine is trusted (the commons'), never client-2's
	msrv := httptest.NewServer((&api.API{S: s, AnyHost: true, Tokens: testChecker{"tok-m": "master"}}).Handler())
	t.Cleanup(msrv.Close)
	v = read(msrv.URL, map[string]string{"Authorization": "Bearer tok-m", "X-Forwarded-For": "10.9.8.6"})
	if v[0]["prompt"] == nil || !bare(v[1]) {
		t.Fatalf("master: slot 0 %v, slot 1 %v", v[0], v[1])
	}
	_ = e
}
