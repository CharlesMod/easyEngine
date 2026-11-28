package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/internal/tokens"
)

// The path table from llama-server's own routes (b12651, tools/server/server.cpp; the table in
// docs/ENGINE.md section 7g): every route it registers, what the door makes of it.
func TestPathTableWalk(t *testing.T) {
	slot := []string{"/completion", "/completions", "/v1/completions", "/chat/completions", "/v1/chat/completions",
		"/v1/messages", "/infill"}
	unconfinable := []string{"/embedding", "/embeddings", "/v1/embeddings", "/rerank", "/reranking", "/v1/rerank",
		"/v1/reranking", "/tts", "/v1/audio/speech", "/audio/speech", "/v1/audio/transcriptions",
		"/audio/transcriptions", "/v1/systemone", "/v1/chat/completions/control", "/lora-adapters", "/props",
		"/predict", "/v1/stream", "/some/path/a/newer/build/adds", "/v1/responses", "/responses"}
	free := []string{"/tokenize", "/detokenize", "/apply-template", "/v1/messages/count_tokens",
		"/chat/completions/input_tokens", "/v1/chat/completions/input_tokens", "/responses/input_tokens",
		"/v1/responses/input_tokens"}
	getFree := []string{"/health", "/v1/health", "/metrics", "/props", "/models", "/v1/models", "/lora-adapters", "/slots", "/"}
	for _, p := range slot {
		if k := classify("POST", p); k != pathSlot {
			t.Errorf("POST %s: %d, want slot", p, k)
		}
		if k := classify("POST", p+"/"); k != pathSlot { // the spelling the server would not route still maps
			t.Errorf("POST %s/: %d", p, k)
		}
	}
	for _, p := range unconfinable {
		if k := classify("POST", p); k != pathUnconfinable {
			t.Errorf("POST %s: %d, want unconfinable", p, k)
		}
	}
	for _, p := range free {
		if k := classify("POST", p); k != pathFree {
			t.Errorf("POST %s: %d, want free", p, k)
		}
	}
	for _, p := range getFree {
		if k := classify("GET", p); k != pathFree {
			t.Errorf("GET %s: %d, want free", p, k)
		}
	}
	if k := classify("GET", "/v1/realtime"); k != pathUnconfinable { // the socket takes a slot
		t.Errorf("GET /v1/realtime: %d", k)
	}
	for _, m := range []string{"POST", "PUT", "DELETE", "PATCH"} {
		if k := classify(m, "/slots/0"); k != pathSlotID {
			t.Errorf("%s /slots/0: %d", m, k)
		}
		if k := classify(m, "/V1/Chat/Completions"); k != pathUnconfinable { // not a route: held, not trusted
			t.Errorf("%s with capitals: %d", m, k)
		}
	}
}

func TestReplyOverTheCapIs502WithACode(t *testing.T) {
	resp := &http.Response{Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(io.LimitReader(zeros{}, maxSamplerBody+1))}
	err := slotsBack(resp, []int{3, 4})
	if _, ok := err.(replyTooLarge); !ok {
		t.Fatalf("err %v", err)
	}
	w := httptest.NewRecorder()
	badGateway(w, httptest.NewRequest("POST", "/v1/chat/completions", nil), err)
	var body struct{ Error struct{ Code string } }
	json.Unmarshal(w.Body.Bytes(), &body)
	if w.Code != 502 || body.Error.Code != "reply_too_large" {
		t.Fatalf("%d %s", w.Code, w.Body)
	}
	// at the cap it is mapped whole
	ok := []byte(`{"id_slot":4,"pad":"` + strings.Repeat("x", 100) + `"}`)
	resp = &http.Response{Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(bytes.NewReader(ok))}
	if err := slotsBack(resp, []int{3, 4}); err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	if !bytes.Contains(b, []byte(`"id_slot":1`)) {
		t.Fatalf("not mapped: %s", b)
	}
}

type zeros struct{}

func (zeros) Read(p []byte) (int, error) {
	clear(p)
	return len(p), nil
}

func TestIdentityOfCallersWithNoIDOfTheirOwn(t *testing.T) {
	req := func(remote, client string, hdr ...string) *http.Request {
		r := httptest.NewRequest("POST", "/engine/v1/attach", nil)
		r.RemoteAddr = remote
		for i := 0; i+1 < len(hdr); i += 2 {
			r.Header.Set(hdr[i], hdr[i+1])
		}
		ctx := context.WithValue(r.Context(), clientKey, client)
		ctx = context.WithValue(ctx, http.LocalAddrContextKey, net.Addr(&net.TCPAddr{IP: net.ParseIP("10.0.0.1"), Port: 47480}))
		return r.WithContext(ctx)
	}
	cases := []struct {
		r    *http.Request
		want string
	}{
		{req("10.0.0.5:4000", ClientOpen), "open@10.0.0.5"},
		{req("10.0.0.6:4000", ClientOpen), "open@10.0.0.6"},
		{req("10.0.0.5:4000", tokens.MasterID), "master@10.0.0.5"},
		{req("127.0.0.1:4000", tokens.MasterID), "loopback"},
		{req("127.0.0.1:4000", ClientOpen), "loopback"},
		{req("127.0.0.1:4000", ClientOpen, "X-Forwarded-For", "10.0.0.9"), "open@127.0.0.1"},
		{req("127.0.0.1:4000", tokens.MasterID, "Via", "1.1 proxy"), "master@127.0.0.1"},
		{req("10.0.0.1:4000", ClientLoopback), "loopback"}, // the listener's own address
		{req("10.0.0.5:4000", "c1"), "c1"},
	}
	for i, c := range cases {
		if got := identity(c.r); got != c.want {
			t.Errorf("%d: %s, want %s", i, got, c.want)
		}
	}
	// a header can only take privilege away
	if localCaller(req("10.0.0.5:4000", ClientLoopback, "X-Forwarded-For", "127.0.0.1")) {
		t.Fatal("a forwarding header made a remote caller local")
	}
}
