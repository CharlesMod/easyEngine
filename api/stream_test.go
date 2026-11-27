package api

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/run"
)

// The streamer through forward, with httptest upstreams standing in for the model server. Every
// wait is a channel receive bounded by the test's context (a network timeout).

const (
	ev1        = `data: {"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}` + "\n\n"
	ev2        = `data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}` + "\n\n"
	evFin      = `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
	evDone     = "data: [DONE]\n\n"
	streamBody = `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`
)

func testCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second) // network timeout
	t.Cleanup(cancel)
	return ctx
}

// front is the engine's side: the proxy's own reverse proxy to up, through forward with f.
func front(t *testing.T, up string, f fwd) *httptest.Server {
	t.Helper()
	target, err := url.Parse(up)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := requestID(r)
		w.Header().Set(RequestIDHeader, id)
		rp := &httputil.ReverseProxy{
			Transport:      transport,
			FlushInterval:  -1,
			Rewrite:        func(pr *httputil.ProxyRequest) { pr.SetURL(target); pr.Out.Host = target.Host },
			ModifyResponse: dropCORS,
			ErrorHandler:   badGateway,
		}
		g := f
		g.id = id
		forward(w, r, rp, g)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// upstream is an httptest server running h.
func upstream(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

// sse starts an event-stream reply the way llama-server does: headers flushed at once.
func sse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(200)
	w.(http.Flusher).Flush()
}

func send(w http.ResponseWriter, s string) {
	io.WriteString(w, s)
	w.(http.Flusher).Flush()
}

// post sends a completion request and returns the reply (the caller closes its body).
func post(t *testing.T, ctx context.Context, base, path, body string, hdr map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, "POST", base+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// readEvent reads one event (up to and including its blank line).
func readEvent(t *testing.T, br *bufio.Reader) string {
	t.Helper()
	var b strings.Builder
	for !strings.HasSuffix(b.String(), "\n\n") {
		c, err := br.ReadByte()
		if err != nil {
			t.Fatalf("reading an event: %v (so far %q)", err, b.String())
		}
		b.WriteByte(c)
	}
	return b.String()
}

func readRest(t *testing.T, r io.Reader) string {
	t.Helper()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("reading the rest: %v (got %q)", err, b)
	}
	return string(b)
}

func local() fwd { return fwd{on: true, yielded: func() (bool, string) { return false, "" }} }

func TestStreamFirstEventBeforeSecond(t *testing.T) {
	ctx := testCtx(t)
	next := make(chan struct{})
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w)
		send(w, ev1)
		select {
		case <-next:
		case <-r.Context().Done():
			return
		}
		send(w, ev2+evFin+evDone)
	})
	resp := post(t, ctx, front(t, up.URL, local()).URL, "/v1/chat/completions", streamBody, nil)
	br := bufio.NewReader(resp.Body)
	if got := readEvent(t, br); got != ev1 {
		t.Fatalf("first event %q", got)
	}
	close(next)
	if got := readRest(t, br); got != ev2+evFin+evDone {
		t.Fatalf("rest %q", got)
	}
	if cc := resp.Header.Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("Cache-Control %q", cc)
	}
}

// dies ends the upstream's reply after ev1: by returning (a clean end of the body) or by
// dropping the connection (a read error at the engine).
func dies(hijack bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sse(w)
		send(w, ev1)
		send(w, `data: {"choices":[{"index":0,"delta":{"content":"half`) // half an event
		if hijack {
			c, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				c.Close()
			}
		}
	}
}

func TestStreamUpstreamDiesLocal(t *testing.T) {
	for _, hj := range []bool{false, true} {
		ctx := testCtx(t)
		up := upstream(t, dies(hj))
		resp := post(t, ctx, front(t, up.URL, local()).URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-1"})
		got := readRest(t, resp.Body)
		want := ev1 + `data: {"error":{"code":"model_server_gone","type":"server_error","message":"The model server stopped before the reply was finished","request_id":"rid-1"}}` + "\n\n"
		if got != want {
			t.Fatalf("hijack=%v:\n got %q\nwant %q", hj, got, want)
		}
		t.Logf("model_server_gone bytes: %q", got[len(ev1):])
	}
}

func TestStreamUpstreamEndsExternal(t *testing.T) {
	ctx := testCtx(t)
	up := upstream(t, dies(false))
	resp := post(t, ctx, front(t, up.URL, fwd{on: true, external: true}).URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-2"})
	got := readRest(t, resp.Body)
	want := ev1 + `data: {"error":{"code":"upstream_ended","type":"upstream_error","message":"The endpoint ended the reply before it was finished","request_id":"rid-2"}}` + "\n\n"
	if got != want {
		t.Fatalf("\n got %q\nwant %q", got, want)
	}
	t.Logf("upstream_ended bytes: %q", got[len(ev1):])
}

func TestStreamYielded(t *testing.T) {
	ctx := testCtx(t)
	up := upstream(t, dies(true))
	f := fwd{on: true, yielded: func() (bool, string) { return true, "Paused while AppId 999 (x) runs" }}
	resp := post(t, ctx, front(t, up.URL, f).URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-3"})
	got := readRest(t, resp.Body)
	want := ev1 + `data: {"error":{"code":"yielded","type":"unavailable_error","message":"Paused while AppId 999 (x) runs","status":503,"request_id":"rid-3"}}` + "\n\n"
	if got != want {
		t.Fatalf("\n got %q\nwant %q", got, want)
	}
	t.Logf("yielded bytes: %q", got[len(ev1):])
}

func TestStreamNativeEndsOnStop(t *testing.T) {
	ctx := testCtx(t)
	stop := `data: {"content":"","stop":true,"stop_type":"eos","id_slot":0}` + "\n\n"
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w)
		send(w, `data: {"content":"Hi","stop":false}`+"\n\n"+stop)
	})
	resp := post(t, ctx, front(t, up.URL, local()).URL, "/completion", `{"prompt":"x","stream":true}`, nil)
	if got := readRest(t, resp.Body); got != `data: {"content":"Hi","stop":false}`+"\n\n"+stop {
		t.Fatalf("got %q", got)
	}
}

// slow holds its reply until release is closed, then answers with h.
func slow(release <-chan struct{}, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		h(w, r)
	}
}

func TestStreamSlowHeadersWaitingWord(t *testing.T) {
	ctx := testCtx(t)
	release := make(chan struct{})
	up := upstream(t, slow(release, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-From-Upstream", "1")
		sse(w)
		send(w, ev1+evFin+evDone)
	}))
	f := local()
	f.open = time.Millisecond
	resp := post(t, ctx, front(t, up.URL, f).URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-4"})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	h := resp.Header
	if h.Get("Content-Type") != "text/event-stream" || h.Get("Cache-Control") != "no-cache" ||
		h.Get("X-Accel-Buffering") != "no" || h.Get(RequestIDHeader) != "rid-4" || h.Get("X-From-Upstream") != "" {
		t.Fatalf("headers %v", h)
	}
	t.Logf("opened headers: %v", h)
	br := bufio.NewReader(resp.Body)
	if got := readEvent(t, br); got != ": easy waiting\n\n" {
		t.Fatalf("first bytes %q", got)
	}
	close(release)
	if got := readRest(t, br); got != ev1+evFin+evDone {
		t.Fatalf("rest %q", got)
	}
}

func TestStreamSlowThenError(t *testing.T) {
	cases := []struct {
		name   string
		h      http.HandlerFunc
		status int
		msg    string
	}{
		{"500 json", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(500)
			io.WriteString(w, `{"error":{"code":500,"message":"boom","type":"server_error"}}`)
		}, 500, "boom"},
		{"connection dropped", func(w http.ResponseWriter, r *http.Request) {
			c, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				c.Close()
			}
		}, 502, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx := testCtx(t)
			release := make(chan struct{})
			up := upstream(t, slow(release, c.h))
			f := local()
			f.open = time.Millisecond
			resp := post(t, ctx, front(t, up.URL, f).URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-5"})
			br := bufio.NewReader(resp.Body)
			if got := readEvent(t, br); got != ": easy waiting\n\n" {
				t.Fatalf("first bytes %q", got)
			}
			close(release)
			got := readRest(t, br)
			if c.msg != "" {
				want := `data: {"error":{"code":"upstream_error","type":"upstream_error","message":"boom","status":500,"request_id":"rid-5"}}` + "\n\n"
				if got != want {
					t.Fatalf("\n got %q\nwant %q", got, want)
				}
				t.Logf("upstream_error bytes: %q", got)
				return
			}
			if !regexp.MustCompile(`^data: \{"error":\{"code":"upstream_error","type":"upstream_error","message":"upstream: [^"]+","status":502,"request_id":"rid-5"\}\}\n\n$`).MatchString(got) {
				t.Fatalf("got %q", got)
			}
		})
	}
}

func TestStreamQuietWord(t *testing.T) {
	ctx := testCtx(t)
	release := make(chan struct{})
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w)
		send(w, ev1)
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		send(w, evDone)
	})
	f := local()
	f.open, f.quiet = time.Millisecond, time.Millisecond
	resp := post(t, ctx, front(t, up.URL, f).URL, "/v1/chat/completions", streamBody, nil)
	br := bufio.NewReader(resp.Body)
	// Before the first upstream byte the word is "waiting"; the upstream's first event may come
	// before or after one of those.
	for {
		got := readEvent(t, br)
		if got == ev1 {
			break
		}
		if got != ": easy waiting\n\n" {
			t.Fatalf("before the first event: %q", got)
		}
	}
	if got := readEvent(t, br); got != ": easy quiet\n\n" {
		t.Fatalf("after the first event: %q", got)
	}
	close(release)
	rest := strings.ReplaceAll(readRest(t, br), ": easy quiet\n\n", "")
	if rest != evDone {
		t.Fatalf("rest %q", rest)
	}
}

func TestStreamFastErrorUntouched(t *testing.T) {
	ctx := testCtx(t)
	body := `{"error":{"code":400,"message":"bad","type":"invalid_request_error"}}`
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(400)
		io.WriteString(w, body)
	})
	resp := post(t, ctx, front(t, up.URL, local()).URL, "/v1/chat/completions", streamBody, nil)
	if resp.StatusCode != 400 || resp.Header.Get("Content-Type") != "application/json; charset=utf-8" {
		t.Fatalf("status %d headers %v", resp.StatusCode, resp.Header)
	}
	if got := readRest(t, resp.Body); got != body {
		t.Fatalf("body %q", got)
	}
}

// untouched: a request the streamer must not handle gets exactly the upstream's bytes, even
// a stream cut short, and no waiting word while the upstream is slow.
func TestStreamUntouched(t *testing.T) {
	cases := []struct {
		name, path, body string
		f                fwd
	}{
		{"not streamed", "/v1/chat/completions", `{"model":"m","stream":false}`, local()},
		{"no body json", "/v1/chat/completions", `not json`, local()},
		{"not a completion", "/v1/embeddings", streamBody, local()},
		{"streamer off", "/v1/chat/completions", streamBody, fwd{on: false}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ctx := testCtx(t)
			arrived, release := make(chan string, 1), make(chan struct{})
			up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				arrived <- string(b)
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				dies(false)(w, r)
			})
			c.f.open = time.Nanosecond // a wrapper, were there one, would speak at once
			got := make(chan *http.Response, 1)
			srv := front(t, up.URL, c.f)
			go func() { // t.Fatal is not for this goroutine: a failed request is a nil reply
				req, _ := http.NewRequestWithContext(ctx, "POST", srv.URL+c.path, strings.NewReader(c.body))
				resp, err := http.DefaultClient.Do(req)
				if err != nil {
					resp = nil
				}
				got <- resp
			}()
			select {
			case b := <-arrived:
				if b != c.body {
					t.Fatalf("upstream got %q", b)
				}
			case <-ctx.Done():
				t.Fatal("no request upstream")
			}
			close(release)
			var resp *http.Response
			select {
			case resp = <-got:
				if resp == nil {
					t.Fatal("the request failed")
				}
				defer resp.Body.Close()
			case <-ctx.Done():
				t.Fatal("no reply")
			}
			want := ev1 + `data: {"choices":[{"index":0,"delta":{"content":"half`
			if b := readRest(t, resp.Body); b != want {
				t.Fatalf("got %q", b)
			}
			if resp.Header.Get("Cache-Control") != "" {
				t.Errorf("Cache-Control added: %q", resp.Header.Get("Cache-Control"))
			}
			if resp.Header.Get(RequestIDHeader) == "" {
				t.Error("no X-Request-Id")
			}
		})
	}
}

func TestStreamContentEncodingUntouched(t *testing.T) {
	ctx := testCtx(t)
	raw := "\x1f\x8b not really gzip, cut short"
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Encoding", "gzip")
		w.WriteHeader(200)
		io.WriteString(w, raw)
	})
	resp := post(t, ctx, front(t, up.URL, local()).URL, "/v1/chat/completions", streamBody, map[string]string{"Accept-Encoding": "gzip"})
	if got := readRest(t, resp.Body); got != raw {
		t.Fatalf("got %q", got)
	}
	if resp.Header.Get("Cache-Control") != "" || resp.Header.Get("Content-Encoding") != "gzip" {
		t.Fatalf("headers %v", resp.Header)
	}
}

func TestStreamClientLeaves(t *testing.T) {
	ctx := testCtx(t)
	upGone := make(chan struct{})
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		sse(w)
		send(w, ev1)
		<-r.Context().Done()
		close(upGone)
	})
	cctx, leave := context.WithCancel(ctx)
	resp := post(t, cctx, front(t, up.URL, local()).URL, "/v1/chat/completions", streamBody, nil)
	if got := readEvent(t, bufio.NewReader(resp.Body)); got != ev1 {
		t.Fatalf("first event %q", got)
	}
	leave()
	select {
	case <-upGone:
	case <-ctx.Done():
		t.Fatal("the upstream call was not cancelled when the client left")
	}
}

func TestRequestID(t *testing.T) {
	mk := func(v string) *http.Request {
		r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		if v != "" {
			r.Header.Set(RequestIDHeader, v)
		}
		return r
	}
	if got := requestID(mk("abc-123")); got != "abc-123" {
		t.Errorf("echo: %q", got)
	}
	hex16 := regexp.MustCompile(`^[0-9a-f]{16}$`)
	for _, bad := range []string{"", "caf\xc3\xa9", strings.Repeat("x", 129), "a\tb"} {
		if got := requestID(mk(bad)); !hex16.MatchString(got) {
			t.Errorf("%q: got %q", bad, got)
		}
	}
	if a, b := requestID(mk("")), requestID(mk("")); a == b {
		t.Errorf("two generated ids are the same: %q", a)
	}
}

// Through the whole handler in external mode (a supervisor that is never run): the sampler's
// header and X-Request-Id are on the reply, the upstream's own id is kept aside, and the stream
// is the upstream's bytes.
func TestStreamThroughHandler(t *testing.T) {
	ctx := testCtx(t)
	up := upstream(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set(RequestIDHeader, "up-1")
		sse(w)
		send(w, ev1+evFin+evDone)
	})
	dir := t.TempDir()
	set := `{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":"","api_key":""}}`
	if err := os.WriteFile(filepath.Join(dir, "settings.json"), []byte(set), 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := run.New(run.Config{DataDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Shutdown)
	srv := httptest.NewServer((&API{S: s}).Handler())
	t.Cleanup(srv.Close)

	resp := post(t, ctx, srv.URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-6"})
	if got := readRest(t, resp.Body); got != ev1+evFin+evDone {
		t.Fatalf("got %q", got)
	}
	h := resp.Header
	if h.Get(AppliedHeader) == "" || h.Get(RequestIDHeader) != "rid-6" || h.Get(upstreamIDHeader) != "up-1" ||
		len(h.Values(RequestIDHeader)) != 1 || h.Get("Cache-Control") != "no-cache" {
		t.Fatalf("headers %v", h)
	}
	t.Logf("upstream-opened stream headers: %v", h)

	resp = post(t, ctx, srv.URL, "/v1/chat/completions", `{"stream":false}`, nil)
	io.Copy(io.Discard, resp.Body)
	if id := resp.Header.Get(RequestIDHeader); !regexp.MustCompile(`^[0-9a-f]{16}$`).MatchString(id) {
		t.Fatalf("generated id %q", id)
	}
}

// The streamer setting: settings.json "streamer": "off" turns the streamer off (a cut stream
// then ends as the upstream ended it, as before the streamer), a PUT of another field keeps the
// key, and an edit of the file applies at once.
func TestStreamerSetting(t *testing.T) {
	ctx := testCtx(t)
	up := upstream(t, dies(false))
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	set := func(streamer string) {
		b := `{"mode":"external","external":{"base_url":"` + up.URL + `/v1","model":"","api_key":""},"streamer":"` + streamer + `"}`
		if err := os.WriteFile(path, []byte(b), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	set("off")
	s, err := run.New(run.Config{DataDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Shutdown)
	srv := httptest.NewServer((&API{S: s}).Handler())
	t.Cleanup(srv.Close)
	cut := ev1 + `data: {"choices":[{"index":0,"delta":{"content":"half`

	resp := post(t, ctx, srv.URL, "/v1/chat/completions", streamBody, nil)
	if got := readRest(t, resp.Body); got != cut {
		t.Fatalf("off: got %q", got)
	}
	if resp.Header.Get(RequestIDHeader) == "" || resp.Header.Get("Cache-Control") != "" {
		t.Fatalf("off: headers %v", resp.Header)
	}

	req, _ := http.NewRequestWithContext(ctx, "PUT", srv.URL+"/engine/v1/settings", strings.NewReader(`{"linger_s":30}`))
	req.Header.Set("X-Easy-Engine", "1")
	put, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, put.Body)
	put.Body.Close()
	if put.StatusCode != 200 {
		t.Fatalf("PUT settings: %d", put.StatusCode)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"streamer": "off"`) || !strings.Contains(string(b), `"linger_s": 30`) {
		t.Fatalf("settings.json after PUT:\n%s", b)
	}
	if s.Streamer() {
		t.Fatal("streamer on after an unrelated PUT")
	}

	set("on")
	s.ReloadSettings()
	resp = post(t, ctx, srv.URL, "/v1/chat/completions", streamBody, nil)
	if got := readRest(t, resp.Body); !strings.HasPrefix(got, ev1+`data: {"error":{"code":"upstream_ended"`) {
		t.Fatalf("on after reload: got %q", got)
	}
}

// The streamWriter sits inside the hearing wrapper: the engine's own waiting comment and its
// terminal event are not heard (a hung model server must not look alive while a client waits);
// the upstream's events are.
func TestStreamOwnWordsAreNotHeard(t *testing.T) {
	ctx := testCtx(t)
	var heard atomic.Int32
	release := make(chan struct{})
	up := upstream(t, slow(release, dies(false)))
	f := local()
	f.open = time.Millisecond
	f.hear = (&API{heard: func() { heard.Add(1) }}).hearing
	resp := post(t, ctx, front(t, up.URL, f).URL, "/v1/chat/completions", streamBody, map[string]string{RequestIDHeader: "rid-7"})
	br := bufio.NewReader(resp.Body)
	if got := readEvent(t, br); got != ": easy waiting\n\n" {
		t.Fatalf("first bytes %q", got)
	}
	if n := heard.Load(); n != 0 {
		t.Fatalf("the waiting comment was heard (%d)", n)
	}
	close(release)
	if got := readEvent(t, br); got != ev1 {
		t.Fatalf("event %q", got)
	}
	if n := heard.Load(); n < 1 {
		t.Fatal("the upstream's event was not heard")
	}
	before := heard.Load()
	if got := readRest(t, br); !strings.HasPrefix(got, `data: {"error":{"code":"model_server_gone"`) {
		t.Fatalf("rest %q", got)
	}
	if n := heard.Load(); n != before {
		t.Fatalf("the terminal event was heard (%d -> %d)", before, n)
	}
}
