package api

import (
	"bufio"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// A streamed reply forwarded through the proxy's writer: Heard is told at least once per chunk,
// and each chunk is flushed to the client as it comes (the writer stays a Flusher).
func TestStreamedReplyIsHeardPerChunk(t *testing.T) {
	const chunks = 4
	step := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < chunks; i++ {
			w.Write([]byte("data: x\n\n"))
			w.(http.Flusher).Flush()
			select {
			case <-step:
			case <-r.Context().Done():
				return
			}
		}
	}))
	defer up.Close()
	target, _ := url.Parse(up.URL)
	var heard atomic.Int32
	a := &API{heard: func() { heard.Add(1) }}
	rp := &httputil.ReverseProxy{FlushInterval: -1, Rewrite: func(pr *httputil.ProxyRequest) { pr.SetURL(target) }}
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { rp.ServeHTTP(a.hearing(w), r) }))
	defer front.Close()
	resp, err := http.Get(front.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	for i := 1; i <= chunks; i++ {
		line, err := br.ReadString('\n')
		if err != nil || !strings.HasPrefix(line, "data:") {
			t.Fatalf("chunk %d: %q %v", i, line, err)
		}
		br.ReadString('\n')
		if got := heard.Load(); got < int32(i) {
			t.Fatalf("after chunk %d heard was told %d times", i, got)
		}
		select {
		case step <- struct{}{}:
		case <-time.After(5 * time.Second): // watchdog for silence: the upstream stopped waiting for the step
			t.Fatalf("upstream not waiting at chunk %d", i)
		}
	}
}
