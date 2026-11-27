package api

import (
	"bufio"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// A streamed reply through the engine's own proxy arrives chunk by chunk: the fake server sends
// each next chunk only once the test has read the one before through the proxy (a control file,
// followed by the kernel's word). A proxy that buffered would hold the first chunk, the test
// would never read it, and the test would fail on its watchdog. Heard is called as the bytes
// flow, so the hang watchdog on the request is put back during the stream.
func TestProxyStreamsChunkByChunkAndIsHeard(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Llama["FAKELLAMA_STREAM_GATE"] = "1"
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	var heard atomic.Int32
	a := &API{S: s}
	a.heard = func() { heard.Add(1); s.Heard() }
	srv := httptest.NewServer(a.Handler())
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // a failed test lets the stream go, so the servers can close
	req, _ := http.NewRequestWithContext(ctx, "POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	lines := make(chan string)
	go func() { // the request too: a buffering proxy holds even the headers
		defer close(lines)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return
		}
		defer resp.Body.Close()
		br := bufio.NewReader(resp.Body)
		for {
			l, err := br.ReadString('\n')
			if strings.HasPrefix(l, "data:") {
				select {
				case lines <- l:
				case <-ctx.Done():
					return
				}
			}
			if err != nil {
				return
			}
		}
	}()
	next := func(i int) string {
		select {
		case l, ok := <-lines:
			if !ok {
				t.Fatalf("stream ended before chunk %d", i)
			}
			return l
		case <-time.After(10 * time.Second): // watchdog for silence: a buffered chunk never comes
			t.Fatalf("chunk %d did not come through the proxy: buffered", i)
		}
		return ""
	}
	for i := 0; i < 5; i++ {
		l := next(i)
		if !strings.Contains(l, "t"+strconv.Itoa(i)+" ") {
			t.Fatalf("chunk %d = %q", i, l)
		}
		if heard.Load() < int32(i+1) {
			t.Fatalf("Heard called %d times by chunk %d", heard.Load(), i)
		}
		if i < 4 { // let the fake send the next one
			if err := os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i+1)), nil, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	if l := next(5); !strings.Contains(l, "[DONE]") {
		t.Fatalf("last = %q", l)
	}
}
