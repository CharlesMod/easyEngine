package app_test

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/internal/tokens"
)

// Several serves started together: exactly one runs the engine, the rest attach to it.
func TestServesStartedTogetherMakeOneEngine(t *testing.T) {
	for round := 0; round < 3; round++ {
		dir := t.TempDir()
		const n = 4
		var parents []*sleeperProc
		for i := 0; i < n; i++ {
			parents = append(parents, newSleeper(t))
		}
		var started atomic.Int32
		var attached atomic.Int32
		var wg sync.WaitGroup
		release := make(chan struct{})
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				<-release
				r, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: []int{parents[i].pid}, Port: 47620, Version: "t",
					Started: func(int) { started.Add(1) }})
				if err != nil {
					t.Errorf("serve %d: %v", i, err)
					return
				}
				if r.Attached {
					attached.Add(1)
				}
			}(i)
		}
		close(release)
		waitFor(t, func() bool { return started.Load()+attached.Load() == n }, "every serve to settle")
		if started.Load() != 1 {
			t.Fatalf("round %d: %d engines started, %d attached", round, started.Load(), attached.Load())
		}
		for _, p := range parents {
			p.kill()
		}
		wg.Wait()
	}
}

// An engine that has begun shutting down must not be attached to: the new serve waits for it
// to end and runs its own engine.
func TestServeDoesNotAttachToStoppingEngine(t *testing.T) {
	dir := t.TempDir()
	p1, p2 := newSleeper(t), newSleeper(t)
	s := startServe(t, dir, []int{p1.pid})
	tg, ok := app.Find(dir)
	if !ok {
		t.Fatal("engine not found")
	}
	req, _ := tg.NewRequest("POST", "/engine/v1/shutdown", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != 200 {
		t.Fatalf("shutdown: %v %v", resp, err)
	}
	resp.Body.Close()
	started := make(chan int, 1)
	res := make(chan app.Result, 1)
	go func() {
		r, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: []int{p2.pid}, Port: 47600, Version: "t",
			Started: func(p int) { started <- p }})
		if err != nil {
			t.Errorf("second serve: %v", err)
		}
		res <- r
	}()
	select {
	case <-started:
	case r := <-res:
		t.Fatalf("second serve returned without starting an engine: %+v", r)
	case <-time.After(20 * time.Second): // watchdog for silence
		t.Fatal("second serve never took over")
	}
	<-s.res
	p2.kill()
	if r := <-res; r.Attached {
		t.Fatal("attached to the stopping engine")
	}
}

func lanOK(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("needs 127.0.0.2 as a second local address")
	}
}

func startHost(t *testing.T, dir, host string, open bool, parent int) (done chan app.Result, port int) {
	t.Helper()
	ch := make(chan int, 1)
	done = make(chan app.Result, 1)
	go func() {
		r, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: []int{parent}, Host: host, Open: open, Port: 47640, Version: "t",
			Started: func(p int) { ch <- p }})
		if err != nil {
			t.Errorf("serve: %v", err)
		}
		done <- r
	}()
	select {
	case port = <-ch:
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("engine did not start")
	}
	return done, port
}

func get(t *testing.T, url, tok string) int {
	t.Helper()
	req, _ := http.NewRequest("GET", url, nil)
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}

func TestTokenedEngineAndItsClients(t *testing.T) {
	lanOK(t)
	dir := t.TempDir()
	p1, p2 := newSleeper(t), newSleeper(t)
	done, port := startHost(t, dir, "127.0.0.2", false, p1.pid)
	url := "http://127.0.0.2:" + itoa(port) + "/engine/v1/status"
	tok, ok := tokens.Open(dir).Existing()
	if !ok {
		t.Fatal("no token made for a non-loopback listener")
	}
	if fi, err := os.Stat(filepath.Join(dir, "tokens.json")); err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("token file: %v %v", fi, err)
	}
	if c := get(t, url, ""); c != 401 {
		t.Fatalf("no token: %d", c)
	}
	if c := get(t, url, tok); c != 200 {
		t.Fatalf("with token: %d", c)
	}
	// the engine's own clients find it through the lock and send the token
	tg, ok := app.Find(dir)
	if !ok || tg.Token != tok || tg.Port != port {
		t.Fatalf("Find: %+v %v", tg, ok)
	}
	if u := app.SetupURL(dir, tg); !strings.HasSuffix(u, "/#token="+tok) {
		t.Fatalf("setup url %q", u)
	}
	// a second serve attaches through the token
	r, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: []int{p2.pid}, Version: "t"})
	if err != nil || !r.Attached || r.Host != "127.0.0.2" {
		t.Fatalf("attach to a tokened engine: %+v %v", r, err)
	}
	p1.kill()
	p2.kill()
	<-done
}

func TestOpenEngineNeedsNoToken(t *testing.T) {
	lanOK(t)
	dir := t.TempDir()
	p := newSleeper(t)
	done, port := startHost(t, dir, "127.0.0.2", true, p.pid)
	if c := get(t, "http://127.0.0.2:"+itoa(port)+"/engine/v1/status", ""); c != 200 {
		t.Fatalf("--open: %d", c)
	}
	if _, err := os.Stat(filepath.Join(dir, "tokens.json")); err == nil {
		t.Fatal("--open made a token file")
	}
	p.kill()
	<-done
}

func TestLoopbackEngineNeedsNoToken(t *testing.T) {
	dir := t.TempDir()
	p := newSleeper(t)
	s := startServe(t, dir, []int{p.pid})
	tg, _ := app.Find(dir)
	if c := get(t, tg.Base()+"/engine/v1/status", ""); c != 200 {
		t.Fatalf("loopback: %d", c)
	}
	if _, err := os.Stat(filepath.Join(dir, "tokens.json")); err == nil {
		t.Fatal("a loopback engine made a token file")
	}
	p.kill()
	<-s.res
}

// An empty lock that is brand new is held (its creator is mid-write), not stale.
func TestFreshEmptyLockIsNotStolen(t *testing.T) {
	dir := t.TempDir()
	lock := filepath.Join(dir, "engine.lock")
	os.WriteFile(lock, nil, 0o644)
	p := newSleeper(t)
	_, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: []int{p.pid}, Port: 47660, Version: "t"})
	if err == nil {
		t.Fatal("took a lock that was being written")
	}
	if _, serr := os.Stat(lock); serr != nil {
		t.Fatal("the fresh lock was removed")
	}
}

// helpers

type sleeperProc struct {
	pid  int
	kill func()
}

func newSleeper(t *testing.T) *sleeperProc {
	c := sleeper(t)
	var once sync.Once
	return &sleeperProc{pid: c.Process.Pid, kill: func() {
		once.Do(func() { c.Process.Kill(); c.Wait() })
	}}
}

func waitFor(t *testing.T, cond func() bool, what string) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for", what)
		}
		runtime.Gosched()
		time.Sleep(20 * time.Millisecond) // watchdog for silence
	}
}
