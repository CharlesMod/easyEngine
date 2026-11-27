package app_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/run"
)

func prof() *run.Profile {
	p := run.Profile{Game: "TestGame", Slots: 3, MinSlots: 1}
	p.CtxPerSlot.Max, p.CtxPerSlot.Floor = 65536, 8192
	return &p
}

func sleeper(t *testing.T) *exec.Cmd {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("unix-only helper")
	}
	c := exec.Command("sleep", "60")
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Process.Kill(); c.Wait() })
	return c
}

type served struct {
	res  chan app.Result
	port chan int
}

func startServe(t *testing.T, dir string, parents []int) *served {
	t.Helper()
	s := &served{res: make(chan app.Result, 1), port: make(chan int, 1)}
	go func() {
		r, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: parents, Port: 47600, Version: "t",
			Started: func(p int) { s.port <- p }})
		if err != nil {
			t.Errorf("serve: %v", err)
		}
		s.res <- r
	}()
	select {
	case <-s.port:
	case <-time.After(10 * time.Second): // watchdog for silence: the engine never says it started
		t.Fatal("engine did not start")
	}
	return s
}

func TestParentDeathEndsEngine(t *testing.T) {
	dir := t.TempDir()
	par := sleeper(t)
	s := startServe(t, dir, []int{par.Process.Pid})
	if _, err := os.Stat(filepath.Join(dir, "engine.lock")); err != nil {
		t.Fatal("no lock:", err)
	}
	par.Process.Kill()
	par.Wait()
	t0 := time.Now()
	select {
	case <-s.res:
	case <-time.After(5 * time.Second): // watchdog for silence: the parent's end never ended the engine
		t.Fatal("engine still running 5 s after its parent died")
	}
	t.Logf("engine ended %v after the parent died", time.Since(t0))
	if _, err := os.Stat(filepath.Join(dir, "engine.lock")); err == nil {
		t.Fatal("lock left behind")
	}
}

func TestSecondServeAttaches(t *testing.T) {
	dir := t.TempDir()
	p1, p2 := sleeper(t), sleeper(t)
	s := startServe(t, dir, []int{p1.Process.Pid})
	r, err := app.Serve(context.Background(), app.Options{DataDir: dir, Profile: prof(), ParentPIDs: []int{p2.Process.Pid}, Port: 47600})
	if err != nil || !r.Attached {
		t.Fatalf("second serve: %+v %v", r, err)
	}
	// still one engine: it lives until BOTH parents are gone
	p1.Process.Kill()
	p1.Wait()
	// The engine still answers, and takes a third parent, after the first parent's end: the
	// reply is the event that tells it is serving at this point (no wait on a clock).
	p3 := sleeper(t)
	tg, ok := app.Find(dir)
	if !ok {
		t.Fatal("the engine does not answer after the first parent died")
	}
	if err := app.PostAttach(tg, p3.Process.Pid); err != nil {
		t.Fatal("attach after the first parent died:", err)
	}
	select {
	case <-s.res:
		t.Fatal("engine ended while the attached parents still live")
	default:
	}
	p2.Process.Kill()
	p2.Wait()
	p3.Process.Kill()
	p3.Wait()
	select {
	case <-s.res:
	case <-time.After(5 * time.Second): // watchdog for silence: the last parent's end never ended the engine
		t.Fatal("engine did not end after the last parent died")
	}
}

func TestStaleLockCleanup(t *testing.T) {
	dir := t.TempDir()
	orphan := sleeper(t) // a child the crashed engine left running
	bystander := sleeper(t)
	ost, err := lifecycle.PidStart(orphan.Process.Pid)
	if err != nil {
		t.Skip("no pid start time:", err)
	}
	// a dead engine pid: a process that already exited
	dead := exec.Command("true")
	dead.Run()
	lock := `{"pid":` + itoa(dead.Process.Pid) + `,"start":"x","port":47601,"children":[` +
		`{"pid":` + itoa(orphan.Process.Pid) + `,"start":"` + ost + `","name":"llama-server"},` +
		`{"pid":` + itoa(bystander.Process.Pid) + `,"start":"not-its-start","name":"llama-server"}],"started":"2026-01-01T00:00:00Z"}`
	os.WriteFile(filepath.Join(dir, "engine.lock"), []byte(lock), 0o644)
	par := sleeper(t)
	s := startServe(t, dir, []int{par.Process.Pid})
	done := make(chan struct{})
	go func() { orphan.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second): // watchdog for silence: the orphan never ended
		t.Fatal("the orphaned child of the crashed engine was not killed")
	}
	if !lifecycle.PidAlive(bystander.Process.Pid) {
		t.Fatal("a process whose start time differs (pid reuse) was killed")
	}
	par.Process.Kill()
	par.Wait()
	<-s.res
}

func itoa(n int) string { return strconv.Itoa(n) }
