package lifecycle

import (
	"context"
	"os"
	"os/exec"
	"testing"
	"time"
)

// TestMain lets the test binary act as a portable child: with the env flag set
// it blocks reading stdin until the parent closes the pipe.
func TestMain(m *testing.M) {
	if os.Getenv("EE_WATCH_CHILD") == "1" {
		buf := make([]byte, 1)
		os.Stdin.Read(buf)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

type child struct {
	cmd   *exec.Cmd
	stdin interface{ Close() error }
}

func spawnParent(t *testing.T) *child {
	t.Helper()
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "EE_WATCH_CHILD=1")
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	c := &child{cmd: cmd, stdin: in}
	t.Cleanup(func() { in.Close(); cmd.Process.Kill(); cmd.Wait() })
	return c
}

// end lets the child exit and reaps it, so the system gives its word.
func (c *child) end() {
	c.stdin.Close()
	c.cmd.Wait()
}

func TestWatcherEndOfParent(t *testing.T) {
	c := spawnParent(t)
	w := NewWatcher()
	defer w.Close()
	if err := w.Add(c.cmd.Process.Pid); err != nil {
		t.Fatal(err)
	}
	if got := w.PIDs(); len(got) != 1 || got[0] != c.cmd.Process.Pid {
		t.Fatalf("PIDs %v", got)
	}
	done := make(chan bool, 1)
	go func() { done <- w.Run(context.Background()) }()
	c.end()
	select {
	case ok := <-done:
		if !ok {
			t.Fatal("Run answered false after the parent ended")
		}
	case <-time.After(30 * time.Second): // watchdog for silence
		t.Fatal("Run did not return after the parent ended")
	}
	if got := w.PIDs(); len(got) != 0 {
		t.Errorf("PIDs after end: %v", got)
	}
}

func TestWatcherWaitsForEveryParent(t *testing.T) {
	a, b := spawnParent(t), spawnParent(t)
	w := NewWatcher()
	defer w.Close()
	for _, c := range []*child{a, b} {
		if err := w.Add(c.cmd.Process.Pid); err != nil {
			t.Fatal(err)
		}
	}
	done := make(chan bool, 1)
	go func() { done <- w.Run(context.Background()) }()
	a.end()
	select {
	case <-done:
		t.Fatal("Run returned with a parent still alive")
	case <-time.After(200 * time.Millisecond): // watchdog for silence: nothing may come
	}
	b.end()
	select {
	case ok := <-done:
		if !ok {
			t.Fatal("Run answered false")
		}
	case <-time.After(30 * time.Second): // watchdog for silence
		t.Fatal("Run did not return after the last parent ended")
	}
}

func TestWatcherRefusesDeadPid(t *testing.T) {
	c := spawnParent(t)
	pid := c.cmd.Process.Pid
	c.end()
	w := NewWatcher()
	defer w.Close()
	if err := w.Add(pid); err == nil {
		t.Error("Add of a dead pid must fail")
	}
	if err := w.Add(0); err == nil {
		t.Error("Add of pid 0 must fail")
	}
	if got := w.PIDs(); len(got) != 0 {
		t.Errorf("PIDs %v", got)
	}
}

func TestWatcherNoParentsWaitsForCtx(t *testing.T) {
	w := NewWatcher()
	defer w.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool, 1)
	go func() { done <- w.Run(ctx) }()
	cancel()
	select {
	case ok := <-done:
		if ok {
			t.Fatal("Run with no parent answered true")
		}
	case <-time.After(30 * time.Second): // watchdog for silence
		t.Fatal("Run ignored ctx")
	}
}

func TestWatcherCloseLetsWatchesGo(t *testing.T) {
	c := spawnParent(t)
	w := NewWatcher()
	if err := w.Add(c.cmd.Process.Pid); err != nil {
		t.Fatal(err)
	}
	w.Close()
	w.Close() // twice is fine
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool, 1)
	go func() { done <- w.Run(ctx) }()
	cancel()
	select {
	case ok := <-done:
		if ok {
			t.Fatal("Run answered true for a parent that is alive")
		}
	case <-time.After(30 * time.Second): // watchdog for silence
		t.Fatal("Run ignored ctx")
	}
}
