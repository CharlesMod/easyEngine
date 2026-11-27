package host

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// bound is the test-only watchdog for silence on every wait below: a broken
// watch fails the test instead of hanging it.
const bound = 10 * time.Second

// within waits for c to give, or fails after bound.
func within(t *testing.T, c <-chan struct{}, what string) {
	t.Helper()
	tm := time.NewTimer(bound) // watchdog for silence: test-only, a broken watch fails instead of hanging
	defer tm.Stop()
	select {
	case <-c:
	case <-tm.C:
		t.Fatalf("%s: nothing heard", what)
	}
}

func empty(c <-chan struct{}) bool {
	select {
	case <-c:
		return false
	default:
		return true
	}
}

// start runs a child that lives until killed, and reaps it when it ends.
func start(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.Command("sleep", "300")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
	return cmd
}

func TestExitsHearsAChildEnd(t *testing.T) {
	cmd := start(t)
	ended, stop, err := Exits(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	if !empty(ended) {
		t.Fatal("a living child is said ended")
	}
	cmd.Process.Kill()
	within(t, ended, "the killed child's end")
	stop()
	stop() // twice, and after the end, is nothing
}

func TestExitsOfAGoneProcessIsEnded(t *testing.T) {
	cmd := exec.Command("true")
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	ended, stop, err := Exits(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	within(t, ended, "a reaped child's end")
}

func TestExitsHearsAnEndBeforeTheReap(t *testing.T) {
	// the child ends and stays a zombie until Wait: its end is still its end
	cmd := exec.Command("true")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Wait()
	ended, stop, err := Exits(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	within(t, ended, "an unreaped child's end")
}

func TestExitsStopLetsGo(t *testing.T) {
	cmd := start(t)
	ended, stop, err := Exits(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	stop()
	cmd.Process.Kill()
	cmd.Wait()
	// after a stop the channel is never closed; the end, heard by the
	// reap, came after the stop, and a closed channel now would be a lie
	if !empty(ended) {
		t.Fatal("a stopped watch said the end")
	}
}

func TestStartOfTellsProcessesApart(t *testing.T) {
	a, b := start(t), start(t)
	ta, err := StartOf(a.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := StartOf(a.Process.Pid); again != ta {
		t.Fatalf("one process, two starts: %q %q", ta, again)
	}
	if _, err := strconv.ParseUint(ta, 10, 64); err != nil {
		t.Fatalf("start %q is not the kernel's ticks", ta)
	}
	if _, err := StartOf(b.Process.Pid); err != nil {
		t.Fatal(err)
	}
	if _, err := StartOf(1 << 30); err == nil {
		t.Fatal("a pid with no process has a start")
	}

	ended, stop, err := ExitsOf(a.Process.Pid, ta)
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	if _, _, err := ExitsOf(a.Process.Pid, ta+"0"); !errors.Is(err, ErrNotThatProcess) {
		t.Fatalf("another start's token: %v, want ErrNotThatProcess", err)
	}
	a.Process.Kill()
	within(t, ended, "ExitsOf the killed child")
}

// changes watches path and hears its directory's names through testSeen.
func changes(t *testing.T, path string) (<-chan struct{}, <-chan string) {
	t.Helper()
	names := make(chan string, 1024)
	testSeen = func(n string) { names <- n }
	ch, stop, err := Changes(path)
	testSeen = nil
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(stop)
	return ch, names
}

// settle makes a marker file and waits until the watch has heard it: the
// kernel says a directory's events in order, so everything before it has
// been acted on.
func settle(t *testing.T, dir string, names <-chan string, n int) {
	t.Helper()
	marker := "marker" + strconv.Itoa(n)
	if err := os.WriteFile(filepath.Join(dir, marker), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	tm := time.NewTimer(bound) // watchdog for silence: test-only, a broken watch fails instead of hanging
	defer tm.Stop()
	for {
		select {
		case got := <-names:
			if got == marker {
				return
			}
		case <-tm.C:
			t.Fatalf("the marker %s was never heard", marker)
		}
	}
}

func TestChangesHearsEachChangeOfOneFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	ch, names := changes(t, path)
	n := 0
	step := func(what string, do func() error, want bool) {
		t.Helper()
		if err := do(); err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		n++
		settle(t, dir, names, n)
		if got := !empty(ch); got != want {
			t.Fatalf("%s: signalled %v, want %v", what, got, want)
		}
	}
	write := func(p, s string) func() error { return func() error { return os.WriteFile(p, []byte(s), 0o644) } }

	step("another file written", write(filepath.Join(dir, "other"), "x"), false)
	step("created", write(path, "a"), true)
	step("nothing", func() error { return nil }, false)
	step("written", write(path, "b"), true)
	step("replaced by a rename", func() error {
		tmp := filepath.Join(dir, "state.json.tmp")
		if err := os.WriteFile(tmp, []byte("c"), 0o644); err != nil {
			return err
		}
		return os.Rename(tmp, path)
	}, true)
	step("another file removed", func() error { return os.Remove(filepath.Join(dir, "other")) }, false)
	step("removed", func() error { return os.Remove(path) }, true)
}

func TestChangesNeedsTheDirectory(t *testing.T) {
	if _, _, err := Changes(filepath.Join(t.TempDir(), "no", "such", "file")); err == nil {
		t.Fatal("a watch in a missing directory was made")
	}
}

func TestOpensHearsAnOpen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "node")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	ch, stop, err := Opens([]string{path})
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	within(t, ch, "the open")
	if _, _, err := Opens([]string{filepath.Join(t.TempDir(), "missing")}); err == nil {
		t.Fatal("Opens of a missing path was made")
	}
}

func TestSilenceFiresOnlyOnSilence(t *testing.T) {
	fired := make(chan struct{})
	s := NewSilence(time.Millisecond, func() { close(fired) })
	within(t, fired, "the silence")
	s.Heard() // after it fired: nothing
	if s.Stop() {
		t.Fatal("a fired watchdog said it was still armed")
	}

	never := NewSilence(time.Hour, func() { t.Error("fired") })
	never.Heard()
	if !never.Stop() {
		t.Fatal("an armed watchdog said it was not")
	}

	done := make(chan struct{})
	Moment(time.Millisecond, func() { close(done) })
	within(t, done, "the named moment")
	if cancel := Moment(time.Hour, func() { t.Error("fired") }); !cancel() {
		t.Fatal("a moment to come could not be cancelled")
	}
}
