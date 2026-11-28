//go:build !windows

package lifecycle

import (
	"context"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// gone reports whether pid no longer runs: absent, or a zombie awaiting its reaper.
func gone(pid int) bool {
	if syscall.Kill(pid, 0) == syscall.ESRCH {
		return true
	}
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return true
	}
	s := string(b)
	i := strings.LastIndexByte(s, ')')
	return i >= 0 && i+2 < len(s) && s[i+2] == 'Z'
}

// A group member that ignores SIGTERM and outlives the leader must not survive Stop.
func TestStopKillsGroupSurvivor(t *testing.T) {
	if _, err := os.Stat("/proc/self/stat"); err != nil {
		t.Skip("needs /proc")
	}
	sink, err := OpenLog("")
	if err != nil {
		t.Fatal(err)
	}
	pidc := make(chan int, 1)
	sink.OnLine(func(line string) {
		if n, err := strconv.Atoi(strings.TrimSpace(line)); err == nil {
			select {
			case pidc <- n:
			default:
			}
		}
	})
	g := NewGroup()
	p, err := g.Launch(Spec{
		Path: "/bin/sh",
		Args: []string{"-c", `(trap "" TERM; sleep 60 >/dev/null 2>&1 & echo $!; exec >/dev/null 2>&1; wait) & exec sleep 60`},
		Log:  sink,
	})
	if err != nil {
		t.Fatal(err)
	}
	var child int
	select {
	case child = <-pidc:
	case <-time.After(10 * time.Second): // watchdog for silence: the shell never printed the child's pid
		p.Kill()
		t.Fatal("no child pid")
	}
	defer syscall.Kill(child, syscall.SIGKILL)
	w := NewWatcher() // the system's word of the child's end (a pidfd), not a poll
	if err := w.Add(child); err != nil {
		t.Fatal(err)
	}

	p.Stop(10 * time.Second)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second) // watchdog for silence: the kill never ended the child
	defer cancel()
	if !w.Run(ctx) || !gone(child) {
		t.Fatalf("child %d survived Stop", child)
	}
}
