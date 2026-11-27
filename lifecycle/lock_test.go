package lifecycle

import (
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCreateLockRace(t *testing.T) {
	for round := 0; round < 30; round++ {
		dir := t.TempDir()
		path := filepath.Join(dir, "engine.lock")
		const n = 8
		var wins, held atomic.Int32
		var wg sync.WaitGroup
		start := make(chan struct{})
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				<-start
				_, err := CreateLock(path, LockData{Pid: 1000 + i, Port: 47480 + i})
				switch err {
				case nil:
					wins.Add(1)
				case ErrLockHeld:
					held.Add(1)
				default:
					t.Error(err)
				}
			}(i)
		}
		close(start)
		wg.Wait()
		if wins.Load() != 1 || held.Load() != n-1 {
			t.Fatalf("round %d: wins=%d held=%d", round, wins.Load(), held.Load())
		}
		if _, err := ReadLock(path); err != nil {
			t.Fatal(err)
		}
		if m, _ := filepath.Glob(filepath.Join(dir, "engine.lock.tmp-*")); len(m) != 0 {
			t.Fatalf("temp files left: %v", m)
		}
	}
}

func TestLockNeverObservedEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "engine.lock")
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // reader
		defer wg.Done()
		for {
			select {
			case <-done:
				return
			default:
			}
			b, err := os.ReadFile(path)
			if err != nil {
				continue // not there yet
			}
			if len(b) == 0 {
				t.Error("observed an empty lock")
				return
			}
			if _, err := ReadLock(path); err != nil {
				t.Errorf("observed an unparseable lock: %v", err)
				return
			}
		}
	}()
	l, err := CreateLock(path, LockData{Pid: 1, Port: 2})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 300; i++ { // rewrites go through a unique temp + rename
		l.SetChildren([]ChildRef{{Pid: i, Name: "c"}})
	}
	close(done)
	wg.Wait()
	l.Remove()
	l.SetChildren(nil) // must not bring a removed lock back
	if _, err := os.Stat(path); err == nil {
		t.Fatal("lock came back after Remove")
	}
}

func TestLockInWrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "engine.lock")
	if LockInWrite(path) {
		t.Fatal("missing lock is not in write")
	}
	os.WriteFile(path, nil, 0o644)
	if !LockInWrite(path) {
		t.Fatal("fresh empty lock is in write")
	}
	os.WriteFile(path, []byte("{}"), 0o644)
	if LockInWrite(path) {
		t.Fatal("filled lock is not in write")
	}
}
