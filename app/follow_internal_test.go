package app

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/tokens"
)

// tokens.json may not exist when the watch is made: its creation (the first pairing's write) and
// a later revoke are both told (host.Changes watches the directory). Each wait is for a call of
// the watch that saw the new state, so a state read without a word from the system proves nothing.
func TestFollowFileHearsCreationAndRevoke(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, tokens.FileName)
	st := tokens.Open(dir)
	var mu sync.Mutex
	var id string
	sawFile, sawRevoked := false, false
	heard := make(chan struct{}, 1)
	stop, err := followFile(path, func() {
		mu.Lock()
		if _, err := os.Stat(path); err == nil {
			sawFile = true
		}
		if id != "" && !st.Valid(id) {
			sawRevoked = true
		}
		mu.Unlock()
		select {
		case heard <- struct{}{}:
		default:
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	until := func(what string, ok func() bool) {
		t.Helper()
		dog := time.NewTimer(20 * time.Second) // watchdog for silence: the change was never told
		defer dog.Stop()
		for {
			mu.Lock()
			done := ok()
			mu.Unlock()
			if done {
				return
			}
			select {
			case <-heard:
			case <-dog.C: // watchdog for silence
				t.Fatalf("never heard %s", what)
			}
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("the file must not exist yet")
	}
	got, _, err := st.Issue("den") // the first pairing creates the file
	if err != nil {
		t.Fatal(err)
	}
	until("the creation", func() bool { return sawFile })
	mu.Lock()
	id = got
	mu.Unlock()
	if err := st.Revoke(got); err != nil {
		t.Fatal(err)
	}
	until("the revoke", func() bool { return sawRevoked })
}
