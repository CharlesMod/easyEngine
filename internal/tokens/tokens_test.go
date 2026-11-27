package tokens

import (
	"encoding/base64"
	"os"
	"strings"
	"sync"
	"testing"
)

func TestMasterCreatedOnceAndPrivate(t *testing.T) {
	dir := t.TempDir()
	s := Open(dir)
	if _, ok := s.Check("anything"); ok {
		t.Fatal("empty store accepts a token")
	}
	m, err := s.Master()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := base64.RawURLEncoding.DecodeString(m)
	if err != nil || len(raw) != 32 {
		t.Fatalf("token %q: %d bytes, %v", m, len(raw), err)
	}
	if m2, _ := s.Master(); m2 != m {
		t.Fatal("Master changed between calls")
	}
	if m3, _ := Open(dir).Master(); m3 != m {
		t.Fatal("a second store sees another master")
	}
	fi, err := os.Stat(s.Path())
	if err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("token file: %v %v", fi, err)
	}
}

func TestMasterRaceMakesOneToken(t *testing.T) {
	for round := 0; round < 20; round++ {
		dir := t.TempDir()
		got := make([]string, 8)
		var wg sync.WaitGroup
		for i := range got {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				m, err := Open(dir).Master() // separate stores: separate mutexes, like separate processes
				if err != nil {
					t.Error(err)
				}
				got[i] = m
			}(i)
		}
		wg.Wait()
		for _, g := range got {
			if g != got[0] || g == "" {
				t.Fatalf("round %d: %v", round, got)
			}
		}
	}
}

func TestCheck(t *testing.T) {
	s := Open(t.TempDir())
	m, _ := s.Master()
	if id, ok := s.Check(m); !ok || id != MasterID {
		t.Fatalf("master: %q %v", id, ok)
	}
	for _, bad := range []string{"", "x", m + "x", m[:len(m)-1], strings.ToUpper(m)} {
		if _, ok := s.Check(bad); ok {
			t.Errorf("accepted %q", bad)
		}
	}
	id, tok, err := s.Issue("game")
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := s.Check(tok); !ok || got != id {
		t.Fatalf("issued: %q %v", got, ok)
	}
	b, _ := os.ReadFile(s.Path())
	if strings.Contains(string(b), tok) {
		t.Fatal("an issued token is stored in clear")
	}
	l, _ := s.List()
	if len(l) != 1 || l[0].Name != "game" || l[0].SHA256 == "" {
		t.Fatalf("list: %+v", l)
	}
	if err := s.Revoke(id); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Check(tok); ok {
		t.Fatal("revoked token accepted")
	}
	if s.Revoke(id) == nil {
		t.Fatal("revoking twice is an error")
	}
	if _, ok := s.Check(m); !ok {
		t.Fatal("master lost")
	}
}

func TestNewMasterReplaces(t *testing.T) {
	s := Open(t.TempDir())
	old, _ := s.Master()
	_, tok, _ := s.Issue("keep")
	n, err := s.NewMaster()
	if err != nil || n == old {
		t.Fatalf("new master %q err %v", n, err)
	}
	if _, ok := s.Check(old); ok {
		t.Fatal("old master still accepted")
	}
	if _, ok := s.Check(n); !ok {
		t.Fatal("new master refused")
	}
	if _, ok := s.Check(tok); !ok {
		t.Fatal("issued token lost by --new")
	}
	if m, _ := Open(s.path[:len(s.path)-len(FileName)-1]).Master(); m != n {
		t.Fatal("another store does not see the new master")
	}
	if fi, _ := os.Stat(s.Path()); fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", fi.Mode())
	}
}

func TestUnreadableFileAcceptsNothing(t *testing.T) {
	dir := t.TempDir()
	s := Open(dir)
	os.WriteFile(s.Path(), []byte("{broken"), 0o600)
	if _, ok := s.Check("x"); ok {
		t.Fatal("accepted with a broken file")
	}
	if _, err := s.Master(); err == nil {
		t.Fatal("Master must not overwrite a broken file")
	}
}
