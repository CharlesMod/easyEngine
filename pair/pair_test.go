package pair

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/event"
)

type fakeIssuer struct{ n int }

func (f *fakeIssuer) Issue(name string) (string, string, error) {
	f.n++
	return "cid-" + name, "secret-token-" + name, nil
}

func newHub(t *testing.T) *event.Hub {
	h, err := event.New([]event.Decl{{Name: FactName, Kind: event.Fact, Keyed: true, What: "test"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	return h
}

func TestAllowIssuesAtCollectionOnce(t *testing.T) {
	hub, iss := newHub(t), &fakeIssuer{}
	p := New(hub, iss, 0)
	req, id, err := p.Ask("mod", "192.0.2.7")
	if err != nil {
		t.Fatal(err)
	}
	if req.Key == "" || req.Key == id || len(req.Code) != 6 || req.From != "192.0.2.7" {
		t.Fatalf("request: %+v %s", req, id)
	}
	f, ok := hub.Fact(FactName, req.Key)
	if !ok || strings.Contains(string(f.Body), id) || strings.Contains(string(f.Body), "token") || !strings.Contains(string(f.Body), req.Code) {
		t.Fatalf("fact: %v %s", ok, f.Body)
	}
	if _, err := p.Answer(id, true); !errors.Is(err, ErrNoRequest) {
		t.Fatal("the secret id answered as a key")
	}
	if _, err := p.Answer(req.Key, true); err != nil {
		t.Fatal(err)
	}
	if iss.n != 0 {
		t.Fatal("token issued before the client collected it")
	}
	if _, ok := hub.Fact(FactName, req.Key); ok {
		t.Fatal("fact stands after the answer")
	}
	r, err := p.Wait(context.Background(), id)
	if err != nil || r.Outcome != Allowed || r.Token != "secret-token-mod" || r.ClientID != "cid-mod" {
		t.Fatalf("collect: %+v %v", r, err)
	}
	if r, _ := p.Wait(context.Background(), id); r.Outcome != Gone || iss.n != 1 {
		t.Fatalf("second collect: %+v issued %d", r, iss.n)
	}
}

func TestWaitKeepsTheRequestWhenTheClientHangsUp(t *testing.T) {
	p := New(nil, &fakeIssuer{}, 0)
	req, id, _ := p.Ask("mod", "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Wait(ctx, id); err == nil {
		t.Fatal("a gone client got an answer")
	}
	if len(p.Waiting()) != 1 {
		t.Fatal("the request did not keep waiting")
	}
	p.Answer(req.Key, false)
	if r, _ := p.Wait(context.Background(), id); r.Outcome != Denied {
		t.Fatalf("deny: %+v", r)
	}
}

func TestCapAndCodes(t *testing.T) {
	p := New(nil, &fakeIssuer{}, 2)
	_, a, _ := p.Ask("a", "")
	_, b, _ := p.Ask("b", "")
	_, c, _ := p.Ask("c", "")
	if r, _ := p.Wait(context.Background(), a); r.Outcome != Gone {
		t.Fatalf("oldest: %+v", r)
	}
	w := p.Waiting()
	if len(w) != 2 || w[0].Name != "b" || w[1].Name != "c" || w[0].Code == w[1].Code {
		t.Fatalf("waiting: %+v", w)
	}
	got, err := p.ByCode(w[0].Code[:3] + " " + w[0].Code[3:])
	if err != nil || got.Name != "b" {
		t.Fatalf("by code: %+v %v", got, err)
	}
	// two requests with one code (Ask never makes this; a hand-made clash) are refused
	p.mu.Lock()
	p.reqs[c].Code = p.reqs[b].Code
	p.mu.Unlock()
	if _, err := p.ByCode(w[0].Code); !errors.Is(err, ErrAmbiguous) {
		t.Fatalf("ambiguous: %v", err)
	}
	if _, err := p.ByCode("000000x"); !errors.Is(err, ErrNoRequest) {
		t.Fatalf("none: %v", err)
	}
}

func TestCloseEndsEveryWaiter(t *testing.T) {
	hub := newHub(t)
	p := New(hub, &fakeIssuer{}, 0)
	req, id, _ := p.Ask("a", "")
	done := make(chan Result, 1)
	go func() { r, _ := p.Wait(context.Background(), id); done <- r }()
	p.Close()
	if r := <-done; r.Outcome != Gone {
		t.Fatalf("after close: %+v", r)
	}
	if _, ok := hub.Fact(FactName, req.Key); ok {
		t.Fatal("fact stands after close")
	}
	if _, _, err := p.Ask("b", ""); !errors.Is(err, ErrClosed) {
		t.Fatalf("ask after close: %v", err)
	}
}

func TestCleanName(t *testing.T) {
	for in, want := range map[string]string{
		"  Kitchen \x07\t PC  ": "Kitchen PC",
		"":                      "unnamed client",
		"\x00\x1b":              "unnamed client",
		strings.Repeat("x", 99): strings.Repeat("x", NameMax),
		"a\u202eb":              "ab", // a direction override is a format character, not shown
	} {
		if got := CleanName(in); got != want {
			t.Errorf("CleanName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEngineIDIsStable(t *testing.T) {
	dir := t.TempDir()
	a, err := EngineID(dir)
	if err != nil || len(a) != 32 {
		t.Fatalf("%q %v", a, err)
	}
	b, err := EngineID(dir)
	if err != nil || a != b {
		t.Fatalf("second read: %q %v", b, err)
	}
	os.WriteFile(filepath.Join(dir, EngineIDFile), []byte("not hex at all"), 0o644)
	if _, err := EngineID(dir); err == nil {
		t.Fatal("a broken id file was taken")
	}
}
