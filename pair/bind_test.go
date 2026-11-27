package pair

import (
	"context"
	"errors"
	"testing"
)

// An answer binds to the code the person compared: a request its source replaced (same key, new
// code) is not answered by a press made for the old code.
func TestAnswerBindsToTheCode(t *testing.T) {
	iss := &fakeIssuer{}
	p := New(nil, iss, 0)
	old, oldID, _ := p.Ask("device A", "192.0.2.5")
	repl, newID, _ := p.Ask("someone else", "192.0.2.5")
	if repl.Key != old.Key || repl.Code == old.Code {
		t.Fatalf("replacement: %+v after %+v", repl, old)
	}
	// the press the person made for the code they compared (the old one)
	if _, err := p.Answer(old.Key, old.Code, true); !errors.Is(err, ErrChanged) {
		t.Fatalf("allow with the old code: %v", err)
	}
	if _, err := p.Answer(old.Key, old.Code, false); !errors.Is(err, ErrChanged) {
		t.Fatalf("deny with the old code: %v", err)
	}
	if iss.n != 0 {
		t.Fatal("a token was issued for a stale press")
	}
	if r, _ := p.Wait(context.Background(), oldID); r.Outcome != Gone {
		t.Fatalf("the replaced client: %+v", r)
	}
	if w := p.Waiting(); len(w) != 1 || w[0].Code != repl.Code || w[0].Quiet {
		t.Fatalf("the new request must still wait, not marked denied: %+v", w)
	}
	// an Allow with the code it shows now answers the new request only
	if _, err := p.Answer(repl.Key, repl.Code[:3]+" "+repl.Code[3:], true); err != nil {
		t.Fatal(err)
	}
	r, _ := p.Wait(context.Background(), newID)
	if r.Outcome != Allowed || r.ClientID != "cid-someone else" || iss.n != 1 {
		t.Fatalf("allow with the new code: %+v issued %d", r, iss.n)
	}
}
