package api_test

import (
	"strings"
	"testing"
)

// The person's answer names the key and the code it showed: a key whose request its source
// replaced since (a new code) is 409 pair_changed and issues nothing.
func TestPairingAnswerBindsToTheCode(t *testing.T) {
	pe := pairServer(t)
	oldID, oldCode := askFrom(t, pe, "127.0.0.41", "device A")
	oldGot := collect(t, pe, oldID)
	key := pe.pr.Waiting()[0].Key
	newID, newCode := askFrom(t, pe, "127.0.0.41", "someone else")
	if r := waitCollected(t, oldGot); r.code != 410 {
		t.Fatalf("the replaced client: %+v", r)
	}
	if c, b := answer(t, pe, bearer(pe.master), `{"key":"`+key+`","allow":true}`); c != 400 {
		t.Fatalf("key without code: %d %s", c, b)
	}
	if c, b := answer(t, pe, bearer(pe.master), `{"key":"`+key+`","code":"`+oldCode+`","allow":true}`); c != 409 || !strings.Contains(b, `"code":"pair_changed"`) {
		t.Fatalf("allow with the old code: %d %s", c, b)
	}
	if list, _ := pe.st.List(); len(list) != 0 {
		t.Fatalf("a token was issued for a stale press: %+v", list)
	}
	newGot := collect(t, pe, newID)
	if c, b := answer(t, pe, bearer(pe.master), `{"key":"`+key+`","code":"`+newCode+`","allow":true}`); c != 200 {
		t.Fatalf("allow with the new code: %d %s", c, b)
	}
	if r := waitCollected(t, newGot); r.code != 200 || !strings.Contains(r.body, `"token"`) {
		t.Fatalf("the new client: %+v", r)
	}
}
