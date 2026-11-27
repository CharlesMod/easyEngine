package discover

import (
	"testing"

	"golang.org/x/net/dns/dnsmessage"
)

// DNS names are compared without regard to ASCII case (RFC 4343): a querier may send any case,
// and some randomise it.
func TestQueriesInAnotherCase(t *testing.T) {
	for _, q := range []struct {
		name string
		typ  dnsmessage.Type
	}{
		{"_EASY-Engine._TCP.Local.", dnsmessage.TypePTR},
		{"STUDY pc [012345]._easy-ENGINE._tcp.LOCAL.", dnsmessage.TypeSRV},
		{"Easy-Engine-01234567.LOCAL.", dnsmessage.TypeA},
	} {
		if _, ok := Answer(testInfo, question(t, 0, q.name, q.typ, dnsmessage.ClassINET), testAddrs, false); !ok {
			t.Errorf("no answer to %s %v", q.name, q.typ)
		}
	}
	// only ASCII folds: a non-ASCII byte is compared as it is
	if foldName("ÄB") != "Äb" {
		t.Errorf("foldName folded beyond ASCII: %q", foldName("ÄB"))
	}
}
