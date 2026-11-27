package tokens

import (
	"os"
	"strings"
	"testing"
)

func TestRevokedIsToldApartFromUnknown(t *testing.T) {
	st := Open(t.TempDir())
	if _, err := st.Master(); err != nil {
		t.Fatal(err)
	}
	id, tok, err := st.Issue("a mod")
	if err != nil {
		t.Fatal(err)
	}
	if got, v := st.Verify(tok); v != Valid || got != id || st.NameOf(id) != "a mod" {
		t.Fatalf("issued: %q %v %q", got, v, st.NameOf(id))
	}
	if err := st.Revoke(id); err != nil {
		t.Fatal(err)
	}
	if got, v := st.Verify(tok); v != Revoked || got != id {
		t.Fatalf("revoked: %q %v", got, v)
	}
	if _, ok := st.Check(tok); ok {
		t.Fatal("Check accepts a revoked token")
	}
	if _, v := st.Verify("never-issued"); v != Unknown {
		t.Fatalf("unknown: %v", v)
	}
	if list, _ := st.List(); len(list) != 0 {
		t.Fatalf("list after revoke: %+v", list)
	}
	b, _ := os.ReadFile(st.Path())
	if strings.Contains(string(b), tok) {
		t.Fatal("an issued token is kept in clear")
	}
}
