package settings

import (
	"strings"
	"testing"
)

// The keep_context key: on by default, a missing key is on, off and always are kept, anything
// else is refused with its line, and a file is always written with the key.
func TestKeepContextKey(t *testing.T) {
	if Defaults().KeepContext != "on" {
		t.Fatalf("default %q", Defaults().KeepContext)
	}
	f, err := Parse([]byte(`{"mode":"local"}`))
	if err != nil || f.KeepContext != "on" {
		t.Fatalf("missing key: %q %v", f.KeepContext, err)
	}
	for _, v := range []string{"off", "always"} {
		f, err = Parse([]byte(`{"keep_context":"` + v + `"}`))
		if err != nil || f.KeepContext != v {
			t.Fatalf("%s: %q %v", v, f.KeepContext, err)
		}
	}
	_, err = Parse([]byte("{\n\"mode\":\"local\",\n\"keep_context\":\"maybe\"\n}"))
	se, ok := err.(*Error)
	if !ok || se.Line != 3 || !strings.Contains(se.Msg, "keep_context must be on, off or always") {
		t.Fatalf("bad value: %v", err)
	}
	if !strings.Contains(string(Encode(Defaults())), `"keep_context": "on"`) {
		t.Fatalf("written file lacks the key:\n%s", Encode(Defaults()))
	}
	a, b := Defaults(), Defaults()
	b.KeepContext = "always"
	if a.Equal(b) {
		t.Fatal("Equal ignores keep_context")
	}
}
