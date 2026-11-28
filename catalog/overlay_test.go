package catalog

import (
	"strings"
	"testing"
)

func TestWithModels(t *testing.T) {
	c, err := Parse(embedded)
	if err != nil {
		t.Fatal(err)
	}
	n := len(c.Models)
	good := Model{ID: "my-model", Name: "Mine", File: "my-model.gguf", URL: "https://example.com/my-model.gguf",
		SHA256: strings.Repeat("ab", 32), Size: 1 << 30}
	imp := Model{ID: "imported", File: "imported.gguf"} // import-only: no URL
	next, err := c.WithModels([]Model{good, imp})
	if err != nil {
		t.Fatal(err)
	}
	if len(next.Models) != n+2 || len(c.Models) != n {
		t.Fatalf("models %d (catalog itself %d)", len(next.Models), len(c.Models))
	}
	if m, ok := next.Model("my-model"); !ok || m.URL != good.URL {
		t.Fatal("not found")
	}
	// the same overlay again adds nothing and is no error
	again, err := next.WithModels([]Model{good})
	if err != nil || len(again.Models) != n+2 {
		t.Fatalf("again: %v %d", err, len(again.Models))
	}
	bad := func(m Model, want string) {
		t.Helper()
		if _, err := next.WithModels([]Model{m}); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%+v: %v (want %q)", m, err, want)
		}
	}
	builtin := c.Models[0]
	builtin.Name = "a replacement"
	bad(builtin, "never replaces")
	other := good
	other.Size++
	bad(other, "never replaces") // an added id is not replaced either
	bad(Model{ID: "x", File: "../x.gguf"}, "model file")
	bad(Model{ID: "a/b", File: "x.gguf"}, "model id")
	bad(Model{ID: "h", File: "h.gguf", URL: "http://example.com/h.gguf", SHA256: strings.Repeat("ab", 32), Size: 1}, "https")
	bad(Model{ID: "s", File: "s.gguf", URL: "https://example.com/s.gguf", Size: 1}, "sha256")
	bad(Model{ID: "z", File: "z.gguf", URL: "https://example.com/z.gguf", SHA256: strings.Repeat("ab", 32)}, "size")
	bad(Model{ID: "f", File: builtin.File}, "already")
	if _, err := c.WithModels([]Model{{ID: "d", File: "d1.gguf"}, {ID: "d", File: "d2.gguf"}}); err == nil {
		t.Error("two of one id in one overlay")
	}
	many := make([]Model, MaxOverlayModels+1)
	if _, err := c.WithModels(many); err == nil {
		t.Error("too many")
	}
}
