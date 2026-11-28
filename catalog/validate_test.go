package catalog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const goodSHA = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func doc(assetID, modelID, file, url, sha string, size int64) string {
	return fmt.Sprintf(`{"version":999,"llamacpp":{"tag":"t","assets":[{"id":%q,"os":"linux","arch":"amd64","url":%q,"sha256":%q,"size":%d}]},
"uv":[],"models":[{"id":%q,"name":"m","file":%q,"url":%q,"sha256":%q,"size":%d}]}`,
		assetID, url, sha, size, modelID, file, url, sha, size)
}

func TestParseValidation(t *testing.T) {
	ok := "https://example.com/a.bin"
	cases := []struct {
		name string
		doc  string
		bad  bool
	}{
		{"good", doc("a-1", "m_1", "m.Q4_K.gguf", ok, goodSHA, 5), false},
		{"loopback http", doc("a", "m", "m.gguf", "http://127.0.0.1:9/x", goodSHA, 5), false},
		{"localhost http", doc("a", "m", "m.gguf", "http://localhost:9/x", goodSHA, 5), false},
		{"asset id slash", doc("a/b", "m", "m.gguf", ok, goodSHA, 5), true},
		{"asset id backslash", doc(`a\b`, "m", "m.gguf", ok, goodSHA, 5), true},
		{"asset id dotdot", doc("..", "m", "m.gguf", ok, goodSHA, 5), true},
		{"asset id dot", doc(".", "m", "m.gguf", ok, goodSHA, 5), true},
		{"asset id empty", doc("", "m", "m.gguf", ok, goodSHA, 5), true},
		{"asset id leading dot", doc(".x", "m", "m.gguf", ok, goodSHA, 5), true},
		{"asset id space", doc("a b", "m", "m.gguf", ok, goodSHA, 5), true},
		{"model id slash", doc("a", "../m", "m.gguf", ok, goodSHA, 5), true},
		{"model id empty", doc("a", "", "m.gguf", ok, goodSHA, 5), true},
		{"model file slash", doc("a", "m", "sub/m.gguf", ok, goodSHA, 5), true},
		{"model file up", doc("a", "m", "../m.gguf", ok, goodSHA, 5), true},
		{"model file backslash", doc("a", "m", `..\m.gguf`, ok, goodSHA, 5), true},
		{"model file empty", doc("a", "m", "", ok, goodSHA, 5), true},
		{"model file leading dot", doc("a", "m", ".hidden", ok, goodSHA, 5), true},
		{"model file NUL", doc("a", "m", "m\\u0000.gguf", ok, goodSHA, 5), true},
		{"http remote", doc("a", "m", "m.gguf", "http://example.com/x", goodSHA, 5), true},
		{"ftp", doc("a", "m", "m.gguf", "ftp://example.com/x", goodSHA, 5), true},
		{"file url", doc("a", "m", "m.gguf", "file:///etc/passwd", goodSHA, 5), true},
		{"no host", doc("a", "m", "m.gguf", "https:///x", goodSHA, 5), true},
		{"no sha", doc("a", "m", "m.gguf", ok, "", 5), true},
		{"short sha", doc("a", "m", "m.gguf", ok, "abc", 5), true},
		{"non-hex sha", doc("a", "m", "m.gguf", ok, strings.Repeat("z", 64), 5), true},
		{"no size", doc("a", "m", "m.gguf", ok, goodSHA, 0), true},
		{"negative size", doc("a", "m", "m.gguf", ok, goodSHA, -1), true},
	}
	for _, c := range cases {
		_, err := Parse([]byte(c.doc))
		if (err != nil) != c.bad {
			t.Errorf("%s: err=%v, want bad=%v", c.name, err, c.bad)
		}
	}
}

func TestImportOnlyModelNeedsNoDownloadFields(t *testing.T) {
	d := `{"version":3,"llamacpp":{"tag":"t","assets":[]},"uv":[],"models":[{"id":"m","name":"m","file":"m.gguf"}]}`
	if _, err := Parse([]byte(d)); err != nil {
		t.Fatal(err)
	}
}

func TestEmbeddedPassesValidation(t *testing.T) {
	c, err := Parse(embedded)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Models) == 0 || len(c.llamaAll()) == 0 {
		t.Fatal("embedded catalog is empty")
	}
}

func TestBadOverrideIgnoredAndReported(t *testing.T) {
	d := t.TempDir()
	emb, _ := Load("")
	bad := doc("../../x", "m", "m.gguf", "https://example.com/a", goodSHA, 5)
	os.WriteFile(filepath.Join(d, "catalog.json"), []byte(bad), 0o644)
	c, note, err := LoadNote(d)
	if err != nil || c.Version != emb.Version {
		t.Fatalf("bad override was used: %v", err)
	}
	if !strings.Contains(note, "ignored") || !strings.Contains(note, "plain name") {
		t.Fatalf("note = %q", note)
	}
	good := doc("a", "m", "m.gguf", "https://example.com/a", goodSHA, 5)
	os.WriteFile(filepath.Join(d, "catalog.json"), []byte(good), 0o644)
	c, note, _ = LoadNote(d)
	if note != "" || c.Version != 999 {
		t.Fatalf("good override: note=%q v=%d", note, c.Version)
	}
}
