package catalog

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// docCodename is doc's catalog with a codename and opt_in on its one model.
func docCodename(codename string, optIn bool) string {
	return fmt.Sprintf(`{"version":999,"llamacpp":{"tag":"t","assets":[{"id":"a","os":"linux","arch":"amd64","url":"https://example.com/a","sha256":%q,"size":5}]},
"uv":[],"models":[{"id":"m","name":"m","file":"m.gguf","url":"https://example.com/m","sha256":%q,"size":5,"codename":%q,"opt_in":%v}]}`,
		goodSHA, goodSHA, codename, optIn)
}

func TestCodenameValidation(t *testing.T) {
	cases := []struct {
		name string
		v    string
		bad  bool
	}{
		{"plain", "Campfire", false},
		{"digit and space", "Bonfire 2", false},
		{"hyphen and apostrophes", "Jack-o'-Lamp", false},
		{"dot", "v1.5", false},
		{"empty", "", false},
		{"24 runes", strings.Repeat("a", 24), false},
		{"25 runes", strings.Repeat("a", 25), true},
		{"leading space", " Campfire", true},
		{"trailing space", "Campfire ", true},
		{"markup", "<b>x</b>", true},
		{"newline", "a\nb", true},
		{"slash", "Camp/fire", true},
		{"emoji", "Camp\U0001F525", true},
	}
	for _, c := range cases {
		_, err := Parse([]byte(docCodename(c.v, false)))
		if (err != nil) != c.bad {
			t.Errorf("%s: codename %q err=%v, want bad=%v", c.name, c.v, err, c.bad)
		}
	}
}

func TestModelLabel(t *testing.T) {
	cases := []struct {
		m    Model
		want string
	}{
		{Model{ID: "id", Name: "Gemma 4 26B RP", Codename: "Campfire"}, "Campfire (Gemma 4 26B RP)"},
		{Model{ID: "id", Name: "Gemma 4 26B RP"}, "Gemma 4 26B RP"},
		{Model{ID: "id", Codename: "Campfire"}, "Campfire"},
		{Model{ID: "id"}, "id"},
	}
	for _, c := range cases {
		if got := c.m.Label(); got != c.want {
			t.Errorf("%+v: Label %q want %q", c.m, got, c.want)
		}
	}
}

func TestOldCatalogHasNoCodenameOrOptIn(t *testing.T) {
	c, err := Parse([]byte(doc("a", "m", "m.gguf", "https://example.com/m", goodSHA, 5)))
	if err != nil {
		t.Fatal(err)
	}
	if m := c.Models[0]; m.Codename != "" || m.OptIn {
		t.Fatalf("old catalog: %+v", m)
	}
	c, err = Parse([]byte(docCodename("Bonfire", true)))
	if err != nil {
		t.Fatal(err)
	}
	if m := c.Models[0]; m.Codename != "Bonfire" || !m.OptIn {
		t.Fatalf("new catalog: %+v", m)
	}
}

func TestEmbeddedCodenames(t *testing.T) {
	c, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	// exactly these code names on exactly these ids (Bonfire joins with its entry)
	want := map[string][2]string{"gemma4-rp-maxxed-q3kxl": {"Campfire", "Campfire-26B"}, "gemma4-e4b-rp-q40": {"Lantern", "Lantern-4B"},
		"gemma4-e2b-q4km": {"Flashlight", "Flashlight-2B"}}
	got := map[string][2]string{}
	for _, m := range c.Models {
		if m.Codename != "" || m.CodenameFull != "" {
			got[m.ID] = [2]string{m.Codename, m.CodenameFull}
		}
		if m.OptIn {
			t.Errorf("embedded model %s is opt_in", m.ID)
		}
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("codenames %v want %v", got, want)
	}
	if m, ok := c.Model("gemma4-e2b-q4km"); !ok || m.Label() != "Flashlight-2B (Gemma 4 E2B (Q4_K_M))" || !m.Fallback {
		t.Errorf("E2B: %q %+v", m.Label(), m)
	}
}

func TestCodenameFullValidation(t *testing.T) {
	doc := func(code, full string) string {
		return fmt.Sprintf(`{"version":999,"llamacpp":{"tag":"t","assets":[]},"uv":[],"models":[{"id":"m","name":"m","file":"m.gguf","url":"https://example.com/m","sha256":%q,"size":5,"codename":%q,"codename_full":%q}]}`, goodSHA, code, full)
	}
	cases := []struct {
		code, full string
		bad        bool
	}{
		{"Campfire", "Campfire-26B", false},
		{"Campfire", "", false},
		{"", "", false},
		{"", "Campfire-26B", true},           // a full name with no code name
		{"Campfire", "Bonfire-27B", true},    // does not begin with the code name
		{"Campfire", "Campfire <26B>", true}, // not plain text
		{"Campfire", "Campfire-" + strings.Repeat("x", 16), true},
	}
	for _, c := range cases {
		if _, err := Parse([]byte(doc(c.code, c.full))); (err != nil) != c.bad {
			t.Errorf("codename %q codename_full %q: err=%v, want bad=%v", c.code, c.full, err, c.bad)
		}
	}
	if l := (Model{Name: "Gemma 4 26B RP", Codename: "Campfire", CodenameFull: "Campfire-26B"}).Label(); l != "Campfire-26B (Gemma 4 26B RP)" {
		t.Errorf("Label %q", l)
	}
}

func TestOptInJSONRoundTrip(t *testing.T) {
	on, err := json.Marshal(Model{ID: "m", Codename: "Bonfire", OptIn: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(on), `"opt_in":true`) || !strings.Contains(string(on), `"codename":"Bonfire"`) {
		t.Errorf("opt-in model JSON: %s", on)
	}
	var back Model
	if err := json.Unmarshal(on, &back); err != nil || !back.OptIn || back.Codename != "Bonfire" {
		t.Errorf("round trip: %+v %v", back, err)
	}
	off, err := json.Marshal(Model{ID: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(off), "opt_in") || strings.Contains(string(off), "codename") {
		t.Errorf("default model JSON writes the new keys: %s", off)
	}
}
