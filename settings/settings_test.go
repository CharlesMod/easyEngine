package settings

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultsRoundTripWhole(t *testing.T) {
	p := filepath.Join(t.TempDir(), FileName)
	if err := Write(p, Defaults()); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	for _, k := range []string{`"mode"`, `"external"`, `"api_key"`, `"linger_s"`, `"kv_type"`, `"sampler_policy"`, `"listen"`, `"host"`, `"port"`, `"open"`,
		`"yield"`, `"to_games"`, `"except": []`, `"ignore": []`, `"floor_mib": 128`, `"min_mib"`, `"grace_s": 15`, `"steady_s": 60`, `"strict"`} {
		if !strings.Contains(string(b), k) {
			t.Errorf("missing %s in\n%s", k, b)
		}
	}
	f, err := Read(p)
	if err != nil || !f.Equal(Defaults()) {
		t.Fatalf("%+v %v", f, err)
	}
	if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", fi.Mode())
	}
}

func TestBadValues(t *testing.T) {
	for _, c := range []struct{ text, want string }{
		{`{"mode":"cloud"}`, "mode must be"},
		{"{\n\"kv_type\":\"x\"}", "line 2: unknown kv_type"},
		{`{"sampler_policy":"loud"}`, "sampler_policy"},
		{`{"linger_s":-1}`, "linger_s"},
		{"{\"listen\":{\n\"port\":70000}}", "line 2: port 70000"},
		{`{"listen":{"port":0}}`, "port 0"},
		{"{\n\"linger_s\": \"soon\"}", "line 2"},
		{"{\"mode\":", "line 1"},
	} {
		_, err := Parse([]byte(c.text))
		var se *Error
		if !errors.As(err, &se) || !strings.Contains(se.Error(), c.want) {
			t.Errorf("%q: %v, want %q", c.text, err, c.want)
		}
	}
}

func TestMissingKeysTakeDefaults(t *testing.T) {
	f, err := Parse([]byte(`{"linger_s": 4, "listen": {"port": 5000}}`))
	if err != nil || f.LingerS != 4 || f.Listen.Port != 5000 || f.Listen.Host != "127.0.0.1" || f.Yield.GraceS != 15 || f.KVType != "q4_0" {
		t.Fatalf("%+v %v", f, err)
	}
}

func TestMigrateOnce(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "state.json"), []byte(`{"mode":"local","linger_s":3,"kv_type":"f16","model":"x"}`), 0o600)
	moved, err := Migrate(dir)
	if !moved || err != nil {
		t.Fatal(moved, err)
	}
	f, err := Read(Path(dir))
	if err != nil || f.LingerS != 3 || f.KVType != "f16" {
		t.Fatalf("%+v %v", f, err)
	}
	b, _ := os.ReadFile(filepath.Join(dir, "state.json"))
	if strings.Contains(string(b), "linger_s") || !strings.Contains(string(b), `"model"`) {
		t.Fatalf("state.json: %s", b)
	}
	if moved, _ := Migrate(dir); moved {
		t.Fatal("moved twice")
	}
}

func TestMigrateKeepsAnExistingSettingsFile(t *testing.T) {
	dir := t.TempDir()
	g := Defaults()
	g.LingerS = 77
	Write(Path(dir), g)
	os.WriteFile(filepath.Join(dir, "state.json"), []byte(`{"linger_s":3,"model":"x"}`), 0o600)
	Migrate(dir)
	if f, _ := Read(Path(dir)); f.LingerS != 77 {
		t.Fatalf("settings.json was replaced: %+v", f)
	}
}

func TestLoadBadFileIsLeftAlone(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(Path(dir), []byte("{oops"), 0o600)
	f, err := Load(dir)
	var se *Error
	if !errors.As(err, &se) || !f.Equal(Defaults()) {
		t.Fatalf("%+v %v", f, err)
	}
	if b, _ := os.ReadFile(Path(dir)); string(b) != "{oops" {
		t.Fatalf("overwritten: %s", b)
	}
}
