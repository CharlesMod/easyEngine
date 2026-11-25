package dirs

import (
	"path/filepath"
	"testing"
)

func TestDataFor(t *testing.T) {
	env := func(m map[string]string) func(string) string { return func(k string) string { return m[k] } }
	cases := []struct {
		goos string
		env  map[string]string
		home string
		want string
	}{
		{"linux", nil, "/home/u", "/home/u/.local/share/hermes-engine"},
		{"linux", map[string]string{"XDG_DATA_HOME": "/x"}, "/home/u", "/x/hermes-engine"},
		{"windows", map[string]string{"LOCALAPPDATA": "/L"}, "/h", filepath.Join("/L", "HermesEngine")},
		{"windows", nil, "/h", filepath.Join("/h", "AppData", "Local", "HermesEngine")},
		{"darwin", nil, "/Users/u", "/Users/u/Library/Application Support/HermesEngine"},
		{"darwin", map[string]string{"HERMES_ENGINE_HOME": "/t"}, "/Users/u", "/t"},
	}
	for _, c := range cases {
		if got := dataFor(c.goos, env(c.env), c.home); got != c.want {
			t.Errorf("%s %v: got %q want %q", c.goos, c.env, got, c.want)
		}
	}
}

func TestDataEnv(t *testing.T) {
	d := t.TempDir()
	t.Setenv("HERMES_ENGINE_HOME", d)
	if Data() != d || Models() != filepath.Join(d, "models") {
		t.Fatal("env override ignored")
	}
}
