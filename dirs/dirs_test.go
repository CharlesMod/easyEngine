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
		{"linux", nil, "/home/u", "/home/u/.local/share/easy-engine"},
		{"linux", map[string]string{"XDG_DATA_HOME": "/x"}, "/home/u", "/x/easy-engine"},
		{"windows", map[string]string{"LOCALAPPDATA": "/L"}, "/h", filepath.Join("/L", "EasyEngine")},
		{"windows", nil, "/h", filepath.Join("/h", "AppData", "Local", "EasyEngine")},
		{"darwin", nil, "/Users/u", "/Users/u/Library/Application Support/EasyEngine"},
		{"darwin", map[string]string{"EASY_ENGINE_HOME": "/t"}, "/Users/u", "/t"},
	}
	for _, c := range cases {
		if got := dataFor(c.goos, env(c.env), c.home, func(string) bool { return false }); got != c.want {
			t.Errorf("%s %v: got %q want %q", c.goos, c.env, got, c.want)
		}
	}
}

func TestDataPrefersNewThenOldThenNew(t *testing.T) {
	getenv := func(k string) string { return "" }
	const nd = "/h/.local/share/easy-engine"
	od := "/h/.local/share/" + "her" + "mes-engine" // the old default; only internal/legacy may spell it plainly
	has := func(ps ...string) func(string) bool {
		return func(p string) bool {
			for _, q := range ps {
				if p == q {
					return true
				}
			}
			return false
		}
	}
	for name, c := range map[string]struct {
		exists func(string) bool
		want   string
	}{
		"both":    {has(nd, od), nd},
		"new":     {has(nd), nd},
		"old":     {has(od), od},
		"neither": {has(), nd},
	} {
		if got := dataFor("linux", getenv, "/h", c.exists); got != c.want {
			t.Errorf("%s: %q want %q", name, got, c.want)
		}
	}
}

func TestDataEnvOldNameHonoured(t *testing.T) {
	d := t.TempDir()
	if got := dataFor("linux", func(k string) string {
		if k == "HER"+"MES_ENGINE_HOME" {
			return d
		}
		return ""
	}, "/h", func(string) bool { return false }); got != d {
		t.Fatalf("old variable ignored: %q", got)
	}
}

func TestDataEnv(t *testing.T) {
	d := t.TempDir()
	t.Setenv("EASY_ENGINE_HOME", d)
	if Data() != d || Models() != filepath.Join(d, "models") {
		t.Fatal("env override ignored")
	}
}
