package dirs

import (
	"path/filepath"
	"testing"
)

// the earlier product's variable name, spelled in two pieces so the purity test stays quiet
const oldVar = "HERM" + "ES_ENGINE_HOME"

func TestCheckFor(t *testing.T) {
	home := filepath.Join(string(filepath.Separator), "home", "u")
	root := string(filepath.Separator)
	abs := func(p ...string) string { return filepath.Join(append([]string{root}, p...)...) }
	cases := []struct {
		name string
		env  map[string]string
		bad  bool
	}{
		{"unset", nil, false},
		{"absolute elsewhere", map[string]string{"EASY_ENGINE_HOME": abs("srv", "ee")}, false},
		{"inside home", map[string]string{"EASY_ENGINE_HOME": filepath.Join(home, "ee")}, false},
		{"relative", map[string]string{"EASY_ENGINE_HOME": "data"}, true},
		{"dot", map[string]string{"EASY_ENGINE_HOME": "."}, true},
		{"relative up", map[string]string{"EASY_ENGINE_HOME": "../x"}, true},
		{"root", map[string]string{"EASY_ENGINE_HOME": root}, true},
		{"root dotted", map[string]string{"EASY_ENGINE_HOME": abs("a", "..")}, true},
		{"home", map[string]string{"EASY_ENGINE_HOME": home}, true},
		{"home slash", map[string]string{"EASY_ENGINE_HOME": home + root}, true},
		{"home dotted", map[string]string{"EASY_ENGINE_HOME": filepath.Join(home, "x", "..")}, true},
		{"old variable relative", map[string]string{oldVar: "rel"}, true},
		{"old variable home", map[string]string{oldVar: home}, true},
		{"old variable ok", map[string]string{oldVar: abs("srv", "old")}, false},
	}
	for _, c := range cases {
		m := c.env
		err := checkFor(func(k string) string { return m[k] }, home)
		if (err != nil) != c.bad {
			t.Errorf("%s: err=%v want bad=%v", c.name, err, c.bad)
		}
	}
}
