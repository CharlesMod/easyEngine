package autostart

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDesktopExec(t *testing.T) {
	cases := []struct {
		name string
		argv []string
		want string
	}{
		{"plain", []string{"/usr/bin/ee"}, `/usr/bin/ee`},
		{"spaces", []string{"/opt/My App/ee"}, `"/opt/My App/ee"`},
		{"percent", []string{"/bin/ee", "100%"}, `/bin/ee 100%%`},
		{"percent quoted", []string{"/bin/ee", "a%b c"}, `/bin/ee "a%%b c"`},
		{"double quote", []string{"/bin/ee", `a"b`}, `/bin/ee "a\\"b"`},
		{"dollar", []string{"/bin/ee", "$HOME"}, `/bin/ee "\\$HOME"`},
		{"backslash", []string{"/bin/ee", `a\b`}, `/bin/ee "a\\\\b"`},
		{"backtick", []string{"/bin/ee", "`x`"}, "/bin/ee \"\\\\`x\\\\`\""},
		{"single quote", []string{"/bin/ee", "it's"}, `/bin/ee "it's"`},
		{"empty arg", []string{"/bin/ee", "", "x"}, `/bin/ee "" x`},
		{"newline", []string{"/bin/ee", "a\nb"}, `/bin/ee "a\nb"`},
		{"separator and glob", []string{"/bin/ee", "a;b*"}, `/bin/ee "a;b*"`},
	}
	for _, c := range cases {
		if got := desktopExec(c.argv); got != c.want {
			t.Errorf("%s: got %s want %s", c.name, got, c.want)
		}
	}
}

func TestDesktopEntry(t *testing.T) {
	got := desktopEntry("easy-engine", "/opt/My App/ee", []string{"--tray"})
	want := "[Desktop Entry]\n" +
		"Type=Application\n" +
		"Name=easy-engine\n" +
		"Exec=\"/opt/My App/ee\" --tray\n" +
		"X-GNOME-Autostart-enabled=true\n" +
		"Terminal=false\n"
	if got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}

func TestDesktopEnabled(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want bool
	}{
		{"plain", "[Desktop Entry]\nType=Application\n", true},
		{"enabled true", "[Desktop Entry]\nX-GNOME-Autostart-enabled=true\n", true},
		{"gnome false", "[Desktop Entry]\nX-GNOME-Autostart-enabled=false\n", false},
		{"gnome False", "[Desktop Entry]\nX-GNOME-Autostart-enabled = False\n", false},
		{"hidden", "[Desktop Entry]\nHidden=true\n", false},
		{"hidden false", "[Desktop Entry]\nHidden=false\n", true},
		{"other group ignored", "[Desktop Entry]\nType=Application\n[Desktop Action x]\nHidden=true\n", true},
		{"empty", "", true},
	}
	for _, c := range cases {
		if got := desktopEnabled(c.in); got != c.want {
			t.Errorf("%s: got %v want %v", c.name, got, c.want)
		}
	}
}

func TestLaunchAgentPlist(t *testing.T) {
	got := launchAgentPlist("com.easyengine.x", "/Apps/ee", []string{"a&b<c>\"d'"})
	want := `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>com.easyengine.x</string>
	<key>ProgramArguments</key>
	<array>
		<string>/Apps/ee</string>
		<string>a&amp;b&lt;c&gt;&quot;d&apos;</string>
	</array>
	<key>RunAtLoad</key>
	<true/>
	<key>ProcessType</key>
	<string>Interactive</string>
</dict>
</plist>
`
	if got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}

func TestPlistLabel(t *testing.T) {
	cases := map[string]string{
		"easy-engine":       "com.easyengine.easy-engine",
		"org.example.thing": "org.example.thing",
		"a.b":               "a.b",
	}
	for in, want := range cases {
		if got := plistLabel(in); got != want {
			t.Errorf("plistLabel(%q) = %q want %q", in, got, want)
		}
	}
}

func TestWindowsCommandLine(t *testing.T) {
	cases := []struct {
		name string
		exe  string
		args []string
		want string
	}{
		{"exe only", `C:\Program Files\ee.exe`, nil, `"C:\Program Files\ee.exe"`},
		{"plain args", `C:\ee.exe`, []string{"--x", "a b"}, `"C:\ee.exe" --x "a b"`},
		{"embedded quotes", `C:\ee.exe`, []string{`say "hi"`}, `"C:\ee.exe" "say \"hi\""`},
		{"trailing backslash", `C:\ee.exe`, []string{`C:\dir with space\`}, `"C:\ee.exe" "C:\dir with space\\"`},
		{"empty arg", `C:\ee.exe`, []string{""}, `"C:\ee.exe" ""`},
		{"backslash plain", `C:\ee.exe`, []string{`a\b`}, `"C:\ee.exe" a\b`},
		{"backslash before quote", `C:\ee.exe`, []string{`a\"b`}, `"C:\ee.exe" "a\\\"b"`},
		{"tab", `C:\ee.exe`, []string{"a\tb"}, "\"C:\\ee.exe\" \"a\tb\""},
	}
	for _, c := range cases {
		if got := windowsCommandLine(c.exe, c.args); got != c.want {
			t.Errorf("%s: got %s want %s", c.name, got, c.want)
		}
	}
}

func TestValidName(t *testing.T) {
	for _, ok := range []string{"easy-engine", "a", "A_b.c-1"} {
		if err := validName(ok); err != nil {
			t.Errorf("validName(%q): %v", ok, err)
		}
	}
	for _, bad := range []string{"", "a b", "a/b", "a\\b", "é", "a\n", "a:b"} {
		if err := validName(bad); err == nil {
			t.Errorf("validName(%q): want error", bad)
		}
	}
}

func TestConfigHome(t *testing.T) {
	if got := configHome("/x/cfg", "/home/u"); got != "/x/cfg" {
		t.Error(got)
	}
	for _, rel := range []string{"", "rel/cfg"} {
		if got := configHome(rel, "/home/u"); got != filepath.Join("/home/u", ".config") {
			t.Errorf("%q: %s", rel, got)
		}
	}
	if got := desktopPath("/c", "n"); got != filepath.Join("/c", "autostart", "n.desktop") {
		t.Error(got)
	}
	if got := plistPath("/h", "l"); got != filepath.Join("/h", "Library", "LaunchAgents", "l.plist") {
		t.Error(got)
	}
}

// mapKey is a fake Run key.
type mapKey map[string]string

func (m mapKey) get(n string) (string, bool, error) { v, ok := m[n]; return v, ok, nil }
func (m mapKey) set(n, v string) error              { m[n] = v; return nil }
func (m mapKey) del(n string) error                 { delete(m, n); return nil }
func (m mapKey) where(n string) string              { return `HKCU\` + runSubkey + `\` + n }

// places returns one place per OS rule, all under temp dirs or fakes.
func places(t *testing.T) map[string]*Place {
	return map[string]*Place{
		"native":  Root(t.TempDir()),
		"desktop": newPlace(kindDesktop, t.TempDir()),
		"plist":   newPlace(kindPlist, t.TempDir()),
		"runfile": newPlace(kindRun, t.TempDir()),
		"runmap":  {kind: kindRun, key: mapKey{}},
	}
}

func TestRoundTrip(t *testing.T) {
	for label, p := range places(t) {
		const name = "easy-engine"
		if on, err := p.Enabled(name); err != nil || on {
			t.Fatalf("%s: before: %v %v", label, on, err)
		}
		if err := p.Disable(name); err != nil {
			t.Fatalf("%s: disable missing: %v", label, err)
		}
		if err := p.Enable(name, "/opt/My App/ee", []string{"--tray", "a b"}); err != nil {
			t.Fatalf("%s: enable: %v", label, err)
		}
		if err := p.Enable(name, "/opt/My App/ee", nil); err != nil {
			t.Fatalf("%s: re-enable: %v", label, err)
		}
		if on, err := p.Enabled(name); err != nil || !on {
			t.Fatalf("%s: after enable: %v %v", label, on, err)
		}
		if err := p.Disable(name); err != nil {
			t.Fatalf("%s: disable: %v", label, err)
		}
		if on, err := p.Enabled(name); err != nil || on {
			t.Fatalf("%s: after disable: %v %v", label, on, err)
		}
		if err := p.Disable(name); err != nil {
			t.Fatalf("%s: disable again: %v", label, err)
		}
	}
}

func TestEnabledHonoursDisabledDesktopFile(t *testing.T) {
	p := newPlace(kindDesktop, t.TempDir())
	if err := p.Enable("x", "/bin/x", nil); err != nil {
		t.Fatal(err)
	}
	path := p.Path("x")
	if err := os.WriteFile(path, []byte("[Desktop Entry]\nHidden=true\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if on, err := p.Enabled("x"); err != nil || on {
		t.Fatalf("hidden entry: %v %v", on, err)
	}
}

func TestFileContentsAndModes(t *testing.T) {
	dir := t.TempDir()
	for _, k := range []kind{kindDesktop, kindPlist, kindRun} {
		p := newPlace(k, filepath.Join(dir, string(rune('a'+int(k)))))
		if err := p.Enable("easy-engine", "/bin/ee", []string{"--x"}); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(p.Path("easy-engine"))
		if err != nil {
			t.Fatal(err)
		}
		var want string
		switch k {
		case kindDesktop:
			want = desktopEntry("easy-engine", "/bin/ee", []string{"--x"})
		case kindPlist:
			want = launchAgentPlist("com.easyengine.easy-engine", "/bin/ee", []string{"--x"})
		default:
			want = windowsCommandLine("/bin/ee", []string{"--x"})
		}
		if string(b) != want {
			t.Errorf("kind %d: got %q want %q", k, b, want)
		}
		st, err := os.Stat(p.Path("easy-engine"))
		if err != nil {
			t.Fatal(err)
		}
		if st.Mode().Perm() != 0o644 && filepath.Separator == '/' {
			t.Errorf("kind %d: mode %v", k, st.Mode().Perm())
		}
		// No temp files left behind.
		ents, _ := os.ReadDir(filepath.Dir(p.Path("easy-engine")))
		if len(ents) != 1 {
			t.Errorf("kind %d: %d entries in dir", k, len(ents))
		}
	}
}

func TestRootPathsStayUnderRoot(t *testing.T) {
	dir := t.TempDir()
	for _, k := range []kind{kindDesktop, kindPlist, kindRun} {
		p := newPlace(k, dir)
		path := p.Path("easy-engine")
		rel, err := filepath.Rel(dir, path)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			t.Errorf("kind %d: %s escapes %s", k, path, dir)
		}
	}
	rel, _ := filepath.Rel(dir, Root(dir).Path("easy-engine"))
	if strings.HasPrefix(rel, "..") {
		t.Errorf("native root path escapes: %s", rel)
	}
	if got := newPlace(kindDesktop, dir).Path("n"); got != filepath.Join(dir, "autostart", "n.desktop") {
		t.Error(got)
	}
	if got := newPlace(kindPlist, dir).Path("n"); got != filepath.Join(dir, "Library", "LaunchAgents", "com.easyengine.n.plist") {
		t.Error(got)
	}
	if got := newPlace(kindRun, dir).Path("n"); got != filepath.Join(dir, "Run", "n.cmd-line") {
		t.Error(got)
	}
	if got := Root(dir).Path("../evil"); got != "" {
		t.Errorf("invalid name path: %q", got)
	}
}

func TestInvalidInputs(t *testing.T) {
	p := Root(t.TempDir())
	if err := p.Enable("bad name", "/bin/x", nil); err == nil {
		t.Error("bad name: want error")
	}
	if err := p.Enable("ok", "", nil); err == nil {
		t.Error("empty exe: want error")
	}
	if _, err := p.Enabled(""); err == nil {
		t.Error("empty name: want error")
	}
	if err := p.Disable("a/b"); err == nil {
		t.Error("slash name: want error")
	}
}

func TestForbiddenWords(t *testing.T) {
	words := []string{"spa" + "wn", "col" + "on", "pa" + "wn", "her" + "mes"}
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		b, err := os.ReadFile(e.Name())
		if err != nil {
			t.Fatal(err)
		}
		low := strings.ToLower(string(b))
		for _, w := range words {
			if strings.Contains(low, w) {
				t.Errorf("%s contains forbidden %q", e.Name(), w)
			}
		}
	}
}
