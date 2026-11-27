package legacy

import (
	"encoding/json"

	"errors"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func env(m map[string]string) func(string) string { return func(k string) string { return m[k] } }

func TestCSRFOK(t *testing.T) {
	cases := []struct {
		name string
		h    map[string]string
		want bool
	}{
		{"new", map[string]string{"X-Easy-Engine": "1"}, true},
		{"old", map[string]string{EngineHeader: "1"}, true},
		{"both", map[string]string{"X-Easy-Engine": "1", EngineHeader: "1"}, true},
		{"neither", nil, false},
		{"wrong value", map[string]string{"X-Easy-Engine": "0", EngineHeader: "yes"}, false},
	}
	for _, c := range cases {
		h := http.Header{}
		for k, v := range c.h {
			h.Set(k, v)
		}
		if got := CSRFOK(h, "X-Easy-Engine"); got != c.want {
			t.Errorf("%s: %v", c.name, got)
		}
	}
}

func TestMarkRequestSetsBothNames(t *testing.T) {
	h := http.Header{}
	MarkRequest(h, "X-Easy-Engine")
	if h.Get("X-Easy-Engine") != "1" || h.Get(EngineHeader) != "1" {
		t.Fatalf("%v", h)
	}
}

func TestSamplerHeaderPrecedenceAndStrip(t *testing.T) {
	h := http.Header{}
	h.Set(SamplerHeader, "structured")
	if got := HeaderOr(h, "X-Easy-Sampler", SamplerHeader); got != "structured" {
		t.Errorf("old only: %q", got)
	}
	h.Set("X-Easy-Sampler", "official")
	if got := HeaderOr(h, "X-Easy-Sampler", SamplerHeader); got != "official" {
		t.Errorf("new wins: %q", got)
	}
	h.Set(EngineHeader, "1")
	StripRequest(h)
	if h.Get(SamplerHeader) != "" || h.Get(EngineHeader) != "" {
		t.Errorf("old headers survive: %v", h)
	}
}

func TestSetAppliedBothNames(t *testing.T) {
	h := http.Header{}
	SetApplied(h, "X-Easy-Sampler-Applied", "rp;enforce")
	if h.Get("X-Easy-Sampler-Applied") != "rp;enforce" || h.Get(AppliedHeader) != "rp;enforce" {
		t.Fatalf("%v", h)
	}
}

func TestEnvPrecedence(t *testing.T) {
	if got := HomeOverride(env(map[string]string{"EASY_ENGINE_HOME": "/n", HomeEnv: "/o"}), "EASY_ENGINE_HOME"); got != "/n" {
		t.Errorf("new must win: %q", got)
	}
	if got := HomeOverride(env(map[string]string{HomeEnv: "/o"}), "EASY_ENGINE_HOME"); got != "/o" {
		t.Errorf("old honoured: %q", got)
	}
	if got := HomeOverride(env(nil), "EASY_ENGINE_HOME"); got != "" {
		t.Errorf("none: %q", got)
	}
}

func TestAgentVariableNames(t *testing.T) {
	if got := LLMURLNames("", "EASY_ENGINE_LLM_URL"); len(got) != 2 || got[0] != "EASY_ENGINE_LLM_URL" || got[1] != LLMURLEnv {
		t.Errorf("default: %v", got)
	}
	if got := LLMURLNames("X_URL", "EASY_ENGINE_LLM_URL"); len(got) != 1 || got[0] != "X_URL" {
		t.Errorf("configured: %v", got)
	}
	if got := EngineURLVars("EASY_ENGINE_URL", "http://h"); got[0] != "EASY_ENGINE_URL=http://h" || got[1] != EngineURLEnv+"=http://h" {
		t.Errorf("engine url: %v", got)
	}
	if LLMPrefix(LLMURLEnv) != strings.TrimSuffix(LLMURLEnv, "_URL") {
		t.Error("prefix")
	}
}

// fakeFS records what Migrate asks of the file system and deletes nothing.
type fakeFS struct {
	exist     map[string]bool
	live      bool
	renameErr error
	calls     []string
}

func (f *fakeFS) Exists(p string) bool { return f.exist[p] }
func (f *fakeFS) LockLive(dir string) bool {
	f.calls = append(f.calls, "lock "+dir)
	return f.live
}
func (f *fakeFS) MkdirAll(p string) error { f.calls = append(f.calls, "mkdir "+p); return nil }
func (f *fakeFS) Rename(a, b string) error {
	f.calls = append(f.calls, "rename "+a+" "+b)
	return f.renameErr
}
func (f *fakeFS) Symlink(target, l string) error {
	f.calls = append(f.calls, "symlink "+target+" "+l)
	return nil
}

func TestDataDirDecisionTable(t *testing.T) {
	const nd, od = "/h/.local/share/easy-engine", "/h/.local/share/old"
	cases := []struct {
		name      string
		goos      string
		env       map[string]string
		exist     []string
		live      bool
		renameErr error
		wantDir   string
		wantCalls []string
	}{
		{"new exists", "linux", nil, []string{nd, od}, false, nil, nd, nil},
		{"only new exists", "linux", nil, []string{nd}, false, nil, nd, nil},
		{"only old, idle: move + symlink", "linux", nil, []string{od}, false, nil, nd,
			[]string{"lock " + od, "mkdir /h/.local/share", "rename " + od + " " + nd, "symlink " + nd + " " + od}},
		{"only old, idle, windows: move, no symlink", "windows", nil, []string{od}, false, nil, nd,
			[]string{"lock " + od, "mkdir /h/.local/share", "rename " + od + " " + nd}},
		{"only old, lock live: use old", "linux", nil, []string{od}, true, nil, od, []string{"lock " + od}},
		{"rename fails: use old", "linux", nil, []string{od}, false, errors.New("EXDEV"), od,
			[]string{"lock " + od, "mkdir /h/.local/share", "rename " + od + " " + nd}},
		{"new env override: untouched", "linux", map[string]string{"EASY_ENGINE_HOME": "/t"}, []string{od}, false, nil, nd, nil},
		{"old env override: untouched", "linux", map[string]string{HomeEnv: "/t"}, []string{od}, false, nil, nd, nil},
		{"neither exists", "linux", nil, nil, false, nil, nd, nil},
	}
	for _, c := range cases {
		f := &fakeFS{exist: map[string]bool{}, live: c.live, renameErr: c.renameErr}
		for _, p := range c.exist {
			f.exist[p] = true
		}
		dir, _ := Migrate(c.goos, env(c.env), "EASY_ENGINE_HOME", nd, od, f)
		if dir != c.wantDir {
			t.Errorf("%s: dir %q want %q", c.name, dir, c.wantDir)
		}
		if strings.Join(f.calls, "|") != strings.Join(c.wantCalls, "|") {
			t.Errorf("%s: calls %v want %v", c.name, f.calls, c.wantCalls)
		}
	}
}

func TestDecideIsPure(t *testing.T) {
	f := &fakeFS{exist: map[string]bool{"/o": true}}
	if Decide(env(nil), "EASY_ENGINE_HOME", "/n", "/o", f) != MoveOld {
		t.Fatal("want MoveOld")
	}
	if len(f.calls) != 1 || f.calls[0] != "lock /o" {
		t.Fatalf("Decide did more than ask: %v", f.calls)
	}
}

func TestOldDefaultDir(t *testing.T) {
	if got := OldDefaultDir("linux", env(nil), "/h"); got != "/h/.local/share/"+dirLinux {
		t.Errorf("linux %q", got)
	}
	if got := OldDefaultDir("linux", env(map[string]string{"XDG_DATA_HOME": "/x"}), "/h"); got != "/x/"+dirLinux {
		t.Errorf("xdg %q", got)
	}
	if got := OldDefaultDir("darwin", env(nil), "/Users/u"); got != "/Users/u/Library/Application Support/"+dirOther {
		t.Errorf("darwin %q", got)
	}
	if got := OldDefaultDir("windows", env(map[string]string{"LOCALAPPDATA": "/L"}), "/h"); got != filepath.Join("/L", dirOther) {
		t.Errorf("windows %q", got)
	}
}

// The move itself, in a temp dir: files arrive, a symlink is left, nothing is deleted.
func TestMigrateMovesTheDirForReal(t *testing.T) {
	root := t.TempDir()
	od := filepath.Join(root, "old")
	nd := filepath.Join(root, "share", "new")
	if err := os.MkdirAll(filepath.Join(od, "models"), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{"state.json": "{}", "engine.lock": "not json", filepath.Join("models", "m.gguf"): "weights"}
	for p, c := range files {
		if err := os.WriteFile(filepath.Join(od, p), []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	dir, note := Migrate(runtime.GOOS, env(nil), "EASY_ENGINE_HOME", nd, od, OSFS{})
	if dir != nd || note == "" {
		t.Fatalf("dir %q note %q", dir, note)
	}
	for p, c := range files {
		b, err := os.ReadFile(filepath.Join(nd, p))
		if err != nil || string(b) != c {
			t.Errorf("%s did not arrive: %v %q", p, err, b)
		}
	}
	if runtime.GOOS != "windows" {
		if l, err := os.Readlink(od); err != nil || l != nd {
			t.Errorf("symlink at the old path: %q %v", l, err)
		}
		if b, err := os.ReadFile(filepath.Join(od, "engine.lock")); err != nil || string(b) != "not json" {
			t.Errorf("engine.lock not reachable through the old path: %v", err)
		}
	}
	// a second run is a no-op: the new dir exists
	if dir2, note2 := Migrate(runtime.GOOS, env(nil), "EASY_ENGINE_HOME", nd, od, OSFS{}); dir2 != nd || note2 != "" {
		t.Errorf("second run: %q %q", dir2, note2)
	}
}

func TestOSFSLockLive(t *testing.T) {
	dir := t.TempDir()
	if (OSFS{}).LockLive(dir) {
		t.Error("no lock file counted as live")
	}
	start, _ := lifecycle.PidStart(os.Getpid())
	live, _ := json.Marshal(lifecycle.LockData{Pid: os.Getpid(), Start: start})
	os.WriteFile(filepath.Join(dir, "engine.lock"), live, 0o644)
	if !(OSFS{}).LockLive(dir) {
		t.Error("own pid not live")
	}
	if start != "" {
		stale, _ := json.Marshal(lifecycle.LockData{Pid: os.Getpid(), Start: "some other start time"})
		os.WriteFile(filepath.Join(dir, "engine.lock"), stale, 0o644)
		if (OSFS{}).LockLive(dir) {
			t.Error("reused pid counted as live")
		}
	}
}

func TestMigrateLeavesOldDirWhenRenameCannotHappen(t *testing.T) {
	root := t.TempDir()
	od := filepath.Join(root, "old")
	nd := filepath.Join(root, "blocker", "new") // "blocker" is a file: MkdirAll of the parent fails
	os.MkdirAll(od, 0o755)
	os.WriteFile(filepath.Join(od, "state.json"), []byte("{}"), 0o644)
	os.WriteFile(filepath.Join(root, "blocker"), []byte("x"), 0o644)
	dir, note := Migrate(runtime.GOOS, env(nil), "EASY_ENGINE_HOME", nd, od, OSFS{})
	if dir != od || !strings.Contains(note, "using it in place") {
		t.Fatalf("dir %q note %q", dir, note)
	}
	if _, err := os.Stat(filepath.Join(od, "state.json")); err != nil {
		t.Errorf("old dir damaged: %v", err)
	}
}
