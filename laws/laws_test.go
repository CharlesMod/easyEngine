package laws

import (
	"flag"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

var writeUnconverted = flag.Bool("write-unconverted", false, "rewrite laws/unconverted.txt from the tree as it stands")

const header = `# The ratchet for the clock law ("Events, not clocks").
#
# These files still hold a sleep, ticker, timer or page timer that no comment
# names as an honest timer (watchdog for silence, network timeout, named
# moment). The list only shrinks: a file that is converted comes off it, and a
# new file never goes on it. When it is empty, the redesign is done and this
# file is deleted along with the ratchet in laws_test.go.
#
# One repo-relative path per line, or a path.Match pattern (cmd/*/main.go)
# for a file whose directory is being renamed; # comments and blank lines are
# allowed.
# Regenerate with:
#   go test ./laws -run TestTheTreeKeepsItsLaws -args -write-unconverted
`

func readList(t *testing.T) []string {
	t.Helper()
	b, err := os.ReadFile("unconverted.txt")
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, l := range strings.Split(string(b), "\n") {
		l = strings.TrimSpace(l)
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		out = append(out, l)
	}
	return out
}

func TestTheTreeKeepsItsLaws(t *testing.T) {
	root := ".."
	found, err := Clock(root)
	if err != nil {
		t.Fatal(err)
	}
	pages, err := PageClock(root)
	if err != nil {
		t.Fatal(err)
	}
	found = append(found, pages...)
	have := map[string][]Finding{}
	for _, f := range found {
		have[f.File] = append(have[f.File], f)
	}

	if *writeUnconverted {
		var files []string
		for f := range have {
			files = append(files, f)
		}
		sort.Strings(files)
		body := header + "\n" + strings.Join(files, "\n")
		if len(files) > 0 {
			body += "\n"
		}
		if err := os.WriteFile("unconverted.txt", []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}

	list := readList(t)
	listed := func(file string) bool {
		for _, p := range list {
			if ok, _ := path.Match(p, file); ok {
				return true
			}
		}
		return false
	}
	for _, p := range list {
		for _, born := range []string{"event/", "host/", "laws/"} {
			if strings.HasPrefix(p, born) {
				t.Errorf("%s is on the list, but %s is born under the law", p, born)
			}
		}
		if m, err := filepath.Glob(filepath.Join(root, filepath.FromSlash(p))); err != nil || len(m) == 0 {
			t.Errorf("%s is on the list but does not exist: remove it from the list", p)
			continue
		}
		n := 0
		for f, fs := range have {
			if ok, _ := path.Match(p, f); ok {
				n += len(fs)
			}
		}
		if n == 0 {
			t.Errorf("%s: converted: remove it from the list", p)
		}
	}
	for _, f := range found {
		if !listed(f.File) {
			t.Errorf("%s", f)
		}
	}
}

// lines gives "file:line" for each finding.
func lines(fs []Finding) []string {
	var out []string
	for _, f := range fs {
		out = append(out, f.File+":"+itoa(f.Line))
	}
	return out
}

func itoa(n int) string { return strconv.Itoa(n) }

func want(t *testing.T, got []Finding, err error, exp ...string) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	g := lines(got)
	if strings.Join(g, ",") != strings.Join(exp, ",") {
		t.Fatalf("findings %v, want %v", g, exp)
	}
}

func TestPlantedSleepInLoopIsCaught(t *testing.T) {
	got, err := Clock("testdata/planted/loop")
	want(t, got, err, "loop.go:7")
	if got[0].What != "time.Sleep with no honest timer named beside it" {
		t.Errorf("what: %q", got[0].What)
	}
	if s := got[0].String(); s != "loop.go:7: time.Sleep with no honest timer named beside it" {
		t.Errorf("string: %q", s)
	}
}

func TestPlantedRenamedImportIsCaught(t *testing.T) {
	got, err := Clock("testdata/planted/renamed")
	want(t, got, err, "r.go:6")
}

func TestHonestTimersPass(t *testing.T) {
	got, err := Clock("testdata/honest")
	want(t, got, err)
}

func TestCommentThreeLinesAboveDoesNotCount(t *testing.T) {
	got, err := Clock("testdata/planted/far")
	want(t, got, err, "far.go:9")
}

func TestNonTimeSleepIsNotReported(t *testing.T) {
	got, err := Clock("testdata/shadow")
	want(t, got, err)
}

func TestPageClock(t *testing.T) {
	got, err := PageClock("testdata/pages")
	want(t, got, err, "bad.html:2")
	if got[0].What != "setInterval with no honest timer named beside it" {
		t.Errorf("what: %q", got[0].What)
	}
}

func TestSkipsKeepRootButNotTestdata(t *testing.T) {
	got, err := Clock(".")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("testdata under the root must be skipped, got %v", got)
	}
}
