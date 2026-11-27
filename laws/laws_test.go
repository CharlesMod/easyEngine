package laws

import (
	"strconv"
	"strings"
	"testing"
)

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
	// The tree is converted (the ratchet list is gone): every file is under the law.
	for _, f := range found {
		t.Errorf("%s", f)
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
