// Package laws keeps the engine's laws as tests keep them: by reading the
// source. The first law is "Events, not clocks": logic is driven by events,
// and the only honest timers are three, each named in a comment where it
// stands: a watchdog for silence, a network timeout, a named moment. Any
// other sleep, ticker or timer is a clock standing in for an event, and
// this package finds it. It reads files and never runs them, and it imports
// only the standard library.
package laws

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Finding is one place that breaks a law.
type Finding struct {
	File string // root-relative, forward slashes
	Line int
	What string
}

// String gives file:line: what.
func (f Finding) String() string { return fmt.Sprintf("%s:%d: %s", f.File, f.Line, f.What) }

// honest are the three timers the clock law keeps, as a site names them.
var honest = []string{"watchdog for silence", "network timeout", "named moment"}

// clocks are the calls in package time that wait on the wall clock.
var clocks = map[string]bool{"Sleep": true, "Tick": true, "NewTicker": true, "After": true, "AfterFunc": true, "NewTimer": true}

// pageClocks are the calls in a page's script that wait on the wall clock.
var pageClocks = []string{"setInterval(", "setTimeout("}

// skipped says whether a directory is outside the law's reach.
func skipped(name string) bool {
	return name == ".git" || name == ".claude" || name == "dist" || name == "testdata"
}

// walk calls fn with the root-relative path and the text of every file under
// root with the given suffix, skipping the directories above. The root itself
// is never skipped, so a test can point at a testdata subtree.
func walk(root, suffix string, fn func(rel string, src []byte) error) error {
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != root && skipped(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, suffix) {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		return fn(filepath.ToSlash(rel), src)
	})
}

// named says whether line n (1-based), or one of the two above it, carries a
// // comment naming one of the honest timers.
func named(lines []string, n int) bool {
	for i := n - 1; i >= 0 && i >= n-3; i-- {
		if i >= len(lines) {
			continue
		}
		if c := strings.Index(lines[i], "//"); c >= 0 {
			for _, h := range honest {
				if strings.Contains(lines[i][c:], h) {
					return true
				}
			}
		}
	}
	return false
}

// timeNames are the local names a file gives the "time" import. A dot import
// or a blank import gives nothing to match (a dot import is ignored: its
// calls are bare names, which cannot be told from the file's own functions).
func timeNames(f *ast.File) map[string]bool {
	names := map[string]bool{}
	for _, im := range f.Imports {
		p, err := strconv.Unquote(im.Path.Value)
		if err != nil || p != "time" {
			continue
		}
		name := "time"
		if im.Name != nil {
			name = im.Name.Name
		}
		if name == "." || name == "_" {
			continue
		}
		names[name] = true
	}
	return names
}

// Clock finds every call to a clock in package time that no comment names as
// an honest timer. It reads every .go file under root (the nested module core/
// included), skipping .git, .claude, dist and testdata directories. A call
// counts only when the file imports "time" and the selector's left side is
// that import's local name, not a local variable of the same name. The
// finding reads "time.X with no honest timer named beside it", with the
// import's local name in place of "time" when the file renames it. A file
// that does not parse is an error naming the file.
func Clock(root string) ([]Finding, error) {
	var out []Finding
	err := walk(root, ".go", func(rel string, src []byte) error {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, rel, src, parser.ParseComments)
		if err != nil {
			return fmt.Errorf("laws: %s: %w", rel, err)
		}
		names := timeNames(f)
		if len(names) == 0 {
			return nil
		}
		lines := strings.Split(string(src), "\n")
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || !clocks[sel.Sel.Name] {
				return true
			}
			x, ok := sel.X.(*ast.Ident)
			if !ok || !names[x.Name] || x.Obj != nil {
				return true
			}
			line := fset.Position(call.Pos()).Line
			if !named(lines, line) {
				out = append(out, Finding{rel, line, x.Name + "." + sel.Sel.Name + " with no honest timer named beside it"})
			}
			return true
		})
		return nil
	})
	sortFindings(out)
	return out, err
}

// PageClock finds every setInterval( and setTimeout( in any .html file under
// root (same skips as Clock) that no // comment on the same line or the two
// above names as an honest timer.
func PageClock(root string) ([]Finding, error) {
	var out []Finding
	err := walk(root, ".html", func(rel string, src []byte) error {
		lines := strings.Split(string(src), "\n")
		for i, l := range lines {
			for _, c := range pageClocks {
				if strings.Contains(l, c) && !named(lines, i+1) {
					out = append(out, Finding{rel, i + 1, strings.TrimSuffix(c, "(") + " with no honest timer named beside it"})
				}
			}
		}
		return nil
	})
	sortFindings(out)
	return out, err
}

func sortFindings(fs []Finding) {
	sort.SliceStable(fs, func(i, j int) bool {
		if fs[i].File != fs[j].File {
			return fs[i].File < fs[j].File
		}
		return fs[i].Line < fs[j].Line
	})
}
