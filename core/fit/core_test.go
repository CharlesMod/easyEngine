package fit

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The core is the sizing library another program imports: it must stay pure Go 1.23 with no
// requirement, and its non-test code must never touch the machine, the clock or a log. This
// test reads the core's own source and go.mod and fails on any import outside the list below
// or any call that reads the clock or arms a timer. Test files are not checked.
var coreImports = map[string]bool{
	"errors": true, "fmt": true, "math": true, "sort": true, "strconv": true, "strings": true,
	"io": true, "bufio": true, "bytes": true, "encoding/binary": true, "slices": true, "maps": true,
	"time": true, // as a type only: the calls below are refused
}

// corePath is the core's module path; its own packages may import each other.
const corePath = "github.com/CharlesMod/easyEngine/core"

var clockCalls = map[string]bool{
	"Now": true, "Since": true, "Until": true, "Sleep": true, "After": true, "AfterFunc": true,
	"Tick": true, "NewTicker": true, "NewTimer": true,
}

func TestCoreStaysPure(t *testing.T) {
	root := ".." // the core module's root: this package is core/fit
	mod, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	var goLine string
	for _, l := range strings.Split(string(mod), "\n") {
		f := strings.Fields(l)
		if len(f) == 0 {
			continue
		}
		switch f[0] {
		case "module":
			if len(f) < 2 || f[1] != corePath {
				t.Errorf("core/go.mod names module %v, want %s", f[1:], corePath)
			}
		case "go":
			goLine = strings.Join(f[1:], " ")
		case "require", "replace", "toolchain":
			t.Errorf("core/go.mod has a %s line: the core needs the standard library alone", f[0])
		}
	}
	if goLine != "1.23" {
		t.Errorf("core/go.mod says go %q; it must stay go 1.23 so callers on Go 1.23 can build it", goLine)
	}

	files := 0
	fset := token.NewFileSet()
	err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return err
		}
		f, err := parser.ParseFile(fset, p, nil, parser.ImportsOnly|parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		files++
		timeName := ""
		for _, im := range f.Imports {
			path, _ := strconv.Unquote(im.Path.Value)
			if !coreImports[path] && !strings.HasPrefix(path, corePath+"/") {
				t.Errorf("%s imports %q: the core may import only pure standard-library packages", p, path)
			}
			// A dot import would let a bare Now() or Sleep() past the selector check below.
			if im.Name != nil && im.Name.Name == "." {
				t.Errorf("%s dot-imports %q: the core names every package it calls", p, path)
			}
			if path == "time" {
				timeName = "time"
				if im.Name != nil {
					timeName = im.Name.Name
				}
			}
		}
		if timeName == "" {
			return nil
		}
		full, err := parser.ParseFile(fset, p, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		ast.Inspect(full, func(n ast.Node) bool {
			if s, ok := n.(*ast.SelectorExpr); ok {
				if x, ok := s.X.(*ast.Ident); ok && x.Name == timeName && clockCalls[s.Sel.Name] {
					t.Errorf("%s uses time.%s: the core never reads the clock or arms a timer; the caller passes a stamp", fset.Position(s.Pos()), s.Sel.Name)
				}
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if files == 0 {
		t.Fatal("read no source files")
	}
}
