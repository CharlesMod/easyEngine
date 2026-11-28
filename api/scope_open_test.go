package api_test

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"net/http/httptest"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
)

// routerNames is every control route name the router dispatches, read from its own source (the
// engine, customRoute and pairRoute functions): each case label, each name == "x" and each
// strings.HasPrefix(name, "x/"), the last as a prefix. A route added later shows up here.
func routerNames(t *testing.T) (names, prefixes []string) {
	t.Helper()
	fs := token.NewFileSet()
	for _, f := range []string{"api.go", "custom.go", "pair.go"} {
		file, err := parser.ParseFile(fs, f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range file.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || (fn.Name.Name != "engine" && fn.Name.Name != "customRoute" && fn.Name.Name != "pairRoute") {
				continue
			}
			lit := func(e ast.Expr) (string, bool) {
				b, ok := e.(*ast.BasicLit)
				if !ok || b.Kind != token.STRING {
					return "", false
				}
				s, err := strconv.Unquote(b.Value)
				return s, err == nil
			}
			isName := func(e ast.Expr) bool { id, ok := e.(*ast.Ident); return ok && id.Name == "name" }
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.SwitchStmt:
					if !isName(x.Tag) {
						return true
					}
					for _, st := range x.Body.List {
						for _, e := range st.(*ast.CaseClause).List {
							if s, ok := lit(e); ok {
								names = append(names, s)
							}
						}
					}
				case *ast.BinaryExpr:
					if x.Op == token.EQL && isName(x.X) {
						if s, ok := lit(x.Y); ok {
							names = append(names, s)
						}
					}
				case *ast.CallExpr:
					if sel, ok := x.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "HasPrefix" && len(x.Args) == 2 && isName(x.Args[0]) {
						if s, ok := lit(x.Args[1]); ok {
							prefixes = append(prefixes, s)
						}
					}
				}
				return true
			})
		}
	}
	sort.Strings(names)
	if len(names) < 20 || len(prefixes) < 3 {
		t.Fatalf("read too few routes from the router's source: %v %v", names, prefixes)
	}
	return names, prefixes
}

// The table the scope tests walk holds every route the router dispatches: a route added to the
// router without a row here fails, so it is judged (and, being off the allow list, refused to
// scoped callers by default).
func TestScopeRoutesCoverTheRouter(t *testing.T) {
	names, prefixes := routerNames(t)
	have := map[string]bool{}
	for _, r := range scopeRoutes {
		have[r.path] = true
	}
	for _, n := range names {
		if !have[n] {
			t.Errorf("route %q is dispatched by the router but not in scopeRoutes", n)
		}
	}
	for _, p := range prefixes {
		found := false
		for _, r := range scopeRoutes {
			found = found || strings.HasPrefix(r.path, p)
		}
		if !found {
			t.Errorf("routes under %q are dispatched by the router but none is in scopeRoutes", p)
		}
	}
}

// openEnv is an engine on an --open listener (no tokens): a peer with a forwarding header is
// another machine; the loopback peer without one is this machine.
func openEnv(t *testing.T) (*httptest.Server, func() map[string]any) {
	t.Helper()
	_, s, _ := readyEnv(t)
	srv := httptest.NewServer((&api.API{S: s, AnyHost: true, Open: true}).Handler())
	t.Cleanup(srv.Close)
	paired := func() map[string]any {
		var m map[string]any
		b, _ := json.Marshal(api.PairedStatusOf(s.Status()))
		json.Unmarshal(b, &m)
		return m
	}
	return srv, paired
}

var fromAfar = map[string]string{"X-Forwarded-For": "10.9.8.7"}

// An --open caller from another machine may do on the control API exactly what a paired token
// may: every other route (and any name the router does not know) is 403 not_allowed, and its
// status is the reduced one.
func TestOpenRemoteCallerIsScopedLikeAPairedToken(t *testing.T) {
	srv, paired := openEnv(t)
	hdr := with(csrf, fromAfar)
	for _, r := range scopeRoutes {
		c, b, _ := do(t, r.method, srv.URL+"/engine/v1/"+r.path, "{}", hdr)
		if r.allowed && notAllowed(c, b) {
			t.Errorf("open remote %s %s refused: %d %s", r.method, r.path, c, b)
		}
		if !r.allowed && !notAllowed(c, b) {
			t.Errorf("open remote %s %s: %d %.200s, want 403 not_allowed", r.method, r.path, c, b)
		}
	}
	for _, p := range []string{"/engine/v1/some-route-added-later", "/engine/v1//sleep", "/engine/v1/status/../stop", "/engine/v1/", "/engine/"} {
		if c, b, _ := do(t, "POST", srv.URL+p, "{}", hdr); !notAllowed(c, b) {
			t.Errorf("open remote POST %s: %d %.200s", p, c, b)
		}
	}
	c, b, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", hdr)
	var got map[string]any
	json.Unmarshal([]byte(b), &got)
	want := paired()
	if c != 200 || len(got) != len(want) {
		t.Fatalf("open remote status: %d, %d fields, want the reduced %d: %.300s", c, len(got), len(want), b)
	}
	for k := range got {
		if _, ok := want[k]; !ok {
			t.Errorf("open remote status has %q, not in the reduced status", k)
		}
	}
}

// The same --open listener keeps everything for a caller of this machine (no route refused for
// who asks, the full status), and the master key from another machine keeps everything too.
func TestOpenLoopbackAndRemoteMasterKeepEverything(t *testing.T) {
	srv, paired := openEnv(t)
	for _, r := range scopeRoutes {
		if r.stream || r.path == "shutdown" {
			continue
		}
		// pairing and pairing/answer are the person's side, answered only on the loopback
		// listener or with the master token (answerer, pair.go), --open or not: as before
		if c, b, _ := do(t, r.method, srv.URL+"/engine/v1/"+r.path, "{}", csrf); notAllowed(c, b) && !strings.HasPrefix(r.path, "pairing") {
			t.Errorf("open loopback %s %s: %d %s", r.method, r.path, c, b)
		}
	}
	c, b, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", csrf)
	var got map[string]any
	json.Unmarshal([]byte(b), &got)
	if c != 200 || len(got) <= len(paired()) {
		t.Fatalf("open loopback status: %d, %d fields (want the full status)", c, len(got))
	}
	pe, _ := scopeEnv(t)
	hdr := with(csrf, bearer(pe.master), fromAfar)
	for _, r := range scopeRoutes {
		if r.stream || r.path == "shutdown" {
			continue
		}
		if c, b, _ := do(t, r.method, pe.srv.URL+"/engine/v1/"+r.path, "{}", hdr); notAllowed(c, b) && r.path != "pairing/answer" {
			t.Errorf("remote master %s %s: %d %s", r.method, r.path, c, b)
		}
	}
}
