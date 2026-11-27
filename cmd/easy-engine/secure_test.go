package main

import (
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/app"
)

func buildBin(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "easy-engine")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	return bin
}

func runBin(t *testing.T, bin string, env []string, args ...string) (string, int) {
	t.Helper()
	cmd := exec.Command(bin, args...)
	cmd.Env = append(os.Environ(), env...)
	out, err := cmd.CombinedOutput()
	code := 0
	if ee, ok := err.(*exec.ExitError); ok {
		code = ee.ExitCode()
	} else if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(out)), code
}

func TestTokenCommand(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix modes")
	}
	bin := buildBin(t)
	home := t.TempDir()
	env := []string{"EASY_ENGINE_HOME=" + home}
	a, code := runBin(t, bin, env, "token")
	if code != 0 || len(a) != 43 {
		t.Fatalf("token: %q exit %d", a, code)
	}
	if b, _ := runBin(t, bin, env, "token"); b != a {
		t.Fatal("token changed between calls")
	}
	fi, err := os.Stat(filepath.Join(home, "tokens.json"))
	if err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("token file: %v %v", fi, err)
	}
	n, code := runBin(t, bin, env, "token", "--new")
	if code != 0 || n == a || len(n) != 43 {
		t.Fatalf("token --new: %q exit %d", n, code)
	}
	if b, _ := runBin(t, bin, env, "token"); b != n {
		t.Fatal("--new was not kept")
	}
}

// A data dir the engine must not own is refused with exit code 2 before anything is written.
func TestDataDirRefused(t *testing.T) {
	bin := buildBin(t)
	home := t.TempDir()
	for name, home2 := range map[string]string{"relative": "data", "dot": ".", "the home dir itself": home} {
		for _, cmd := range [][]string{{"token"}, {"install", "--model", "x"}, {"import", "--path", "x"}, {"setup"}, {"serve", "--parent-pid", "1"}} {
			out, code := runBin(t, bin, []string{"HOME=" + home, "XDG_DATA_HOME=" + filepath.Join(home, "xdg"), "EASY_ENGINE_HOME=" + home2}, cmd...)
			if code != 2 {
				t.Errorf("%s %v: exit %d: %s", name, cmd, code, out)
			}
		}
	}
	if _, err := os.Stat(filepath.Join(home, "tokens.json")); err == nil {
		t.Fatal("a refused command wrote into the dir")
	}
}

// installVia must not report success when the engine cannot be read.
func TestInstallViaFailsWhenEngineDies(t *testing.T) {
	cases := map[string]http.HandlerFunc{
		"non-200 status": func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				w.WriteHeader(202)
				return
			}
			http.Error(w, "boom", 500)
		},
		"connection dropped": func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				w.WriteHeader(202)
				return
			}
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
		},
		"not JSON": func(w http.ResponseWriter, r *http.Request) {
			if r.Method == "POST" {
				w.WriteHeader(202)
				return
			}
			w.Write([]byte("<html>"))
		},
	}
	for name, h := range cases {
		srv := httptest.NewServer(h)
		_, p, _ := net.SplitHostPort(srv.Listener.Addr().String())
		port, _ := strconv.Atoi(p)
		if code := installVia(app.Target{Host: "127.0.0.1", Port: port}, "m", ""); code != 1 {
			t.Errorf("%s: exit %d, want 1", name, code)
		}
		srv.Close()
	}
}
