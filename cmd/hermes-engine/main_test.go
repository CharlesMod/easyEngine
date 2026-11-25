package main

import (
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"hermesengine/lifecycle"
)

// Serve mode with stdout and stderr as pipes whose read end is closed (the game
// exited or never reads): logging must not kill the engine with SIGPIPE, and it
// still shuts down cleanly when its parent dies.
func TestServeSurvivesClosedStdioPipes(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix pipes")
	}
	home := t.TempDir()
	bin := filepath.Join(t.TempDir(), "hermes-engine")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	parent := exec.Command("sleep", "60")
	if err := parent.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { parent.Process.Kill(); parent.Wait() }()

	or, ow, _ := os.Pipe()
	er, ew, _ := os.Pipe()
	cmd := exec.Command(bin, "serve", "--port", "47650", "--parent-pid", strconv.Itoa(parent.Process.Pid))
	cmd.Env = append(os.Environ(), "HERMES_ENGINE_HOME="+home)
	cmd.Stdout, cmd.Stderr = ow, ew
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	ow.Close()
	ew.Close()
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	defer cmd.Process.Kill()

	base := ""
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) && base == "" {
		if d, err := lifecycle.ReadLock(filepath.Join(home, "engine.lock")); err == nil && d.Port > 0 {
			if r, err := http.Get("http://127.0.0.1:" + strconv.Itoa(d.Port) + "/engine/v1/status"); err == nil {
				r.Body.Close()
				base = "http://127.0.0.1:" + strconv.Itoa(d.Port)
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	if base == "" {
		t.Fatal("engine did not come up")
	}
	// the readers go away, then the engine logs
	or.Close()
	er.Close()
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequest("POST", base+"/engine/v1/sleep", nil)
		req.Header.Set("X-Hermes-Engine", "1")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("engine died after its pipes closed: %v", err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	select {
	case err := <-exited:
		t.Fatalf("engine exited after the pipes closed: %v", err)
	case <-time.After(300 * time.Millisecond):
	}
	b, _ := os.ReadFile(filepath.Join(home, "logs", "engine.log"))
	if !strings.Contains(string(b), "phase") {
		t.Fatalf("engine.log lacks the log lines:\n%s", b)
	}
	parent.Process.Kill()
	parent.Wait()
	select {
	case err := <-exited:
		if err != nil {
			t.Fatalf("unclean exit: %v", err)
		}
	case <-time.After(6 * time.Second):
		t.Fatal("engine did not exit after its parent died")
	}
	if _, err := os.Stat(filepath.Join(home, "engine.lock")); err == nil {
		t.Fatal("lock left behind")
	}
}
