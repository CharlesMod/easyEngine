package run_test

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

func sum(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func tgz(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, data := range files {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(data)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		tw.Write(data)
	}
	tw.Close()
	gz.Close()
	return buf.Bytes()
}

func wheel(t *testing.T, version string) []byte {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("hello_agent-" + version + ".dist-info/entry_points.txt")
	w.Write([]byte("[console_scripts]\nhello = hello_agent.cli:main\n"))
	w, _ = zw.Create("hello_agent/cli.py")
	w.Write([]byte("# " + version))
	zw.Close()
	return buf.Bytes()
}

const fakeUV = `#!/bin/sh
echo "$*" >> "$FAKE_UV_LOG"
echo "pref=$UV_PYTHON_PREFERENCE" >> "$FAKE_UV_LOG"
case "$1" in
  --version) echo "uv 0.0.0-fake" ;;
  venv)  # uv venv --python 3.12 DIR
    mkdir -p "$4/bin"
    cat > "$4/bin/python" <<'EOS'
#!/bin/sh
echo "$(date +%s%N) url=$EASY_ENGINE_LLM_URL mode=$EASY_ENGINE_LLM_MODE args=$*" >> "$FAKE_AGENT_LOG"
if [ -f "$FAKE_AGENT_DIE" ]; then exit 3; fi
exec sleep 30
EOS
    chmod +x "$4/bin/python" ;;
  python)
    case "$2" in
      install) : ;;
      find)  # the managed base interpreter (no-venv layout)
        b="$UV_PYTHON_INSTALL_DIR/base/bin"; mkdir -p "$b"
        cat > "$b/python" <<'EOS'
#!/bin/sh
echo "$(date +%s%N) url=$EASY_ENGINE_LLM_URL pythonpath=$PYTHONPATH nousersite=$PYTHONNOUSERSITE args=$*" >> "$FAKE_AGENT_LOG"
exec sleep 30
EOS
        chmod +x "$b/python"; echo "$b/python" ;;
    esac ;;
  pip) : ;;
esac
`

type site struct {
	e      *testkit.Env
	srv    *httptest.Server
	cat    *catalog.Catalog
	uvLog  string
	agent  string
	wheelP string
	model  []byte
	slow   int32 // when 1, the model download blocks until cancelled
}

func newSite(t *testing.T) *site {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	s := &site{e: e, uvLog: filepath.Join(e.Ctl, "uv.log"), agent: filepath.Join(e.Ctl, "agent.log")}
	e.WatchFile(s.uvLog) // waits on these files are woken by their changes
	e.WatchFile(s.agent)
	t.Setenv("FAKE_UV_LOG", s.uvLog)
	t.Setenv("FAKE_AGENT_LOG", s.agent)
	t.Setenv("FAKE_AGENT_DIE", filepath.Join(e.Ctl, "die"))
	fake := testkit.FakeLlama(t)
	llamaBin, _ := os.ReadFile(fake)
	cudaBin := []byte("#!/bin/sh\n[ \"$1\" = --version ] && { echo broken cuda; exit 0; }\nexit 1\n")
	cudaTar := tgz(t, map[string][]byte{"build/bin/llama-server": cudaBin})
	cpuTar := tgz(t, map[string][]byte{"build/bin/llama-server": llamaBin, "build/bin/libfake.so": []byte("lib")})
	uvTar := tgz(t, map[string][]byte{"uv-x/uv": []byte(fakeUV)})
	s.model = []byte(strings.Repeat("modeldata", 20000))
	s.wheelP = filepath.Join(e.Ctl, "hello_agent-0.1.0-py3-none-any.whl")
	os.WriteFile(s.wheelP, wheel(t, "0.1.0"), 0o644)
	mux := http.NewServeMux()
	serve := func(b []byte) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) { w.Write(b) }
	}
	mux.HandleFunc("/cuda.tar.gz", serve(cudaTar))
	mux.HandleFunc("/cpu.tar.gz", serve(cpuTar))
	mux.HandleFunc("/uv.tar.gz", serve(uvTar))
	mux.HandleFunc("/model.gguf", func(w http.ResponseWriter, r *http.Request) {
		if atomic.LoadInt32(&s.slow) == 1 {
			w.Write(s.model[:1000])
			w.(http.Flusher).Flush()
			<-r.Context().Done()
			return
		}
		for i := 0; i < len(s.model); i += len(s.model) / 8 {
			end := i + len(s.model)/8
			if end > len(s.model) {
				end = len(s.model)
			}
			w.Write(s.model[i:end])
			w.(http.Flusher).Flush()
		}
	})
	s.srv = httptest.NewServer(mux)
	t.Cleanup(s.srv.Close)
	s.cat = &catalog.Catalog{
		Version: 1,
		LlamaCpp: catalog.LlamaCpp{Tag: "bTEST", Assets: []catalog.Asset{
			{ID: "cuda-test", OS: runtime.GOOS, Arch: runtime.GOARCH, Accel: "cuda", URL: s.srv.URL + "/cuda.tar.gz", SHA256: sum(cudaTar), Size: int64(len(cudaTar))},
			{ID: "cpu-test", OS: runtime.GOOS, Arch: runtime.GOARCH, Accel: "cpu", URL: s.srv.URL + "/cpu.tar.gz", SHA256: sum(cpuTar), Size: int64(len(cpuTar))},
		}},
		UV: []catalog.Asset{{ID: "uv", OS: runtime.GOOS, Arch: runtime.GOARCH, URL: s.srv.URL + "/uv.tar.gz", SHA256: sum(uvTar), Size: int64(len(uvTar))}},
		Models: []catalog.Model{
			{ID: "dl", Name: "Downloadable", File: "dl.gguf", URL: s.srv.URL + "/model.gguf", SHA256: sum(s.model), Size: int64(len(s.model))},
			{ID: "imp", Name: "Import only", File: "imp.gguf", SHA256: sum([]byte("imported bytes")), Size: int64(len("imported bytes"))},
		},
	}
	e.Cat = s.cat
	e.Prof.Agent = &run.AgentSpec{ID: "hello-agent", Wheel: s.wheelP, Args: []string{"run"}}
	e.Agent = []string{"FAKE_UV_LOG=" + s.uvLog, "FAKE_AGENT_LOG=" + s.agent, "FAKE_AGENT_DIE=" + filepath.Join(e.Ctl, "die")}
	return s
}

func (s *site) sup(t *testing.T) *run.Supervisor { return s.supWith(t, nil) }

// supWith is sup with the config changed first (mod may be nil).
func (s *site) supWith(t *testing.T, mod func(*run.Config)) *run.Supervisor {
	cfg := s.e.Config()
	if mod != nil {
		mod(&cfg)
	}
	sup := s.e.StartSup(cfg)
	sup.SetPort(4242)
	return sup
}

func lines(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil || len(bytes.TrimSpace(b)) == 0 {
		return nil
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

func TestInstallJob(t *testing.T) {
	s := newSite(t)
	sup := s.sup(t)
	if sup.Phase() != "not_installed" {
		t.Fatalf("phase %s", sup.Phase())
	}
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	if err := sup.Install("dl", ""); err != run.ErrBusy {
		t.Fatalf("second install: %v (want ErrBusy, the API answers 409)", err)
	}
	sawProgress, sawDownloading := false, false
	testkit.WaitFor(t, 30*time.Second, "install to finish and the server to be ready", func() bool {
		st := sup.Status()
		if st.Phase == "downloading" {
			sawDownloading = true
		}
		if st.Progress != nil && st.Progress.TotalBytes > 0 && st.Progress.DoneBytes > 0 {
			sawProgress = true
		}
		return st.Phase == "ready"
	})
	if !sawDownloading || !sawProgress {
		t.Errorf("progress not shown: downloading=%v progress=%v", sawDownloading, sawProgress)
	}
	st := run.LoadState(s.e.Data)
	// the cuda build failed its --list-devices smoke test, so the fallback (cpu) was chosen
	if st.Model != "dl" || st.Backend != "cpu-test" {
		t.Fatalf("state = %+v", st)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "dl.gguf")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "runtime", "uv")); err != nil {
		t.Fatal("uv not installed")
	}
	// unpacked backend archives are not kept (on Windows they were about 0.8 GB of dead weight)
	if left, _ := filepath.Glob(filepath.Join(s.e.Data, "downloads", "*")); len(left) != 0 {
		t.Fatalf("backend archives left after unpacking: %v", left)
	}
	uv := lines(s.uvLog)
	var venvLine, pipLine string
	for _, l := range uv {
		if strings.HasPrefix(l, "venv ") {
			venvLine = l
		}
		if strings.HasPrefix(l, "pip ") {
			pipLine = l
		}
	}
	for i, l := range uv { // venv and pip run with our own Python only, never one found on PATH
		if (strings.HasPrefix(l, "venv ") || strings.HasPrefix(l, "pip ")) && (i+1 >= len(uv) || uv[i+1] != "pref=only-managed") {
			t.Fatalf("uv %q ran without UV_PYTHON_PREFERENCE=only-managed: %v", l, uv)
		}
	}
	if !strings.Contains(venvLine, "--python 3.12") || !strings.Contains(pipLine, "install --python") || !strings.Contains(pipLine, "hello_agent-0.1.0") {
		t.Fatalf("uv calls: %v", uv)
	}
	// the agent runs with the engine URL and the module from the wheel's console script
	testkit.WaitFor(t, 5*time.Second, "agent to start", func() bool { return len(lines(s.agent)) >= 1 })
	l := lines(s.agent)[0]
	if !strings.Contains(l, "url=http://127.0.0.1:4242") || !strings.Contains(l, "args=-m hello_agent.cli run") {
		t.Fatalf("agent line: %s", l)
	}
	if sup.Status().Agent.Phase != "running" {
		t.Fatalf("agent status %+v", sup.Status().Agent)
	}
	// installing the same wheel again does not reinstall; a new wheel does
	n := len(lines(s.uvLog))
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	// the job's end is a fact: the "job" fact stands from the install call (it said it) until the job ended
	testkit.WaitFor(t, 20*time.Second, "the job to end", func() bool {
		_, running := sup.Hub().Fact("job", "")
		return !running
	})
	testkit.WaitPhase(t, sup, "ready", 20*time.Second)
	if got := len(lines(s.uvLog)); got != n {
		t.Fatalf("uv was called again without a wheel change: %v", lines(s.uvLog))
	}
	os.WriteFile(s.wheelP, wheel(t, "0.2.0"), 0o644)
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 20*time.Second, "reinstall after the wheel changed", func() bool {
		return len(lines(s.uvLog)) > n && sup.Phase() == "ready"
	})
}

func TestBackendFallbackAllFail(t *testing.T) {
	s := newSite(t)
	s.cat.LlamaCpp.Assets = s.cat.LlamaCpp.Assets[:1] // only the broken cuda build
	sup := s.sup(t)
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, sup, "error", 20*time.Second)
	if !strings.Contains(sup.Status().Message, "smoke test") {
		t.Fatalf("message: %s", sup.Status().Message)
	}
}

func TestInstallCancelKeepsPart(t *testing.T) {
	s := newSite(t)
	atomic.StoreInt32(&s.slow, 1)
	sup := s.sup(t)
	if err := sup.Install("dl", "cpu-test"); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 10*time.Second, "download to start", func() bool {
		st := sup.Status()
		return st.Progress != nil && strings.HasPrefix(st.Progress.Step, "model")
	})
	sup.Cancel()
	testkit.WaitFor(t, 10*time.Second, "job to end", func() bool { return sup.Phase() != "downloading" })
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "dl.gguf.part")); err != nil {
		t.Fatalf("partial download was not kept for resume: %v", err)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "dl.gguf")); err == nil {
		t.Fatal("cancelled download produced a model file")
	}
	// a new job may start
	atomic.StoreInt32(&s.slow, 0)
	if err := sup.Install("dl", "cpu-test"); err != nil {
		t.Fatalf("install after cancel: %v", err)
	}
	testkit.WaitPhase(t, sup, "ready", 30*time.Second)
}

func TestImportOnlyModel(t *testing.T) {
	s := newSite(t)
	sup := s.sup(t)
	if err := sup.Install("imp", "cpu-test"); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, sup, "error", 20*time.Second)
	if !strings.Contains(sup.Status().Message, "import-only") {
		t.Fatalf("message: %s", sup.Status().Message)
	}
	src := filepath.Join(s.e.Ctl, "my-model.gguf")
	os.WriteFile(src, []byte("imported bytes"), 0o644)
	if err := sup.ImportSync(t.Context(), src); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(s.e.Data, "models", "imp.gguf")); string(b) != "imported bytes" {
		t.Fatalf("model not imported: %q", b)
	}
	bad := filepath.Join(s.e.Ctl, "other.gguf")
	os.WriteFile(bad, []byte("something else"), 0o644)
	if err := sup.ImportSync(t.Context(), bad); err == nil {
		t.Fatal("a file that is not in the catalog was imported")
	}
}

// An agent that exits at once is restarted once at once; a second quick failure in a row waits for a
// word (no clock); a run of AgentStable resets the count.
func TestAgentRestartOnceThenWaitForWord(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	wheelP := filepath.Join(e.Ctl, "a-0.1-py3-none-any.whl")
	os.WriteFile(wheelP, []byte("wheel"), 0o644)
	e.Prof.Agent = &run.AgentSpec{ID: "bo", Wheel: wheelP, Module: "m", Args: []string{"run"}}
	// a ready venv by hand: every start adds a line to the log; the third start stays up long
	// enough to count as stable, every other one exits at once
	venv := filepath.Join(e.Data, "runtime", "venvs", "bo")
	os.MkdirAll(filepath.Join(venv, "bin"), 0o755)
	logf := filepath.Join(e.Ctl, "agent.log")
	script := "#!/bin/sh\necho start >> " + logf + "\nn=$(wc -l < " + logf + ")\nif [ \"$n\" -eq 3 ]; then sleep 1; fi\nexit 3\n"
	os.WriteFile(filepath.Join(venv, "bin", "python"), []byte(script), 0o755)
	os.WriteFile(filepath.Join(venv, ".wheel-sha"), []byte(sum([]byte("wheel"))), 0o644)
	cfg := e.Config()
	cfg.AgentStable = 300 * time.Millisecond
	sup := e.StartSup(cfg)
	waiting := func(restarts int) func() bool {
		return func() bool {
			st := sup.Status().Agent
			return st.Phase == "error" && st.Restarts == restarts && strings.Contains(st.Message, "waiting for a request or a word from a client")
		}
	}
	// start 1 fails: restarted at once (start 2); start 2 fails soon after: waits
	testkit.WaitFor(t, 10*time.Second, "agent waiting after two quick exits", waiting(2))
	if n := len(lines(logf)); n != 2 {
		t.Fatalf("%d agent starts before a word, want 2", n)
	}
	// a word gives exactly one more start (the third, which runs stable); a stable run resets the
	// count, so its exit is restarted at once (the fourth), whose quick exit waits again
	sup.Nudge("x")
	testkit.WaitFor(t, 10*time.Second, "agent waiting after the stable run", waiting(4))
	if n := len(lines(logf)); n != 4 {
		t.Fatalf("%d agent starts after the stable run, want 4", n)
	}
	// and a request at the door is a word too (the model is ready, the agent waits): exactly one
	// start and no more
	sup.Begin()()
	testkit.WaitFor(t, 10*time.Second, "agent waiting after one more start", waiting(5))
	if n := len(lines(logf)); n != 5 {
		t.Fatalf("%d agent starts after a word, want 5", n)
	}
}

// Windows layout: no venv (its python.exe launcher was blocked by Smart App Control); the managed base
// Python runs the agent with the packages in <env>/site on PYTHONPATH, and uv puts no shim in ~/.local/bin.
func TestInstallJobNoVenv(t *testing.T) {
	defer run.SetAgentNoVenv(true)()
	s := newSite(t)
	sup := s.sup(t)
	if err := sup.Install("dl", ""); err != nil {
		t.Fatal(err)
	}
	testkit.WaitPhase(t, sup, "ready", 30*time.Second)
	var sawInstall, sawFind, sawTarget, sawVenv bool
	for _, l := range lines(s.uvLog) {
		sawInstall = sawInstall || strings.HasPrefix(l, "python install --no-bin")
		sawFind = sawFind || strings.HasPrefix(l, "python find")
		sawTarget = sawTarget || (strings.HasPrefix(l, "pip install --python ") && strings.Contains(l, "--target "+filepath.Join(s.e.Data, "runtime", "venvs", "hello-agent", "site")))
		sawVenv = sawVenv || strings.HasPrefix(l, "venv ")
	}
	if !sawInstall || !sawFind || !sawTarget || sawVenv {
		t.Fatalf("uv calls (install=%v find=%v target=%v venv=%v): %v", sawInstall, sawFind, sawTarget, sawVenv, lines(s.uvLog))
	}
	testkit.WaitFor(t, 5*time.Second, "agent to start", func() bool { return len(lines(s.agent)) >= 1 })
	l := lines(s.agent)[0]
	site := filepath.Join(s.e.Data, "runtime", "venvs", "hello-agent", "site")
	if !strings.Contains(l, "pythonpath="+site) || !strings.Contains(l, "nousersite=1") || !strings.Contains(l, "args=-m hello_agent.cli run") {
		t.Fatalf("agent line: %s", l)
	}
	if sup.Status().Agent.Phase != "running" {
		t.Fatalf("agent status %+v", sup.Status().Agent)
	}
}
