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

	"hermesengine/catalog"
	"hermesengine/internal/testkit"
	"hermesengine/run"
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
case "$1" in
  --version) echo "uv 0.0.0-fake" ;;
  venv)  # uv venv --python 3.12 DIR
    mkdir -p "$4/bin"
    cat > "$4/bin/python" <<'EOS'
#!/bin/sh
echo "$(date +%s%N) url=$HERMESVOICE_LLM_URL mode=$HERMESVOICE_LLM_MODE args=$*" >> "$FAKE_AGENT_LOG"
if [ -f "$FAKE_AGENT_DIE" ]; then exit 3; fi
exec sleep 30
EOS
    chmod +x "$4/bin/python" ;;
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
			time.Sleep(30 * time.Millisecond)
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

func (s *site) sup(t *testing.T) *run.Supervisor {
	cfg := s.e.Config()
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
	time.Sleep(300 * time.Millisecond)
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

func TestAgentRestartWithBackoff(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	wheelP := filepath.Join(e.Ctl, "a-0.1-py3-none-any.whl")
	os.WriteFile(wheelP, []byte("wheel"), 0o644)
	e.Prof.Agent = &run.AgentSpec{ID: "bo", Wheel: wheelP, Module: "m", Args: []string{"run"}}
	// a ready venv by hand: the agent exits at once every time
	venv := filepath.Join(e.Data, "runtime", "venvs", "bo")
	os.MkdirAll(filepath.Join(venv, "bin"), 0o755)
	logf := filepath.Join(e.Ctl, "agent.log")
	os.WriteFile(filepath.Join(venv, "bin", "python"), []byte("#!/bin/sh\necho \"$(date +%s%N)\" >> "+logf+"\nexit 3\n"), 0o755)
	os.WriteFile(filepath.Join(venv, ".wheel-sha"), []byte(sum([]byte("wheel"))), 0o644)
	cfg := e.Config()
	cfg.AgentBackoffBase = 60 * time.Millisecond
	cfg.AgentBackoffMax = 250 * time.Millisecond
	sup := e.StartSup(cfg)
	testkit.WaitFor(t, 10*time.Second, "5 agent starts", func() bool { return len(lines(logf)) >= 6 })
	ts := lines(logf)
	var gaps []time.Duration
	for i := 1; i < len(ts); i++ {
		var a, b int64
		for _, c := range ts[i-1] {
			a = a*10 + int64(c-'0')
		}
		for _, c := range ts[i] {
			b = b*10 + int64(c-'0')
		}
		gaps = append(gaps, time.Duration(b-a))
	}
	// 60, 120, 240, 250 (capped), 250: each gap at least about the backoff, growing then flat
	if !(gaps[1] > gaps[0]+30*time.Millisecond && gaps[2] > gaps[1]+60*time.Millisecond) {
		t.Fatalf("backoff did not grow: %v", gaps)
	}
	if gaps[4] > 700*time.Millisecond {
		t.Fatalf("backoff not capped: %v", gaps)
	}
	if st := sup.Status().Agent; st.Restarts < 3 {
		t.Fatalf("agent status %+v", st)
	}
}
