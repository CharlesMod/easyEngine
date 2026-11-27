package run

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/fetch"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
)

func (s *Supervisor) jobProgress(step string) func(fetch.Progress) {
	return func(p fetch.Progress) {
		s.setProgress(stepLabel(step, p), &Progress{DoneBytes: p.Done, TotalBytes: p.Total, RateBps: p.RateBps, ETAS: int(p.ETA.Seconds())})
	}
}

// stepLabel keeps fetch's "verifying" pass visible: after a download the file is hashed and the
// byte count starts again from zero, which would read as the download going backwards.
func stepLabel(step string, p fetch.Progress) string {
	if p.Step == "verifying" {
		return "verifying " + step
	}
	return step
}

// Install starts the install job: backend, then model, then uv and the agent.
func (s *Supervisor) Install(model, backend string) error {
	if handled, err := s.trySwitch(model, backend); handled {
		return err
	}
	if err := s.prepareInstall(model); err != nil {
		return err
	}
	s.mu.Lock()
	if s.jobRunning {
		s.mu.Unlock()
		return ErrBusy
	}
	s.jobRunning = true
	ctx, cancel := context.WithCancel(s.root)
	s.jobCancel = cancel
	s.mu.Unlock()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		err := s.runInstall(ctx, model, backend)
		s.endJob(ctx, err, "Install")
	}()
	return nil
}

// InstallSync runs the install job and waits (the CLI uses it).
func (s *Supervisor) InstallSync(ctx context.Context, model, backend string) error {
	if err := s.prepareInstall(model); err != nil {
		return err
	}
	s.mu.Lock()
	if s.jobRunning {
		s.mu.Unlock()
		return ErrBusy
	}
	s.jobRunning = true
	jctx, cancel := context.WithCancel(ctx)
	s.jobCancel = cancel
	s.mu.Unlock()
	err := s.runInstall(jctx, model, backend)
	s.endJob(jctx, err, "Install")
	return err
}

func (s *Supervisor) endJob(ctx context.Context, err error, what string) {
	s.mu.Lock()
	s.jobRunning = false
	s.jobCancel = nil
	s.progress = nil
	s.mu.Unlock()
	switch {
	case err == nil:
		s.logf("%s finished", what)
		s.settle()
		if !s.cfg.NoAutoStart {
			s.startAgent()
			if s.Status().Mode == "local" {
				go s.restartServer()
			}
		}
	case errors.Is(err, context.Canceled) || ctx.Err() != nil:
		s.addError("%s cancelled", what)
		s.settle()
	default:
		s.addError("%s failed: %v", what, err)
		s.setPhase("error", fmt.Sprintf("%s failed: %v", what, err))
	}
}

// Cancel stops the running job (partial downloads stay for resume).
func (s *Supervisor) Cancel() {
	s.mu.Lock()
	c := s.jobCancel
	s.mu.Unlock()
	if c != nil {
		c()
	}
	s.cancelUpgrade()
	s.forgetSwitch()
}

func archiveExt(url string) string {
	u := strings.ToLower(strings.SplitN(url, "?", 2)[0])
	if strings.HasSuffix(u, ".tar.gz") || strings.HasSuffix(u, ".tgz") {
		return ".tar.gz"
	}
	return ".zip"
}

func (s *Supervisor) runInstall(ctx context.Context, modelID, backendID string) error {
	cat, err := s.catalog()
	if err != nil {
		return err
	}
	m, ok := cat.Model(modelID)
	if !ok {
		return fmt.Errorf("unknown model %q", modelID)
	}
	if m.URL == "" && !s.modelInstalled(m) {
		return fmt.Errorf("%s is import-only: run `easy-engine import <path>` (or POST /engine/v1/import)", m.Name)
	}
	hw, err := s.Hardware(ctx, false)
	if err != nil {
		return err
	}
	// The whole setup up front (the build, the model, the agent): refuse before writing a byte.
	if err := s.checkNeed(s.setupNeed(cat, hw, backendID, m), m.Name); err != nil {
		return err
	}
	s.setPhase("downloading", "Installing the model server")
	be, err := s.ensureBackend(ctx, cat, hw, backendID, m)
	if err != nil {
		return err
	}
	serve, target := m, ""
	if m.URL != "" && !s.modelInstalled(m) {
		if st := s.starterFor(ctx, cat, hw, m); st != nil {
			// First run: a small model first so the first line comes in minutes; the target
			// downloads in the background afterwards (starter.go).
			serve, target = *st, m.ID
			s.beginQueued(m)
			defer s.dropQueued()
			if !s.modelInstalled(serve) {
				s.setPhase("downloading", "Downloading "+serve.Name+" first")
				if err := fetch.Download(ctx, serve.URL, s.modelPath(serve), serve.SHA256, serve.Size, s.jobProgress("model "+serve.ID)); err != nil {
					return err
				}
				s.mu.Lock()
				s.st.StarterOwned = serve.ID // ours to delete after the swap
				s.mu.Unlock()
			}
		} else {
			if err := s.checkDisk(m); err != nil {
				return err
			}
			s.setPhase("downloading", "Downloading "+m.Name)
			if err := fetch.Download(ctx, m.URL, s.modelPath(m), m.SHA256, m.Size, s.jobProgress("model "+m.ID)); err != nil {
				return err
			}
		}
	}
	if s.cfg.Profile.Agent != nil && s.cfg.Profile.Agent.Wheel != "" {
		if err := s.ensureAgent(ctx, cat); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.st.Model, s.st.Backend, s.st.Target = serve.ID, be, target
	s.mu.Unlock()
	s.save()
	if target != "" {
		s.beginUpgrade(m)
	}
	return nil
}

// ensureBackend installs the first build that passes its smoke test; models are what the setup
// installs after it (for the space check before each fallback build).
func (s *Supervisor) ensureBackend(ctx context.Context, cat *catalog.Catalog, hw probe.Hardware, want string, models ...catalog.Model) (string, error) {
	var list []catalog.Asset
	if want != "" {
		a, ok := cat.Asset(want)
		if !ok {
			return "", fmt.Errorf("unknown backend %q", want)
		}
		list = []catalog.Asset{a}
	} else {
		list = cat.Backends(hw)
	}
	if len(list) == 0 {
		return "", errors.New("no llama.cpp build in the catalog for this machine")
	}
	var last error
	for i, a := range list {
		if i > 0 { // a fallback build: the failed one stays on disk; check again from here
			if err := s.checkNeed(s.setupNeed(cat, hw, a.ID, models...), a.ID); err != nil {
				return "", err
			}
		}
		if err := s.installBackend(ctx, cat, a); err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			s.addError("backend %s: %v", a.ID, err)
			if errors.Is(err, ErrAppControl) {
				return "", err // every build from this release is blocked alike: don't fetch the others
			}
			last = err
			continue
		}
		return a.ID, nil
	}
	return "", fmt.Errorf("no backend passed its smoke test: %w", last)
}

func (s *Supervisor) installBackend(ctx context.Context, cat *catalog.Catalog, a catalog.Asset) error {
	dir := filepath.Join(s.dir, "backends", a.ID)
	marker := filepath.Join(dir, ".installed")
	if _, err := os.Stat(marker); err != nil || findServer(dir) == "" {
		os.RemoveAll(dir) // only a path we build under our own data dir
		arch := filepath.Join(s.dir, "downloads", a.ID+archiveExt(a.URL))
		if err := fetch.Download(ctx, a.URL, arch, a.SHA256, a.Size, s.jobProgress("backend "+a.ID)); err != nil {
			return err
		}
		s.setPhase("installing", "Unpacking "+a.ID)
		if err := fetch.Unpack(arch, dir); err != nil {
			return err
		}
		os.Remove(arch) // unpacked: the archive is only dead weight (hundreds of MB)
		for _, x := range a.Extra {
			ea, ok := cat.Asset(x)
			if !ok {
				return fmt.Errorf("catalog has no asset %q", x)
			}
			earch := filepath.Join(s.dir, "downloads", ea.ID+archiveExt(ea.URL))
			if err := fetch.Download(ctx, ea.URL, earch, ea.SHA256, ea.Size, s.jobProgress("backend "+ea.ID)); err != nil {
				return err
			}
			tmp := filepath.Join(s.dir, "downloads", ea.ID+"-unpacked")
			os.RemoveAll(tmp)
			if err := fetch.Unpack(earch, tmp); err != nil {
				return err
			}
			os.Remove(earch)
			bin := findServer(dir)
			if bin == "" {
				return errors.New("llama-server not found in the unpacked backend")
			}
			err := copyFlat(tmp, filepath.Dir(bin))
			os.RemoveAll(tmp)
			if err != nil {
				return err
			}
		}
	}
	bin := findServer(dir)
	if bin == "" {
		return errors.New("llama-server not found in the unpacked backend")
	}
	if runtime.GOOS != "windows" {
		os.Chmod(bin, 0o755)
	}
	if err := smokeTest(ctx, bin, a.Accel); err != nil {
		os.Remove(marker)
		return err
	}
	return os.WriteFile(marker, []byte(a.SHA256), 0o644)
}

// smokeTest: `--version` must run, and a GPU backend must list its device.
//
// Each step has its own time limit, and a generous one: on a fresh Windows install the first run of a CUDA build
// loads several hundred MB of cuBLAS DLLs that the antivirus scans on first load. In the first Windows test
// (RTX 3050 laptop, Defender on) that took longer than the old shared 30 s, the kill looked like "exit status 1",
// and the engine quietly fell back to the much slower Vulkan build. A run later is fast (the scan is cached).
var smokeVersionTimeout, smokeDevicesTimeout = 60 * time.Second, 180 * time.Second

func smokeTest(ctx context.Context, bin, accel string) error {
	out, err := smokeRun(ctx, smokeVersionTimeout, bin, "--version")
	if err != nil {
		if appControlBlocked(err) {
			return fmt.Errorf("%w (%s)", ErrAppControl, filepath.Base(bin))
		}
		return fmt.Errorf("%s --version failed: %v: %s", filepath.Base(bin), err, lastLine(out))
	}
	want := map[string]string{"cuda": "cuda", "vulkan": "vulkan", "metal": "metal"}[accel]
	if want == "" {
		return nil
	}
	out, err = smokeRun(ctx, smokeDevicesTimeout, bin, "--list-devices")
	if err != nil {
		if appControlBlocked(err) {
			return fmt.Errorf("%w (%s)", ErrAppControl, filepath.Base(bin))
		}
		return fmt.Errorf("--list-devices failed: %v: %s", err, lastLine(out))
	}
	if !strings.Contains(strings.ToLower(out), want) {
		return fmt.Errorf("--list-devices shows no %s device", accel)
	}
	return nil
}

// smokeRun runs bin with args under its own time limit; a timeout is reported as one, not as the exit status
// the kill leaves behind (1 on Windows).
func smokeRun(ctx context.Context, limit time.Duration, bin string, args ...string) (string, error) {
	cctx, cancel := context.WithTimeout(ctx, limit)
	defer cancel()
	cmd := exec.CommandContext(cctx, bin, args...)
	cmd.Dir = filepath.Dir(bin)
	cmd.Env = llamaEnv(bin, nil)
	b, err := cmd.CombinedOutput()
	if err != nil && ctx.Err() == nil && cctx.Err() == context.DeadlineExceeded {
		err = fmt.Errorf("timed out after %s", limit)
	}
	return string(b), err
}

func copyFlat(src, dst string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		in, err := os.Open(p)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(filepath.Join(dst, d.Name()), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			return err
		}
		defer out.Close()
		_, err = io.Copy(out, in)
		return err
	})
}

func (s *Supervisor) runCmd(ctx context.Context, env []string, name string, args ...string) error {
	_, err := s.runCombined(ctx, env, false, name, args...)
	return err
}

// runOut is runCmd that returns stdout alone (stderr is only logged).
func (s *Supervisor) runOut(ctx context.Context, env []string, name string, args ...string) (string, error) {
	return s.runCombined(ctx, env, true, name, args...)
}

func (s *Supervisor) runCombined(ctx context.Context, env []string, stdoutOnly bool, name string, args ...string) (string, error) {
	cctx, cancel := context.WithTimeout(ctx, s.cfg.CmdTimeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, name, args...)
	cmd.Env = append(os.Environ(), env...)
	var stdout bytes.Buffer
	all := &lockedBuf{}
	cmd.Stdout = io.MultiWriter(&stdout, all)
	cmd.Stderr = all
	err := cmd.Run()
	out := all.String()
	s.logf("%s %s: %v\n%s", path.Base(name), strings.Join(args, " "), err, strings.TrimSpace(out))
	if err != nil {
		return "", fmt.Errorf("%s %s: %v: %s", filepath.Base(name), args[0], err, lastLine(out))
	}
	if stdoutOnly {
		return stdout.String(), nil
	}
	return out, nil
}

// ensureUV installs the pinned uv into runtime/.
func (s *Supervisor) ensureUV(ctx context.Context, cat *catalog.Catalog) (string, error) {
	uv := uvPath(s.dir)
	if _, err := os.Stat(uv); err == nil {
		if s.runCmd(ctx, nil, uv, "--version") == nil {
			return uv, nil
		}
	}
	var a *catalog.Asset
	for i := range cat.UV {
		if cat.UV[i].OS == runtime.GOOS && cat.UV[i].Arch == runtime.GOARCH {
			a = &cat.UV[i]
			break
		}
	}
	if a == nil {
		return "", errors.New("no uv build in the catalog for this machine")
	}
	arch := filepath.Join(s.dir, "downloads", "uv"+archiveExt(a.URL))
	if err := fetch.Download(ctx, a.URL, arch, a.SHA256, a.Size, s.jobProgress("uv")); err != nil {
		return "", err
	}
	s.setPhase("installing", "Installing uv")
	tmp := filepath.Join(s.dir, "downloads", "uv-unpacked")
	os.RemoveAll(tmp)
	defer os.RemoveAll(tmp)
	if err := fetch.Unpack(arch, tmp); err != nil {
		return "", err
	}
	os.Remove(arch) // unpacked; uv.exe is copied out below
	var found string
	filepath.WalkDir(tmp, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && (d.Name() == "uv" || d.Name() == "uv.exe") && found == "" {
			found = p
		}
		return nil
	})
	if found == "" {
		return "", errors.New("uv not found in its archive")
	}
	os.MkdirAll(filepath.Dir(uv), 0o755)
	os.Remove(uv)
	in, err := os.Open(found)
	if err != nil {
		return "", err
	}
	defer in.Close()
	out, err := os.OpenFile(uv, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return "", err
	}
	out.Close()
	if err := s.runCmd(ctx, nil, uv, "--version"); err != nil {
		return "", err
	}
	return uv, nil
}

// ensureAgent installs the agent wheel into its venv, reinstalling when the wheel's sha changes.
func (s *Supervisor) ensureAgent(ctx context.Context, cat *catalog.Catalog) error {
	ag := s.cfg.Profile.Agent
	if s.agentReady() {
		return nil
	}
	sha, err := fileSHA(ag.Wheel)
	if err != nil {
		return fmt.Errorf("agent wheel: %w", err)
	}
	uv, err := s.ensureUV(ctx, cat)
	if err != nil {
		return err
	}
	s.agent.set("installing", "installing the agent", 0)
	s.setPhase("installing", "Installing the agent")
	s.setProgress("agent", nil)
	venv := venvDir(s.dir, ag)
	os.RemoveAll(venv)
	env := []string{
		"UV_CACHE_DIR=" + filepath.Join(s.dir, "runtime", "uv-cache"),
		"UV_PYTHON_INSTALL_DIR=" + filepath.Join(s.dir, "runtime", "python"),
		// Always our own Python under runtime\python, never one the player happens to have installed
		// (seen on Windows: uv picked a user's Python312 from PATH), so every machine gets the same agent
		// and removing the data dir removes all of it.
		"UV_PYTHON_PREFERENCE=only-managed",
	}
	if agentNoVenv {
		// Managed base Python (no shim in the player's ~/.local/bin), packages into <env>/site.
		if err := s.runCmd(ctx, env, uv, "python", "install", "--no-bin", ag.Python); err != nil {
			return err
		}
		out, err := s.runOut(ctx, env, uv, "python", "find", ag.Python)
		if err != nil {
			return err
		}
		base := lastLine(out)
		if base == "" {
			return errors.New("uv python find: no interpreter")
		}
		if err := s.runCmd(ctx, env, uv, "pip", "install", "--python", base, "--target", filepath.Join(venv, "site"), ag.Wheel); err != nil {
			return err
		}
		os.MkdirAll(venv, 0o755)
		if err := os.WriteFile(filepath.Join(venv, basePythonFile), []byte(base), 0o644); err != nil {
			return err
		}
	} else {
		if err := s.runCmd(ctx, env, uv, "venv", "--python", ag.Python, venv); err != nil {
			return err
		}
		if err := s.runCmd(ctx, env, uv, "pip", "install", "--python", venvPython(venv), ag.Wheel); err != nil {
			return err
		}
	}
	s.agent.set("stopped", "", 0)
	return os.WriteFile(filepath.Join(venv, ".wheel-sha"), []byte(sha), 0o644)
}

// Import copies or hard-links a local model file (import-only models) after
// checking its sha256 against the catalog. It runs as a job.
func (s *Supervisor) Import(src string) error {
	s.mu.Lock()
	if s.jobRunning {
		s.mu.Unlock()
		return ErrBusy
	}
	s.jobRunning = true
	ctx, cancel := context.WithCancel(s.root)
	s.jobCancel = cancel
	s.mu.Unlock()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.endJob(ctx, s.runImport(ctx, src), "Import")
	}()
	return nil
}

// ImportSync imports and waits.
func (s *Supervisor) ImportSync(ctx context.Context, src string) error {
	s.mu.Lock()
	if s.jobRunning {
		s.mu.Unlock()
		return ErrBusy
	}
	s.jobRunning = true
	jctx, cancel := context.WithCancel(ctx)
	s.jobCancel = cancel
	s.mu.Unlock()
	err := s.runImport(jctx, src)
	s.endJob(jctx, err, "Import")
	return err
}

func (s *Supervisor) runImport(ctx context.Context, src string) error {
	cat, err := s.catalog()
	if err != nil {
		return err
	}
	s.setPhase("installing", "Importing "+filepath.Base(src))
	s.setProgress("hashing "+filepath.Base(src), nil)
	sum, err := fileSHA(src)
	if err != nil {
		return err
	}
	var m *catalog.Model
	for i := range cat.Models {
		if strings.EqualFold(cat.Models[i].SHA256, sum) {
			m = &cat.Models[i]
		}
	}
	if m == nil {
		return fmt.Errorf("%s is not a model in the catalog (sha256 %s)", filepath.Base(src), sum)
	}
	room := func(n int64) error { // a copy, not a link: the whole file lands on our disk
		return s.checkNeed(plan.NeedOf([]plan.DiskStep{plan.ModelStep(m.Name, n, 0)}, s.logRoom(), s.hostFiles()), "the copy of "+m.Name)
	}
	if err := fetch.ImportCheck(src, s.modelPath(*m), sum, room); err != nil {
		return err
	}
	s.mu.Lock()
	if s.st.Model == "" {
		s.st.Model = m.ID
	}
	s.mu.Unlock()
	s.save()
	return nil
}

// restartServer (re)starts the model server, e.g. after an install selected a model.
func (s *Supervisor) restartServer() {
	s.mu.Lock()
	s.userStopped = false
	lc := s.loadCancel
	s.mu.Unlock()
	if lc != nil {
		lc()
	}
	s.opMu.Lock()
	defer s.opMu.Unlock()
	s.stopServerLocked()
	s.startLocked()
}

// lockedBuf is a bytes.Buffer safe for exec's two copying goroutines (stdout and stderr).
type lockedBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedBuf) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *lockedBuf) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}
