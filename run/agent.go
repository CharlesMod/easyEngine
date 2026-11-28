package run

import (
	"archive/zip"
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/CharlesMod/easyEngine/internal/legacy"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
)

type agentMgr struct {
	s        *Supervisor
	mu       sync.Mutex
	cancel   context.CancelFunc
	done     chan struct{}
	phase    string
	msg      string
	pid      int
	restarts int
	proc     *lifecycle.Proc
}

// AgentStatus is the agent part of the status.
type AgentStatus struct {
	Phase    string `json:"phase"` // not_installed|installing|running|restarting|stopped|error
	Pid      int    `json:"pid"`
	Restarts int    `json:"restarts"`
	Message  string `json:"message"`
}

func (a *agentMgr) status() AgentStatus {
	a.mu.Lock()
	defer a.mu.Unlock()
	p := a.phase
	if p == "" {
		p = "not_installed"
	}
	return AgentStatus{Phase: p, Pid: a.pid, Restarts: a.restarts, Message: a.msg}
}

// set records the status and says the agent fact (key: the profile's game name, else "agent").
func (a *agentMgr) set(phase, msg string, pid int) {
	a.mu.Lock()
	a.phase, a.msg, a.pid = phase, msg, pid
	a.mu.Unlock()
	a.s.say("agent", a.s.agentKey(), a.status())
}

func venvDir(dataDir string, ag *AgentSpec) string {
	return filepath.Join(dataDir, "runtime", "venvs", ag.ID)
}

// agentNoVenv: on Windows the agent runs from the managed base Python with its packages in <env>\site
// (PYTHONPATH), not from a venv. A venv's python.exe there is a small unsigned launcher copy that Smart App
// Control blocked in the first Windows test (os error 4551) while the base interpreter it starts ran fine.
var agentNoVenv = runtime.GOOS == "windows"

// basePythonFile, in the agent env dir, holds the base interpreter's path (no-venv layout).
const basePythonFile = "base-python.txt"

// agentPython is the interpreter that runs the agent.
func agentPython(env string) string {
	if agentNoVenv {
		b, err := os.ReadFile(filepath.Join(env, basePythonFile))
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(b))
	}
	return venvPython(env)
}

func venvPython(venv string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(venv, "Scripts", "python.exe")
	}
	return filepath.Join(venv, "bin", "python")
}

func uvPath(dataDir string) string {
	n := "uv"
	if runtime.GOOS == "windows" {
		n = "uv.exe"
	}
	return filepath.Join(dataDir, "runtime", n)
}

// fileSHA is the sha256 of a file.
func fileSHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// wheelModule reads the first console script of a wheel and returns its module
// ("pkg.cli" from "name = pkg.cli:main").
func wheelModule(wheel string) (string, error) {
	zr, err := zip.OpenReader(wheel)
	if err != nil {
		return "", err
	}
	defer zr.Close()
	for _, f := range zr.File {
		if !strings.HasSuffix(f.Name, ".dist-info/entry_points.txt") {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return "", err
		}
		defer rc.Close()
		sc := bufio.NewScanner(rc)
		in := false
		for sc.Scan() {
			l := strings.TrimSpace(sc.Text())
			if strings.HasPrefix(l, "[") {
				in = l == "[console_scripts]"
				continue
			}
			if in && strings.Contains(l, "=") {
				v := strings.TrimSpace(l[strings.Index(l, "=")+1:])
				v = strings.TrimSpace(strings.SplitN(v, "[", 2)[0])
				return strings.TrimSpace(strings.SplitN(v, ":", 2)[0]), nil
			}
		}
	}
	return "", fmt.Errorf("no console script in %s", filepath.Base(wheel))
}

// agentReady is true when the venv holds the current wheel.
func (s *Supervisor) agentReady() bool {
	ag := s.agentSpec()
	if ag == nil || ag.Wheel == "" {
		return false
	}
	sha, err := fileSHA(ag.Wheel)
	if err != nil {
		return false
	}
	v := venvDir(s.dir, ag)
	b, err := os.ReadFile(filepath.Join(v, ".wheel-sha"))
	if err != nil || strings.TrimSpace(string(b)) != sha {
		return false
	}
	py := agentPython(v)
	if py == "" {
		return false
	}
	_, err = os.Stat(py)
	return err == nil
}

// agentURLEnv names the variables that carry the LLM URL to the agent: the profile's
// llm_url_env, else EASY_ENGINE_LLM_URL and its old twin (internal/legacy).
func (s *Supervisor) agentURLEnv() []string {
	configured := ""
	if ag := s.agentSpec(); ag != nil {
		configured = ag.LLMURLEnv
	}
	return legacy.LLMURLNames(configured, "EASY_ENGINE_LLM_URL")
}

func (s *Supervisor) agentEnv() []string {
	ag := s.agentSpec()
	env := os.Environ()
	for k, v := range ag.Env {
		env = append(env, k+"="+v)
	}
	s.mu.Lock()
	port, mode, ext := s.port, s.st.Mode, s.st.External
	s.mu.Unlock()
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	url := base
	if mode == "external" {
		url = base + "/v1"
	}
	for _, urlEnv := range s.agentURLEnv() {
		if mode == "external" {
			prefix := legacy.LLMPrefix(urlEnv)
			env = append(env, prefix+"_MODE=openai")
			if ext.Model != "" {
				env = append(env, prefix+"_MODEL="+ext.Model)
			}
		}
		env = append(env, urlEnv+"="+url)
	}
	env = append(env, legacy.EngineURLVars("EASY_ENGINE_URL", base)...)
	if agentNoVenv {
		env = append(env, "PYTHONPATH="+filepath.Join(venvDir(s.dir, ag), "site"), "PYTHONNOUSERSITE=1")
	}
	return append(env, s.cfg.AgentEnv...)
}

// startAgent runs the agent (restarting with backoff) if it is installed.
func (s *Supervisor) startAgent() {
	if !s.agentReady() {
		return
	}
	a := &s.agent
	a.mu.Lock()
	if a.cancel != nil {
		a.mu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(s.root)
	done := make(chan struct{})
	a.cancel, a.done = cancel, done
	a.mu.Unlock()
	go func() {
		defer close(done)
		s.agentLoop(ctx)
	}()
}

func (s *Supervisor) stopAgent() {
	a := &s.agent
	a.mu.Lock()
	cancel, done := a.cancel, a.done
	a.cancel, a.done = nil, nil
	a.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
	a.set("stopped", "", 0)
}

func (s *Supervisor) restartAgent() {
	if s.cfg.NoAutoStart {
		return
	}
	s.stopAgent()
	s.startAgent()
}

// agentWanted says whether the profile has an agent the engine installs (a wheel).
func (s *Supervisor) agentWanted() bool {
	ag := s.agentSpec()
	return ag != nil && ag.Wheel != ""
}

// installAgent installs the agent by itself, as a job of its own: uv, its Python and the wheel,
// with no model and no model server. The model install job installs the agent on its way, but
// in external mode no model is ever installed, and a data dir can lose its agent while its model
// stays; without this the agent stayed not_installed for good and nothing answered the game.
//
// It shows the phase and progress the install job shows ("Installing the agent"), and its disk
// need is the agent's alone. A failure is said in the status errors and in the agent's own
// status and blocks nothing: the phase settles as if no job had run. It does nothing when there
// is no agent to install, when it is installed, or when another job runs (that job installs it).
// It returns when the attempt has ended: whether to try again (a failure that was not a cancel),
// for installAgentThen, which waits for a word before it does. w is its listener: a failure
// counts it as waiting before the status says so, so a request seen after the status is heard.
func (s *Supervisor) installAgent(w *word) (again bool) {
	if !s.agentWanted() || s.agentReady() {
		return false
	}
	s.mu.Lock()
	if s.jobRunning {
		s.mu.Unlock()
		return false
	}
	s.jobRunning = true
	ctx, cancel := context.WithCancel(s.root)
	s.jobCancel = cancel
	s.refreshLocked() // the job fact
	s.mu.Unlock()
	defer cancel()
	err := func() error {
		cat, err := s.catalog()
		if err != nil {
			return err
		}
		if err := s.checkNeed(plan.NeedOf(s.agentSteps(cat), s.logRoom(), s.hostFiles()), "the agent"); err != nil {
			return err
		}
		return s.ensureAgent(ctx, cat)
	}()
	s.mu.Lock()
	s.jobRunning = false
	s.jobCancel = nil
	s.progress = nil
	sayPtr(s, "install.progress", "", s.progress)
	s.refreshLocked()               // the job fact
	mine := s.phase == "installing" // ensureAgent's own phase: nothing else took the phase since
	s.mu.Unlock()
	switch {
	case err == nil:
		s.logf("Agent install finished")
	case errors.Is(err, context.Canceled) || ctx.Err() != nil:
		s.agent.set("", "", 0)
		s.addError("Agent install cancelled")
	default:
		w.count() // it waits now: a request is a word for it, before the status says so
		s.agent.set("error", "install failed: "+err.Error()+" (waiting for a request, the network or a word from a client)", 0)
		s.addError("Agent install failed: %v", err)
		again = true
	}
	if mine {
		s.settle()
	}
	return again
}

// installAgentThen installs a missing agent (installAgent), then calls then (start it, start the
// model); a failed install does not hold then back. It tries again with no cadence: after a
// failure it waits for a word that could change the outcome (a request at the door, a client's
// settings, install or nudge, the network coming back), then runs again, until it is installed,
// is cancelled, or the engine ends. Run it in a goroutine counted in s.wg.
func (s *Supervisor) installAgentThen(then func()) {
	for {
		w, err := s.listenQuiet() // before the attempt, so a word said while it runs is heard
		if err != nil {
			if then != nil {
				then()
			}
			return
		}
		again := s.installAgent(w)
		if then != nil {
			then()
			then = nil
		}
		if !again {
			w.Close()
			return
		}
		w.count() // already counted when the failure was said; this holds for any other path
		if s.awaitWord(s.root, w) != nil {
			return
		}
		if s.agentReady() || !s.agentWanted() {
			return
		}
		s.agent.set("", "", 0) // trying again: the install's own status takes over
		then = s.startAgent    // an install that works starts the agent, as the first would have
	}
}

// restartAgentInstalled restarts the agent, installing it first when the profile has one that
// is not installed (external mode: the settings are the only word that it is wanted). The
// install runs in the background; a settings change is also how a failed install is tried again.
func (s *Supervisor) restartAgentInstalled() {
	if s.cfg.NoAutoStart {
		return
	}
	s.stopAgent()
	if !s.agentWanted() || s.agentReady() {
		s.startAgent()
		return
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.installAgentThen(s.startAgent)
	}()
}

func (s *Supervisor) agentLoop(ctx context.Context) {
	ag := s.agentSpec()
	a := &s.agent
	module := ag.Module
	if module == "" {
		if m, err := wheelModule(ag.Wheel); err == nil {
			module = m
		} else {
			module = strings.ReplaceAll(ag.ID, "-", "_")
		}
	}
	// quick counts failures in a row: a run shorter than AgentStable (a launch that failed is a
	// run of no length). The first is restarted at once; from the second on the loop waits for a
	// word that could change the outcome (docs/EVENTS.md section 3).
	quick := 0
	for ctx.Err() == nil {
		sink, _ := lifecycle.OpenLog(filepath.Join(s.dir, "logs", "agent.log"))
		args := append([]string{"-m", module}, ag.Args...)
		started := s.cfg.Now()
		var why string // what ended the run, in plain words
		proc, err := s.cfg.Group.Launch(lifecycle.Spec{Name: "agent", Path: agentPython(venvDir(s.dir, ag)), Args: args, Env: s.agentEnv(), Dir: s.dir, Log: sink})
		if err != nil {
			why = fmt.Sprintf("could not start: %v", err)
			s.addError("agent start: %v", err)
		} else {
			a.mu.Lock()
			a.proc = proc
			a.mu.Unlock()
			a.set("running", "", proc.Pid)
			select {
			case <-proc.Done():
			case <-ctx.Done():
				proc.Stop(s.cfg.StopGrace)
			}
			a.mu.Lock()
			a.proc = nil
			a.mu.Unlock()
			why = fmt.Sprintf("exited (%v)", proc.Err())
			if ctx.Err() == nil {
				s.addError("agent %s", why)
			}
		}
		if sink != nil {
			sink.Close()
		}
		if ctx.Err() != nil {
			return
		}
		if s.cfg.Now().Sub(started) >= s.cfg.AgentStable {
			quick = 0
		}
		quick++
		a.mu.Lock()
		a.restarts++
		a.mu.Unlock()
		if quick < 2 {
			a.set("restarting", "agent "+why+": restarting at once", 0)
			continue
		}
		w, err := s.listenWord() // listen before saying that we wait
		if err != nil {
			return
		}
		a.set("error", "agent "+why+" twice soon after starting: waiting for a request or a word from a client", 0)
		if s.awaitWord(ctx, w) != nil {
			return
		}
	}
}
