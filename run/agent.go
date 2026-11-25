package run

import (
	"archive/zip"
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"hermesengine/lifecycle"
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

func (a *agentMgr) set(phase, msg string, pid int) {
	a.mu.Lock()
	a.phase, a.msg, a.pid = phase, msg, pid
	a.mu.Unlock()
}

func venvDir(dataDir string, ag *AgentSpec) string {
	return filepath.Join(dataDir, "runtime", "venvs", ag.ID)
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
	ag := s.cfg.Profile.Agent
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
	_, err = os.Stat(venvPython(v))
	return err == nil
}

func (s *Supervisor) agentURLEnv() string {
	if ag := s.cfg.Profile.Agent; ag != nil && ag.LLMURLEnv != "" {
		return ag.LLMURLEnv
	}
	return "HERMESVOICE_LLM_URL"
}

func (s *Supervisor) agentEnv() []string {
	ag := s.cfg.Profile.Agent
	env := os.Environ()
	for k, v := range ag.Env {
		env = append(env, k+"="+v)
	}
	s.mu.Lock()
	port, mode, ext := s.port, s.st.Mode, s.st.External
	s.mu.Unlock()
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	urlEnv := s.agentURLEnv()
	prefix := strings.TrimSuffix(urlEnv, "_URL")
	url := base
	if mode == "external" {
		url = base + "/v1"
		env = append(env, prefix+"_MODE=openai")
		if ext.Model != "" {
			env = append(env, prefix+"_MODEL="+ext.Model)
		}
	}
	env = append(env, urlEnv+"="+url, "HERMES_ENGINE_URL="+base)
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

func (s *Supervisor) agentLoop(ctx context.Context) {
	ag := s.cfg.Profile.Agent
	a := &s.agent
	module := ag.Module
	if module == "" {
		if m, err := wheelModule(ag.Wheel); err == nil {
			module = m
		} else {
			module = strings.ReplaceAll(ag.ID, "-", "_")
		}
	}
	backoff := s.cfg.AgentBackoffBase
	for ctx.Err() == nil {
		sink, _ := lifecycle.OpenLog(filepath.Join(s.dir, "logs", "agent.log"))
		args := append([]string{"-m", module}, ag.Args...)
		started := s.cfg.Now()
		proc, err := s.cfg.Group.Launch(lifecycle.Spec{Name: "agent", Path: venvPython(venvDir(s.dir, ag)), Args: args, Env: s.agentEnv(), Dir: s.dir, Log: sink})
		if err != nil {
			s.addError("agent start: %v", err)
			a.set("error", err.Error(), 0)
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
			if ctx.Err() == nil {
				s.addError("agent exited (%v): restarting in %s", proc.Err(), backoff)
			}
		}
		if sink != nil {
			sink.Close()
		}
		if ctx.Err() != nil {
			return
		}
		if s.cfg.Now().Sub(started) >= s.cfg.AgentStable {
			backoff = s.cfg.AgentBackoffBase
		}
		a.mu.Lock()
		a.restarts++
		a.mu.Unlock()
		a.set("restarting", "restarting in "+backoff.String(), 0)
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > s.cfg.AgentBackoffMax {
			backoff = s.cfg.AgentBackoffMax
		}
	}
}
