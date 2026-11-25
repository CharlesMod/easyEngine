package lifecycle

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"
)

// Spec describes a child to start.
type Spec struct {
	Path string
	Args []string
	Name string   // "llama-server" | "agent": recorded in engine.lock
	Env  []string // full environment; nil means inherit
	Dir  string
	Log  *LogSink // stdout and stderr go here (nil: discarded)
}

// Proc is a running child.
type Proc struct {
	Pid   int
	Start string // process start time token (for pid-reuse-proof checks)
	cmd   *exec.Cmd
	done  chan struct{}
	err   error
	log   *LogSink
	name  string
	g     *Group
}

// Done is closed when the child has exited.
func (p *Proc) Done() <-chan struct{} { return p.done }

// Err is the exit error, valid after Done is closed.
func (p *Proc) Err() error { return p.err }

// Exited reports whether the child has exited.
func (p *Proc) Exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

// Tail returns the child's recent output.
func (p *Proc) Tail() string {
	if p.log == nil {
		return ""
	}
	return p.log.Tail()
}

// Stop asks the child (and its process group) to end, then kills it after grace.
func (p *Proc) Stop(grace time.Duration) {
	if p.Exited() {
		return
	}
	terminate(p, false)
	select {
	case <-p.done:
	case <-time.After(grace):
		terminate(p, true)
		select {
		case <-p.done:
		case <-time.After(5 * time.Second):
		}
	}
}

// Kill ends the child at once.
func (p *Proc) Kill() {
	if p.Exited() {
		return
	}
	terminate(p, true)
	select {
	case <-p.done:
	case <-time.After(5 * time.Second):
	}
}

// Group tracks children so they can all be killed on any exit path.
type Group struct {
	mu       sync.Mutex
	procs    map[int]*Proc
	OnChange func(children []ChildRef) // called (outside the lock) when the set changes
}

// ChildRef identifies a child for engine.lock.
type ChildRef struct {
	Pid   int    `json:"pid"`
	Start string `json:"start"`
	Name  string `json:"name,omitempty"`
}

func NewGroup() *Group { return &Group{procs: map[int]*Proc{}} }

// Refs lists live children.
func (g *Group) Refs() []ChildRef {
	g.mu.Lock()
	defer g.mu.Unlock()
	var out []ChildRef
	for _, p := range g.procs {
		out = append(out, ChildRef{Pid: p.Pid, Start: p.Start, Name: p.name})
	}
	return out
}

func (g *Group) changed() {
	if g.OnChange != nil {
		g.OnChange(g.Refs())
	}
}

// Launch starts a child that dies with the engine.
func (g *Group) Launch(spec Spec) (*Proc, error) {
	if spec.Path == "" {
		return nil, errors.New("empty path")
	}
	cmd := exec.Command(spec.Path, spec.Args...)
	cmd.Dir = spec.Dir
	if spec.Env != nil {
		cmd.Env = spec.Env
	}
	if spec.Log != nil {
		cmd.Stdout, cmd.Stderr = spec.Log, spec.Log
	}
	if devnull, err := os.Open(os.DevNull); err == nil {
		cmd.Stdin = devnull
		defer devnull.Close()
	}
	prepare(cmd)
	if err := startChild(cmd); err != nil {
		return nil, fmt.Errorf("start %s: %w", spec.Path, err)
	}
	p := &Proc{Pid: cmd.Process.Pid, cmd: cmd, done: make(chan struct{}), log: spec.Log, name: spec.Name, g: g}
	p.Start, _ = PidStart(p.Pid)
	adopt(p)
	g.mu.Lock()
	g.procs[p.Pid] = p
	g.mu.Unlock()
	g.changed()
	go func() {
		p.err = cmd.Wait()
		close(p.done)
		g.mu.Lock()
		delete(g.procs, p.Pid)
		g.mu.Unlock()
		g.changed()
	}()
	return p, nil
}

// KillAll kills every live child at once.
func (g *Group) KillAll() {
	g.mu.Lock()
	var ps []*Proc
	for _, p := range g.procs {
		ps = append(ps, p)
	}
	g.mu.Unlock()
	for _, p := range ps {
		p.Kill()
	}
}
