package run

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
	"github.com/CharlesMod/easyEngine/probe"
)

// ErrNoSpace wraps every refusal for lack of disk space.
var ErrNoSpace = errors.New("not enough disk space")

// rotatingLogs are the engine's capped logs under logs/ (lifecycle.LogSink: each file and its
// two old copies).
var rotatingLogs = []string{"engine.log", "llama-server.log", "agent.log"}

// sinkWriteMax is the most one write to a LogSink adds past the cap before the next write
// rotates: a child's output reaches the sink in os/exec's 32 KiB pipe reads; the engine's own
// lines are far shorter.
const sinkWriteMax = 32 << 10

func roundBlock(n int64) int64 { return (n + 4095) / 4096 * 4096 }

// logCeiling is the most one rotating log takes on disk: the file and two old copies, each at
// most the cap plus one write.
func logCeiling() int64 { return 3 * roundBlock(lifecycle.MaxLogSize+sinkWriteMax) }

// logRoom is the disk the logs may still take before every one sits at its ceiling, from their
// current sizes: the rotating logs and the stdio log (app.detachStdio, lifecycle.StdioLog).
func (s *Supervisor) logRoom() int64 {
	dir := filepath.Join(s.dir, "logs")
	var room int64
	have := func(name string, copies int) int64 {
		var n int64
		for i := 0; i <= copies; i++ {
			p := filepath.Join(dir, name)
			if i > 0 {
				p = fmt.Sprintf("%s.%d", p, i)
			}
			if st, err := os.Stat(p); err == nil {
				n += roundBlock(st.Size())
			}
		}
		return n
	}
	for _, name := range rotatingLogs {
		room += max(logCeiling()-have(name, 2), 0)
	}
	room += max(lifecycle.StdioLogCeiling()-have(lifecycle.StdioLogName, 1), 0)
	return room
}

// hostFiles is what the profile declares the host's own files may still grow by.
func (s *Supervisor) hostFiles() int64 { return int64(s.cfg.Profile.HostFilesMiB) << 20 }

// setupNeed is the free disk the setup of models (in install order: a starter, then the
// target) needs from the current state, counting only what is missing: the llama.cpp build
// (backendID, or the one an install would pick first) with its extras, each model's download
// less its .part, uv, the agent's runtime, then the logs to their caps, the host's files and
// plan.DiskHeadroomMiB. Every disk check and the plan's disk verdict use it.
func (s *Supervisor) setupNeed(cat *catalog.Catalog, hw probe.Hardware, backendID string, models ...catalog.Model) plan.DiskNeed {
	var steps []plan.DiskStep
	steps = append(steps, s.backendSteps(cat, hw, backendID)...)
	for _, m := range models {
		if s.modelInstalled(m) {
			continue
		}
		var part int64
		if st, err := os.Stat(s.modelPath(m) + ".part"); err == nil {
			part = st.Size()
		}
		steps = append(steps, plan.ModelStep(m.Name, m.Size, part))
	}
	steps = append(steps, s.agentSteps(cat)...)
	return plan.NeedOf(steps, s.logRoom(), s.hostFiles())
}

// backendSteps are the steps of the llama.cpp build an install would put in place, or none when
// it is installed. An installed build the state names (a past install) counts as in place.
func (s *Supervisor) backendSteps(cat *catalog.Catalog, hw probe.Hardware, want string) []plan.DiskStep {
	if want == "" {
		s.mu.Lock()
		cur, tag := s.st.Backend, s.st.Build
		s.mu.Unlock()
		if cur != "" && backendInstalled(s.dir, cur, tag) {
			return nil
		}
		if bs := cat.Backends(hw); len(bs) > 0 {
			want = bs[0].ID
		}
	}
	a, ok := cat.Asset(want)
	if !ok || backendInstalled(s.dir, a.ID, a.Tag) {
		return nil
	}
	var extras []catalog.Asset
	for _, x := range a.Extra {
		if ea, ok := cat.LlamaAsset(a.Tag, x); ok {
			extras = append(extras, ea)
		}
	}
	return plan.BackendSteps(a, extras)
}

// backendInstalled reports whether build id@tag is installed whole (backendDir, its marker and
// its server).
func backendInstalled(dataDir, id, tag string) bool {
	dir := backendDir(dataDir, id, tag)
	_, err := os.Stat(filepath.Join(dir, ".installed"))
	return err == nil && findServer(dir) != ""
}

// agentSteps are uv and the agent's runtime when the profile has an agent that is not set up.
func (s *Supervisor) agentSteps(cat *catalog.Catalog) []plan.DiskStep {
	ag := s.cfg.Profile.Agent
	if ag == nil || ag.Wheel == "" || s.agentReady() {
		return nil
	}
	var steps []plan.DiskStep
	if _, err := os.Stat(uvPath(s.dir)); err != nil {
		if a, ok := cat.UVFor(runtime.GOOS, runtime.GOARCH); ok {
			steps = append(steps, plan.UVStep(a))
		}
	}
	rt := agentRuntime(cat, ag)
	rtDir := filepath.Join(s.dir, "runtime")
	// What of it is on disk already: a reinstall removes the venv first and reuses the rest.
	have := diskUsage(filepath.Join(rtDir, "python"), filepath.Join(rtDir, "uv-cache"), venvDir(s.dir, ag))
	return append(steps, plan.AgentStep(rt, have))
}

// agentRuntime is the agent setup's measured size: the profile's own figure, else the
// catalog's measurement for this OS, else the largest measurement the catalog has.
func agentRuntime(cat *catalog.Catalog, ag *AgentSpec) catalog.RuntimeSize {
	if ag.DiskBytes > 0 {
		return catalog.RuntimeSize{PeakBytes: ag.DiskBytes, KeepBytes: ag.DiskBytes}
	}
	if rt, ok := cat.RuntimeFor(runtime.GOOS, runtime.GOARCH); ok {
		return rt
	}
	var big catalog.RuntimeSize
	for _, r := range cat.AgentRuntime {
		if r.PeakBytes > big.PeakBytes {
			big = r
		}
	}
	return big
}

// diskUsage is the disk the trees under paths take, each hard-linked file once (uv links the
// venv's packages to its cache).
func diskUsage(paths ...string) int64 {
	var total int64
	seen := map[fileKey]bool{}
	for _, root := range paths {
		filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			fi, err := d.Info()
			if err != nil {
				return nil
			}
			n, key, linked := allocOf(p, fi)
			if linked {
				if seen[key] {
					return nil
				}
				seen[key] = true
			}
			total += n
			return nil
		})
	}
	return total
}

// checkNeed refuses when the free disk is short of n, naming what needs it. A free space that
// cannot be read is said in the status (an error line) and does not block.
func (s *Supervisor) checkNeed(n plan.DiskNeed, what string) error {
	if !n.Downloads() {
		return nil
	}
	free, err := s.cfg.DiskFree(s.dir)
	if err != nil {
		s.addError("could not read the free disk space (%v): installing %s without a space check", err, what)
		return nil
	}
	fb := int64(free) << 20
	if kb := s.kept.bytes(); n.Short(fb) && kb > 0 { // the disk runs short: kept contexts go first
		s.letGoContexts("the disk runs short (" + what + ")")
		if f2, err := s.cfg.DiskFree(s.dir); err == nil {
			fb = int64(f2) << 20
		}
	}
	if n.Short(fb) {
		return fmt.Errorf("%w for %s: %s, %s free (%s short)", ErrNoSpace, what, n.Explain(), plan.GB(fb), plan.GB(n.Total-fb))
	}
	return nil
}

// liveDisk returns hw with DiskFreeMiB read now (0 when unreadable: no disk verdict).
func (s *Supervisor) liveDisk(hw probe.Hardware) probe.Hardware {
	hw.DiskFreeMiB = 0
	if free, err := s.cfg.DiskFree(s.dir); err == nil {
		hw.DiskFreeMiB = max(free, 1) // 0 would read as unknown
	}
	return hw
}
