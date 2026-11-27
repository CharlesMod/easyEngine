package yield

import (
	"bytes"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/CharlesMod/easyEngine/host"
)

// gpuNode says whether a /proc/<pid>/fd link names a GPU device a program must open before it
// can allocate: NVIDIA's per-card nodes and DRM render or card nodes. nvidiactl alone is not
// enough (NVML monitors open it), nor is nvidia-uvm without a card node.
func gpuNode(link string) bool {
	if rest, ok := strings.CutPrefix(link, "/dev/nvidia"); ok {
		return digits(rest)
	}
	return strings.HasPrefix(link, "/dev/dri/renderD") || strings.HasPrefix(link, "/dev/dri/card")
}

type linuxOS struct {
	root string
	sys  string // /sys and /proc/driver roots, for mapping nodes to an adapter
	drv  string
	dev  string

	mu    sync.Mutex      // pci and nodes: Adapter (the level's read) and Watch (the reactor) meet
	nodes map[string]bool // the engine adapter's nodes; nil: any GPU node
	pci   string

	seen map[int]seenProc
	now  func() time.Time
}

// seenProc caches what a pid's argv said, so a scan reads only new pids' cmdline. A pid is
// read again while it is young: a launcher forks, then execs the game's wrapper.
type seenProc struct {
	first  time.Time
	launch *Launch
}

const youngProc = 2 * time.Second

// NewOS watches the real machine. On Linux, device opens are heard by Watch (host.Opens) on
// /dev/nvidia<N>, /dev/dri/renderD* and /dev/dri/card*.
func NewOS() OS {
	o := newLinuxOS("/proc", "/dev")
	o.sys, o.drv = "/sys/class/drm", "/proc/driver/nvidia/gpus"
	return o
}

func newLinuxOS(root, dev string) *linuxOS {
	return &linuxOS{root: root, dev: dev, seen: map[int]seenProc{}, now: time.Now}
}

func (o *linuxOS) Close() {}

// Nodes are the device nodes whose opens say another program took the card: the engine's
// adapter's (once Adapter knew it), else every NVIDIA and DRM node under /dev.
func (o *linuxOS) Nodes() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.nodesLocked()
}

func (o *linuxOS) nodesLocked() []string {
	if len(o.nodes) > 0 {
		out := make([]string, 0, len(o.nodes))
		for n := range o.nodes {
			out = append(out, n)
		}
		sort.Strings(out)
		return out
	}
	var out []string
	for _, pat := range []string{"nvidia[0-9]*", "dri/renderD*", "dri/card*"} {
		ms, _ := filepath.Glob(filepath.Join(o.dev, pat))
		for _, m := range ms {
			if gpuNode(filepath.Join("/dev", strings.TrimPrefix(m, o.dev))) {
				out = append(out, m)
			}
		}
	}
	sort.Strings(out)
	return out
}

// Watch is host.Opens on Nodes: no time cap, the kernel says when a program opens one.
func (o *linuxOS) Watch() Words {
	paths := o.Nodes()
	if len(paths) == 0 {
		return Words{}
	}
	ch, stop, err := host.Opens(paths)
	if err != nil {
		return Words{}
	}
	return Words{Opens: ch, Stop: stop, Set: strings.Join(paths, ",")}
}

// NodeSet names the nodes Watch would watch now, in the form Words.Set has: the caller compares
// the two to know the watch is on other nodes than the adapter's now.
func (o *linuxOS) NodeSet() string { return strings.Join(o.Nodes(), ",") }

func (o *linuxOS) pids() []int {
	ents, err := os.ReadDir(o.root)
	if err != nil {
		return nil
	}
	var out []int
	for _, e := range ents {
		if pid, err := strconv.Atoi(e.Name()); err == nil {
			out = append(out, pid)
		}
	}
	return out
}

// Launches is cheap (the host watchdog reads it every period): one readdir of /proc, and cmdline reads only
// for pids that are new or young.
func (o *linuxOS) Launches() []Launch {
	now := o.now()
	live := map[int]bool{}
	var out []Launch
	for _, pid := range o.pids() {
		live[pid] = true
		sp, ok := o.seen[pid]
		if !ok {
			sp.first = now
		}
		if !ok || now.Sub(sp.first) < youngProc {
			sp.launch = nil
			b, err := os.ReadFile(filepath.Join(o.root, strconv.Itoa(pid), "cmdline"))
			if err == nil && len(b) > 0 {
				argv := strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
				if l, ok := MatchArgv(argv); ok {
					l.PID = pid
					sp.launch = &l
				}
			}
			o.seen[pid] = sp
		}
		if sp.launch != nil {
			out = append(out, *sp.launch)
		}
	}
	for pid := range o.seen {
		if !live[pid] {
			delete(o.seen, pid)
		}
	}
	return out
}

// Adapter maps the PCI address to its nodes: the NVIDIA minor from
// /proc/driver/nvidia/gpus/<pci>/information, DRM nodes whose /sys/class/drm/<node>/device
// resolves to that address.
func (o *linuxOS) Adapter(pci string) {
	pci = strings.ToLower(strings.TrimSpace(pci))
	o.mu.Lock()
	same, known := pci == o.pci, o.nodes != nil
	o.mu.Unlock()
	if same && known { // an address whose nodes were not found is looked at again: they may appear
		return
	}
	var found map[string]bool
	if pci != "" && o.sys != "" {
		found = o.adapterNodes(pci)
	}
	o.mu.Lock()
	o.pci, o.nodes = pci, found
	o.mu.Unlock()
}

// adapterNodes reads the nodes of the adapter at pci (nil: none found).
func (o *linuxOS) adapterNodes(pci string) map[string]bool {
	nodes := map[string]bool{}
	if b, err := os.ReadFile(filepath.Join(o.drv, pci, "information")); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if v, ok := strings.CutPrefix(line, "Device Minor:"); ok {
				nodes["/dev/nvidia"+strings.TrimSpace(v)] = true
			}
		}
	}
	ents, _ := os.ReadDir(o.sys)
	for _, e := range ents {
		n := e.Name()
		if !strings.HasPrefix(n, "card") && !strings.HasPrefix(n, "renderD") || strings.Contains(n, "-") {
			continue
		}
		if dev, err := filepath.EvalSymlinks(filepath.Join(o.sys, n, "device")); err == nil && strings.EqualFold(filepath.Base(dev), pci) {
			nodes["/dev/dri/"+n] = true
		}
	}
	if len(nodes) > 0 {
		return nodes
	}
	return nil
}

func (o *linuxOS) wanted(link string) bool {
	if !gpuNode(link) {
		return false
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.nodes == nil || o.nodes[link]
}

func (o *linuxOS) Openers(skip func(pid int) bool) map[int]string {
	out := map[int]string{}
	for _, pid := range o.pids() {
		dir := filepath.Join(o.root, strconv.Itoa(pid), "fd")
		ents, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range ents {
			if l, err := os.Readlink(filepath.Join(dir, e.Name())); err == nil && o.wanted(l) {
				// skip is asked only of a holder: it may walk the process tree (Under), which
				// costs more than the one readdir that rules out nearly every process
				if skip == nil || !skip(pid) {
					out[pid] = o.name(pid)
				}
				break
			}
		}
	}
	return out
}

func (o *linuxOS) name(pid int) string {
	b, _ := os.ReadFile(filepath.Join(o.root, strconv.Itoa(pid), "comm"))
	return strings.TrimSpace(string(b))
}

// parent reads the ppid from /proc/<pid>/stat (the name in parentheses may hold spaces).
func (o *linuxOS) parent(pid int) int {
	b, err := os.ReadFile(filepath.Join(o.root, strconv.Itoa(pid), "stat"))
	if err != nil {
		return 0
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return 0
	}
	f := strings.Fields(string(b[i+1:]))
	if len(f) < 2 {
		return 0
	}
	pp, _ := strconv.Atoi(f[1])
	return pp
}

func (o *linuxOS) Under(pid, anc int) bool {
	for i := 0; i < 64 && pid > 1; i++ {
		if pid == anc {
			return true
		}
		pid = o.parent(pid)
	}
	return pid == anc
}
