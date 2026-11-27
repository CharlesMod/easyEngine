package yield

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
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
	ino  int // inotify fd, -1 when unavailable
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

// NewOS watches the real machine. On Linux, device opens wake Wait through inotify IN_OPEN
// on /dev/nvidia<N>, /dev/dri/renderD* and /dev/dri/card*.
func NewOS() OS { return newLinuxOS("/proc", "/dev") }

func newLinuxOS(root, dev string) *linuxOS {
	o := &linuxOS{root: root, ino: -1, seen: map[int]seenProc{}, now: time.Now}
	fd, err := unix.InotifyInit1(unix.IN_CLOEXEC | unix.IN_NONBLOCK)
	if err != nil {
		return o
	}
	n := 0
	for _, pat := range []string{"nvidia[0-9]*", "dri/renderD*", "dri/card*"} {
		ms, _ := filepath.Glob(filepath.Join(dev, pat))
		for _, m := range ms {
			if _, err := unix.InotifyAddWatch(fd, m, unix.IN_OPEN); err == nil {
				n++
			}
		}
	}
	if n == 0 {
		unix.Close(fd)
		return o
	}
	o.ino = fd
	return o
}

func (o *linuxOS) Close() {
	if o.ino >= 0 {
		unix.Close(o.ino)
		o.ino = -1
	}
}

func (o *linuxOS) Drain() {
	if o.ino < 0 {
		return
	}
	buf := make([]byte, 4096)
	for {
		if n, err := unix.Read(o.ino, buf); n <= 0 || err != nil {
			return
		}
	}
}

func (o *linuxOS) Wait(ctx context.Context, d time.Duration) bool {
	if o.ino < 0 {
		return sleepCtx(ctx, d)
	}
	end := time.Now().Add(d)
	for {
		left := time.Until(end)
		if left <= 0 || ctx.Err() != nil {
			return false
		}
		ms := int(min(left, 250*time.Millisecond) / time.Millisecond)
		fds := []unix.PollFd{{Fd: int32(o.ino), Events: unix.POLLIN}}
		n, err := unix.Poll(fds, max(ms, 1))
		if err != nil && err != syscall.EINTR {
			return sleepCtx(ctx, left)
		}
		if n > 0 {
			o.Drain()
			return true
		}
	}
}

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

// Launches is cheap enough for every 250 ms: one readdir of /proc, and cmdline reads only
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

func (o *linuxOS) Openers(skip func(pid int) bool) map[int]string {
	out := map[int]string{}
	for _, pid := range o.pids() {
		if skip != nil && skip(pid) {
			continue
		}
		dir := filepath.Join(o.root, strconv.Itoa(pid), "fd")
		ents, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range ents {
			if l, err := os.Readlink(filepath.Join(dir, e.Name())); err == nil && gpuNode(l) {
				out[pid] = o.name(pid)
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
