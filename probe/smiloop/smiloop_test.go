package smiloop

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

// fakeSMI is an nvidia-smi that prints one block for each word written to its control fifo
// (csv, xml, pids), ends on "exit", and logs every start to ./spawns. Without -lms it is a
// one-shot: it prints the csv block's line and ends.
const fakeSMI = `#!/bin/sh
d=$(dirname "$0")
echo "$$ $*" >> "$d/spawns"
case "$*" in
  *-lms*) ;;
  *) echo "0, Fake GPU, 16384, 4321"; exit 0 ;;
esac
case "$1$2" in
  --query-gpu*) k=csv ;;
  -q-x) k=xml ;;
  *) k=pids ;;
esac
while read -r w < "$d/ctl.$k"; do
  case "$w" in
    csv) echo "1, 0, Fake GPU, 16384, 1234" ;;
    xml) printf '<?xml version="1.0" ?>\n<nvidia_smi_log>\n<gpu/>\n</nvidia_smi_log>\n' ;;
    pids) printf '\n==============NVSMI LOG==============\n\nAttached GPUs : 1\nGPU 00000000:01:00.0\n    Processes\n        Process ID                        : 7\n            Type                          : G\n\n' ;;
    exit) exit 0 ;;
  esac
done
`

type rig struct {
	t     *testing.T
	dir   string
	m     *Manager
	heard chan Kind
	ended chan Kind
}

func newRig(t *testing.T) *rig {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("a shell fake")
	}
	dir := t.TempDir()
	for _, k := range []string{"csv", "xml", "pids"} { // one control fifo per query
		if err := syscall.Mkfifo(filepath.Join(dir, "ctl."+k), 0o600); err != nil {
			t.Skip(err)
		}
	}
	path := filepath.Join(dir, "nvidia-smi")
	if err := os.WriteFile(path, []byte(fakeSMI), 0o755); err != nil {
		t.Fatal(err)
	}
	r := &rig{t: t, dir: dir, heard: make(chan Kind, 64), ended: make(chan Kind, 64)}
	r.m = NewManager(Config{
		Path:  path,
		Heard: func(k Kind) { r.heard <- k },
		Ended: func(k Kind) { r.ended <- k },
		Logf:  func(f string, a ...any) { t.Logf(f, a...) },
	})
	t.Cleanup(r.m.Close)
	return r
}

// say makes the fake of query w print its block (the child must be alive and waiting); "exit"
// goes to the csv child.
func (r *rig) say(w string) {
	k := w
	if w == "exit" {
		k = "csv"
	}
	f, err := os.OpenFile(filepath.Join(r.dir, "ctl."+k), os.O_WRONLY, 0)
	if err != nil {
		r.t.Fatal(err)
	}
	f.WriteString(w + "\n")
	f.Close()
}

func (r *rig) spawns() int {
	b, _ := os.ReadFile(filepath.Join(r.dir, "spawns"))
	return strings.Count(string(b), "\n")
}

func (r *rig) proc(k Kind, fields string) interface{ Exited() bool } {
	rd := r.m.readers[k.String()+"|"+fields]
	rd.mu.Lock()
	defer rd.mu.Unlock()
	return rd.proc
}

var (
	csvArgs  = []string{"--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"}
	xmlArgs  = []string{"-q", "-x"}
	pidsArgs = []string{"-q", "-d", "PIDS"}
)

func TestManyReadsOneChild(t *testing.T) {
	r := newRig(t)
	release := r.m.Hold()
	defer release()
	if _, ok := r.m.Read(csvArgs); ok {
		t.Fatal("a read before the first block is the one-shot's")
	}
	if r.m.Starts() != 1 { // the child is started by the first read
		t.Fatal("no child started by the first read")
	}
	r.say("csv")
	<-r.heard
	for i := 0; i < 20; i++ {
		b, ok := r.m.Read(csvArgs)
		if !ok || string(b) != "0, Fake GPU, 16384, 1234\n" {
			t.Fatalf("read %d: %q %v", i, b, ok)
		}
	}
	if n := r.spawns(); n != 1 {
		t.Fatalf("21 reads started %d children", n)
	}
	// a different query is another child, another shape of block
	if _, ok := r.m.Read(xmlArgs); ok {
		t.Fatal("xml before its block")
	}
	r.say("xml")
	<-r.heard
	b, ok := r.m.Read(xmlArgs)
	if !ok || !strings.HasPrefix(string(b), "<?xml") || !strings.Contains(string(b), "</nvidia_smi_log>") {
		t.Fatalf("xml block: %q %v", b, ok)
	}
	r.m.Read(pidsArgs)
	r.say("pids")
	<-r.heard
	b, ok = r.m.Read(pidsArgs)
	if !ok || !strings.Contains(string(b), "Process ID") || !strings.HasPrefix(string(b), "==============NVSMI LOG") {
		t.Fatalf("pids block: %q %v", b, ok)
	}
	if n := r.spawns(); n != 3 {
		t.Fatalf("three queries started %d children", n)
	}
}

func TestNotHeldOrNotLoopable(t *testing.T) {
	r := newRig(t)
	if _, ok := r.m.Read(csvArgs); ok || r.m.Starts() != 0 {
		t.Fatal("no user holds the loop: no sample and no child")
	}
	release := r.m.Hold()
	defer release()
	if _, ok := r.m.Read([]string{"-i", "0", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"}); ok || r.m.Starts() != 0 {
		t.Fatal("a query that is not looped is the one-shot's")
	}
}

func TestStaleSampleIsNotUsed(t *testing.T) {
	r := newRig(t)
	defer r.m.Hold()()
	r.m.Read(csvArgs)
	r.say("csv")
	<-r.heard
	if _, ok := r.m.Read(csvArgs); !ok {
		t.Fatal("fresh sample")
	}
	r.m.Stale()
	if _, ok := r.m.Read(csvArgs); ok {
		t.Fatal("a sample read before Stale is not used after it")
	}
	r.say("csv")
	<-r.heard
	if _, ok := r.m.Read(csvArgs); !ok {
		t.Fatal("a block after Stale is")
	}
	if r.spawns() != 1 {
		t.Fatalf("spawns %d", r.spawns())
	}
}

func TestDeadChildRestartedOnceThenOneShot(t *testing.T) {
	r := newRig(t)
	defer r.m.Hold()()
	r.m.Read(csvArgs)
	p1 := r.proc(CSV, "index,name,memory.total,memory.used")
	r.say("exit")
	<-r.ended
	if !p1.Exited() {
		t.Fatal("the first child is gone")
	}
	if _, ok := r.m.Read(csvArgs); ok { // the event that finds it dead starts it again
		t.Fatal("no block yet")
	}
	if n := r.m.Starts(); n != 2 {
		t.Fatalf("restart: %d starts", n)
	}
	r.say("exit")
	<-r.ended
	for i := 0; i < 5; i++ {
		if _, ok := r.m.Read(csvArgs); ok {
			t.Fatal("it gave up: reads are the one-shot's")
		}
	}
	if n := r.m.Starts(); n != 2 || r.spawns() != 2 {
		t.Fatalf("after giving up: %d starts, %d on disk", n, r.spawns())
	}
}

func TestChildGoneWhenLastUserLetsGoAndAtClose(t *testing.T) {
	r := newRig(t)
	a, b := r.m.Hold(), r.m.Hold()
	r.m.Read(csvArgs)
	r.say("csv")
	<-r.heard
	p := r.proc(CSV, "index,name,memory.total,memory.used")
	a()
	a() // twice is once
	if p.Exited() {
		t.Fatal("one user still holds the loop")
	}
	if _, ok := r.m.Read(csvArgs); !ok {
		t.Fatal("still answering")
	}
	b() // returns when the child has ended
	if !p.Exited() {
		t.Fatal("the last user let go: the child must be gone")
	}
	if _, ok := r.m.Read(csvArgs); ok {
		t.Fatal("no user: one-shot")
	}
	// the engine's end
	defer r.m.Hold()()
	r.m.Read(csvArgs)
	q := r.proc(CSV, "index,name,memory.total,memory.used")
	r.m.Close()
	if !q.Exited() {
		t.Fatal("Close ends the child")
	}
}

func TestParsers(t *testing.T) {
	// two GPUs: a csv block is whole at the count
	c := &csvParser{}
	if _, ok := c.feed("2, 0, A, 1, 2"); ok {
		t.Fatal("half")
	}
	b, ok := c.feed("2, 1, B, 3, 4")
	if !ok || string(b) != "0, A, 1, 2\n1, B, 3, 4\n" {
		t.Fatalf("csv: %q", b)
	}
	// a pids block with two GPUs, one without processes
	p := &pidsParser{}
	lines := strings.Split("\n==============NVSMI LOG==============\n\nTimestamp : x\nAttached GPUs : 2\nGPU 0000:01:00.0\n    Processes\n        Process ID : 7\n\nGPU 0000:02:00.0\n    Processes                             : None\n\n", "\n")
	var got []byte
	for _, l := range lines {
		if b, ok := p.feed(l); ok {
			if got != nil {
				t.Fatal("one block")
			}
			got = b
		}
	}
	if got == nil || !strings.Contains(string(got), "0000:02:00.0") || !strings.Contains(string(got), "None") {
		t.Fatalf("pids: %q", got)
	}
}
