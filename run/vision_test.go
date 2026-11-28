//go:build !windows

package run_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
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
	"github.com/CharlesMod/easyEngine/probe"
	"github.com/CharlesMod/easyEngine/run"
)

// Vision (run/vision.go): the served model's projector loaded beside it. Every wait is on a fact
// (status, the hub, starts.log), never on a clock.

type visEnv struct {
	e    *testkit.Env
	proj []byte
	p    *catalog.Projector
	file string // the projector's path in models/
}

// newVis installs the fake and gives model m1 a projector of the given load; the projector file
// is not written (write it with put).
func newVis(t *testing.T, loadMiB int) *visEnv {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	v := &visEnv{e: e, proj: []byte("not a real projector, the fake server only needs the file")}
	sum := sha256.Sum256(v.proj)
	v.p = &catalog.Projector{File: "m1-mmproj.gguf", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(v.proj)),
		LoadMiB: loadMiB, ImageMiB: 200, MaxImageTokens: 512, Detail: catalog.Detail{Low: 256, High: 512, Max: 1024}}
	e.Cat.Models[0].Projector = v.p
	v.file = filepath.Join(e.Data, "models", v.p.File)
	return v
}

func (v *visEnv) put(t *testing.T) {
	t.Helper()
	if err := os.WriteFile(v.file, v.proj, 0o644); err != nil {
		t.Fatal(err)
	}
}

func (v *visEnv) start(t *testing.T, mod func(*run.Config)) *run.Supervisor {
	t.Helper()
	cfg := v.e.Config()
	if mod != nil {
		mod(&cfg)
	}
	s := v.e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	return s
}

func (v *visEnv) hasMM(line string) bool { return strings.Contains(line, "--mmproj "+v.file) }

func visSet(t *testing.T, s *run.Supervisor, on bool) {
	t.Helper()
	if err := s.SetSettings(run.Settings{Vision: &on}); err != nil {
		t.Fatal(err)
	}
}

// visReady waits for a ready server whose vision is on (or off).
func visReady(t *testing.T, s *run.Supervisor, on bool) run.Status {
	t.Helper()
	return waitStatus(t, s, "ready with vision "+map[bool]string{true: "on", false: "off"}[on], func(st run.Status) bool {
		return st.Phase == "ready" && st.Vision == on && st.Shape != nil
	})
}

func visFact(t *testing.T, s *run.Supervisor) run.VisionInfo {
	t.Helper()
	f, ok := s.Hub().Fact("vision", "")
	if !ok {
		t.Fatal("no vision fact")
	}
	var vi run.VisionInfo
	if err := json.Unmarshal(f.Body, &vi); err != nil {
		t.Fatal(err)
	}
	return vi
}

func waitStarts(t *testing.T, e *testkit.Env, n int) {
	t.Helper()
	testkit.WaitFor(t, 20*time.Second, "the server to be started again", func() bool { return len(e.Starts()) >= n })
}

// 1. Off by default.
func TestVisionOffByDefault(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, nil)
	st := s.Status()
	if st.Vision || st.VisionNote != "" || st.VisionProgress != nil {
		t.Fatalf("status: vision %v note %q progress %v", st.Vision, st.VisionNote, st.VisionProgress)
	}
	if strings.Contains(v.e.Starts()[0], "--mmproj") {
		t.Fatalf("--mmproj without a want: %s", v.e.Starts()[0])
	}
	if vi := visFact(t, s); vi.Vision || vi.Note != "" {
		t.Fatalf("fact %+v", vi)
	}
	if k := s.CtxKeyString(); strings.Contains(k, "mmproj") {
		t.Fatalf("ctx key %q", k)
	}
}

// 2. The setting on reloads the idle server with the projector; off unloads it.
func TestVisionSettingOnAndOff(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, nil)
	visSet(t, s, true)
	st := visReady(t, s, true)
	if st.VisionNote != "" {
		t.Errorf("note %q", st.VisionNote)
	}
	starts := v.e.Starts()
	if len(starts) != 2 || strings.Contains(starts[0], "--mmproj") || !v.hasMM(starts[1]) {
		t.Fatalf("starts: %v", starts)
	}
	if vi := visFact(t, s); !vi.Vision || vi.Note != "" {
		t.Fatalf("fact %+v", vi)
	}
	if k := s.CtxKeyString(); !strings.Contains(k, "|mmproj:"+v.p.SHA256) {
		t.Fatalf("ctx key %q lacks the projector", k)
	}
	if got := s.Settings().Vision; got == nil || !*got {
		t.Fatalf("settings vision %v", got)
	}
	visSet(t, s, false)
	visReady(t, s, false)
	starts = v.e.Starts()
	if len(starts) != 3 || strings.Contains(starts[2], "--mmproj") {
		t.Fatalf("starts after off: %v", starts)
	}
	if vi := visFact(t, s); vi.Vision || vi.Note != "" {
		t.Fatalf("fact after off %+v", vi)
	}
	if k := s.CtxKeyString(); strings.Contains(k, "mmproj") {
		t.Fatalf("ctx key after off %q", k)
	}
}

// 3. serve --vision (Config.Vision) wants it at the first start.
func TestVisionConfigFlag(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := s.Status()
	if !st.Vision {
		t.Fatalf("not loaded: note %q", st.VisionNote)
	}
	if starts := v.e.Starts(); len(starts) != 1 || !v.hasMM(starts[0]) {
		t.Fatalf("starts: %v", starts)
	}
}

// 4. The same card gives a smaller shape with the projector than without.
func TestVisionShrinksTheShape(t *testing.T) {
	ctxOf := func(vision bool) string {
		v := newVis(t, 1500)
		v.put(t)
		v.start(t, func(c *run.Config) { c.Vision = vision })
		starts := v.e.Starts()
		if len(starts) != 1 || v.hasMM(starts[0]) != vision {
			t.Fatalf("vision %v: starts %v", vision, starts)
		}
		return testkit.Field(starts[0], "c")
	}
	without, with := ctxOf(false), ctxOf(true)
	if without != "196608" {
		t.Fatalf("without vision c=%s, the test's card should hold the full shape", without)
	}
	if with == without || atoi(t, with) >= atoi(t, without) {
		t.Fatalf("c=%s with vision, c=%s without: the projector's room did not shrink the shape", with, without)
	}
}

// 5. A projector too big for the card: text is served, nobody is left out.
func TestVisionTooBigFallsBackToText(t *testing.T) {
	v := newVis(t, 20000)
	v.put(t)
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := s.Status()
	if st.Vision || !strings.Contains(st.VisionNote, "does not fit") {
		t.Fatalf("vision %v note %q", st.Vision, st.VisionNote)
	}
	starts := v.e.Starts()
	if len(starts) != 1 || strings.Contains(starts[0], "--mmproj") || testkit.Field(starts[0], "c") != "196608" {
		t.Fatalf("starts: %v", starts)
	}
	for _, a := range st.Attachments {
		if a.Slots.Unserved != "" || a.Slots.Given == 0 {
			t.Errorf("a client was left out for vision: %+v", a)
		}
	}
	if vi := visFact(t, s); vi.Vision || !strings.Contains(vi.Note, "does not fit") {
		t.Fatalf("fact %+v", vi)
	}
}

// 6. Out of memory loading the projector: the next launch is text at once.
func TestVisionOOMLoadingProjector(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	v.e.Llama["FAKELLAMA_MMPROJ_OOM"] = "1"
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := s.Status()
	if st.Vision || !strings.Contains(st.VisionNote, "out of memory") {
		t.Fatalf("vision %v note %q", st.Vision, st.VisionNote)
	}
	starts := v.e.Starts()
	if len(starts) != 2 || !v.hasMM(starts[0]) || strings.Contains(starts[1], "--mmproj") {
		t.Fatalf("starts: %v", starts)
	}
	if testkit.Field(starts[0], "c") != testkit.Field(starts[1], "c") {
		t.Errorf("the retry changed the shape: %v", starts)
	}
}

// 7. A build that does not know --mmproj; a model with no projector.
func TestVisionBuildWithoutTheFlag(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	v.e.Llama["FAKELLAMA_HIDE"] = "--mmproj"
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := s.Status()
	if st.Vision || !strings.Contains(st.VisionNote, "does not know --mmproj") {
		t.Fatalf("vision %v note %q", st.Vision, st.VisionNote)
	}
	if strings.Contains(v.e.Starts()[0], "--mmproj") {
		t.Fatalf("passed an unknown flag: %v", v.e.Starts())
	}
}

func TestVisionModelWithoutProjector(t *testing.T) {
	v := newVis(t, 800)
	v.e.Cat.Models[0].Projector = nil
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := s.Status()
	if st.Vision || !strings.Contains(st.VisionNote, "has no vision projector") {
		t.Fatalf("vision %v note %q", st.Vision, st.VisionNote)
	}
	if strings.Contains(v.e.Starts()[0], "--mmproj") {
		t.Fatalf("starts: %v", v.e.Starts())
	}
}

// projSite serves the projector. gate (when set) holds the body after its first half until the
// test closes it; hits says each request; bad serves wrong bytes of the right size.
type projSite struct {
	srv  *httptest.Server
	gate chan struct{}
	hits chan struct{}
	n    atomic.Int32
	bad  atomic.Bool
	gone atomic.Bool // a request was cancelled by its client while the body was held
}

func newProjSite(t *testing.T, body []byte, gated bool) *projSite {
	ps := &projSite{hits: make(chan struct{}, 16)}
	if gated {
		ps.gate = make(chan struct{})
	}
	ps.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ps.n.Add(1)
		ps.hits <- struct{}{}
		b := body
		if ps.bad.Load() {
			b = []byte(strings.Repeat("x", len(body)))
		}
		w.Write(b[:len(b)/2])
		w.(http.Flusher).Flush()
		if ps.gate != nil {
			select {
			case <-ps.gate:
			case <-r.Context().Done():
				ps.gone.Store(true)
				return
			}
		}
		w.Write(b[len(b)/2:])
	}))
	t.Cleanup(ps.srv.Close)
	return ps
}

// 8. The projector is downloaded when wanted: text serves meanwhile, then it reloads with it.
func TestVisionDownload(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, true)
	v.p.URL = ps.srv.URL + "/proj.gguf"
	s := v.start(t, nil)
	visSet(t, s, true)
	st := waitStatus(t, s, "the projector's download in progress", func(st run.Status) bool { return st.VisionProgress != nil })
	if st.Phase != "ready" || st.Vision || !strings.Contains(st.VisionNote, "downloading") {
		t.Fatalf("while downloading: phase %s vision %v note %q", st.Phase, st.Vision, st.VisionNote)
	}
	if !strings.Contains(st.VisionProgress.Step, "vision projector") || st.VisionProgress.TotalBytes != int64(len(v.proj)) {
		t.Errorf("progress %+v", st.VisionProgress)
	}
	if n := len(v.e.Starts()); n != 1 {
		t.Fatalf("restarted while downloading: %v", v.e.Starts())
	}
	close(ps.gate)
	st = visReady(t, s, true)
	if st.VisionProgress != nil || st.VisionNote != "" {
		t.Errorf("after: progress %v note %q", st.VisionProgress, st.VisionNote)
	}
	if got, err := os.ReadFile(v.file); err != nil || string(got) != string(v.proj) {
		t.Fatalf("projector file: %v", err)
	}
	if starts := v.e.Starts(); len(starts) != 2 || !v.hasMM(starts[1]) {
		t.Fatalf("starts: %v", starts)
	}
	if _, err := os.Stat(v.file + ".part"); err == nil {
		t.Error("a .part is left")
	}
}

// A download of wrong bytes fails once; text serves; only a new want asks again.
func TestVisionDownloadWrongBytes(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, false)
	ps.bad.Store(true)
	v.p.URL = ps.srv.URL + "/proj.gguf"
	s := v.start(t, nil)
	visSet(t, s, true)
	st := waitStatus(t, s, "the download's failure said", func(st run.Status) bool {
		return strings.Contains(st.VisionNote, "could not be downloaded") && st.VisionProgress == nil && len(st.Errors) > 0
	})
	if st.Phase != "ready" || st.Vision {
		t.Fatalf("phase %s vision %v", st.Phase, st.Vision)
	}
	if !strings.Contains(strings.Join(st.Errors, "\n"), "sha256") {
		t.Errorf("errors do not say why: %v", st.Errors)
	}
	if strings.Contains(st.VisionNote, ps.srv.URL) || strings.Contains(st.VisionNote, v.e.Data) {
		t.Errorf("the note names a url or path: %q", st.VisionNote)
	}
	if _, err := os.Stat(v.file); err == nil {
		t.Fatal("a file of wrong bytes is in place")
	}
	// nothing asks again of its own: another word that wants vision (the nudge of a settings
	// call that changes nothing) is not a new want
	<-ps.hits
	s.Nudge("test")
	testkit.Mark(t, s, "after the nudge")
	if ps.n.Load() != 1 {
		t.Fatalf("downloaded %d times", ps.n.Load())
	}
	// off then on is a new want: asked again (now with the right bytes)
	ps.bad.Store(false)
	visSet(t, s, false)
	waitStatus(t, s, "vision no longer wanted", func(st run.Status) bool { return st.VisionNote == "" })
	visSet(t, s, true)
	<-ps.hits
	visReady(t, s, true)
	if ps.n.Load() != 2 {
		t.Fatalf("downloaded %d times, want 2", ps.n.Load())
	}
}

// 9. Storage and the projector.
func TestVisionStorage(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	other := func(id string, proj *catalog.Projector) {
		m := catalog.Model{ID: id, Name: id, File: id + ".gguf", SHA256: strings.Repeat("a", 64), Size: 5, Projector: proj}
		v.e.Cat.Models = append(v.e.Cat.Models, m)
		os.WriteFile(filepath.Join(v.e.Data, "models", m.File), []byte("12345"), 0o644)
	}
	p2 := *v.p
	p2.File = "m2-mmproj.gguf"
	other("m2", &p2)
	os.WriteFile(filepath.Join(v.e.Data, "models", p2.File), v.proj, 0o644)
	other("m3", v.p) // shares m1's projector
	s := v.start(t, nil)
	var listed int
	for _, m := range s.Storage(true).Models {
		if m.Projector {
			listed++
			if m.File == v.p.File && m.ID != "m1" || m.File == p2.File && m.ID != "m2" {
				t.Errorf("projector listed with the wrong model: %+v", m)
			}
		}
	}
	if listed != 2 {
		t.Fatalf("projectors listed: %d in %+v", listed, s.Storage(true).Models)
	}
	// wanted: protected
	visSet(t, s, true)
	visReady(t, s, true)
	if _, err := s.DeleteModel("", v.p.File); !errors.Is(err, run.ErrInUse) {
		t.Fatalf("delete the loaded projector: %v", err)
	}
	// a non-serving model goes with its own projector, and leaves the one another model shares
	if _, err := s.DeleteModel("m2", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(v.e.Data, "models", p2.File)); err == nil {
		t.Error("m2's projector stays after m2 was deleted")
	}
	if _, err := s.DeleteModel("m3", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(v.file); err != nil {
		t.Errorf("a projector m1 serves with went with m3: %v", err)
	}
	// off and unloaded: free to delete
	visSet(t, s, false)
	visReady(t, s, false)
	freed, err := s.DeleteModel("", v.p.File)
	if err != nil || freed != int64(len(v.proj)) {
		t.Fatalf("delete the idle projector: %d %v", freed, err)
	}
	if _, err := os.Stat(v.file); err == nil {
		t.Error("the projector is still there")
	}
}

// the second one while a download holds its file
func TestVisionStorageProtectsDownload(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, true)
	v.p.URL = ps.srv.URL + "/proj.gguf"
	s := v.start(t, nil)
	visSet(t, s, true)
	waitStatus(t, s, "the download", func(st run.Status) bool { return st.VisionProgress != nil })
	if _, err := s.DeleteModel("", v.p.File+".part"); !errors.Is(err, run.ErrInUse) {
		t.Fatalf("delete the part under download: %v", err)
	}
	close(ps.gate)
	visReady(t, s, true)
}

// 10. Attachments.
func visProf(vision bool) *run.Profile {
	p := clProf(1, 8192, 4096)
	p.Vision = vision
	return p
}

func TestVisionLocalAttach(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, nil)
	k := newKid(t)
	av, _ := clAttach(t, s, clReq("eyes", k, false, visProf(true)))
	visReady(t, s, true)
	starts := v.e.Starts()
	if len(starts) != 2 || !v.hasMM(starts[1]) {
		t.Fatalf("starts: %v", starts)
	}
	if e := s.Detach(av.Attachment, "loopback"); e != nil {
		t.Fatal(e)
	}
	visReady(t, s, false)
	if starts = v.e.Starts(); len(starts) != 3 || strings.Contains(starts[2], "--mmproj") {
		t.Fatalf("starts after leave: %v", starts)
	}
}

func remVis(name string, vision bool) run.AttachRequest {
	return run.AttachRequest{Name: name, Hold: true, Profile: visProf(vision), Identity: "client-1", Local: false}
}

// A remote attach with vision while the model runs restarts it with the projector and is charged;
// its leave restarts nothing (the projector stays loaded) and is not charged.
func TestVisionRemoteAttachIsCharged(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, nil)
	av, _ := clAttach(t, s, remVis("r", true))
	if av.RestartsLeft == nil || *av.RestartsLeft != 4 {
		t.Fatalf("restarts left at attach: %v", av.RestartsLeft)
	}
	visReady(t, s, true)
	starts := v.e.Starts()
	if len(starts) != 2 || !v.hasMM(starts[1]) {
		t.Fatalf("starts: %v", starts)
	}
	if left := clView(t, s, av.Attachment).RestartsLeft; left == nil || *left != 3 {
		t.Fatalf("restarts left after the vision restart: %v", ptrOf(left))
	}
	if e := s.Detach(av.Attachment, "client-1"); e != nil {
		t.Fatal(e)
	}
	s.React()
	testkit.Mark(t, s, "after the remote leave and the reactor's look")
	if st := s.Status(); st.Phase != "ready" || !st.Vision {
		t.Fatalf("after a remote leave: phase %s vision %v", st.Phase, st.Vision)
	}
	if got := len(v.e.Starts()); got != 2 {
		t.Fatalf("a remote leave started the server: %d starts", got)
	}
	if got := s.RestartsOf("client-1"); got != 1 {
		t.Fatalf("charged %d after the leave, want the attach's 1", got)
	}
}

// With the budget spent the vision change waits (Slots.Why restart_budget_spent) and text keeps serving.
func TestVisionRemoteBudgetSpentWaits(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, nil)
	spend(t, s, "client-1", "x") // four changes of wants, none of them vision
	n := len(v.e.Starts())
	av, _ := clAttach(t, s, remVis("v", true))
	testkit.WaitFor(t, 10*time.Second, "the change to wait on the budget", func() bool {
		return clView(t, s, av.Attachment).Slots.Why == "restart_budget_spent"
	})
	s.React()
	testkit.Mark(t, s, "after the reactor's look")
	st := s.Status()
	if st.Phase != "ready" || st.Vision || len(v.e.Starts()) != n {
		t.Fatalf("phase %s vision %v starts %d (was %d)", st.Phase, st.Vision, len(v.e.Starts()), n)
	}
	if got := s.RestartsOf("client-1"); got != 4 {
		t.Fatalf("charged %d, want 4", got)
	}
}

// A remote attach with vision while the model is unloaded is not charged, and the start the wake
// makes loads the projector.
func TestVisionRemoteAttachWhileUnloadedIsFree(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, nil)
	var av run.AttachView
	sleepThenWake(t, s, func() {
		av, _ = clAttach(t, s, remVis("r", true))
		if av.RestartsLeft == nil || *av.RestartsLeft != 4 {
			t.Fatalf("restarts left at attach: %v", ptrOf(av.RestartsLeft))
		}
	})
	visReady(t, s, true)
	starts := v.e.Starts()
	if len(starts) != 2 || !v.hasMM(starts[1]) {
		t.Fatalf("starts: %v", starts)
	}
	if left := clView(t, s, av.Attachment).RestartsLeft; left == nil || *left != 4 || s.RestartsOf("client-1") != 0 {
		t.Fatalf("restarts left %v, charged %d: a start from unloaded is never charged", ptrOf(left), s.RestartsOf("client-1"))
	}
}

// 11. The book holds the model alone: the reading less the projector's load.
func TestVisionCalibrationSubtractsProjector(t *testing.T) {
	const reading = 15500 // near the seed (15330), so the book takes it: 170 MiB over
	fixed := func(vision bool) float64 {
		v := newVis(t, 800)
		v.put(t)
		got := reading
		if vision {
			got += v.p.LoadMiB
		}
		v.start(t, func(c *run.Config) {
			c.Vision = vision
			c.Measure = func(ctx context.Context, pid int) (int, error) { return got, nil }
			// a card that holds the full shape either way, so both runs have the same shape
			c.Probe = func(ctx context.Context) (probe.Hardware, error) {
				return probe.Hardware{OS: runtime.GOOS, Arch: runtime.GOARCH, Cores: 8, RAMTotalMiB: 65536, RAMAvailMiB: 60000,
					GPUs: []probe.GPU{{Vendor: "nvidia", Name: "Fake GPU", VRAMTotalMiB: 65536}}}, nil
			}
		})
		var f float64
		testkit.WaitFor(t, 10*time.Second, "the book's reading", func() bool {
			for _, c := range run.LoadState(v.e.Data).Book.Entries {
				if c.FixedMiB > 0 {
					f = c.FixedMiB
					return true
				}
			}
			return false
		})
		if starts := v.e.Starts(); len(starts) != 1 || v.hasMM(starts[0]) != vision {
			t.Fatalf("starts %v", starts)
		}
		return f
	}
	plain, with := fixed(false), fixed(true)
	if plain != with {
		t.Fatalf("the book's fixed part is %v with the projector and %v without: the projector's load was learnt into the model", with, plain)
	}
}

func atoi(t *testing.T, s string) int {
	t.Helper()
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			t.Fatalf("not a number: %q", s)
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func ptrOf(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

// Task 3a. No room on the disk: the projector's download does not start, the note says so without
// a path, text keeps serving; a later want checks again and downloads once there is room.
func TestVisionNoRoomForTheProjector(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, false)
	v.p.URL = ps.srv.URL + "/proj.gguf"
	var free atomic.Int64
	free.Store(1 << 20)
	s := v.start(t, func(c *run.Config) {
		c.DiskFree = func(string) (int, error) { return int(free.Load()), nil }
	})
	free.Store(1) // 1 MiB: short of the headroom alone
	visSet(t, s, true)
	st := waitStatus(t, s, "the no-room note", func(st run.Status) bool { return strings.Contains(st.VisionNote, "no room on disk") })
	if st.Phase != "ready" || st.Vision || st.VisionProgress != nil {
		t.Fatalf("phase %s vision %v progress %v", st.Phase, st.Vision, st.VisionProgress)
	}
	if !strings.HasPrefix(st.VisionNote, "no room on disk for the vision projector (needs ") || !strings.HasSuffix(st.VisionNote, " MiB more)") ||
		strings.Contains(st.VisionNote, v.e.Data) || strings.Contains(st.VisionNote, "/") {
		t.Fatalf("note %q", st.VisionNote)
	}
	if ps.n.Load() != 0 {
		t.Fatalf("the download was asked for with no room: %d requests", ps.n.Load())
	}
	// a later want asks again: with room it downloads
	free.Store(1 << 20)
	visSet(t, s, false)
	visReady(t, s, false)
	visSet(t, s, true)
	st = visReady(t, s, true)
	if ps.n.Load() != 1 || st.VisionNote != "" {
		t.Fatalf("requests %d note %q", ps.n.Load(), st.VisionNote)
	}
	if got, err := os.ReadFile(v.file); err != nil || string(got) != string(v.proj) {
		t.Fatalf("projector file: %v", err)
	}
}

// Task 3b. Remove downloads lets a projector download in flight go: the request is cancelled,
// the .part is gone, and it is not recorded as a failure.
func TestVisionRemoveDownloadsCancelsTheProjector(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, true)
	v.p.URL = ps.srv.URL + "/proj.gguf"
	s := v.start(t, nil)
	visSet(t, s, true)
	waitStatus(t, s, "the download", func(st run.Status) bool { return st.VisionProgress != nil })
	if _, err := os.Stat(v.file + ".part"); err != nil {
		t.Fatalf("no .part while downloading: %v", err)
	}
	if err := s.RemoveDownloads(); err != nil {
		t.Fatal(err)
	}
	st := waitStatus(t, s, "removed", func(st run.Status) bool {
		return st.Phase == "not_installed" && strings.HasPrefix(st.Message, "Removed")
	})
	if _, err := os.Stat(v.file + ".part"); err == nil {
		t.Error("the .part is still there")
	}
	testkit.WaitFor(t, 20*time.Second, "the projector request to be cancelled", ps.gone.Load)
	if st.VisionProgress != nil || strings.Contains(st.VisionNote, "could not be downloaded") {
		t.Errorf("progress %v note %q", st.VisionProgress, st.VisionNote)
	}
	for _, e := range st.Errors {
		if strings.Contains(e, "vision projector") {
			t.Errorf("recorded as failed: %q", e)
		}
	}
	if ents, _ := os.ReadDir(filepath.Join(v.e.Data, "models")); len(ents) != 0 {
		t.Errorf("models left: %v", ents)
	}
}

// Task 4. serve --vision holds vision on: turning it off through PUT is refused (409 vision_forced)
// with nothing changed; turning it on is accepted; GET says it is forced.
func TestVisionForcedRefusesOff(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	if got := s.Settings(); !got.VisionForced || got.Vision == nil || *got.Vision {
		t.Fatalf("settings %+v", got)
	}
	off, on := false, true
	ce := s.CheckVisionOff(&off)
	if ce == nil || ce.Status != 409 || ce.Code != "vision_forced" || ce.Msg != run.VisionForcedMsg {
		t.Fatalf("check: %+v", ce)
	}
	if s.CheckVisionOff(&on) != nil || s.CheckVisionOff(nil) != nil {
		t.Fatal("on or absent refused")
	}
	if err := s.SetSettings(run.Settings{Vision: &on}); err != nil {
		t.Fatal(err)
	}
	err := s.SetSettings(run.Settings{Vision: &off})
	var got *run.ClientError
	if !errors.As(err, &got) || got.Code != "vision_forced" {
		t.Fatalf("set off: %v", err)
	}
	if st := s.Settings(); st.Vision == nil || !*st.Vision {
		t.Fatalf("stored vision moved: %+v", st.Vision)
	}
	if st := s.Status(); !st.Vision {
		t.Fatal("vision went off")
	}
}
