//go:build !windows

package run_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// The model server's environment loses every LLAMA_ARG_* of the parent (the engine gives the
// flags), and the engine log names the dropped variables once.
func TestParentLlamaArgEnvNeverReachesTheChild(t *testing.T) {
	names := []string{"LLAMA_ARG_HF_REPO", "LLAMA_ARG_MODEL_URL", "LLAMA_ARG_DOCKER_REPO", "LLAMA_ARG_MMPROJ"}
	for _, k := range names {
		t.Setenv(k, "x")
	}
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	log, err := os.ReadFile(filepath.Join(e.Data, "logs", "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(log), "left out of the parent's environment"); n != 1 {
		t.Errorf("the log names the dropped variables %d times", n)
	}
	for _, k := range names {
		if !strings.Contains(string(log), k) {
			t.Errorf("the log does not name %s", k)
		}
	}
	for _, l := range e.Starts() {
		if strings.Contains(l, "envarg=") && !strings.Contains(l, "envarg=none") {
			t.Errorf("a LLAMA_ARG_* reached the child: %s", l)
		}
	}
}

// mmStarts counts the starts that carried --mmproj and those that did not.
func mmStarts(e *testkit.Env) (with, without int) {
	for _, l := range e.Starts() {
		if strings.Contains(l, "--mmproj") {
			with++
		} else {
			without++
		}
	}
	return
}

// Finding 3: a projector that fails to load for any reason but memory leaves text serving; the
// start that failed is retried at once without --mmproj, the projector is not tried again on a
// client's word or a remote attach, and a person asking again (the setting turned on) tries once more.
func TestVisionProjectorLoadFailureServesText(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	v.e.Llama["FAKELLAMA_MMPROJ_BAD"] = "1"
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := s.Status()
	if st.Phase != "ready" || st.Vision || !strings.HasPrefix(st.VisionNote, "the vision projector failed to load: ") {
		t.Fatalf("phase %s vision %v note %q", st.Phase, st.Vision, st.VisionNote)
	}
	if strings.Contains(st.VisionNote, v.e.Data) || strings.Contains(st.VisionNote, "/") {
		t.Errorf("the note names a path: %q", st.VisionNote)
	}
	if with, without := mmStarts(v.e); with != 1 || without != 1 {
		t.Fatalf("starts with --mmproj %d, without %d: %v", with, without, v.e.Starts())
	}
	// a client's word and a remote attach with vision: no start carries --mmproj again
	s.Nudge("a client's word")
	av, _ := clAttach(t, s, remVis("r", true))
	s.React()
	testkit.Mark(t, s, "after a remote attach with vision and the reactor's look")
	if st := s.Status(); st.Phase != "ready" || st.Vision {
		t.Fatalf("after the attach: phase %s vision %v note %q", st.Phase, st.Vision, st.VisionNote)
	}
	if with, _ := mmStarts(v.e); with != 1 {
		t.Fatalf("a later start carried --mmproj: %v", v.e.Starts())
	}
	if left := clView(t, s, av.Attachment).RestartsLeft; left == nil || *left != 4 {
		t.Errorf("the attach was charged for a restart that did not happen: %v", ptrOf(left))
	}
	if e := s.Detach(av.Attachment, "client-1"); e != nil {
		t.Fatal(e)
	}
	// a start made for another reason (the KV type changed) does not carry --mmproj either: the
	// projector is marked not loadable, not merely skipped once
	n := len(v.e.Starts())
	if err := s.SetSettings(run.Settings{KVType: "q8_0"}); err != nil {
		t.Fatal(err)
	}
	waitStarts(t, v.e, n+1)
	st = visReady(t, s, false)
	if with, _ := mmStarts(v.e); with != 1 {
		t.Fatalf("the restart for the KV type carried --mmproj: %v", v.e.Starts())
	}
	if !strings.HasPrefix(st.VisionNote, "the vision projector failed to load: ") {
		t.Errorf("note after the restart: %q", st.VisionNote)
	}
}

func TestVisionProjectorLoadFailureAskedAgain(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	v.e.Llama["FAKELLAMA_MMPROJ_BAD"] = "1"
	s := v.start(t, nil)
	visSet(t, s, true) // the first ask: one start with --mmproj fails, text again
	waitStarts(t, v.e, 3)
	visReady(t, s, false)
	if with, without := mmStarts(v.e); with != 1 || without != 2 {
		t.Fatalf("starts with --mmproj %d, without %d: %v", with, without, v.e.Starts())
	}
	visSet(t, s, false)
	visSet(t, s, true) // a person asks again: once more
	waitStarts(t, v.e, 5)
	visReady(t, s, false)
	if with, without := mmStarts(v.e); with != 2 || without != 3 {
		t.Fatalf("after asking again: starts with --mmproj %d, without %d: %v", with, without, v.e.Starts())
	}
}

// Finding 6: a projector placed by hand with the right content is hashed, then loaded.
func TestVisionHandPlacedProjectorIsHashedThenLoaded(t *testing.T) {
	v := newVis(t, 800)
	v.putHand(t)
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	v.e.WaitProjectorVerified(v.p.File)
	visReady(t, s, true)
	starts := v.e.Starts()
	if len(starts) != 2 || strings.Contains(starts[0], "--mmproj") || !v.hasMM(starts[1]) {
		t.Fatalf("starts: %v", starts)
	}
}

// ... and the same size with other content is never loaded: the note says it did not match.
func TestVisionWrongContentProjectorIsNotLoaded(t *testing.T) {
	v := newVis(t, 800)
	v.proj = []byte(strings.Repeat("x", len(v.proj))) // the file; the catalog names the real content's hash
	v.putHand(t)
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	st := waitStatus(t, s, "the mismatch note", func(st run.Status) bool { return strings.Contains(st.VisionNote, "did not match") })
	if st.Vision || strings.Contains(st.VisionNote, "/") {
		t.Fatalf("vision %v note %q", st.Vision, st.VisionNote)
	}
	s.Nudge("a client's word")
	s.React()
	testkit.Mark(t, s, "after the reactor's look")
	if with, _ := mmStarts(v.e); with != 0 {
		t.Fatalf("a start carried the unverified projector: %v", v.e.Starts())
	}
}

// An overlay (a catalog model of a client) cannot make the engine load, or delete, a file it does
// not own: a model's .gguf or anyone's file named as its projector is hashed, does not match, and
// stays when the model is deleted.
func TestVisionForeignFileNamedAsProjectorIsNeverLoadedOrDeleted(t *testing.T) {
	v := newVis(t, 800)
	foreign := []byte("someone else's model, not a projector............")
	if err := os.WriteFile(filepath.Join(v.e.Data, "models", "user.gguf"), foreign, 0o644); err != nil {
		t.Fatal(err)
	}
	v.p.File, v.p.Size = "user.gguf", int64(len(foreign)) // right size, other content
	v.file = filepath.Join(v.e.Data, "models", "user.gguf")
	// a second model whose projector is another foreign file
	if err := os.WriteFile(filepath.Join(v.e.Data, "models", "user2.gguf"), foreign, 0o644); err != nil {
		t.Fatal(err)
	}
	p2 := *v.p
	p2.File = "user2.gguf"
	v.e.Cat.Models = append(v.e.Cat.Models, catalog.Model{ID: "m2", Name: "m2", File: "m2.gguf", SHA256: strings.Repeat("a", 64), Size: 5, Projector: &p2})
	if err := os.WriteFile(filepath.Join(v.e.Data, "models", "m2.gguf"), []byte("12345"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	waitStatus(t, s, "the mismatch note", func(st run.Status) bool { return strings.Contains(st.VisionNote, "did not match") })
	if with, _ := mmStarts(v.e); with != 0 {
		t.Fatalf("the foreign file was loaded: %v", v.e.Starts())
	}
	if _, err := s.DeleteModel("m2", ""); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"user.gguf", "user2.gguf"} {
		if _, err := os.Stat(filepath.Join(v.e.Data, "models", f)); err != nil {
			t.Errorf("%s was deleted: %v", f, err)
		}
	}
}

// A file whose name is the projector's but whose size is not the catalog's is not overwritten by
// the download either (it is not the engine's).
func TestVisionForeignFileOfOtherSizeIsNotOverwritten(t *testing.T) {
	v := newVis(t, 800)
	if err := os.WriteFile(v.file, []byte("mine"), 0o644); err != nil {
		t.Fatal(err)
	}
	ps := newProjSite(t, v.proj, false)
	v.p.URL = ps.srv.URL + "/p.gguf"
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	waitStatus(t, s, "the mismatch note", func(st run.Status) bool { return strings.Contains(st.VisionNote, "did not match") })
	if b, _ := os.ReadFile(v.file); string(b) != "mine" || ps.n.Load() != 0 {
		t.Fatalf("the file was replaced or fetched: %q, %d requests", b, ps.n.Load())
	}
}

// Finding 5: the reload after a projector download is charged. A remote-only want: one restart,
// charged to that client.
func TestVisionDownloadReloadChargesTheRemoteWant(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, true)
	v.p.URL = ps.srv.URL + "/p.gguf"
	s := v.start(t, nil)
	av, _ := clAttach(t, s, remVis("r", true)) // the projector is not on disk: nothing to restart for, nothing charged
	waitStatus(t, s, "the download", func(st run.Status) bool { return st.VisionProgress != nil })
	if left := clView(t, s, av.Attachment).RestartsLeft; left == nil || *left != 4 {
		t.Fatalf("charged before the projector came: %v", ptrOf(left))
	}
	close(ps.gate)
	visReady(t, s, true)
	if left := clView(t, s, av.Attachment).RestartsLeft; left == nil || *left != 3 {
		t.Fatalf("restarts left after the reload: %v", ptrOf(left))
	}
	if got := s.RestartsOf("client-1"); got != 1 {
		t.Fatalf("charged %d, want 1", got)
	}
	if with, _ := mmStarts(v.e); with != 1 {
		t.Fatalf("starts: %v", v.e.Starts())
	}
}

// A remote-only want with the budget spent: no restart, and the note says it waits for the next start.
func TestVisionDownloadReloadWithSpentBudgetWaits(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, true)
	v.p.URL = ps.srv.URL + "/p.gguf"
	s := v.start(t, nil)
	spend(t, s, "client-1", "x")
	clAttach(t, s, remVis("r", true))
	waitStatus(t, s, "the download", func(st run.Status) bool { return st.VisionProgress != nil })
	n := len(v.e.Starts())
	close(ps.gate)
	st := waitStatus(t, s, "the note that it waits", func(st run.Status) bool {
		return st.VisionProgress == nil && strings.Contains(st.VisionNote, "next restart") && strings.Contains(st.VisionNote, "no attached client has a restart left")
	})
	s.React()
	testkit.Mark(t, s, "after the reactor's look")
	if st = s.Status(); st.Vision || st.Phase != "ready" || len(v.e.Starts()) != n {
		t.Fatalf("vision %v phase %s starts %d (was %d)", st.Vision, st.Phase, len(v.e.Starts()), n)
	}
	if got := s.RestartsOf("client-1"); got != 4 {
		t.Fatalf("charged %d, want 4", got)
	}
}

// A local want (here a local attachment with vision, beside a remote one) reloads when idle and
// charges nobody.
func TestVisionDownloadReloadForALocalWantChargesNobody(t *testing.T) {
	v := newVis(t, 800)
	ps := newProjSite(t, v.proj, true)
	v.p.URL = ps.srv.URL + "/p.gguf"
	s := v.start(t, nil)
	k := newKid(t)
	clAttach(t, s, clReq("eyes", k, false, visProf(true)))
	clAttach(t, s, remVis("r", true))
	waitStatus(t, s, "the download", func(st run.Status) bool { return st.VisionProgress != nil })
	close(ps.gate)
	visReady(t, s, true)
	if got := s.RestartsOf("client-1"); got != 0 {
		t.Fatalf("charged %d, want nobody", got)
	}
}

// A projector marked not loadable is tried again once its file changes: the file written again
// (a new mtime) is hashed again, its verified identity is new, and the next start carries
// --mmproj once more (here it fails again, and text comes back up).
func TestVisionProjectorLoadFailureChangedFileTriedAgain(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	v.e.Llama["FAKELLAMA_MMPROJ_BAD"] = "1"
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	if with, without := mmStarts(v.e); with != 1 || without != 1 {
		t.Fatalf("starts with --mmproj %d, without %d: %v", with, without, v.e.Starts())
	}
	if err := os.WriteFile(v.file, v.proj, 0o644); err != nil {
		t.Fatal(err)
	}
	later := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC) // a fixed mtime unlike the file's own
	if err := os.Chtimes(v.file, later, later); err != nil {
		t.Fatal(err)
	}
	if err := s.SetSettings(run.Settings{KVType: "q8_0"}); err != nil { // a start for another reason
		t.Fatal(err)
	}
	testkit.WaitFor(t, 20*time.Second, "a start with --mmproj again", func() bool { with, _ := mmStarts(v.e); return with == 2 })
	st := visReady(t, s, false)
	if !strings.HasPrefix(st.VisionNote, "the vision projector failed to load: ") {
		t.Errorf("note: %q", st.VisionNote)
	}
}

// The mark never outlives the engine process: the next life of the engine on the same data tries
// the projector again (and here, failing again, comes up with text).
func TestVisionProjectorLoadFailureForgottenAtRestart(t *testing.T) {
	v := newVis(t, 800)
	v.put(t)
	v.e.Llama["FAKELLAMA_MMPROJ_BAD"] = "1"
	s := v.start(t, func(c *run.Config) { c.Vision = true })
	if with, _ := mmStarts(v.e); with != 1 {
		t.Fatalf("starts: %v", v.e.Starts())
	}
	s.Shutdown()
	s2 := v.start(t, func(c *run.Config) { c.Vision = true })
	visReady(t, s2, false)
	if with, without := mmStarts(v.e); with != 2 || without != 2 {
		t.Fatalf("the next life: starts with --mmproj %d, without %d: %v", with, without, v.e.Starts())
	}
}
