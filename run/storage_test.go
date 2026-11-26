package run_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hermesengine/run"
)

// serveStarter installs the small model alone (nothing smaller exists, so no staging) and waits for ready.
func serveStarter(t *testing.T, s *stage) *run.Supervisor {
	t.Helper()
	sup := s.start(t)
	if err := sup.Install("starter", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "starter ready", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "starter" })
	return sup
}

func TestSwitchKeepsServingThenSwaps(t *testing.T) {
	s := newStage(t)
	sup := serveStarter(t, s)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	st := waitStatus(t, sup, "background download", func(st run.Status) bool {
		return st.Upgrade != nil && st.Upgrade.State == "downloading" && s.order() != "starter"
	})
	if st.Phase != "ready" || st.Model != "starter" || st.ModelTier != "full" || st.ModelFallback || !st.Upgrade.Switch {
		t.Fatalf("a switch must keep serving and not look like a starter: %+v", st)
	}
	if st.Previous != nil {
		t.Errorf("previous before the swap: %+v", st.Previous)
	}
	// cancel keeps the old model serving and forgets the target
	sup.Cancel()
	st = waitStatus(t, sup, "cancelled", func(st run.Status) bool { return st.Upgrade != nil && st.Upgrade.State == "cancelled" })
	if st.Phase != "ready" || st.Model != "starter" {
		t.Errorf("after cancel: %+v", st)
	}
	if run.LoadState(s.e.Data).Target != "" {
		t.Error("a cancelled switch left state.target set")
	}
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	s.release()
	st = waitStatus(t, sup, "swap", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "target" && st.Previous != nil })
	if p := st.Previous; p.ID != "starter" || p.File != "starter.gguf" || p.Name != "Starter" || p.Bytes != int64(len(s.starter)) {
		t.Errorf("previous = %+v", p)
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "models", "starter.gguf")); err != nil {
		t.Errorf("the old model file must stay: %v", err)
	}
	if run.LoadState(s.e.Data).Previous != "starter" {
		t.Error("previous not persisted")
	}
	// the serving file cannot be deleted; the previous one can, which clears previous
	if _, err := sup.DeleteModel("target", ""); !errors.Is(err, run.ErrInUse) {
		t.Errorf("delete serving: %v", err)
	}
	freed, err := sup.DeleteModel("", "starter.gguf")
	if err != nil || freed != int64(len(s.starter)) {
		t.Fatalf("delete previous: %d %v", freed, err)
	}
	if sup.Status().Previous != nil {
		t.Error("previous not cleared by delete-model")
	}
	if _, err := sup.DeleteModel("starter", ""); !errors.Is(err, run.ErrNoModel) {
		t.Errorf("second delete: %v", err)
	}
	for _, bad := range []string{"../state.json", "a/b.gguf"} {
		if _, err := sup.DeleteModel("", bad); err == nil || errors.Is(err, run.ErrNoModel) {
			t.Errorf("delete %q: %v", bad, err)
		}
	}
	if _, err := os.Stat(filepath.Join(s.e.Data, "state.json")); err != nil {
		t.Error("state.json must survive")
	}
}

func TestSwitchToInstalledSwapsDirectly(t *testing.T) {
	s := newStage(t)
	s.release()
	sup := serveStarter(t, s)
	sup.Install("target", "")
	waitStatus(t, sup, "target serving", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "target" && st.Previous != nil })
	before := s.order()
	if err := sup.Install("starter", ""); err != nil {
		t.Fatal(err)
	}
	st := waitStatus(t, sup, "back to starter", func(st run.Status) bool {
		return st.Phase == "ready" && st.Model == "starter" && st.Previous != nil && st.Previous.ID == "target"
	})
	if s.order() != before {
		t.Errorf("a downloaded model must not be fetched again: %q -> %q", before, s.order())
	}
	if st.Previous.Bytes != int64(len(s.target)) {
		t.Errorf("previous %+v", st.Previous)
	}
}

func TestSwitchRefusedWithoutDisk(t *testing.T) {
	s := newStage(t)
	sup := serveStarter(t, s)
	s.free = 100 // MiB, less than the target plus headroom
	err := sup.Install("target", "")
	if err == nil || !strings.Contains(err.Error(), "needs") || !strings.Contains(err.Error(), "GB") || !strings.Contains(err.Error(), "free") {
		t.Fatalf("want a needs-X-GB error, got %v", err)
	}
	st := sup.Status()
	if st.Phase != "ready" || st.Model != "starter" || st.Upgrade != nil {
		t.Errorf("a refused switch must change nothing: %+v", st)
	}
}

func TestPlanDiskFields(t *testing.T) {
	s := newStage(t)
	sup := serveStarter(t, s)
	res, err := sup.Plan(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if res.DiskFreeBytes != int64(s.free)<<20 {
		t.Errorf("disk_free_bytes = %d", res.DiskFreeBytes)
	}
	for _, v := range res.Models {
		switch v.ID {
		case "starter":
			if v.DiskNeedBytes != 0 {
				t.Errorf("installed model needs %d", v.DiskNeedBytes)
			}
		case "target":
			if v.DiskNeedBytes < int64(len(s.target)) {
				t.Errorf("target need %d < its size", v.DiskNeedBytes)
			}
		}
	}
	// a partial download counts
	var before int64
	for _, v := range res.Models {
		if v.ID == "target" {
			before = v.DiskNeedBytes
		}
	}
	os.WriteFile(filepath.Join(s.e.Data, "models", "target.gguf.part"), make([]byte, 100000), 0o644)
	res, _ = sup.Plan(t.Context())
	for _, v := range res.Models {
		if v.ID == "target" && before-v.DiskNeedBytes < 100000-4096 {
			t.Errorf("partial not subtracted: %d before, %d after", before, v.DiskNeedBytes)
		}
	}
}

func TestStorageListing(t *testing.T) {
	s := newStage(t)
	sup := serveStarter(t, s)
	os.WriteFile(filepath.Join(s.e.Data, "models", "target.gguf.part"), make([]byte, 1234), 0o644)
	os.WriteFile(filepath.Join(s.e.Data, "models", "stray.gguf"), make([]byte, 10), 0o644)
	in := sup.Storage(true)
	if in.DataDir != s.e.Data || in.FreeBytes <= 0 || in.TotalBytes < int64(len(s.starter)) {
		t.Fatalf("%+v", in)
	}
	names := map[string]int64{}
	for _, p := range in.Parts {
		names[p.Name] = p.Bytes
	}
	if len(in.Parts) != 5 || names["models"] != int64(len(s.starter))+1234+10 || names["backends"] == 0 {
		t.Errorf("parts %+v", in.Parts)
	}
	got := map[string]run.StorageModel{}
	for _, m := range in.Models {
		got[m.File] = m
	}
	if m := got["starter.gguf"]; m.ID != "starter" || !m.InUse || m.Partial {
		t.Errorf("starter %+v", m)
	}
	if m := got["target.gguf.part"]; m.ID != "target" || !m.Partial || m.InUse {
		t.Errorf("part %+v", m)
	}
	if m := got["stray.gguf"]; m.ID != "" || m.InUse {
		t.Errorf("stray %+v", m)
	}
	if _, err := sup.DeleteModel("", "stray.gguf"); err != nil {
		t.Error(err)
	}
}

func TestRemoveDownloadsThenReinstall(t *testing.T) {
	s := newStage(t)
	s.release()
	sup := serveStarter(t, s)
	data := s.e.Data
	// a symlink pointing out of the data dir must be unlinked, never followed
	outside := t.TempDir()
	keep := filepath.Join(outside, "keep.txt")
	os.WriteFile(keep, []byte("mine"), 0o644)
	if err := os.Symlink(outside, filepath.Join(data, "downloads", "escape")); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(data, "logs", "mark.txt"), []byte("x"), 0o644)
	if err := sup.SetSettings(run.Settings{Mode: "local", LingerS: 7, KVType: "q8_0"}); err != nil {
		t.Fatal(err)
	}
	if err := sup.RemoveDownloads(); err != nil {
		t.Fatal(err)
	}
	st := waitStatus(t, sup, "removed", func(st run.Status) bool {
		return st.Phase == "not_installed" && strings.HasPrefix(st.Message, "Removed downloaded files")
	})
	if !strings.Contains(st.Message, "GB freed") || st.Model != "" || st.Backend != "" || st.Previous != nil {
		t.Errorf("%+v", st)
	}
	if b, _ := os.ReadFile(keep); string(b) != "mine" {
		t.Fatal("remove-downloads followed a symlink out of the data dir")
	}
	for _, d := range []string{"models", "backends", "runtime", "downloads"} {
		ents, err := os.ReadDir(filepath.Join(data, d))
		if err != nil || len(ents) != 0 {
			t.Errorf("%s: %v %v", d, ents, err)
		}
	}
	if _, err := os.Stat(filepath.Join(data, "logs", "mark.txt")); err != nil {
		t.Error("logs must stay")
	}
	ss := run.LoadState(data)
	if ss.Model != "" || ss.Target != "" || ss.StarterOwned != "" || ss.Backend != "" || len(ss.Book.Entries) != 0 {
		t.Errorf("state %+v", ss)
	}
	if ss.LingerS != 7 || ss.KVType != "q8_0" || ss.Mode != "local" {
		t.Errorf("settings lost: %+v", ss)
	}
	if got := sup.Storage(true); got.TotalBytes > 1<<20 {
		t.Errorf("storage after: %+v", got.Parts)
	}
	// install works from scratch
	if err := sup.Install("starter", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "ready again", func(st run.Status) bool { return st.Phase == "ready" && st.Model == "starter" })
}

func TestRemoveDownloadsCancelsInstall(t *testing.T) {
	s := newStage(t) // the target download is gated: an install job is running
	sup := s.start(t)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "starter serving, target downloading", func(st run.Status) bool { return st.Phase == "ready" && st.ModelStage == "target_downloading" })
	if err := sup.RemoveDownloads(); err != nil {
		t.Fatal(err)
	}
	st := waitStatus(t, sup, "removed", func(st run.Status) bool {
		return st.Phase == "not_installed" && strings.HasPrefix(st.Message, "Removed")
	})
	if st.Upgrade != nil || st.Model != "" {
		t.Errorf("%+v", st)
	}
	if ents, _ := os.ReadDir(filepath.Join(s.e.Data, "models")); len(ents) != 0 {
		t.Errorf("models left: %v", ents)
	}
}
