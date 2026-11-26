package run

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Storage: what the engine keeps on disk, removing it, and deleting one model file
// (docs/ENGINE.md, section "Storage").

// ErrInUse means the model file is serving or being downloaded.
var ErrInUse = errors.New("that model file is in use")

// ErrNoModel means there is no such file in models/.
var ErrNoModel = errors.New("no such model file")

// downloadDirs are the directories remove-downloads empties (logs/ and state.json stay).
var downloadDirs = []string{"models", "backends", "runtime", "downloads"}

// storageParts are the directories GET storage reports.
var storageParts = []string{"models", "backends", "runtime", "downloads", "logs"}

const (
	storageWalkCap  = 2 * time.Second
	storageCacheTTL = 10 * time.Second
)

// StoragePart is the size of one directory under the data dir.
type StoragePart struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	Bytes int64  `json:"bytes"`
}

// StorageModel is one file in models/.
type StorageModel struct {
	File    string `json:"file"`
	ID      string `json:"id"` // "" when the file is not a catalog model
	Bytes   int64  `json:"bytes"`
	InUse   bool   `json:"in_use"`
	Partial bool   `json:"partial"`
}

// StorageInfo is GET /engine/v1/storage.
type StorageInfo struct {
	DataDir    string         `json:"data_dir"`
	TotalBytes int64          `json:"total_bytes"`
	FreeBytes  int64          `json:"free_bytes"` // -1 unknown
	Parts      []StoragePart  `json:"parts"`
	Models     []StorageModel `json:"models"`
	// Approximate is true when the 2 s walk cap hit and the sizes are a lower bound.
	Approximate bool `json:"approximate"`
}

type storageCache struct {
	at   time.Time
	info StorageInfo
}

// dirSize sums the regular files under root without following links. ok is false when ctx ended.
func dirSize(ctx context.Context, root string) (n int64, ok bool) {
	ok = true
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if ctx.Err() != nil {
			ok = false
			return ctx.Err()
		}
		if err != nil {
			return nil
		}
		if d.Type().IsRegular() {
			if fi, e := d.Info(); e == nil {
				n += fi.Size()
			}
		}
		return nil
	})
	return n, ok
}

// Storage reports disk use. Walks are capped at 2 s; results are cached 10 s (refresh skips the cache).
func (s *Supervisor) Storage(refresh bool) StorageInfo {
	s.mu.Lock()
	c := s.stoCache
	s.mu.Unlock()
	if !refresh && c != nil && time.Since(c.at) < storageCacheTTL {
		info := c.info
		info.Models = s.storageModels() // in_use changes fast; the listing is cheap
		return info
	}
	ctx, cancel := context.WithTimeout(context.Background(), storageWalkCap)
	defer cancel()
	info := StorageInfo{DataDir: s.dir, FreeBytes: -1, Parts: []StoragePart{}}
	if free, err := s.cfg.DiskFree(s.dir); err == nil {
		info.FreeBytes = int64(free) << 20
	}
	for _, name := range storageParts {
		p := filepath.Join(s.dir, name)
		n, ok := dirSize(ctx, p)
		if !ok {
			info.Approximate = true
		}
		info.Parts = append(info.Parts, StoragePart{Name: name, Path: p, Bytes: n})
		info.TotalBytes += n
	}
	if n, ok := dirSize(ctx, s.dir); ok {
		info.TotalBytes = n // everything, state.json included
	} else {
		info.Approximate = true
	}
	info.Models = s.storageModels()
	if !info.Approximate {
		s.mu.Lock()
		s.stoCache = &storageCache{at: time.Now(), info: info}
		s.mu.Unlock()
	}
	return info
}

// modelUse says which files in models/ are not to be touched: the selected local model, the
// target of a running upgrade, and (while an install job runs) any partial download.
func (s *Supervisor) modelUse() (inUse map[string]bool, anyPartial bool) {
	inUse = map[string]bool{}
	cat, _ := s.catalog()
	s.mu.Lock()
	defer s.mu.Unlock()
	anyPartial = s.jobRunning
	if cat == nil {
		return
	}
	if s.st.Mode != "external" && s.st.Model != "" {
		if m, ok := cat.Model(s.st.Model); ok {
			inUse[m.File] = true
		}
	}
	if s.upg != nil && s.upg.activeState() {
		inUse[s.upg.target.File] = true
		inUse[s.upg.target.File+".part"] = true
	}
	return
}

func (s *Supervisor) storageModels() []StorageModel {
	out := []StorageModel{}
	ents, err := os.ReadDir(filepath.Join(s.dir, "models"))
	if err != nil {
		return out
	}
	cat, _ := s.catalog()
	use, anyPartial := s.modelUse()
	for _, e := range ents {
		if !e.Type().IsRegular() {
			continue
		}
		fi, err := e.Info()
		if err != nil {
			continue
		}
		name := e.Name()
		partial := strings.HasSuffix(name, ".part")
		m := StorageModel{File: name, Bytes: fi.Size(), Partial: partial, InUse: use[name] || partial && anyPartial}
		if cat != nil {
			base := strings.TrimSuffix(name, ".part")
			for _, cm := range cat.Models {
				if cm.File == base {
					m.ID = cm.ID
					break
				}
			}
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].File < out[j].File })
	return out
}

// DeleteModel deletes one file in models/ named by catalog id or by file name. It never touches
// the serving model or an active download (ErrInUse) and returns the bytes freed.
func (s *Supervisor) DeleteModel(id, file string) (int64, error) {
	cat, err := s.catalog()
	if err != nil {
		return 0, err
	}
	var catID string
	if file == "" {
		m, ok := cat.Model(id)
		if !ok || m.File == "" {
			return 0, ErrNoModel
		}
		file, catID = m.File, m.ID
	}
	if file != filepath.Base(file) || file == "." || file == ".." || strings.ContainsAny(file, `/\`) {
		return 0, errors.New("file must be a plain file name in models/")
	}
	use, anyPartial := s.modelUse()
	if use[file] || strings.HasSuffix(file, ".part") && anyPartial {
		return 0, ErrInUse
	}
	// A job running now may be downloading this very file.
	s.mu.Lock()
	busy := s.jobRunning
	s.mu.Unlock()
	p := filepath.Join(s.dir, "models", file)
	fi, err := os.Lstat(p)
	if err != nil || !fi.Mode().IsRegular() {
		return 0, ErrNoModel
	}
	if busy {
		for _, cm := range cat.Models {
			if cm.File == file && !s.modelInstalled(cm) {
				return 0, ErrInUse
			}
		}
	}
	if err := os.Remove(p); err != nil {
		return 0, err
	}
	freed := fi.Size()
	base := strings.TrimSuffix(file, ".part")
	if !strings.HasSuffix(file, ".part") {
		if st, err := os.Lstat(p + ".part"); err == nil && st.Mode().IsRegular() && os.Remove(p+".part") == nil {
			freed += st.Size()
		}
	}
	s.mu.Lock()
	if catID == "" {
		for _, cm := range cat.Models {
			if cm.File == base {
				catID = cm.ID
			}
		}
	}
	if catID != "" && s.st.Previous == catID {
		s.st.Previous = ""
	}
	s.stoCache = nil
	s.mu.Unlock()
	s.save()
	s.logf("deleted model file %s (%d MiB)", file, freed>>20)
	return freed, nil
}

// forgetDownloadsLocked clears what points at downloaded files. Caller holds s.mu.
// NOTE for the merge with the custom-GGUF work: forget state.custom here too (never delete its path).
func (s *Supervisor) forgetDownloadsLocked() {
	s.st.Model, s.st.Target, s.st.StarterOwned, s.st.Backend, s.st.Previous = "", "", "", "", ""
	s.st.Book.Entries = nil
	s.st.defaults()
}

// RemoveDownloads starts the job that deletes everything downloaded. The caller checked confirm.
func (s *Supervisor) RemoveDownloads() error {
	s.mu.Lock()
	if s.removing {
		s.mu.Unlock()
		return ErrBusy
	}
	s.removing = true
	s.mu.Unlock()
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.runRemoveDownloads()
		s.mu.Lock()
		s.removing = false
		s.mu.Unlock()
	}()
	return nil
}

func (s *Supervisor) waitUntil(d time.Duration, f func() bool) bool {
	end := time.Now().Add(d)
	for {
		if f() {
			return true
		}
		if time.Now().After(end) {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (s *Supervisor) runRemoveDownloads() {
	s.setPhase("removing", "Removing downloaded files")
	// Cancel the install job and any upgrade, then take the job slot so no install starts.
	s.Cancel()
	s.waitUntil(30*time.Second, func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.jobRunning || s.upg != nil && s.upg.activeState() {
			return false
		}
		s.jobRunning = true
		return true
	})
	s.mu.Lock()
	s.userStopped = true
	lc := s.loadCancel
	s.mu.Unlock()
	if lc != nil {
		lc()
	}
	s.stopAgent()
	s.opMu.Lock()
	s.stopServerLocked()
	s.opMu.Unlock()

	var freed int64
	var left []string
	for _, d := range downloadDirs {
		f, l := s.emptyDir(filepath.Join(s.dir, d))
		freed += f
		left = append(left, l...)
	}
	s.mu.Lock()
	s.forgetDownloadsLocked()
	s.upg = nil
	s.swapping = false
	s.progress = nil
	s.tokS = 0
	s.userStopped = false
	s.jobRunning = false
	s.jobCancel = nil
	s.stoCache = nil
	s.mu.Unlock()
	s.save()
	msg := fmt.Sprintf("Removed downloaded files (%.1f GB freed)", float64(freed)/1e9)
	if len(left) > 0 {
		msg += fmt.Sprintf("; %d item(s) could not be removed (in use?): %s", len(left), strings.Join(left, ", "))
		s.addError("%s", msg)
	}
	s.setPhase("not_installed", msg)
}

// emptyDir deletes the contents of dir (a directory directly under the data dir), never following
// a symlink or junction, and returns the bytes freed and the paths that remain.
func (s *Supervisor) emptyDir(dir string) (freed int64, left []string) {
	fi, err := os.Lstat(dir)
	if err != nil {
		os.MkdirAll(dir, 0o755)
		return 0, nil
	}
	if !fi.IsDir() { // a link or file where the directory should be: leave it, it may point out
		return 0, []string{filepath.Base(dir)}
	}
	ents, _ := os.ReadDir(dir)
	for _, e := range ents {
		f, l := s.removeEntry(filepath.Join(dir, e.Name()))
		freed += f
		left = append(left, l...)
	}
	return freed, left
}

// removeEntry removes a file, link or tree. Links are unlinked, never entered.
func (s *Supervisor) removeEntry(p string) (freed int64, left []string) {
	fi, err := os.Lstat(p)
	if err != nil {
		return 0, nil
	}
	if fi.IsDir() && fi.Mode()&fs.ModeSymlink == 0 && fi.Mode()&fs.ModeIrregular == 0 {
		ents, _ := os.ReadDir(p)
		for _, e := range ents {
			f, l := s.removeEntry(filepath.Join(p, e.Name()))
			freed += f
			left = append(left, l...)
		}
		if len(left) == 0 {
			if !retry(func() error { return os.Remove(p) }) {
				left = append(left, p)
			}
		}
		return freed, left
	}
	size := int64(0)
	if fi.Mode().IsRegular() {
		size = fi.Size()
	}
	if !retry(func() error { return os.Remove(p) }) {
		return 0, []string{p}
	}
	return size, nil
}

// retry runs f a few times: on Windows a file just released by a stopped process can stay locked briefly.
func retry(f func() error) bool {
	for i := 0; i < 8; i++ {
		err := f()
		if err == nil || os.IsNotExist(err) {
			return true
		}
		time.Sleep(150 * time.Millisecond)
	}
	return false
}
