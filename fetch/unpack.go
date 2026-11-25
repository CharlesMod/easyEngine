package fetch

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

const maxUnpackBytes = 40 << 30 // zip-bomb guard

// Unpack extracts a .zip, .tar.gz or .tgz into destDir. It extracts into a temp dir beside
// destDir and renames it into place, so destDir is either the old content or the full new one.
// Entries that would land outside destDir are refused; symlinks are kept when their target is
// relative and stays inside destDir; exec bits are kept.
func Unpack(archive, destDir string) error {
	destDir = filepath.Clean(destDir)
	parent := filepath.Dir(destDir)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(parent, "."+filepath.Base(destDir)+".unpack-*")
	if err != nil {
		return err
	}
	ok := false
	defer func() {
		if !ok {
			os.RemoveAll(tmp)
		}
	}()
	root, err := filepath.EvalSymlinks(tmp)
	if err != nil {
		return err
	}
	x := &extractor{root: root}
	lower := strings.ToLower(archive)
	switch {
	case strings.HasSuffix(lower, ".zip"):
		err = x.zip(archive)
	case strings.HasSuffix(lower, ".tar.gz"), strings.HasSuffix(lower, ".tgz"):
		err = x.targz(archive)
	default:
		err = fmt.Errorf("unpack: unknown archive type %q", filepath.Base(archive))
	}
	if err != nil {
		return err
	}
	var old string
	if _, serr := os.Lstat(destDir); serr == nil {
		old = destDir + ".old-" + filepath.Base(tmp)
		if err := os.Rename(destDir, old); err != nil {
			return err
		}
	}
	if err := os.Rename(tmp, destDir); err != nil {
		if old != "" {
			os.Rename(old, destDir)
		}
		return err
	}
	ok = true
	if old != "" {
		os.RemoveAll(old)
	}
	return nil
}

type extractor struct {
	root    string
	written int64
}

// target returns the absolute path for an archive entry name, refusing escapes.
func (x *extractor) target(name string) (string, error) {
	n := strings.ReplaceAll(name, "\\", "/")
	if n == "" || strings.HasPrefix(n, "/") || filepath.IsAbs(n) || (len(n) > 1 && n[1] == ':') {
		return "", fmt.Errorf("unpack: absolute path %q in archive", name)
	}
	for _, part := range strings.Split(n, "/") {
		if part == ".." {
			return "", fmt.Errorf("unpack: path %q escapes the destination", name)
		}
	}
	p := filepath.Join(x.root, filepath.FromSlash(n))
	if p != x.root && !strings.HasPrefix(p, x.root+string(filepath.Separator)) {
		return "", fmt.Errorf("unpack: path %q escapes the destination", name)
	}
	return p, nil
}

// safeParent creates the parent dirs of p and checks that, after resolving symlinks already
// written, the parent is still inside root.
func (x *extractor) safeParent(p string) error {
	dir := filepath.Dir(p)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return err
	}
	if real != x.root && !strings.HasPrefix(real, x.root+string(filepath.Separator)) {
		return fmt.Errorf("unpack: %q resolves outside the destination", p)
	}
	return nil
}

func (x *extractor) symlink(name, linkname string) error {
	p, err := x.target(name)
	if err != nil {
		return err
	}
	l := strings.ReplaceAll(linkname, "\\", "/")
	if strings.HasPrefix(l, "/") || filepath.IsAbs(l) {
		return fmt.Errorf("unpack: symlink %q has absolute target %q", name, linkname)
	}
	resolved := filepath.Join(filepath.Dir(p), filepath.FromSlash(l))
	if resolved != x.root && !strings.HasPrefix(resolved, x.root+string(filepath.Separator)) {
		return fmt.Errorf("unpack: symlink %q points outside the destination (%q)", name, linkname)
	}
	if err := x.safeParent(p); err != nil {
		return err
	}
	os.Remove(p)
	return os.Symlink(filepath.FromSlash(l), p)
}

func (x *extractor) file(name string, mode fs.FileMode, r io.Reader) error {
	p, err := x.target(name)
	if err != nil {
		return err
	}
	if err := x.safeParent(p); err != nil {
		return err
	}
	if st, err := os.Lstat(p); err == nil && st.Mode()&os.ModeSymlink != 0 {
		os.Remove(p) // never write through a symlink
	}
	perm := mode.Perm() | 0o600
	f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	n, err := io.Copy(f, io.LimitReader(r, maxUnpackBytes-x.written+1))
	x.written += n
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil && x.written > maxUnpackBytes {
		err = errors.New("unpack: archive expands beyond the size limit")
	}
	if err != nil {
		return err
	}
	return os.Chmod(p, perm) // umask must not strip exec bits
}

func (x *extractor) dir(name string, mode fs.FileMode) error {
	p, err := x.target(strings.TrimSuffix(name, "/"))
	if err != nil {
		return err
	}
	if err := x.safeParent(p); err != nil {
		return err
	}
	if err := os.MkdirAll(p, 0o755); err != nil {
		return err
	}
	return nil
}

func (x *extractor) zip(archive string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	// directories and files first, symlinks last, so a link can never redirect a later write
	var links []*zip.File
	for _, f := range zr.File {
		m := f.Mode()
		switch {
		case m&fs.ModeSymlink != 0:
			links = append(links, f)
		case f.FileInfo().IsDir() || strings.HasSuffix(f.Name, "/"):
			if err := x.dir(f.Name, m); err != nil {
				return err
			}
		case m.IsRegular():
			rc, err := f.Open()
			if err != nil {
				return err
			}
			err = x.file(f.Name, m, rc)
			rc.Close()
			if err != nil {
				return err
			}
		default:
			return fmt.Errorf("unpack: unsupported entry %q (mode %v)", f.Name, m)
		}
	}
	for _, f := range links {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		b, err := io.ReadAll(io.LimitReader(rc, 4096))
		rc.Close()
		if err != nil {
			return err
		}
		if err := x.symlink(f.Name, string(b)); err != nil {
			return err
		}
	}
	return nil
}

func (x *extractor) targz(archive string) error {
	fh, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer fh.Close()
	gz, err := gzip.NewReader(fh)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	type link struct{ name, target string }
	var links, hard []link
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := x.dir(h.Name, fs.FileMode(h.Mode)); err != nil {
				return err
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := x.file(h.Name, fs.FileMode(h.Mode), tr); err != nil {
				return err
			}
		case tar.TypeSymlink:
			links = append(links, link{h.Name, h.Linkname})
		case tar.TypeLink:
			hard = append(hard, link{h.Name, h.Linkname})
		case tar.TypeXGlobalHeader, tar.TypeXHeader:
		default:
			return fmt.Errorf("unpack: unsupported entry %q (type %c)", h.Name, h.Typeflag)
		}
	}
	for _, l := range hard { // hard links become copies of the (already extracted) target
		src, err := x.target(l.target)
		if err != nil {
			return err
		}
		st, err := os.Stat(src)
		if err != nil {
			return fmt.Errorf("unpack: hard link %q target %q missing", l.name, l.target)
		}
		in, err := os.Open(src)
		if err != nil {
			return err
		}
		err = x.file(l.name, st.Mode(), in)
		in.Close()
		if err != nil {
			return err
		}
	}
	for _, l := range links {
		if err := x.symlink(l.name, l.target); err != nil {
			return err
		}
	}
	return nil
}

// FindFile returns the first file named name under root (shallowest first), for locating
// llama-server inside an unpacked release whatever its top-level directory is called.
func FindFile(root, name string) (string, bool) {
	best, bestDepth := "", 1<<30
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || d.Name() != name {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		if depth := strings.Count(rel, string(filepath.Separator)); depth < bestDepth {
			best, bestDepth = p, depth
		}
		return nil
	})
	return best, best != ""
}
