// STUB: replaced by eng/fetch
package fetch

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Progress struct {
	Step        string
	Done, Total int64
	RateBps     float64
	ETA         time.Duration
}

func Download(ctx context.Context, url, dest, sum string, size int64, onProgress func(Progress)) error {
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}
	os.MkdirAll(filepath.Dir(dest), 0o755)
	f, err := os.Create(dest + ".part")
	if err != nil {
		return err
	}
	h := sha256.New()
	var done int64
	buf := make([]byte, 64<<10)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			f.Write(buf[:n])
			h.Write(buf[:n])
			done += int64(n)
			if onProgress != nil {
				onProgress(Progress{Done: done, Total: size})
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			return rerr
		}
	}
	f.Close()
	if sum != "" && hex.EncodeToString(h.Sum(nil)) != strings.ToLower(sum) {
		os.Remove(dest + ".part")
		return fmt.Errorf("sha256 mismatch for %s", url)
	}
	return os.Rename(dest+".part", dest)
}

func Unpack(archive, destDir string) error {
	os.MkdirAll(destDir, 0o755)
	if strings.HasSuffix(archive, ".zip") {
		zr, err := zip.OpenReader(archive)
		if err != nil {
			return err
		}
		defer zr.Close()
		for _, f := range zr.File {
			p := filepath.Join(destDir, filepath.Clean("/"+f.Name))
			if f.FileInfo().IsDir() {
				os.MkdirAll(p, 0o755)
				continue
			}
			os.MkdirAll(filepath.Dir(p), 0o755)
			rc, _ := f.Open()
			w, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, f.Mode()|0o600)
			if err != nil {
				return err
			}
			io.Copy(w, rc)
			w.Close()
			rc.Close()
		}
		return nil
	}
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		p := filepath.Join(destDir, filepath.Clean("/"+h.Name))
		if h.Typeflag == tar.TypeDir {
			os.MkdirAll(p, 0o755)
			continue
		}
		os.MkdirAll(filepath.Dir(p), 0o755)
		w, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(h.Mode)|0o600)
		if err != nil {
			return err
		}
		io.Copy(w, tr)
		w.Close()
	}
}

func Import(src, dest, sum string) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	h := sha256.New()
	io.Copy(h, f)
	f.Close()
	if sum != "" && hex.EncodeToString(h.Sum(nil)) != strings.ToLower(sum) {
		return fmt.Errorf("sha256 mismatch for %s", src)
	}
	os.MkdirAll(filepath.Dir(dest), 0o755)
	os.Remove(dest)
	if os.Link(src, dest) == nil {
		return nil
	}
	in, _ := os.Open(src)
	defer in.Close()
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
