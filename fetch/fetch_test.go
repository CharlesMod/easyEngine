package fetch

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func init() {
	backoffBase = time.Millisecond
	backoffMax = 5 * time.Millisecond
	progressEvery = time.Millisecond
}

func payload(n int) ([]byte, string) {
	b := make([]byte, n)
	rand.New(rand.NewSource(1)).Read(b)
	s := sha256.Sum256(b)
	return b, hex.EncodeToString(s[:])
}

func serve(t *testing.T, h http.Handler) string {
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	return s.URL
}

func TestDownloadPlain(t *testing.T) {
	data, sum := payload(3 << 20)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	dest := filepath.Join(t.TempDir(), "sub", "f.bin")
	var last Progress
	var steps = map[string]bool{}
	err := Download(context.Background(), url, dest, sum, int64(len(data)), func(p Progress) { last = p; steps[p.Step] = true })
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, data) {
		t.Fatal("content differs")
	}
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Error(".part left behind")
	}
	if last.Done != int64(len(data)) || last.Total != int64(len(data)) || !steps["verifying"] || !steps["downloading"] {
		t.Errorf("progress %+v %v", last, steps)
	}
	// second call: already there, no request needed
	url2 := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("unexpected request") }))
	if err := Download(context.Background(), url2, dest, sum, int64(len(data)), nil); err != nil {
		t.Fatal(err)
	}
}

func TestDownloadResumesAfterCut(t *testing.T) {
	data, sum := payload(2 << 20)
	var reqs atomic.Int32
	var ranges []string
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := reqs.Add(1)
		ranges = append(ranges, r.Header.Get("Range"))
		if n == 1 {
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			w.WriteHeader(200)
			w.Write(data[:700000])
			w.(http.Flusher).Flush()
			c, _, _ := w.(http.Hijacker).Hijack()
			c.Close()
			return
		}
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	if err := Download(context.Background(), url, dest, sum, int64(len(data)), nil); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, data) {
		t.Fatal("content differs after resume")
	}
	if len(ranges) < 2 || ranges[1] != "bytes=700000-" {
		t.Errorf("second request ranges: %q", ranges)
	}
}

func TestDownloadResumesFromExistingPart(t *testing.T) {
	data, sum := payload(1 << 20)
	var gotRange string
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRange = r.Header.Get("Range")
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	os.WriteFile(dest+".part", data[:400000], 0o644)
	if err := Download(context.Background(), url, dest, sum, 0, nil); err != nil {
		t.Fatal(err)
	}
	if gotRange != "bytes=400000-" {
		t.Errorf("range %q", gotRange)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, data) {
		t.Fatal("content differs")
	}
}

func TestDownloadWrongHash(t *testing.T) {
	data, _ := payload(100000)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	err := Download(context.Background(), url, dest, strings.Repeat("0", 64), int64(len(data)), nil)
	if !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("want hash mismatch, got %v", err)
	}
	for _, p := range []string{dest, dest + ".part"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s must not exist after a bad hash", p)
		}
	}
}

func TestDownload416WhenPartIsComplete(t *testing.T) {
	data, sum := payload(50000)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data)) // 416 for bytes=50000-
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	os.WriteFile(dest+".part", data, 0o644)
	if err := Download(context.Background(), url, dest, sum, 0, nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, data) {
		t.Fatal("content differs")
	}
}

func TestDownload416StalePartRestarts(t *testing.T) {
	data, sum := payload(50000)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	os.WriteFile(dest+".part", append(append([]byte{}, data...), make([]byte, 999)...), 0o644) // longer than the file
	if err := Download(context.Background(), url, dest, sum, 0, nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, data) {
		t.Fatal("content differs")
	}
}

func TestDownloadNoRangeServer(t *testing.T) {
	data, sum := payload(300000)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data))) // ignores Range, always 200
		w.Write(data)
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	os.WriteFile(dest+".part", data[:1000], 0o644)
	if err := Download(context.Background(), url, dest, sum, int64(len(data)), nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, data) {
		t.Fatal("content differs (200 reply must restart, not append)")
	}
}

func TestDownload404IsPermanent(t *testing.T) {
	var n atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { n.Add(1); http.NotFound(w, r) }))
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil)
	if err == nil || n.Load() != 1 {
		t.Fatalf("err=%v requests=%d", err, n.Load())
	}
}

func TestDownloadRetries5xx(t *testing.T) {
	data, sum := payload(10000)
	var n atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) < 3 {
			http.Error(w, "busy", 503)
			return
		}
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	if err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), sum, 0, nil); err != nil || n.Load() != 3 {
		t.Fatalf("err=%v requests=%d", err, n.Load())
	}
}

func TestDownloadCancelKeepsPart(t *testing.T) {
	data, sum := payload(2 << 20)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.Write(data[:600000])
		w.(http.Flusher).Flush()
		<-r.Context().Done() // stalls until the client leaves
	}))
	dest := filepath.Join(t.TempDir(), "f.bin")
	ctx, cancel := context.WithCancel(context.Background())
	go func() { // cancel once the .part holds a good chunk (progress callbacks are throttled)
		for {
			if st, err := os.Stat(dest + ".part"); err == nil && st.Size() >= 500000 {
				cancel()
				return
			}
			time.Sleep(2 * time.Millisecond)
		}
	}()
	err := Download(ctx, url, dest, sum, int64(len(data)), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
	st, serr := os.Stat(dest + ".part")
	if serr != nil || st.Size() < 500000 {
		t.Fatalf(".part missing or short: %v %v", st, serr)
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Error("dest must not exist after cancel")
	}
}

func TestDownloadStallGivesUp(t *testing.T) {
	old := stallTimeout
	stallTimeout = 100 * time.Millisecond
	defer func() { stallTimeout = old }()
	oldAttempts := maxAttempts
	maxAttempts = 2
	defer func() { maxAttempts = oldAttempts }()
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000000")
		w.Write([]byte("x"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	if err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil); err == nil {
		t.Fatal("expected failure from stalled server")
	}
}

func TestProgressJSON(t *testing.T) {
	b, _ := Progress{Step: "downloading", Done: 5, Total: 10, RateBps: 2.5, ETA: 2 * time.Second}.MarshalJSON()
	if string(b) != `{"step":"downloading","done_bytes":5,"total_bytes":10,"rate_bps":2.5,"eta_s":2}` {
		t.Error(string(b))
	}
}

// ---- Unpack ----

func tgz(t *testing.T, entries func(tw *tar.Writer)) string {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	entries(tw)
	tw.Close()
	gz.Close()
	p := filepath.Join(t.TempDir(), "a.tar.gz")
	os.WriteFile(p, buf.Bytes(), 0o644)
	return p
}

func tfile(tw *tar.Writer, name string, mode int64, body string) {
	tw.WriteHeader(&tar.Header{Name: name, Mode: mode, Size: int64(len(body)), Typeflag: tar.TypeReg})
	tw.Write([]byte(body))
}

func TestUnpackTarGz(t *testing.T) {
	a := tgz(t, func(tw *tar.Writer) {
		tw.WriteHeader(&tar.Header{Name: "llama-b1/", Mode: 0o755, Typeflag: tar.TypeDir})
		tfile(tw, "llama-b1/llama-server", 0o755, "#!bin")
		tfile(tw, "llama-b1/libggml.so.0.9.1", 0o644, "lib")
		tw.WriteHeader(&tar.Header{Name: "llama-b1/libggml.so.0", Linkname: "libggml.so.0.9.1", Typeflag: tar.TypeSymlink})
		tw.WriteHeader(&tar.Header{Name: "llama-b1/sub/up", Linkname: "../libggml.so.0", Typeflag: tar.TypeSymlink})
		tw.WriteHeader(&tar.Header{Name: "llama-b1/hard", Linkname: "llama-b1/libggml.so.0.9.1", Typeflag: tar.TypeLink})
	})
	dest := filepath.Join(t.TempDir(), "backends", "cuda")
	if err := Unpack(a, dest); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(filepath.Join(dest, "llama-b1", "llama-server"))
	if err != nil || st.Mode().Perm()&0o100 == 0 {
		t.Fatalf("exec bit lost: %v %v", st, err)
	}
	if l, err := os.Readlink(filepath.Join(dest, "llama-b1", "libggml.so.0")); err != nil || l != "libggml.so.0.9.1" {
		t.Errorf("symlink %q %v", l, err)
	}
	if b, err := os.ReadFile(filepath.Join(dest, "llama-b1", "sub", "up")); err != nil || string(b) != "lib" {
		t.Errorf("nested symlink %q %v", b, err)
	}
	if b, _ := os.ReadFile(filepath.Join(dest, "llama-b1", "hard")); string(b) != "lib" {
		t.Errorf("hard link copy %q", b)
	}
	if p, ok := FindFile(dest, "llama-server"); !ok || filepath.Base(filepath.Dir(p)) != "llama-b1" {
		t.Errorf("FindFile %q %v", p, ok)
	}
	// replacing is atomic: old content gone, no temp dirs left beside it
	a2 := tgz(t, func(tw *tar.Writer) { tfile(tw, "new.txt", 0o644, "n") })
	if err := Unpack(a2, dest); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dest, "llama-b1")); !os.IsNotExist(err) {
		t.Error("old content must be replaced")
	}
	ents, _ := os.ReadDir(filepath.Dir(dest))
	if len(ents) != 1 {
		t.Errorf("leftovers: %v", ents)
	}
}

func TestUnpackTraversalRefused(t *testing.T) {
	bad := map[string]func(tw *tar.Writer){
		"dotdot":   func(tw *tar.Writer) { tfile(tw, "../evil", 0o644, "x") },
		"nested":   func(tw *tar.Writer) { tfile(tw, "a/../../evil", 0o644, "x") },
		"absolute": func(tw *tar.Writer) { tfile(tw, "/tmp/evil", 0o644, "x") },
		"linkout": func(tw *tar.Writer) {
			tw.WriteHeader(&tar.Header{Name: "l", Linkname: "../x", Typeflag: tar.TypeSymlink})
		},
		"linkabs": func(tw *tar.Writer) {
			tw.WriteHeader(&tar.Header{Name: "l", Linkname: "/etc/passwd", Typeflag: tar.TypeSymlink})
		},
		"linkdeepout": func(tw *tar.Writer) {
			tw.WriteHeader(&tar.Header{Name: "a/b/l", Linkname: "../../../x", Typeflag: tar.TypeSymlink})
		},
		"hardout": func(tw *tar.Writer) { tw.WriteHeader(&tar.Header{Name: "h", Linkname: "../x", Typeflag: tar.TypeLink}) },
		"device":  func(tw *tar.Writer) { tw.WriteHeader(&tar.Header{Name: "d", Typeflag: tar.TypeChar}) },
	}
	for name, f := range bad {
		base := t.TempDir()
		dest := filepath.Join(base, "out")
		if err := Unpack(tgz(t, f), dest); err == nil {
			t.Errorf("%s: expected refusal", name)
		}
		if _, err := os.Stat(dest); !os.IsNotExist(err) {
			t.Errorf("%s: dest must not exist after a refused archive", name)
		}
		if _, err := os.Stat(filepath.Join(base, "evil")); err == nil {
			t.Errorf("%s: wrote outside", name)
		}
		if ents, _ := os.ReadDir(base); len(ents) != 0 {
			t.Errorf("%s: temp dir left: %v", name, ents)
		}
	}
}

func TestUnpackZip(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("bin/")
	_ = w
	hdr := &zip.FileHeader{Name: "bin/llama-server.exe", Method: zip.Deflate}
	hdr.SetMode(0o755)
	fw, _ := zw.CreateHeader(hdr)
	fw.Write([]byte("exe"))
	lh := &zip.FileHeader{Name: "bin/link"}
	lh.SetMode(0o777 | os.ModeSymlink)
	lw, _ := zw.CreateHeader(lh)
	lw.Write([]byte("llama-server.exe"))
	zw.Close()
	a := filepath.Join(t.TempDir(), "a.zip")
	os.WriteFile(a, buf.Bytes(), 0o644)
	dest := filepath.Join(t.TempDir(), "z")
	if err := Unpack(a, dest); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(filepath.Join(dest, "bin", "llama-server.exe"))
	if err != nil || st.Mode().Perm()&0o100 == 0 {
		t.Fatalf("%v %v", st, err)
	}
	if l, _ := os.Readlink(filepath.Join(dest, "bin", "link")); l != "llama-server.exe" {
		t.Errorf("zip symlink %q", l)
	}
}

func TestUnpackZipTraversal(t *testing.T) {
	for _, name := range []string{"../evil", "a/../../evil", "/abs/evil", `..\evil`} {
		var buf bytes.Buffer
		zw := zip.NewWriter(&buf)
		fw, _ := zw.Create(name)
		fw.Write([]byte("x"))
		zw.Close()
		base := t.TempDir()
		a := filepath.Join(base, "a.zip")
		os.WriteFile(a, buf.Bytes(), 0o644)
		if err := Unpack(a, filepath.Join(base, "out")); err == nil {
			t.Errorf("%q: expected refusal", name)
		}
	}
}

func TestUnpackSymlinkChainCannotRedirect(t *testing.T) {
	// "a" -> "." is fine (inside); a file written through a/ stays inside
	a := tgz(t, func(tw *tar.Writer) {
		tw.WriteHeader(&tar.Header{Name: "a", Linkname: ".", Typeflag: tar.TypeSymlink})
		tfile(tw, "b/x", 0o644, "x")
	})
	dest := filepath.Join(t.TempDir(), "o")
	if err := Unpack(a, dest); err != nil {
		t.Fatal(err)
	}
}

func TestUnpackUnknownType(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.rar")
	os.WriteFile(p, []byte("x"), 0o644)
	if err := Unpack(p, filepath.Join(t.TempDir(), "o")); err == nil {
		t.Error("expected error")
	}
}

// ---- Import ----

func TestImport(t *testing.T) {
	data, sum := payload(200000)
	d := t.TempDir()
	src := filepath.Join(d, "src.gguf")
	os.WriteFile(src, data, 0o644)
	dest := filepath.Join(d, "models", "m.gguf")
	if err := Import(src, dest, sum); err != nil {
		t.Fatal(err)
	}
	a, _ := os.Stat(src)
	b, _ := os.Stat(dest)
	if !os.SameFile(a, b) {
		t.Log("not hard-linked (copied); fine on this filesystem")
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, data) {
		t.Fatal("content")
	}
	if err := Import(src, dest, sum); err != nil { // idempotent
		t.Fatal(err)
	}
	if err := Import(src, filepath.Join(d, "models", "bad.gguf"), strings.Repeat("1", 64)); !errors.Is(err, ErrHashMismatch) {
		t.Errorf("want mismatch, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(d, "models", "bad.gguf")); !os.IsNotExist(err) {
		t.Error("bad import must leave nothing")
	}
	if err := Import(filepath.Join(d, "nope"), dest, ""); err == nil {
		t.Error("missing source")
	}
	if err := Import(d, filepath.Join(d, "x"), ""); err == nil {
		t.Error("directory source")
	}
}
