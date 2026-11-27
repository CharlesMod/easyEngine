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
	err := Download(context.Background(), url, dest, sum, int64(len(data)), func(p Progress) { last = p; steps[p.Step] = true }, nil)
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
	if err := Download(context.Background(), url2, dest, sum, int64(len(data)), nil, nil); err != nil {
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
	// a cut after bytes arrived resumes at once: again must never be asked
	noAgain := func(ctx context.Context, err error) error {
		t.Errorf("again asked after progress: %v", err)
		return err
	}
	if err := Download(context.Background(), url, dest, sum, int64(len(data)), nil, noAgain); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, data) {
		t.Fatal("content differs after resume")
	}
	if len(ranges) != 2 || ranges[1] != "bytes=700000-" {
		t.Errorf("request ranges: %q", ranges)
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
	if err := Download(context.Background(), url, dest, sum, 0, nil, nil); err != nil {
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
	err := Download(context.Background(), url, dest, strings.Repeat("0", 64), int64(len(data)), nil, nil)
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
	if err := Download(context.Background(), url, dest, sum, 0, nil, nil); err != nil {
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
	if err := Download(context.Background(), url, dest, sum, 0, nil, nil); err != nil {
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
	if err := Download(context.Background(), url, dest, sum, int64(len(data)), nil, nil); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(dest); !bytes.Equal(got, data) {
		t.Fatal("content differs (200 reply must restart, not append)")
	}
}

func TestDownload404IsPermanent(t *testing.T) {
	var n atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { n.Add(1); http.NotFound(w, r) }))
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil, nil)
	if err == nil || n.Load() != 1 {
		t.Fatalf("err=%v requests=%d", err, n.Load())
	}
}

// again answers nil for the first ok calls, then fail; calls counts how often it was asked.
func againAnswers(ok int, fail error, calls *atomic.Int32) Again {
	return func(ctx context.Context, err error) error {
		if int(calls.Add(1)) > ok {
			return fail
		}
		return nil
	}
}

func TestDownloadNoProgressAsksAgain(t *testing.T) {
	data, sum := payload(10000)
	var n, calls atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) < 3 {
			http.Error(w, "busy", 503)
			return
		}
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), sum, 0, nil, againAnswers(5, errors.New("no"), &calls))
	if err != nil || n.Load() != 3 || calls.Load() != 2 {
		t.Fatalf("err=%v requests=%d again=%d", err, n.Load(), calls.Load())
	}
}

// A server that ignores Range and sends only the first k bytes each time is
// progress once (past the old .part) and then none: Download must ask Again
// instead of going round at once for ever.
func TestDownloadRangeIgnoredNoEndlessLoop(t *testing.T) {
	data, sum := payload(10000)
	const k = 4000
	var n, calls atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.WriteHeader(200)
		w.Write(data[:k])
		w.(http.Flusher).Flush()
		conn, _, err := w.(http.Hijacker).Hijack()
		if err == nil {
			conn.Close()
		}
	}))
	dest := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(dest+".part", data[:1000], 0o644); err != nil {
		t.Fatal(err)
	}
	stop := errors.New("client said stop")
	err := Download(context.Background(), url, dest, sum, int64(len(data)), nil, againAnswers(0, stop, &calls))
	if !errors.Is(err, stop) {
		t.Fatalf("err=%v", err)
	}
	// attempt 1 grew the .part past 1000 (progress); attempt 2 reached no further, so Again was asked once
	if n.Load() != 2 || calls.Load() != 1 {
		t.Fatalf("requests=%d again=%d", n.Load(), calls.Load())
	}
}

func TestDownloadAgainErrorEnds(t *testing.T) {
	var n, calls atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { n.Add(1); http.Error(w, "busy", 503) }))
	stop := errors.New("client said stop")
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil, againAnswers(2, stop, &calls))
	if !errors.Is(err, stop) || n.Load() != 3 || calls.Load() != 3 {
		t.Fatalf("err=%v requests=%d again=%d", err, n.Load(), calls.Load())
	}
}

func TestDownloadNilAgainEndsAtFirstFailure(t *testing.T) {
	var n atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { n.Add(1); http.Error(w, "busy", 503) }))
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "503") || n.Load() != 1 {
		t.Fatalf("err=%v requests=%d", err, n.Load())
	}
}

func TestDownloadAgainNotAskedForPermanent(t *testing.T) {
	var calls atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) }))
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil, againAnswers(9, nil, &calls))
	if err == nil || calls.Load() != 0 {
		t.Fatalf("err=%v again=%d", err, calls.Load())
	}
}

func TestDownloadProgressPerStep(t *testing.T) {
	data, sum := payload(3 << 20)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(data))
	}))
	total := int64(len(data))
	var dl []int64
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), sum, total, func(p Progress) {
		if p.Step == "downloading" {
			dl = append(dl, p.Done)
		}
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	step := stepOf(total)
	if dl[0] != 0 || dl[len(dl)-1] != total {
		t.Fatalf("first/last reports %d, %d", dl[0], dl[len(dl)-1])
	}
	for i := 1; i < len(dl); i++ {
		if dl[i] < total && dl[i]-dl[i-1] < step {
			t.Errorf("report %d (%d) came less than a step (%d) after %d", i, dl[i], step, dl[i-1])
		}
	}
	if len(dl) > 105 {
		t.Errorf("%d reports for 100 steps", len(dl))
	}
}

func TestDownloadProgressUnknownTotal(t *testing.T) {
	old := unknownStep
	unknownStep = 100000
	defer func() { unknownStep = old }()
	data, _ := payload(1 << 20)
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for off := 0; off < len(data); off += 32768 { // chunked: no length
			w.Write(data[off:min(off+32768, len(data))])
			w.(http.Flusher).Flush()
		}
	}))
	var dl []int64
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, func(p Progress) { dl = append(dl, p.Done) }, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(dl); i++ {
		if dl[i] != int64(len(data)) && dl[i]-dl[i-1] < unknownStep {
			t.Errorf("report %d (%d) less than a step after %d", i, dl[i], dl[i-1])
		}
	}
	if dl[len(dl)-1] != int64(len(data)) || len(dl) > 14 {
		t.Errorf("reports %v", dl)
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
	defer cancel()
	err := Download(ctx, url, dest, sum, int64(len(data)), func(p Progress) {
		if p.Done >= 500000 { // the .part holds the bytes already: they are written before they are reported
			cancel()
		}
	}, nil)
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

func TestDownloadStallEndsAttempt(t *testing.T) {
	old := stallTimeout
	stallTimeout = 100 * time.Millisecond // the watchdog for silence, shrunk
	defer func() { stallTimeout = old }()
	var reqs, calls atomic.Int32
	url := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.Add(1)
		w.Header().Set("Content-Length", "1000000")
		w.Write([]byte("x"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	stop := errors.New("stop")
	// attempt 1 moves a byte and resumes at once; attempt 2 (the server ignores Range and
	// restarts) moves no more than the .part held, so again is asked, and ends it.
	err := Download(context.Background(), url, filepath.Join(t.TempDir(), "f"), "", 0, nil, againAnswers(0, stop, &calls))
	if !errors.Is(err, stop) || reqs.Load() != 2 || calls.Load() != 1 {
		t.Fatalf("err=%v requests=%d again=%d", err, reqs.Load(), calls.Load())
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
