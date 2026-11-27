// Package fetch downloads (resumable, hash-checked), unpacks and imports files.
package fetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Progress is reported while a download runs. It marshals as the control API's progress
// object: {step, done_bytes, total_bytes, rate_bps, eta_s}.
type Progress struct {
	Step    string
	Done    int64
	Total   int64
	RateBps float64
	ETA     time.Duration
}

func (p Progress) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Step    string  `json:"step"`
		Done    int64   `json:"done_bytes"`
		Total   int64   `json:"total_bytes"`
		RateBps float64 `json:"rate_bps"`
		ETAs    float64 `json:"eta_s"`
	}{p.Step, p.Done, p.Total, p.RateBps, p.ETA.Seconds()})
}

// ErrHashMismatch is returned (wrapped) when the finished file does not match the expected sha256.
var ErrHashMismatch = errors.New("sha256 mismatch")

// Tunables (variables so tests can shrink them).
var (
	stallTimeout = 60 * time.Second
	// unknownStep is the progress step, in bytes, when the total is not known.
	unknownStep int64 = 4 << 20
	httpClient        = &http.Client{Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 20 * time.Second}).DialContext,
		TLSHandshakeTimeout:   20 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
	}}
)

type permanentError struct{ error }

func (p permanentError) Unwrap() error { return p.error }

// Again is asked after an attempt that failed having moved no bytes. It answers nil once something has happened that could change the outcome (the network came back, a client's word), and the download tries again at once; an error ends the download with it. A nil Again ends the download with the attempt's error.
type Again func(ctx context.Context, err error) error

// stepOf is how many bytes make one progress step: 1% of total, or 4 MiB when the total is unknown.
func stepOf(total int64) int64 {
	if total <= 0 {
		return unknownStep
	}
	return max(total/100, 1)
}

// Download fetches url into dest through dest+".part", resuming with an HTTP Range request.
// sha256 (hex) and size (bytes) are checked when non-empty / positive. Cancelling ctx returns
// ctx.Err() and keeps the .part file so a later call resumes. A finished file already at dest
// with the right hash is left alone.
//
// An attempt that wrote bytes past the furthest byte this call has held and then failed resumes at once (that bounds it: a server that ignores Range and re-sends the same prefix is no progress). An attempt
// that wrote none asks again; see Again. Progress is reported per step (1% of the total, or every
// 4 MiB when the total is unknown), never on a clock.
func Download(ctx context.Context, url, dest, sha256hex string, size int64, onProgress func(Progress), again Again) error {
	if onProgress == nil {
		onProgress = func(Progress) {}
	}
	sha256hex = strings.ToLower(strings.TrimSpace(sha256hex))
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	if st, err := os.Stat(dest); err == nil && st.Mode().IsRegular() && (size <= 0 || st.Size() == size) {
		if sha256hex == "" {
			onProgress(Progress{Step: "downloading", Done: st.Size(), Total: st.Size()})
			return nil
		}
		if got, err := hashFile(ctx, dest, nil); err == nil && got == sha256hex {
			onProgress(Progress{Step: "downloading", Done: st.Size(), Total: st.Size()})
			return nil
		}
	}
	part := dest + ".part"
	var reach int64 // the furthest byte this Download has held in the .part, across attempts: progress is growth beyond it
	for {
		progressed, err := attemptDownload(ctx, url, part, size, &reach, onProgress)
		if err == nil {
			return finish(ctx, part, dest, sha256hex, size, onProgress)
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		var pe permanentError
		if errors.As(err, &pe) {
			return pe.error
		}
		err = fmt.Errorf("download %s: %w", url, err)
		if progressed {
			continue // the .part grew: resume at once
		}
		if again == nil {
			return err
		}
		if aerr := again(ctx, err); aerr != nil {
			return aerr
		}
	}
}

func finish(ctx context.Context, part, dest, sha256hex string, size int64, onProgress func(Progress)) error {
	st, err := os.Stat(part)
	if err != nil {
		return err
	}
	if size > 0 && st.Size() != size {
		os.Remove(part)
		return fmt.Errorf("size mismatch: got %d bytes, want %d", st.Size(), size)
	}
	if sha256hex != "" {
		start := time.Now()
		got, err := hashFile(ctx, part, func(done int64) {
			onProgress(Progress{Step: "verifying", Done: done, Total: st.Size(), RateBps: float64(done) / (time.Since(start).Seconds() + 1e-9)})
		})
		if err != nil {
			return err // cancelled: keep the .part
		}
		if got != sha256hex {
			os.Remove(part)
			return fmt.Errorf("%w: got %s, want %s", ErrHashMismatch, got, sha256hex)
		}
	}
	if err := os.Rename(part, dest); err != nil {
		return err
	}
	onProgress(Progress{Step: "downloading", Done: st.Size(), Total: st.Size()})
	return nil
}

func hashFile(ctx context.Context, path string, progress func(int64)) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, 1<<20)
	var done int64
	var total int64
	if st, err := f.Stat(); err == nil {
		total = st.Size()
	}
	step := stepOf(total)
	var reported int64
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := f.Read(buf)
		if n > 0 {
			h.Write(buf[:n])
			done += int64(n)
			if progress != nil && done-reported >= step {
				progress(done)
				reported = done
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
	}
	if progress != nil {
		progress(done)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// attemptDownload makes one request, appending to part. It returns nil when part holds the
// whole file (all bytes of the response were written and, if size is known, it is complete).
func attemptDownload(ctx context.Context, url, part string, size int64, reach *int64, onProgress func(Progress)) (progressed bool, err error) {
	var have int64
	if st, err := os.Stat(part); err == nil {
		have = st.Size()
	}
	if size > 0 && have == size {
		return false, nil // a previous run finished the bytes; go straight to verification
	}
	if size > 0 && have > size {
		os.Remove(part)
		have = 0
	}
	*reach = max(*reach, have)      // what the .part held at the start counts as reached
	grew := func(done int64) bool { // progress is growth past the furthest byte ever held, not past this attempt's start
		g := done > *reach
		*reach = max(*reach, done)
		return g
	}
	reqCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "GET", url, nil)
	if err != nil {
		return false, permanentError{err}
	}
	req.Header.Set("User-Agent", "easy-engine")
	if have > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(have, 10)+"-")
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		start, ok := contentRangeStart(resp.Header.Get("Content-Range"))
		if !ok || start != have {
			return false, fmt.Errorf("server answered a different range (%q) than asked (%d-)", resp.Header.Get("Content-Range"), have)
		}
		flags |= os.O_APPEND
	case http.StatusOK:
		// no Range support (or the file is fresh): start over
		flags |= os.O_TRUNC
		have = 0
	case http.StatusRequestedRangeNotSatisfiable:
		if size > 0 && have == size {
			return false, nil
		}
		// our .part is longer than the file or the file changed: drop it and retry from zero.
		// That changes the outcome, so it counts as progress: the caller starts again at once.
		os.Remove(part)
		return true, errors.New("416 range not satisfiable; restarting from zero")
	default:
		if resp.StatusCode >= 400 && resp.StatusCode < 500 && resp.StatusCode != http.StatusRequestTimeout && resp.StatusCode != http.StatusTooManyRequests {
			return false, permanentError{fmt.Errorf("download %s: %s", url, resp.Status)}
		}
		return false, fmt.Errorf("download %s: %s", url, resp.Status)
	}

	total := size
	if total <= 0 {
		if resp.StatusCode == http.StatusPartialContent {
			total = rangeTotal(resp.Header.Get("Content-Range"))
		} else if resp.ContentLength > 0 {
			total = resp.ContentLength
		}
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return false, permanentError{err}
	}
	defer f.Close()

	// watchdog for silence: a stream that sends no byte for stallTimeout is cut and the attempt fails.
	stall := time.AfterFunc(stallTimeout, cancel)
	defer stall.Stop()
	start := time.Now()
	startBytes := have
	done := have
	step := stepOf(total)
	var reported int64
	first := true
	report := func(final bool) {
		if !final && !first && done-reported < step {
			return
		}
		first = false
		reported = done
		rate := float64(done-startBytes) / (time.Since(start).Seconds() + 1e-9)
		var eta time.Duration
		if total > done && rate > 0 {
			eta = time.Duration(float64(total-done) / rate * float64(time.Second))
		}
		onProgress(Progress{Step: "downloading", Done: done, Total: total, RateBps: rate, ETA: eta})
	}
	report(true)
	buf := make([]byte, 256<<10)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			stall.Reset(stallTimeout)
			if _, werr := f.Write(buf[:n]); werr != nil {
				return false, permanentError{werr}
			}
			done += int64(n)
			report(false)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Sync()
			return grew(done), rerr
		}
	}
	report(true)
	if total > 0 && done < total {
		return grew(done), fmt.Errorf("connection ended at %d of %d bytes", done, total)
	}
	return true, nil
}

func contentRangeStart(h string) (int64, bool) {
	rest, ok := strings.CutPrefix(h, "bytes ")
	if !ok {
		return 0, false
	}
	a, _, ok := strings.Cut(rest, "-")
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(a, 10, 64)
	return n, err == nil
}

func rangeTotal(h string) int64 {
	_, t, ok := strings.Cut(h, "/")
	if !ok {
		return 0
	}
	n, _ := strconv.ParseInt(t, 10, 64)
	return n
}
