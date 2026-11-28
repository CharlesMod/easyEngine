package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"math/rand"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
	"unsafe"
)

// inside says whether p's bytes lie within b's.
func inside(p, b []byte) bool {
	if len(p) == 0 {
		return false
	}
	lo := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	x := uintptr(unsafe.Pointer(unsafe.SliceData(p)))
	return x >= lo && x+uintptr(len(p)) <= lo+uintptr(len(b))
}

// Each image's base64 is a view of the body, not a copy; an escaped character (which base64
// never needs) is the one case that copies.
func TestParseLookHoldsNoCopy(t *testing.T) {
	body := []byte(`{"images":[{"data":"iVBORw0KGgo=","label":"a"},{"label":"b","data":"/9j/4AAQ"}],"question":"q"}`)
	q, ce := parseLook(body)
	if ce != nil {
		t.Fatal(ce.Msg)
	}
	for i, im := range q.images {
		if !inside(im.data, body) {
			t.Fatalf("image %d's data is a copy", i)
		}
	}
	if string(q.images[0].data) != "iVBORw0KGgo=" || q.images[1].label != "b" || string(q.images[1].data) != "/9j/4AAQ" {
		t.Fatalf("parsed %+v", q.images)
	}
	esc := []byte(`{"images":[{"data":"\/9j\/4AAQ"}],"question":"q"}`)
	q, ce = parseLook(esc)
	if ce != nil || string(q.images[0].data) != "/9j/4AAQ" || inside(q.images[0].data, esc) {
		t.Fatalf("escaped data: %v %q", ce, q.images)
	}
}

// The walk agrees with encoding/json on every body the strict parse takes, and never panics.
func FuzzParseLook(f *testing.F) {
	for _, s := range []string{
		`{"images":[{"data":"AAAA"}],"question":"q"}`,
		`{"images":[{"data":"AAAA","label":"x\"y"}],"question":"qé","detail":"low","max_tokens":5,"id_slot":0,"stream":true}`,
		` { "images" : [ { "data" : "A\/A" } ] , "question" : "q" } `,
		`{"images":[],"question":"q"}`,
		`{"images":[{"data":"A"}],"question":"q","question":"r"}`,
		`{"images":[{"data":"A"}],"question":"q"} {}`,
		`[1]`, `{"a":{"b":[1,{"c":"]}"}]}}`,
	} {
		f.Add([]byte(s))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		q, ce := parseLook(body)
		if ce != nil {
			return
		}
		var ref struct {
			Images []struct {
				Data  string `json:"data"`
				Label string `json:"label"`
			} `json:"images"`
			Question string `json:"question"`
		}
		if err := json.Unmarshal(body, &ref); err != nil {
			t.Fatalf("taken but not JSON: %v", err)
		}
		if len(ref.Images) != len(q.images) || ref.Question != q.question {
			t.Fatalf("walk %+v, json %+v", q, ref)
		}
		for i, im := range q.images {
			// bytes not UTF-8 stay as sent in the view (encoding/json would replace them); base64
			// decoding refuses them either way
			if (utf8.Valid(im.data) && string(im.data) != ref.Images[i].Data) || im.label != ref.Images[i].Label {
				t.Fatalf("image %d: walk %q %q, json %q %q", i, im.data, im.label, ref.Images[i].Data, ref.Images[i].Label)
			}
		}
	})
}

// Measured memory and time of a look's parse and build at the remote body bound: a 16 MiB body
// of four 4096x4096 PNGs (each 2.9 MiB, partly noise so the file is dense), built at edge 2048.
// The parse allocates next to nothing (the body is walked in place); each image's step is bounded
// from above by the heap live before it plus everything it allocates (GC off within the step).
// Figures are logged for docs/ENGINE.md section 7h; the test fails over 128 MiB per step beyond
// the body, or over 1 MiB for the parse.
func TestLookBuildMemory(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 4096, 4096))
	rng := rand.New(rand.NewSource(1))
	for i := range img.Pix {
		img.Pix[i] = 200
	}
	rng.Read(img.Pix[:len(img.Pix)/26]) // noise rows on top: the file is about 2.9 MiB
	var file bytes.Buffer
	if err := png.Encode(&file, img); err != nil {
		t.Fatal(err)
	}
	img = nil
	b64 := base64.StdEncoding.EncodeToString(file.Bytes())
	file = bytes.Buffer{}
	parts := make([]string, 4)
	for i := range parts {
		parts[i] = fmt.Sprintf(`{"data":%q}`, b64)
	}
	src := `{"images":[` + strings.Join(parts, ",") + `],"question":"what is here?"}`
	b64, parts = "", nil
	if len(src) > maxRemoteBody {
		t.Fatalf("the body is %d bytes, over the remote bound", len(src))
	}

	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	var m0, m1 runtime.MemStats
	body := []byte(src)
	src = ""
	runtime.GC()
	runtime.ReadMemStats(&m0)
	base := m0.HeapAlloc - uint64(cap(body)) // the heap before the look: the body is its first
	q, ce := parseLook(body)
	runtime.ReadMemStats(&m1)
	if ce != nil {
		t.Fatal(ce.Msg)
	}
	parseAlloc := m1.TotalAlloc - m0.TotalAlloc
	t.Logf("body %.1f MiB; parse allocates %d KiB", float64(len(body))/(1<<20), parseAlloc>>10)
	if parseAlloc > 1<<20 {
		t.Errorf("the parse allocates %d bytes", parseAlloc)
	}
	var buf lookBuf
	var peak uint64
	var slow time.Duration
	for i := range q.images {
		runtime.GC()
		runtime.ReadMemStats(&m0)
		start := time.Now()
		if ce := lookPart(context.Background(), &buf, i, &q.images[i], 2048, false); ce != nil {
			t.Fatal(ce.Msg)
		}
		took := time.Since(start)
		runtime.ReadMemStats(&m1)
		step := m0.HeapAlloc - base + (m1.TotalAlloc - m0.TotalAlloc)
		t.Logf("image %d: live before %.1f MiB, step allocates %.1f MiB, bound %.1f MiB, %v", i,
			float64(m0.HeapAlloc-base)/(1<<20), float64(m1.TotalAlloc-m0.TotalAlloc)/(1<<20), float64(step)/(1<<20), took)
		peak, slow = max(peak, step), max(slow, took)
	}
	body = nil
	runtime.KeepAlive(body)
	t.Logf("peak bound over the look %.1f MiB (body included); built %.1f MiB; slowest image %v", float64(peak)/(1<<20), float64(buf.Len())/(1<<20), slow)
	if peak > uint64(maxRemoteBody)+128<<20 {
		t.Errorf("a step's bound is %d MiB", peak>>20)
	}
}
