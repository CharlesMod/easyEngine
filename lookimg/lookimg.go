// Package lookimg is the engine's own image intake for the look call. A
// remote client's images reach the model only through Prepare, so every
// input is treated as hostile: each check runs before the step that costs
// more memory, and a decoder is never handed bytes the limits have not
// passed.
//
// Limits (Default) and why:
//
//   - MaxBase64 = 11 MiB of base64 text, which is about 8 MiB decoded. The
//     text is refused by length before any byte of it is decoded.
//   - MaxBytes = 8 MiB of decoded file. Checked from the base64 length
//     before the decode buffer is allocated.
//   - MaxPixels = 16 MiPixel (4096 x 4096), read from the header with
//     image.DecodeConfig before any pixel is decoded. Measured (below), the
//     worst allowed kind at this limit allocates about 93 MiB in one
//     Prepare, under the 120 MiB ceiling, so the limit stays at 16 Mi.
//   - MaxSide = 8192 pixels on either side, so a one-pixel-high strip of
//     absurd length is refused even though its pixel count is small.
//
// Only PNG and JPEG are accepted: the format is sniffed from the magic
// bytes, only the png and jpeg decoders are linked into this package, and
// GIF, WebP, BMP and anything else is refused. Before any decode, the file's
// own structure is walked (structure.go; rules there) and refused with
// image_format unless it is baseline 8-bit JPEG (SOF0, 1 or 3 components, at
// most 3 scans, no byte after EOI) or non-interlaced 8-bit non-animated PNG
// (colour type 0, 2, 3, 4 or 6, no byte after IEND, bounded chunk counts).
// Progressive, CMYK, 16-bit, Adam7 and APNG inputs, whose decode cost the file
// size does not bound, never reach a decoder. Every type the allowed inputs
// decode to (YCbCr in all six subsample ratios, Gray, RGBA, NRGBA, Paletted)
// has a typed scaler; any other type is image_format. The image is decoded in
// full and re-encoded (PNG in, PNG out; JPEG in, JPEG out at quality 90), so
// the model never sees the caller's bytes and all metadata is dropped.
//
// Prepare takes a context and asks it before the decode, after it, after the
// scale and after the encode; a done context is image_cancelled. The caller
// checks between images. No timer is used.
//
// Resource measurements (TestResourceBoundsAtPixelLimit; go1.27, one core
// used, GC off so TotalAlloc bounds the live peak from above). Each file is
// 4096 x 4096 and noise is sized to just fit Default.MaxBytes (5-7 MiB files),
// which is the densest data a caller can send. Time is the whole Prepare
// (decode, scale, encode); the decode is most of it.
//
//	kind            file MiB  edge 1024        edge 2048
//	png-rgb         6.4       76 MiB  0.29 s   90 MiB  0.43 s
//	png-rgba        6.9       77 MiB  0.33 s   93 MiB  0.56 s  (worst heap)
//	png-gray        6.2       28 MiB  0.18 s   44 MiB  0.34 s
//	png-palette     5.3       29 MiB  0.15 s   48 MiB  0.42 s
//	jpeg-420        6.6       35 MiB  0.28 s   54 MiB  0.41 s
//	jpeg-444        7.1       60 MiB  0.45 s   79 MiB  0.58 s  (worst time)
//	jpeg-gray       6.8       27 MiB  0.20 s   46 MiB  0.30 s
//	png-rgba flat   0.4       69 MiB  0.13 s   81 MiB  0.17 s
//
// The test fails if any kind allocates over 112 MiB. The heap figure includes
// the 8 MiB decoded-file buffer and the full decoded image; the caller's
// base64 string (up to 11 MiB) is held outside it.
package lookimg

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
)

// Limits bounds what Prepare will accept.
type Limits struct {
	MaxBase64 int // max length of the base64 text (bytes)
	MaxBytes  int // max decoded file bytes
	MaxPixels int // max width*height read from the header
	MaxSide   int // max width or height
}

// Default is the limit set the look call uses; see the package comment.
var Default = Limits{MaxBase64: 11 << 20, MaxBytes: 8 << 20, MaxPixels: 16 << 20, MaxSide: 8192}

// Error is a refusal with a stable Code: image_too_large, image_bad,
// image_format, image_too_many_pixels or image_cancelled.
type Error struct{ Code, Msg string }

func (e *Error) Error() string { return e.Code + ": " + e.Msg }

func fail(code, format string, a ...any) *Error {
	return &Error{Code: code, Msg: fmt.Sprintf(format, a...)}
}

func cancelled() *Error {
	return fail("image_cancelled", "the request ended before the image was prepared")
}

// Result is a prepared image: the re-encoded bytes, their MIME type, the
// size they have, and the size of the source.
type Result struct {
	Data       []byte
	MIME       string
	W, H       int
	SrcW, SrcH int
}

var (
	pngMagic  = []byte("\x89PNG\r\n\x1a\n")
	jpegMagic = []byte("\xff\xd8\xff")
)

var dataPrefixes = []string{"data:image/png;base64,", "data:image/jpeg;base64,"}

// Prepare decodes one image given as standard base64 (padding optional; a
// data:image/png;base64, or data:image/jpeg;base64, prefix is stripped, any
// other data: prefix is refused), checks the limits, decodes PNG or JPEG
// only, scales it down so its long edge is at most longEdge (never up;
// aspect kept, each side at least 1) and re-encodes it. Re-encoding drops
// all metadata (EXIF, text chunks, profiles). The caller's bytes are never
// returned. A panic inside a decoder is recovered and reported as image_bad.
func Prepare(ctx context.Context, b64 string, longEdge int, lim Limits) (Result, *Error) {
	if len(b64) > lim.MaxBase64 {
		return Result{}, fail("image_too_large", "base64 text is %d bytes, limit %d", len(b64), lim.MaxBase64)
	}
	return PrepareBytes(ctx, []byte(b64), longEdge, lim)
}

// PrepareBytes is Prepare on the base64 text as bytes, which it only reads: a caller holding the
// text inside a larger buffer (the look body) passes a view of it and no copy is made.
func PrepareBytes(ctx context.Context, b64 []byte, longEdge int, lim Limits) (res Result, perr *Error) {
	defer func() {
		if r := recover(); r != nil {
			res, perr = Result{}, fail("image_bad", "image could not be decoded")
		}
	}()
	if longEdge <= 0 {
		return Result{}, fail("image_bad", "longEdge must be positive, got %d", longEdge)
	}
	if len(b64) > lim.MaxBase64 {
		return Result{}, fail("image_too_large", "base64 text is %d bytes, limit %d", len(b64), lim.MaxBase64)
	}
	if bytes.HasPrefix(b64, []byte("data:")) {
		stripped := false
		for _, p := range dataPrefixes {
			if bytes.HasPrefix(b64, []byte(p)) {
				b64, stripped = b64[len(p):], true
				break
			}
		}
		if !stripped {
			return Result{}, fail("image_format", "only data:image/png;base64 and data:image/jpeg;base64 prefixes are accepted")
		}
	}
	raw, e := decodeBase64(b64, lim.MaxBytes)
	if e != nil {
		return Result{}, e
	}
	var want, mime string
	switch {
	case bytes.HasPrefix(raw, pngMagic):
		want, mime = "png", "image/png"
	case bytes.HasPrefix(raw, jpegMagic):
		want, mime = "jpeg", "image/jpeg"
	default:
		return Result{}, fail("image_format", "only PNG and JPEG images are accepted")
	}
	if want == "png" {
		e = checkPNG(raw)
	} else {
		e = checkJPEG(raw)
	}
	if e != nil {
		return Result{}, e
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return Result{}, fail("image_bad", "image header could not be read")
	}
	if format != want {
		return Result{}, fail("image_bad", "image content does not match its format")
	}
	w, h := cfg.Width, cfg.Height
	if w <= 0 || h <= 0 {
		return Result{}, fail("image_bad", "image has no size")
	}
	if w > lim.MaxSide || h > lim.MaxSide {
		return Result{}, fail("image_too_many_pixels", "image is %dx%d, side limit %d", w, h, lim.MaxSide)
	}
	if int64(w)*int64(h) > int64(lim.MaxPixels) {
		return Result{}, fail("image_too_many_pixels", "image is %dx%d, pixel limit %d", w, h, lim.MaxPixels)
	}
	if ctx.Err() != nil {
		return Result{}, cancelled()
	}
	src, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return Result{}, fail("image_bad", "image could not be decoded")
	}
	raw = nil
	if ctx.Err() != nil {
		return Result{}, cancelled()
	}
	if b := src.Bounds(); b.Dx() != w || b.Dy() != h {
		return Result{}, fail("image_bad", "image size differs from its header")
	}
	dw, dh := fit(w, h, longEdge)
	dst, e := scale(src, dw, dh)
	src = nil
	if e != nil {
		return Result{}, e
	}
	if ctx.Err() != nil {
		return Result{}, cancelled()
	}
	var out bytes.Buffer
	if want == "png" {
		err = png.Encode(&out, dst)
	} else {
		err = jpeg.Encode(&out, dst, &jpeg.Options{Quality: 90})
	}
	if err != nil {
		return Result{}, fail("image_bad", "image could not be encoded")
	}
	if ctx.Err() != nil {
		return Result{}, cancelled()
	}
	return Result{Data: out.Bytes(), MIME: mime, W: dw, H: dh, SrcW: w, SrcH: h}, nil
}

// decodeBase64 decodes standard base64 with optional padding, refusing it
// before allocating when the decoded size would pass max.
func decodeBase64(s []byte, max int) ([]byte, *Error) {
	n := len(s)
	for pad := 0; pad < 2 && n > 0 && s[n-1] == '='; pad++ {
		n--
	}
	s = s[:n]
	if size := base64.RawStdEncoding.DecodedLen(n); size > max {
		return nil, fail("image_too_large", "image is about %d bytes, limit %d", size, max)
	}
	buf := make([]byte, base64.RawStdEncoding.DecodedLen(n))
	m, err := base64.RawStdEncoding.Decode(buf, s)
	if err != nil {
		return nil, fail("image_bad", "invalid base64")
	}
	return buf[:m], nil
}

// fit gives the size w x h scaled down to a long edge of at most long.
func fit(w, h, long int) (int, int) {
	if w <= long && h <= long {
		return w, h
	}
	if w >= h {
		return long, max(1, int(int64(h)*int64(long)/int64(w)))
	}
	return max(1, int(int64(w)*int64(long)/int64(h))), long
}
