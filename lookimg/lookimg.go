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
//   - MaxPixels = 16 MiPixel (16.7 MP), read from the header with
//     image.DecodeConfig before any pixel is decoded. A 16 MP image
//     decodes to at most 64 MiB of RGBA (4 bytes a pixel), which is the
//     most memory the full decode may ask for; the downscaled copy is
//     small beside it.
//   - MaxSide = 8192 pixels on either side, so a one-pixel-high strip of
//     absurd length is refused even though its pixel count is small.
//
// Only PNG and JPEG are accepted: the format is sniffed from the magic
// bytes, only the png and jpeg decoders are linked into this package, and
// GIF, WebP, BMP and anything else is refused. The image is decoded in full
// and re-encoded (PNG in, PNG out; JPEG in, JPEG out at quality 90), so the
// model never sees the caller's bytes and all metadata (EXIF, text chunks,
// ICC profiles, trailing data) is dropped by the re-encode.
package lookimg

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
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
// image_format or image_too_many_pixels.
type Error struct{ Code, Msg string }

func (e *Error) Error() string { return e.Code + ": " + e.Msg }

func fail(code, format string, a ...any) *Error {
	return &Error{Code: code, Msg: fmt.Sprintf(format, a...)}
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
func Prepare(b64 string, longEdge int, lim Limits) (res Result, perr *Error) {
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
	if strings.HasPrefix(b64, "data:") {
		stripped := false
		for _, p := range dataPrefixes {
			if strings.HasPrefix(b64, p) {
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
	src, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return Result{}, fail("image_bad", "image could not be decoded")
	}
	raw = nil
	if b := src.Bounds(); b.Dx() != w || b.Dy() != h {
		return Result{}, fail("image_bad", "image size differs from its header")
	}
	dw, dh := fit(w, h, longEdge)
	dst := scale(src, dw, dh)
	src = nil
	var out bytes.Buffer
	if want == "png" {
		err = png.Encode(&out, dst)
	} else {
		err = jpeg.Encode(&out, dst, &jpeg.Options{Quality: 90})
	}
	if err != nil {
		return Result{}, fail("image_bad", "image could not be encoded")
	}
	return Result{Data: out.Bytes(), MIME: mime, W: dw, H: dh, SrcW: w, SrcH: h}, nil
}

// decodeBase64 decodes standard base64 with optional padding, refusing it
// before allocating when the decoded size would pass max.
func decodeBase64(s string, max int) ([]byte, *Error) {
	n := len(s)
	for pad := 0; pad < 2 && n > 0 && s[n-1] == '='; pad++ {
		n--
	}
	s = s[:n]
	if size := base64.RawStdEncoding.DecodedLen(n); size > max {
		return nil, fail("image_too_large", "image is about %d bytes, limit %d", size, max)
	}
	buf := make([]byte, base64.RawStdEncoding.DecodedLen(n))
	m, err := base64.RawStdEncoding.Decode(buf, []byte(s))
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

// scale box-filters src into a new dw x dh RGBA (dw <= source width,
// dh <= source height). Averaging is done on premultiplied values, so alpha
// stays straight when the PNG encoder un-premultiplies.
func scale(src image.Image, dw, dh int) *image.RGBA {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	px := pixelReader(src)
	sum := make([]uint64, dw*4)
	for dy := 0; dy < dh; dy++ {
		y0, y1 := dy*sh/dh, (dy+1)*sh/dh
		for i := range sum {
			sum[i] = 0
		}
		for y := y0; y < y1; y++ {
			sy := b.Min.Y + y
			for dx := 0; dx < dw; dx++ {
				x0, x1 := dx*sw/dw, (dx+1)*sw/dw
				s := sum[dx*4 : dx*4+4 : dx*4+4]
				for x := x0; x < x1; x++ {
					r, g, bl, a := px(b.Min.X+x, sy)
					s[0] += uint64(r)
					s[1] += uint64(g)
					s[2] += uint64(bl)
					s[3] += uint64(a)
				}
			}
		}
		rows := uint64(y1 - y0)
		for dx := 0; dx < dw; dx++ {
			n := rows * uint64((dx+1)*sw/dw-dx*sw/dw)
			s := sum[dx*4 : dx*4+4]
			o := dst.PixOffset(dx, dy)
			dst.Pix[o+0] = uint8((s[0]/n + 128) / 257)
			dst.Pix[o+1] = uint8((s[1]/n + 128) / 257)
			dst.Pix[o+2] = uint8((s[2]/n + 128) / 257)
			dst.Pix[o+3] = uint8((s[3]/n + 128) / 257)
		}
	}
	return dst
}

// pixelReader returns a reader of premultiplied 16-bit RGBA, with a fast
// path for the common concrete types and At() for the rest.
func pixelReader(img image.Image) func(x, y int) (r, g, b, a uint32) {
	switch m := img.(type) {
	case *image.YCbCr:
		return func(x, y int) (uint32, uint32, uint32, uint32) {
			yy := m.Y[m.YOffset(x, y)]
			ci := m.COffset(x, y)
			r, g, b := color.YCbCrToRGB(yy, m.Cb[ci], m.Cr[ci])
			return uint32(r) * 0x101, uint32(g) * 0x101, uint32(b) * 0x101, 0xffff
		}
	case *image.RGBA:
		return func(x, y int) (uint32, uint32, uint32, uint32) {
			p := m.Pix[m.PixOffset(x, y):]
			return uint32(p[0]) * 0x101, uint32(p[1]) * 0x101, uint32(p[2]) * 0x101, uint32(p[3]) * 0x101
		}
	case *image.NRGBA:
		return func(x, y int) (uint32, uint32, uint32, uint32) {
			p := m.Pix[m.PixOffset(x, y):]
			a := uint32(p[3]) * 0x101
			return uint32(p[0]) * 0x101 * a / 0xffff, uint32(p[1]) * 0x101 * a / 0xffff, uint32(p[2]) * 0x101 * a / 0xffff, a
		}
	case *image.Gray:
		return func(x, y int) (uint32, uint32, uint32, uint32) {
			v := uint32(m.Pix[m.PixOffset(x, y)]) * 0x101
			return v, v, v, 0xffff
		}
	}
	return func(x, y int) (uint32, uint32, uint32, uint32) { return img.At(x, y).RGBA() }
}
