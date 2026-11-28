package lookimg

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/draw"
	"strings"
	"testing"
	"time"
)

func refused(t *testing.T, name string, in []byte, code, msgPart string) {
	t.Helper()
	_, e := Prepare(ctx0, b64(in), 64, Default)
	if e == nil || e.Code != code || !strings.Contains(e.Msg, msgPart) {
		t.Errorf("%s: got %v, want %s containing %q", name, e, code, msgPart)
	}
}

func ihdrPNG(w, h uint32, depth, ctype, interlace byte) []byte {
	d := make([]byte, 13)
	binary.BigEndian.PutUint32(d[0:], w)
	binary.BigEndian.PutUint32(d[4:], h)
	d[8], d[9], d[12] = depth, ctype, interlace
	return chunk("IHDR", d)
}

func TestReviewerFilesRefused(t *testing.T) {
	for name, want := range map[string]string{
		"cmyk_prog.jpg":    "SOF2",
		"rgb_prog444.jpg":  "SOF2",
		"rgb16.png":        "bit depth 16",
		"rgb16_inter.png":  "bit depth 16",
		"rgba16_inter.png": "bit depth 16",
	} {
		refused(t, name, testdata(t, name), "image_format", want)
	}
}

// TestManyScansRefusedAtTheFrameHeader builds the review's hour-long file
// (a progressive JPEG whose last scan is repeated to 8 MiB) and proves the
// structure walk stops at the SOF2 without ever reaching the repeats: it is
// refused in a small fraction of the time one pass of the decoder would take.
func TestManyScansRefusedAtTheFrameHeader(t *testing.T) {
	p := testdata(t, "rgb_prog444.jpg")
	last := bytes.LastIndex(p, []byte{0xff, 0xda})
	scan := p[last : len(p)-2]
	in := append([]byte{}, p[:len(p)-2]...)
	for len(in)+len(scan) < 8<<20-2 {
		in = append(in, scan...)
	}
	in = append(in, 0xff, 0xd9)
	s := b64(in)
	t0 := time.Now()
	_, e := Prepare(ctx0, s, 64, Default)
	d := time.Since(t0)
	wantCode(t, e, "image_format")
	if d > 300*time.Millisecond {
		t.Fatalf("refusal took %v for a %d byte file", d, len(in))
	}
}

func TestJPEGStructure(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 16, 16))
	base := jpegBytes(t, g, 75)
	// the plain grey file is fine, and so are 1 to 3 scans of a 3-component file
	for k := 1; k <= 3; k++ {
		if _, e := Prepare(ctx0, b64(grayTo444(t, base, k)), 64, Default); e != nil {
			t.Errorf("%d scans: %v", k, e)
		}
	}
	refused(t, "4 scans", grayTo444(t, base, 4), "image_format", "more than 3 scans")
	refused(t, "byte after EOI", append(append([]byte{}, base...), 0), "image_format", "after its end marker")
	refused(t, "second image after EOI", append(append([]byte{}, base...), base...), "image_format", "after its end marker")

	sof := bytes.Index(base, []byte{0xff, 0xc0})
	patch := func(f func(b []byte)) []byte {
		b := append([]byte{}, base...)
		f(b)
		return b
	}
	refused(t, "12-bit", patch(func(b []byte) { b[sof+4] = 12 }), "image_format", "8-bit")
	refused(t, "SOF1", patch(func(b []byte) { b[sof+1] = 0xc1 }), "image_format", "SOF1")
	refused(t, "SOF2", patch(func(b []byte) { b[sof+1] = 0xc2 }), "image_format", "SOF2")
	refused(t, "SOF3", patch(func(b []byte) { b[sof+1] = 0xc3 }), "image_format", "SOF3")
	refused(t, "SOF9", patch(func(b []byte) { b[sof+1] = 0xc9 }), "image_format", "SOF9")
	refused(t, "DNL", patch(func(b []byte) { b[sof+1] = 0xdc }), "image_format", "0xFFDC")
	refused(t, "DHP", patch(func(b []byte) { b[sof+1] = 0xde }), "image_format", "0xFFDE")
	refused(t, "EXP", patch(func(b []byte) { b[sof+1] = 0xdf }), "image_format", "0xFFDF")
	refused(t, "JPG", patch(func(b []byte) { b[sof+1] = 0xc8 }), "image_format", "0xFFC8")

	// four and two components
	for _, nc := range []int{2, 4} {
		in := append([]byte{}, base[:sof]...)
		in = append(in, 0xff, 0xc0, 0, byte(8+3*nc), 8, 0, 16, 0, 16, byte(nc))
		for c := 0; c < nc; c++ {
			in = append(in, byte(c+1), 0x11, 0)
		}
		in = append(in, base[sof+2+int(binary.BigEndian.Uint16(base[sof+2:])):]...)
		refused(t, "components", in, "image_format", "components")
	}

	// two frame headers
	two := append(append(append([]byte{}, base[:sof]...), base[sof:sof+2+int(binary.BigEndian.Uint16(base[sof+2:]))]...), base[sof:]...)
	refused(t, "two SOF", two, "image_format", "more than one frame")

	// too many segments, too much metadata
	seg := []byte{0xff, 0xfe, 0, 6, 'a', 'b', 'c', 'd'}
	many := append(append([]byte{}, base[:2]...), bytes.Repeat(seg, maxJPEGSegments+1)...)
	refused(t, "65 segments", append(many, base[2:]...), "image_format", "segments")
	big := append([]byte{0xff, 0xd8}, bytes.Repeat(append([]byte{0xff, 0xe1, 0xff, 0xff}, make([]byte, 0xffff-2)...), 17)...)
	refused(t, "APP bytes", append(big, base[2:]...), "image_format", "APP/COM")
	// a stray byte between segments, a stray RST, a scan before its frame
	refused(t, "stray", append(append([]byte{}, base[:sof]...), append([]byte{0}, base[sof:]...)...), "image_bad", "stray")
	refused(t, "RST marker", append(append([]byte{}, base[:2]...), append([]byte{0xff, 0xd0}, base[2:]...)...), "image_format", "0xFFD0")
}

// TestJPEGEntropyDataIsWalked checks stuffing and restart markers inside scan
// data do not end the walk early, and a forbidden marker after data does.
func TestJPEGEntropyDataIsWalked(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 16, 16))
	for i := range g.Pix {
		g.Pix[i] = byte(i * 37)
	}
	base := jpegBytes(t, g, 75)
	if !bytes.Contains(base, []byte{0xff, 0x00}) {
		t.Skip("fixture has no stuffed byte")
	}
	// a progressive frame header hidden after real scan data is still found
	tail := append(append([]byte{}, base[:len(base)-2]...), 0xff, 0xc2, 0, 2)
	refused(t, "SOF2 after scan", tail, "image_format", "SOF2")
	// and trailing data after a scan with the stuffed bytes intact
	refused(t, "trailing", append(append([]byte{}, base...), 0xff, 0xd9), "image_format", "after its end marker")
}

func TestPNGStructure(t *testing.T) {
	good := encPNG(t, gradient(8, 8))
	if _, e := Prepare(ctx0, b64(good), 64, Default); e != nil {
		t.Fatal(e)
	}
	refused(t, "after IEND", append(append([]byte{}, good...), 0), "image_format", "after its end chunk")
	refused(t, "two images", append(append([]byte{}, good...), good...), "image_format", "after its end chunk")

	iend := len(good) - 12
	// APNG: acTL after IHDR, and fcTL, fdAT
	for _, c := range []string{"acTL", "fcTL", "fdAT"} {
		in := append(append(append([]byte{}, good[:33]...), chunk(c, make([]byte, 8))...), good[33:]...)
		refused(t, c, in, "image_format", "animated")
	}
	// 16-bit, 1-, 2-, 4-bit and interlaced headers, even claiming 8x8
	for _, d := range []byte{16, 1, 2, 4} {
		in := append(append([]byte{}, pngMagic...), ihdrPNG(8, 8, d, 0, 0)...)
		in = append(in, chunk("IDAT", []byte{0x78, 0x9c, 0, 0})...)
		in = append(in, chunk("IEND", nil)...)
		refused(t, "depth", in, "image_format", "8-bit")
	}
	in := append(append([]byte{}, pngMagic...), ihdrPNG(8, 8, 8, 2, 1)...)
	in = append(append(in, chunk("IDAT", []byte{0x78, 0x9c, 0, 0})...), chunk("IEND", nil)...)
	refused(t, "adam7", in, "image_format", "interlaced")
	// ancillary and total chunk counts
	anc := func(n int) []byte {
		var b []byte
		for i := 0; i < n; i++ {
			b = append(b, chunk("teXt", []byte("k\x00v"))...)
		}
		return append(append(append([]byte{}, good[:33]...), b...), good[33:iend+12]...)
	}
	if _, e := Prepare(ctx0, b64(anc(maxPNGAncillary)), 64, Default); e != nil {
		t.Errorf("64 ancillary: %v", e)
	}
	refused(t, "65 ancillary", anc(maxPNGAncillary+1), "image_format", "ancillary")
	var idats []byte
	for i := 0; i < maxPNGChunks; i++ {
		idats = append(idats, chunk("IDAT", nil)...)
	}
	refused(t, "chunks", append(append(append([]byte{}, good[:33]...), idats...), good[33:]...), "image_format", "more than 4096 chunks")
	// not IHDR first, bad chunk type
	refused(t, "no IHDR", append(append([]byte{}, pngMagic...), chunk("IDAT", nil)...), "image_bad", "IHDR")
	refused(t, "type", append(append([]byte{}, pngMagic...), chunk("ID\x01T", nil)...), "image_bad", "type")
}

func TestCancelledContext(t *testing.T) {
	in := b64(encPNG(t, gradient(64, 64)))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, e := Prepare(ctx, in, 32, Default)
	wantCode(t, e, "image_cancelled")
	// Prepare asks the context at four points: before the decode, after it,
	// after the scale and after the encode. A context that turns done at its
	// n-th question is refused for n = 1..4 and not for n = 5.
	for n := 1; n <= 5; n++ {
		c := &nthCtx{Context: context.Background(), n: n}
		_, e := Prepare(c, in, 32, Default)
		if n <= 4 {
			wantCode(t, e, "image_cancelled")
		} else if e != nil || c.calls != 4 {
			t.Fatalf("n=%d: %v after %d questions", n, e, c.calls)
		}
	}
}

// nthCtx reports done from its n-th Err call on.
type nthCtx struct {
	context.Context
	n, calls int
}

func (c *nthCtx) Err() error {
	c.calls++
	if c.calls >= c.n {
		return context.Canceled
	}
	return nil
}

// TestEveryAllowedKindHasAFastPath decodes a small file of every allowed kind
// and checks the decoder's type has a typed scaler, then runs Prepare on it.
func TestEveryAllowedKindHasAFastPath(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 40, 30))
	rgba := image.NewNRGBA(image.Rect(0, 0, 40, 30))
	draw.Draw(rgba, rgba.Bounds(), &image.Uniform{color.NRGBA{10, 20, 30, 200}}, image.Point{}, draw.Src)
	opaque := image.NewRGBA(image.Rect(0, 0, 40, 30))
	draw.Draw(opaque, opaque.Bounds(), &image.Uniform{color.RGBA{10, 20, 30, 255}}, image.Point{}, draw.Src)
	pal := image.NewPaletted(image.Rect(0, 0, 40, 30), color.Palette{color.RGBA{1, 2, 3, 255}, color.RGBA{9, 8, 7, 255}})
	// 8-bit palette: pad to 256 entries so the encoder writes depth 8
	for len(pal.Palette) < 256 {
		pal.Palette = append(pal.Palette, color.RGBA{uint8(len(pal.Palette)), 0, 0, 255})
	}
	// 8-bit gray + alpha PNG (colour type 4), crafted: decodes to NRGBA
	ga := append([]byte{}, pngMagic...)
	ga = append(ga, ihdrPNG(2, 1, 8, 4, 0)...)
	ga = append(ga, chunk("IDAT", zlibStored([]byte{0, 100, 255, 200, 128}))...)
	ga = append(ga, chunk("IEND", nil)...)
	// gray + tRNS and rgb + tRNS decode to NRGBA
	trnsG := append([]byte{}, pngMagic...)
	trnsG = append(trnsG, ihdrPNG(2, 1, 8, 0, 0)...)
	trnsG = append(trnsG, chunk("tRNS", []byte{0, 100})...)
	trnsG = append(trnsG, chunk("IDAT", zlibStored([]byte{0, 100, 200}))...)
	trnsG = append(trnsG, chunk("IEND", nil)...)
	files := map[string][]byte{
		"png-rgb":       encPNG(t, opaque),
		"png-nrgba":     encPNG(t, rgba),
		"png-gray":      encPNG(t, g),
		"png-gray-a":    ga,
		"png-gray-trns": trnsG,
		"png-palette":   encPNG(t, pal),
		"jpeg-420":      encJPEG(t, opaque),
		"jpeg-gray":     encJPEG(t, g),
		"jpeg-444":      grayTo444(t, encJPEG(t, g), 3),
	}
	for name, f := range files {
		img, _, err := image.Decode(bytes.NewReader(f))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if rowAdder(img) == nil {
			t.Errorf("%s decodes to %T, which has no typed scaler", name, img)
		}
		if _, e := Prepare(ctx0, b64(f), 16, Default); e != nil {
			t.Errorf("%s: %v", name, e)
		}
	}
	// an unexpected type is refused, not read through At()
	if _, e := scale(image.NewCMYK(image.Rect(0, 0, 4, 4)), 2, 2); e == nil || e.Code != "image_format" {
		t.Fatalf("CMYK image: %v", e)
	}
	if _, e := scale(image.NewRGBA64(image.Rect(0, 0, 4, 4)), 2, 2); e == nil || e.Code != "image_format" {
		t.Fatalf("RGBA64 image: %v", e)
	}
}

func zlibStored(raw []byte) []byte {
	b := []byte{0x78, 0x01, 1, byte(len(raw)), 0, ^byte(len(raw)), 0xff}
	b = append(b, raw...)
	a, c := uint32(1), uint32(0)
	for _, v := range raw {
		a = (a + uint32(v)) % 65521
		c = (c + a) % 65521
	}
	return binary.BigEndian.AppendUint32(b, c<<16|a)
}

// TestScaleMatchesReference compares each typed path with the generic At()
// average over odd sizes, every YCbCr subsample ratio included.
func TestScaleMatchesReference(t *testing.T) {
	r := rng()
	fill := func(p []byte) {
		for i := range p {
			p[i] = byte(r.Uint32())
		}
	}
	rect := image.Rect(0, 0, 37, 23)
	var imgs []image.Image
	for _, ratio := range []image.YCbCrSubsampleRatio{image.YCbCrSubsampleRatio444, image.YCbCrSubsampleRatio422, image.YCbCrSubsampleRatio420, image.YCbCrSubsampleRatio440, image.YCbCrSubsampleRatio411, image.YCbCrSubsampleRatio410} {
		m := image.NewYCbCr(rect, ratio)
		fill(m.Y)
		fill(m.Cb)
		fill(m.Cr)
		imgs = append(imgs, m)
	}
	gr := image.NewGray(rect)
	fill(gr.Pix)
	rg := image.NewRGBA(rect)
	fill(rg.Pix)
	for i := 0; i < len(rg.Pix); i += 4 { // valid premultiplied pixels
		for c := 0; c < 3; c++ {
			rg.Pix[i+c] = uint8(int(rg.Pix[i+c]) * int(rg.Pix[i+3]) / 255)
		}
	}
	ng := image.NewNRGBA(rect)
	fill(ng.Pix)
	pl := image.NewPaletted(rect, color.Palette{color.RGBA{1, 2, 3, 255}, color.NRGBA{200, 100, 50, 77}, color.Gray{9}})
	for i := range pl.Pix {
		pl.Pix[i] = byte(r.Uint32() % 3)
	}
	imgs = append(imgs, gr, rg, ng, pl)
	for _, img := range imgs {
		for _, d := range [][2]int{{37, 23}, {20, 11}, {5, 3}, {1, 1}, {36, 1}} {
			got, e := scale(img, d[0], d[1])
			if e != nil {
				t.Fatalf("%T: %v", img, e)
			}
			want := refScale(img, d[0], d[1])
			if !bytes.Equal(got.Pix, want.Pix) {
				t.Errorf("%T to %v differs from the At() reference", img, d)
			}
		}
	}
}

func refScale(src image.Image, dw, dh int) *image.RGBA {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for dy := 0; dy < dh; dy++ {
		for dx := 0; dx < dw; dx++ {
			var s [4]uint64
			var n uint64
			for y := dy * sh / dh; y < (dy+1)*sh/dh; y++ {
				for x := dx * sw / dw; x < (dx+1)*sw/dw; x++ {
					r, g, bl, a := src.At(x, y).RGBA()
					if m, ok := src.(*image.YCbCr); ok { // At() converts at 16 bits; the scaler at 8
						c := m.YCbCrAt(x, y)
						r8, g8, b8 := color.YCbCrToRGB(c.Y, c.Cb, c.Cr)
						r, g, bl, a = uint32(r8)*0x101, uint32(g8)*0x101, uint32(b8)*0x101, 0xffff
					}
					s[0] += uint64(r)
					s[1] += uint64(g)
					s[2] += uint64(bl)
					s[3] += uint64(a)
					n++
				}
			}
			o := dst.PixOffset(dx, dy)
			for c := 0; c < 4; c++ {
				dst.Pix[o+c] = uint8((s[c]/n + 128) / 257)
			}
		}
	}
	return dst
}

var _ = crc32.ChecksumIEEE
