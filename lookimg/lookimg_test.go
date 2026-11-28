package lookimg

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"runtime"
	"strings"
	"testing"
)

func b64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

func gradient(w, h int) *image.RGBA {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetRGBA(x, y, color.RGBA{uint8(x * 255 / w), uint8(y * 255 / h), 90, 255})
		}
	}
	return m
}

func encPNG(t testing.TB, m image.Image) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, m); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func encJPEG(t testing.TB, m image.Image) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := jpeg.Encode(&b, m, nil); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func wantCode(t *testing.T, e *Error, code string) {
	t.Helper()
	if e == nil || e.Code != code {
		t.Fatalf("got %v, want code %s", e, code)
	}
}

func chunk(typ string, data []byte) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.BigEndian, uint32(len(data)))
	b.WriteString(typ)
	b.Write(data)
	binary.Write(&b, binary.BigEndian, crc32.ChecksumIEEE(append([]byte(typ), data...)))
	return b.Bytes()
}

func hugePNG(w, h uint32) []byte {
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:], w)
	binary.BigEndian.PutUint32(ihdr[4:], h)
	ihdr[8], ihdr[9] = 8, 2
	out := append([]byte{}, pngMagic...)
	out = append(out, chunk("IHDR", ihdr)...)
	out = append(out, chunk("IDAT", []byte{0x78, 0x9c, 0, 0})...)
	return append(out, chunk("IEND", nil)...)
}

// hugeJPEG is a header-only JPEG claiming w x h; jfif says whether it opens
// with a JFIF segment (Go's DecodeConfig stops at the frame header only then)
// or runs on to the scan header.
func hugeJPEG(w, h uint16, jfif bool) []byte {
	out := []byte{0xff, 0xd8}
	if jfif {
		out = append(out, 0xff, 0xe0, 0, 16, 'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0)
	}
	sof := []byte{8, byte(h >> 8), byte(h), byte(w >> 8), byte(w), 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1}
	out = append(out, 0xff, 0xc0, 0, byte(len(sof)+2))
	out = append(out, sof...)
	if !jfif {
		out = append(out, 0xff, 0xda, 0, 12, 3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0)
	}
	return out
}

func TestPNGDownscale(t *testing.T) {
	r, e := Prepare(b64(encPNG(t, gradient(1000, 500))), 512, Default)
	if e != nil {
		t.Fatal(e)
	}
	if r.W != 512 || r.H != 256 || r.SrcW != 1000 || r.SrcH != 500 || r.MIME != "image/png" {
		t.Fatalf("%+v", r)
	}
	img, f, err := image.Decode(bytes.NewReader(r.Data))
	if err != nil || f != "png" || img.Bounds().Dx() != 512 || img.Bounds().Dy() != 256 {
		t.Fatalf("%v %s", err, f)
	}
}

func TestJPEGDownscale(t *testing.T) {
	r, e := Prepare(b64(encJPEG(t, gradient(500, 1000))), 512, Default)
	if e != nil {
		t.Fatal(e)
	}
	if r.W != 256 || r.H != 512 || r.MIME != "image/jpeg" {
		t.Fatalf("%+v", r)
	}
	if _, f, err := image.Decode(bytes.NewReader(r.Data)); err != nil || f != "jpeg" {
		t.Fatalf("%v %s", err, f)
	}
}

func TestBoxFilterAverages(t *testing.T) {
	m := image.NewRGBA(image.Rect(0, 0, 2, 2))
	m.SetRGBA(0, 0, color.RGBA{255, 255, 255, 255})
	m.SetRGBA(1, 0, color.RGBA{0, 0, 0, 255})
	m.SetRGBA(0, 1, color.RGBA{0, 0, 0, 255})
	m.SetRGBA(1, 1, color.RGBA{255, 255, 255, 255})
	r, e := Prepare(b64(encPNG(t, m)), 1, Default)
	if e != nil || r.W != 1 || r.H != 1 {
		t.Fatalf("%v %+v", e, r)
	}
	img, _ := png.Decode(bytes.NewReader(r.Data))
	c := color.RGBAModel.Convert(img.At(0, 0)).(color.RGBA)
	if c.R < 126 || c.R > 129 {
		t.Fatalf("average %v", c)
	}
}

func TestNoUpscaleStillReencoded(t *testing.T) {
	in := encPNG(t, gradient(300, 200))
	r, e := Prepare(b64(in), 1024, Default)
	if e != nil || r.W != 300 || r.H != 200 {
		t.Fatalf("%v %+v", e, r)
	}
	if _, err := png.Decode(bytes.NewReader(r.Data)); err != nil {
		t.Fatal(err)
	}
	if len(r.Data) > 0 && &r.Data[0] == &in[0] {
		t.Fatal("returned caller bytes")
	}
}

func TestExtremeAspectKeepsOnePixel(t *testing.T) {
	r, e := Prepare(b64(encPNG(t, gradient(4000, 2))), 100, Default)
	if e != nil || r.W != 100 || r.H != 1 {
		t.Fatalf("%v %+v", e, r)
	}
}

func TestDataPrefix(t *testing.T) {
	p := b64(encPNG(t, gradient(8, 8)))
	if _, e := Prepare("data:image/png;base64,"+p, 64, Default); e != nil {
		t.Fatal(e)
	}
	j := b64(encJPEG(t, gradient(8, 8)))
	if _, e := Prepare("data:image/jpeg;base64,"+j, 64, Default); e != nil {
		t.Fatal(e)
	}
	for _, pre := range []string{"data:image/gif;base64,", "data:text/html;base64,", "data:,", "data:image/png,"} {
		_, e := Prepare(pre+p, 64, Default)
		wantCode(t, e, "image_format")
	}
}

func TestPaddingOptional(t *testing.T) {
	p := b64(encPNG(t, gradient(7, 9)))
	if _, e := Prepare(strings.TrimRight(p, "="), 64, Default); e != nil {
		t.Fatal(e)
	}
}

func TestInvalidBase64(t *testing.T) {
	for _, s := range []string{"@@@@", "iVBO Rw0K", "ab=c", ""} {
		_, e := Prepare(s, 64, Default)
		if s == "" {
			wantCode(t, e, "image_format")
			continue
		}
		wantCode(t, e, "image_bad")
	}
}

func TestLimits(t *testing.T) {
	_, e := Prepare(strings.Repeat("A", Default.MaxBase64+1), 64, Default)
	wantCode(t, e, "image_too_large")
	p := b64(encPNG(t, gradient(64, 64)))
	_, e = Prepare(p, 64, Limits{MaxBase64: len(p) - 1, MaxBytes: 1 << 20, MaxPixels: 1 << 20, MaxSide: 8192})
	wantCode(t, e, "image_too_large")
	_, e = Prepare(p, 64, Limits{MaxBase64: len(p), MaxBytes: 100, MaxPixels: 1 << 20, MaxSide: 8192})
	wantCode(t, e, "image_too_large")
}

func TestNotPNGOrJPEG(t *testing.T) {
	gifBytes := []byte("GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")
	bmp := append([]byte("BM"), make([]byte, 60)...)
	webp := append([]byte("RIFF\x24\x00\x00\x00WEBPVP8 "), make([]byte, 40)...)
	for name, in := range map[string][]byte{"gif": gifBytes, "bmp": bmp, "webp": webp, "text": []byte("hello, I am not an image"), "short": {0xff}} {
		_, e := Prepare(b64(in), 64, Default)
		if e == nil || e.Code != "image_format" {
			t.Errorf("%s: %v", name, e)
		}
	}
}

func TestMagicWithGarbageIsBad(t *testing.T) {
	_, e := Prepare(b64(append(append([]byte{}, pngMagic...), "garbage garbage"...)), 64, Default)
	wantCode(t, e, "image_bad")
	_, e = Prepare(b64(append(append([]byte{}, jpegMagic...), "garbage garbage"...)), 64, Default)
	wantCode(t, e, "image_bad")
}

func allocated(f func()) uint64 {
	var a, b runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&a)
	f()
	runtime.ReadMemStats(&b)
	return b.TotalAlloc - a.TotalAlloc
}

func TestHugeHeaderRefusedWithoutDecoding(t *testing.T) {
	for name, in := range map[string][]byte{"png": hugePNG(100000, 100000), "jpeg": hugeJPEG(65535, 65535, true), "jpeg-nojfif": hugeJPEG(65535, 65535, false), "png-pixels": hugePNG(8000, 8000)} {
		if len(in) > 1000 {
			t.Fatalf("%s fixture is %d bytes", name, len(in))
		}
		s := b64(in)
		var e *Error
		n := allocated(func() { _, e = Prepare(s, 512, Default) })
		if e == nil || e.Code != "image_too_many_pixels" {
			t.Fatalf("%s: %v", name, e)
		}
		if n > 1<<20 {
			t.Errorf("%s: allocated %d bytes", name, n)
		}
	}
}

func TestMaxSideWithFewPixels(t *testing.T) {
	_, e := Prepare(b64(hugePNG(9000, 1)), 512, Default)
	wantCode(t, e, "image_too_many_pixels")
	_, e = Prepare(b64(encPNG(t, gradient(300, 1))), 512, Limits{MaxBase64: 1 << 20, MaxBytes: 1 << 20, MaxPixels: 1 << 20, MaxSide: 200})
	wantCode(t, e, "image_too_many_pixels")
}

func TestTruncated(t *testing.T) {
	p := encPNG(t, gradient(200, 200))
	_, e := Prepare(b64(p[:len(p)/2]), 64, Default)
	wantCode(t, e, "image_bad")
	j := encJPEG(t, gradient(200, 200))
	_, e = Prepare(b64(j[:len(j)/2]), 64, Default)
	wantCode(t, e, "image_bad")
}

func TestGrayPNG(t *testing.T) {
	g := image.NewGray(image.Rect(0, 0, 40, 20))
	for i := range g.Pix {
		g.Pix[i] = 200
	}
	r, e := Prepare(b64(encPNG(t, g)), 20, Default)
	if e != nil || r.W != 20 || r.H != 10 {
		t.Fatalf("%v %+v", e, r)
	}
	img, _ := png.Decode(bytes.NewReader(r.Data))
	if c := color.RGBAModel.Convert(img.At(3, 3)).(color.RGBA); c.R != 200 || c.G != 200 || c.A != 255 {
		t.Fatalf("%v", c)
	}
}

func TestAlphaKept(t *testing.T) {
	m := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			m.SetNRGBA(x, y, color.NRGBA{200, 50, 10, 128})
		}
	}
	m.SetNRGBA(0, 0, color.NRGBA{0, 0, 0, 0})
	r, e := Prepare(b64(encPNG(t, m)), 4, Default)
	if e != nil {
		t.Fatal(e)
	}
	img, _ := png.Decode(bytes.NewReader(r.Data))
	c := color.NRGBAModel.Convert(img.At(3, 3)).(color.NRGBA)
	if c.A < 127 || c.A > 129 || c.R < 195 || c.R > 205 || c.G < 46 || c.G > 54 {
		t.Fatalf("alpha or colour lost: %v", c)
	}
	// Same size, so the transparent pixel is only averaged with itself.
	r, _ = Prepare(b64(encPNG(t, m)), 8, Default)
	img, _ = png.Decode(bytes.NewReader(r.Data))
	if c := color.NRGBAModel.Convert(img.At(0, 0)).(color.NRGBA); c.A != 0 {
		t.Fatalf("transparent pixel became %v", c)
	}
}

func TestJPEGKinds(t *testing.T) {
	// image/jpeg writes 4:2:0 YCbCr; a gray image writes a gray JPEG.
	y := encJPEG(t, gradient(64, 48))
	if img, _ := jpeg.Decode(bytes.NewReader(y)); img.(*image.YCbCr).SubsampleRatio != image.YCbCrSubsampleRatio420 {
		t.Fatal("fixture is not 4:2:0")
	}
	if r, e := Prepare(b64(y), 32, Default); e != nil || r.W != 32 || r.H != 24 {
		t.Fatalf("%v %+v", e, r)
	}
	g := image.NewGray(image.Rect(0, 0, 30, 30))
	if r, e := Prepare(b64(encJPEG(t, g)), 10, Default); e != nil || r.W != 10 {
		t.Fatalf("%v %+v", e, r)
	}
}

func TestCMYKJPEG(t *testing.T) {
	// A 1x1 Adobe CMYK baseline JPEG would be needed as a fixture; the
	// generic At() path is exercised through a CMYK image directly.
	c := image.NewCMYK(image.Rect(0, 0, 10, 10))
	for i := range c.Pix {
		c.Pix[i] = 0
	}
	d := scale(c, 5, 5)
	if p := d.RGBAAt(2, 2); p.R != 255 || p.A != 255 {
		t.Fatalf("%v", p)
	}
}

func TestMetadataDropped(t *testing.T) {
	p := encPNG(t, gradient(16, 16))
	text := chunk("tEXt", []byte("Comment\x00SECRET-METADATA"))
	// Insert after IHDR (8 magic + 25 IHDR bytes).
	in := append(append(append([]byte{}, p[:33]...), text...), p[33:]...)
	if !bytes.Contains(in, []byte("SECRET-METADATA")) {
		t.Fatal("fixture lacks tEXt")
	}
	r, e := Prepare(b64(in), 16, Default)
	if e != nil {
		t.Fatal(e)
	}
	if bytes.Contains(r.Data, []byte("SECRET-METADATA")) || bytes.Contains(r.Data, []byte("tEXt")) {
		t.Fatal("metadata survived")
	}
}

func TestBadLongEdge(t *testing.T) {
	for _, n := range []int{0, -5} {
		_, e := Prepare(b64(encPNG(t, gradient(4, 4))), n, Default)
		wantCode(t, e, "image_bad")
	}
}

func FuzzPrepare(f *testing.F) {
	f.Add(b64(encPNG(f, gradient(40, 20))))
	f.Add("data:image/jpeg;base64," + b64(encJPEG(f, gradient(40, 20))))
	f.Add(b64(hugePNG(100000, 100000)))
	f.Add(b64(hugeJPEG(65535, 65535, true)))
	f.Add(b64(hugeJPEG(65535, 65535, false)))
	f.Add(b64([]byte("GIF89a")))
	f.Add("data:text/html;base64,AAAA")
	f.Add("")
	p := encPNG(f, gradient(40, 20))
	f.Add(b64(p[:len(p)/2]))
	lim := Limits{MaxBase64: 1 << 20, MaxBytes: 1 << 20, MaxPixels: 1 << 20, MaxSide: 4096}
	f.Fuzz(func(t *testing.T, s string) {
		r, e := Prepare(s, 16, lim)
		if e != nil {
			return
		}
		img, _, err := image.Decode(bytes.NewReader(r.Data))
		if err != nil {
			t.Fatalf("output does not decode: %v", err)
		}
		if b := img.Bounds(); max(b.Dx(), b.Dy()) > 16 || b.Dx() != r.W || b.Dy() != r.H {
			t.Fatalf("bad size %v for %+v", b, r)
		}
	})
}
