package lookimg

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

// noiseSide is the side of the pixel-limit fixtures: 4096 x 4096 = 16 Mi px.
const noiseSide = 4096

func rng() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

func testdata(t testing.TB, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// kind is one allowed input kind, built at a given side with noise sized so
// that the file stays inside Default.MaxBytes at 4096 x 4096.
type kind struct {
	name string
	make func(t testing.TB, side int) []byte
}

func pngBytes(t testing.TB, m image.Image) []byte {
	t.Helper()
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	var b bytes.Buffer
	if err := enc.Encode(&b, m); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func jpegBytes(t testing.TB, m image.Image, q int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := jpeg.Encode(&b, m, &jpeg.Options{Quality: q}); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

var kinds = []kind{
	{"png-rgb", func(t testing.TB, n int) []byte {
		r := rng()
		m := image.NewRGBA(image.Rect(0, 0, n, n))
		for i := 0; i < len(m.Pix); i += 4 {
			v := r.Uint32()
			m.Pix[i], m.Pix[i+1], m.Pix[i+2], m.Pix[i+3] = byte(v&1), 5, 7, 255
		}
		return pngBytes(t, m) // opaque: written as colour type 2
	}},
	{"png-rgba", func(t testing.TB, n int) []byte {
		r := rng()
		m := image.NewNRGBA(image.Rect(0, 0, n, n))
		for i := 0; i < len(m.Pix); i += 4 {
			v := r.Uint32()
			m.Pix[i], m.Pix[i+1], m.Pix[i+2], m.Pix[i+3] = byte(v&1), 9, 7, 255
		}
		m.Pix[3] = 100 // one see-through pixel: colour type 6
		return pngBytes(t, m)
	}},
	{"png-gray", func(t testing.TB, n int) []byte {
		r := rng()
		m := image.NewGray(image.Rect(0, 0, n, n))
		for i := range m.Pix {
			m.Pix[i] = byte(r.Uint32() & 3)
		}
		return pngBytes(t, m)
	}},
	{"png-palette", func(t testing.TB, n int) []byte {
		r := rng()
		pal := make(color.Palette, 256)
		for i := range pal {
			pal[i] = color.RGBA{uint8(i), uint8(i * 7), uint8(255 - i), 255}
		}
		m := image.NewPaletted(image.Rect(0, 0, n, n), pal)
		for i := range m.Pix {
			m.Pix[i] = byte(r.Uint32() & 3)
		}
		return pngBytes(t, m)
	}},
	{"jpeg-420", func(t testing.TB, n int) []byte {
		r := rng()
		m := image.NewRGBA(image.Rect(0, 0, n, n))
		for i := 0; i < len(m.Pix); i += 4 {
			v := r.Uint32()
			m.Pix[i], m.Pix[i+1], m.Pix[i+2], m.Pix[i+3] = byte(v), byte(v>>8), byte(v>>16), 255
		}
		return jpegBytes(t, m, 50)
	}},
	{"jpeg-444", func(t testing.TB, n int) []byte {
		r := rng()
		m := image.NewGray(image.Rect(0, 0, n, n))
		for i := range m.Pix {
			m.Pix[i] = byte(r.Uint32())
		}
		return grayTo444(t, jpegBytes(t, m, 5), 3)
	}},
	{"jpeg-gray", func(t testing.TB, n int) []byte {
		r := rng()
		m := image.NewGray(image.Rect(0, 0, n, n))
		for i := range m.Pix {
			m.Pix[i] = byte(r.Uint32())
		}
		return jpegBytes(t, m, 50)
	}},
}

// grayTo444 turns the grey JPEG the Go encoder writes into a baseline 4:4:4
// three-component file made of k non-interleaved scans (k = 1..3 components
// get a scan each; with k > 3 the extra scans repeat component 3, which is a
// structure test, not a decodable picture). Every scan reuses the grey scan
// data, which is valid for a 1x1-sampled component.
func grayTo444(t testing.TB, g []byte, scans int) []byte {
	t.Helper()
	sof := bytes.Index(g, []byte{0xff, 0xc0})
	sos := bytes.Index(g, []byte{0xff, 0xda})
	if sof < 0 || sos < 0 {
		t.Fatal("no SOF0/SOS in grey JPEG")
	}
	sofEnd := sof + 2 + int(binary.BigEndian.Uint16(g[sof+2:]))
	sosEnd := sos + 2 + int(binary.BigEndian.Uint16(g[sos+2:]))
	data := g[sosEnd : len(g)-2]
	out := append([]byte{}, g[:sof]...)
	out = append(out, 0xff, 0xc0, 0, 17, 8)
	out = append(out, g[sof+5:sof+9]...) // height, width
	out = append(out, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0)
	out = append(out, g[sofEnd:sos]...)
	for i := 0; i < scans; i++ {
		c := byte(min(i+1, 3))
		out = append(out, 0xff, 0xda, 0, 8, 1, c, 0, 0, 63, 0)
		out = append(out, data...)
	}
	return append(out, 0xff, 0xd9)
}
