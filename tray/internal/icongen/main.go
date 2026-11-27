// Command icongen draws the tray's default icon and writes it to tray/icons/: a full colour PNG
// (Linux), an ICO with the small sizes Windows asks for, and a black-on-transparent template PNG
// (macOS menu bar). The mark is plain geometry, so it reads at 16 px on light and dark trays: a
// teal disc with a dark rim, a white ring inside it and a white dot in the middle.
// Run from the repo root: go run ./tray/internal/icongen
package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
)

// Radii as fractions of half the icon size.
const (
	discR   = 1.00
	rimW    = 0.10 // dark rim inside the disc edge, so the disc holds on a light tray
	ringOut = 0.66
	ringIn  = 0.46
	dotR    = 0.22
)

var (
	teal  = [3]float64{0x14, 0xa3, 0x8b}
	rim   = [3]float64{0x0b, 0x5e, 0x50}
	white = [3]float64{0xff, 0xff, 0xff}
)

// shade says what covers the point at distance r (in half-sizes) from the centre: the colour
// and whether it is part of the template's ink.
func shade(r float64) (c [3]float64, alpha float64, ink bool) {
	switch {
	case r > discR:
		return c, 0, false
	case r <= dotR:
		return white, 1, false
	case r <= ringIn:
		return teal, 1, true
	case r <= ringOut:
		return white, 1, false
	case r <= discR-rimW:
		return teal, 1, true
	default:
		return rim, 1, true
	}
}

// draw renders the mark at n x n with 8 x 8 supersampling. template draws black ink only.
func draw(n int, template bool) *image.NRGBA {
	const ss = 8
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	half := float64(n) / 2
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			var sr, sg, sb, sa float64
			for j := 0; j < ss; j++ {
				for i := 0; i < ss; i++ {
					px := (float64(x) + (float64(i)+0.5)/ss - half) / half
					py := (float64(y) + (float64(j)+0.5)/ss - half) / half
					c, a, ink := shade(math.Hypot(px, py))
					if template {
						if ink {
							sa++
						}
						continue
					}
					sr += c[0] * a
					sg += c[1] * a
					sb += c[2] * a
					sa += a
				}
			}
			if sa == 0 {
				continue
			}
			px := color.NRGBA{A: uint8(math.Round(sa / (ss * ss) * 255))}
			if !template {
				px.R = uint8(math.Round(sr / sa))
				px.G = uint8(math.Round(sg / sa))
				px.B = uint8(math.Round(sb / sa))
			}
			img.SetNRGBA(x, y, px)
		}
	}
	return img
}

func encode(img image.Image) []byte {
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		log.Fatal(err)
	}
	return b.Bytes()
}

// ico packs PNG images into an .ico file (PNG entries are valid since Windows Vista).
func ico(sizes []int) []byte {
	var head, body bytes.Buffer
	binary.Write(&head, binary.LittleEndian, [3]uint16{0, 1, uint16(len(sizes))})
	off := 6 + 16*len(sizes)
	for _, n := range sizes {
		p := encode(draw(n, false))
		w := uint8(n)
		if n >= 256 {
			w = 0
		}
		head.Write([]byte{w, w, 0, 0})
		binary.Write(&head, binary.LittleEndian, [2]uint16{1, 32})
		binary.Write(&head, binary.LittleEndian, [2]uint32{uint32(len(p)), uint32(off)})
		body.Write(p)
		off += len(p)
	}
	return append(head.Bytes(), body.Bytes()...)
}

func main() {
	dir := flag.String("out", filepath.Join("tray", "icons"), "directory to write the icons to")
	flag.Parse()
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		log.Fatal(err)
	}
	files := map[string][]byte{
		"tray.png":     encode(draw(64, false)),
		"tray.ico":     ico([]int{16, 20, 24, 32, 40, 48, 64}),
		"template.png": encode(draw(36, true)), // 18 pt at 2x
	}
	for name, b := range files {
		if err := os.WriteFile(filepath.Join(*dir, name), b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
