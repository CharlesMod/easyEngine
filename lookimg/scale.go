package lookimg

import (
	"image"
	"image/color"
)

// adder adds the source row y (relative to the image origin) into sum, one
// RGBA quadruple of 16-bit premultiplied values per destination column; column
// dx covers source columns xs[dx] to xs[dx+1].
type adder func(y int, xs []int, sum []uint64)

// scale box-filters src into a new dw x dh RGBA (dw <= source width,
// dh <= source height). Averaging is done on premultiplied values, so alpha
// stays straight when the PNG encoder un-premultiplies. It allocates the
// target image plus three tables of size dw (and a 256-entry palette table);
// nothing per row or per pixel. A type the decoders can not yield for an
// allowed input is image_format.
func scale(src image.Image, dw, dh int) (*image.RGBA, *Error) {
	b := src.Bounds()
	if b.Min != (image.Point{}) {
		return nil, fail("image_format", "decoded image has an unexpected origin")
	}
	sw, sh := b.Dx(), b.Dy()
	add := rowAdder(src)
	if add == nil {
		return nil, fail("image_format", "decoded image has an unsupported pixel type")
	}
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	xs := make([]int, dw+1)
	for i := range xs {
		xs[i] = i * sw / dw
	}
	sum := make([]uint64, dw*4)
	for dy := 0; dy < dh; dy++ {
		y0, y1 := dy*sh/dh, (dy+1)*sh/dh
		for i := range sum {
			sum[i] = 0
		}
		for y := y0; y < y1; y++ {
			add(y, xs, sum)
		}
		rows := uint64(y1 - y0)
		o := dst.PixOffset(0, dy)
		for dx := 0; dx < dw; dx++ {
			n := rows * uint64(xs[dx+1]-xs[dx])
			s := sum[dx*4 : dx*4+4 : dx*4+4]
			p := dst.Pix[o+dx*4 : o+dx*4+4 : o+dx*4+4]
			p[0] = uint8((s[0]/n + 128) / 257)
			p[1] = uint8((s[1]/n + 128) / 257)
			p[2] = uint8((s[2]/n + 128) / 257)
			p[3] = uint8((s[3]/n + 128) / 257)
		}
	}
	return dst, nil
}

// rowAdder returns the typed row reader for every type the allowed inputs
// decode to, or nil.
func rowAdder(img image.Image) adder {
	switch m := img.(type) {
	case *image.YCbCr:
		// chroma subsampling: horizontal shift and vertical halving per ratio
		var cxs uint
		vhalf := false
		switch m.SubsampleRatio {
		case image.YCbCrSubsampleRatio444:
		case image.YCbCrSubsampleRatio422:
			cxs = 1
		case image.YCbCrSubsampleRatio420:
			cxs, vhalf = 1, true
		case image.YCbCrSubsampleRatio440:
			vhalf = true
		case image.YCbCrSubsampleRatio411:
			cxs = 2
		case image.YCbCrSubsampleRatio410:
			cxs, vhalf = 2, true
		default:
			return nil
		}
		return func(y int, xs []int, sum []uint64) {
			yr := m.Y[y*m.YStride:]
			cy := y
			if vhalf {
				cy = y >> 1
			}
			cb := m.Cb[cy*m.CStride:]
			cr := m.Cr[cy*m.CStride:]
			for dx := 0; dx < len(xs)-1; dx++ {
				var r0, g0, b0 uint64
				for x := xs[dx]; x < xs[dx+1]; x++ {
					r, g, bl := color.YCbCrToRGB(yr[x], cb[x>>cxs], cr[x>>cxs])
					r0 += uint64(r)
					g0 += uint64(g)
					b0 += uint64(bl)
				}
				n := uint64(xs[dx+1] - xs[dx])
				s := sum[dx*4 : dx*4+4 : dx*4+4]
				s[0] += r0 * 0x101
				s[1] += g0 * 0x101
				s[2] += b0 * 0x101
				s[3] += n * 0xffff
			}
		}
	case *image.Gray:
		return func(y int, xs []int, sum []uint64) {
			row := m.Pix[y*m.Stride:]
			for dx := 0; dx < len(xs)-1; dx++ {
				var v uint64
				for x := xs[dx]; x < xs[dx+1]; x++ {
					v += uint64(row[x])
				}
				n := uint64(xs[dx+1] - xs[dx])
				s := sum[dx*4 : dx*4+4 : dx*4+4]
				s[0] += v * 0x101
				s[1] += v * 0x101
				s[2] += v * 0x101
				s[3] += n * 0xffff
			}
		}
	case *image.RGBA:
		return func(y int, xs []int, sum []uint64) {
			row := m.Pix[y*m.Stride:]
			for dx := 0; dx < len(xs)-1; dx++ {
				var r, g, bl, a uint64
				for x := xs[dx]; x < xs[dx+1]; x++ {
					p := row[x*4 : x*4+4 : x*4+4]
					r += uint64(p[0])
					g += uint64(p[1])
					bl += uint64(p[2])
					a += uint64(p[3])
				}
				s := sum[dx*4 : dx*4+4 : dx*4+4]
				s[0] += r * 0x101
				s[1] += g * 0x101
				s[2] += bl * 0x101
				s[3] += a * 0x101
			}
		}
	case *image.NRGBA:
		return func(y int, xs []int, sum []uint64) {
			row := m.Pix[y*m.Stride:]
			for dx := 0; dx < len(xs)-1; dx++ {
				var r, g, bl, a uint64
				for x := xs[dx]; x < xs[dx+1]; x++ {
					p := row[x*4 : x*4+4 : x*4+4]
					a16 := uint64(p[3]) * 0x101
					r += uint64(p[0]) * 0x101 * a16 / 0xffff
					g += uint64(p[1]) * 0x101 * a16 / 0xffff
					bl += uint64(p[2]) * 0x101 * a16 / 0xffff
					a += a16
				}
				s := sum[dx*4 : dx*4+4 : dx*4+4]
				s[0] += r
				s[1] += g
				s[2] += bl
				s[3] += a
			}
		}
	case *image.Paletted:
		var tab [256][4]uint64
		for i, c := range m.Palette {
			if i >= len(tab) {
				break
			}
			r, g, b, a := c.RGBA()
			tab[i] = [4]uint64{uint64(r), uint64(g), uint64(b), uint64(a)}
		}
		return func(y int, xs []int, sum []uint64) {
			row := m.Pix[y*m.Stride:]
			for dx := 0; dx < len(xs)-1; dx++ {
				var r, g, bl, a uint64
				for x := xs[dx]; x < xs[dx+1]; x++ {
					t := &tab[row[x]]
					r += t[0]
					g += t[1]
					bl += t[2]
					a += t[3]
				}
				s := sum[dx*4 : dx*4+4 : dx*4+4]
				s[0] += r
				s[1] += g
				s[2] += bl
				s[3] += a
			}
		}
	}
	return nil
}
