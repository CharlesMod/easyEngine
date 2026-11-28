package lookimg

import (
	"bytes"
	"encoding/binary"
)

// Structure rules. The walk reads the file's own markers or chunks, never a
// pixel, and runs before image.DecodeConfig. It refuses (Code image_format)
// what the engine does not need and what image/jpeg or image/png would spend
// unbounded work on. A file that simply ends early (truncated) is not refused
// here: the walk covers every byte that is present, so nothing unchecked is
// ever handed to a decoder, and the decode itself then fails with image_bad.
//
// JPEG: SOI first. Then only APPn, COM, DQT, DHT, DRI segments (each length
// bounds-checked), exactly one SOF, which must be SOF0 (baseline DCT) with
// 8-bit precision and 1 or 3 components, and at most 3 SOS scans, each
// followed by its entropy-coded data (0xFF00 stuffing, 0xFF fill bytes and
// RST0-RST7 are data). EOI ends the file: no byte may follow it. At most 64
// segments in all and 1 MiB of APPn and COM payload. Every other marker
// (SOF1-15 including progressive SOF2, DHP, EXP, DNL, JPG, TEM, a second SOI,
// stray RSTn, 0xFF00) is refused.
//
// PNG: signature, then chunks. IHDR first with bit depth 8, colour type 0, 2,
// 3, 4 or 6, compression 0, filter 0 and interlace 0 (no Adam7). acTL, fcTL
// and fdAT (APNG) refused. At most 4096 chunks, at most 64 of them ancillary
// (lowercase first letter). IEND is the last chunk: no byte may follow it.
const (
	maxJPEGSegments = 64
	maxJPEGMeta     = 1 << 20
	maxJPEGScans    = 3
	maxPNGChunks    = 4096
	maxPNGAncillary = 64
)

func checkJPEG(raw []byte) *Error {
	n := len(raw)
	pos := 2 // past SOI (the magic sniff proved FF D8)
	segs, meta, scans := 0, 0, 0
	sof := false
	for {
		// find the marker: 0xFF, any number of 0xFF fill bytes, then a code
		if pos+2 > n {
			return nil // truncated
		}
		if raw[pos] != 0xFF {
			return fail("image_bad", "JPEG has stray bytes between segments")
		}
		pos++
		for pos < n && raw[pos] == 0xFF {
			pos++
		}
		if pos >= n {
			return nil
		}
		m := raw[pos]
		pos++
		switch {
		case m == 0xD9: // EOI
			if pos != n {
				return fail("image_format", "JPEG has %d bytes after its end marker", n-pos)
			}
			return nil
		case m == 0xC0, m == 0xDA, m == 0xDB, m == 0xC4, m == 0xDD, m == 0xFE, m >= 0xE0 && m <= 0xEF:
		case m >= 0xC1 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC:
			return fail("image_format", "only baseline JPEG is accepted (frame type SOF%d refused)", m-0xC0)
		case m == 0x00 || (m >= 0x02 && m <= 0xBF):
			return fail("image_bad", "JPEG has an invalid marker 0xFF%02X", m)
		default:
			return fail("image_format", "JPEG marker 0xFF%02X is not accepted", m)
		}
		segs++
		if segs > maxJPEGSegments {
			return fail("image_format", "JPEG has more than %d segments", maxJPEGSegments)
		}
		if pos+2 > n {
			return nil
		}
		l := int(binary.BigEndian.Uint16(raw[pos:]))
		if l < 2 {
			return fail("image_bad", "JPEG segment length is invalid")
		}
		if pos+l > n {
			return nil // truncated inside a segment
		}
		seg := raw[pos+2 : pos+l]
		pos += l
		switch {
		case m == 0xC0:
			if sof {
				return fail("image_format", "JPEG has more than one frame header")
			}
			sof = true
			if len(seg) < 6 {
				return fail("image_bad", "JPEG frame header is short")
			}
			if seg[0] != 8 {
				return fail("image_format", "only 8-bit JPEG is accepted (precision %d)", seg[0])
			}
			if nc := int(seg[5]); nc != 1 && nc != 3 {
				return fail("image_format", "only grey or 3-component JPEG is accepted (%d components)", nc)
			} else if len(seg) != 6+3*nc {
				return fail("image_bad", "JPEG frame header length is invalid")
			}
		case m == 0xDA:
			if !sof {
				return fail("image_bad", "JPEG scan before its frame header")
			}
			scans++
			if scans > maxJPEGScans {
				return fail("image_format", "JPEG has more than %d scans", maxJPEGScans)
			}
			if len(seg) < 1 || int(seg[0]) < 1 || int(seg[0]) > 3 {
				return fail("image_bad", "JPEG scan header is invalid")
			}
			// entropy-coded data up to the next real marker
			for {
				i := bytes.IndexByte(raw[pos:], 0xFF)
				if i < 0 {
					return nil // truncated in data
				}
				pos += i
				if pos+1 >= n {
					return nil
				}
				c := raw[pos+1]
				if c == 0x00 || (c >= 0xD0 && c <= 0xD7) {
					pos += 2
					continue
				}
				if c == 0xFF {
					pos++
					continue
				}
				break // pos is at the next marker's 0xFF
			}
		case m >= 0xE0 && m <= 0xEF, m == 0xFE:
			meta += len(seg)
			if meta > maxJPEGMeta {
				return fail("image_format", "JPEG carries more than %d bytes of APP/COM data", maxJPEGMeta)
			}
		}
	}
}

func checkPNG(raw []byte) *Error {
	n := len(raw)
	pos := len(pngMagic)
	chunks, anc := 0, 0
	for {
		if pos+8 > n {
			return nil // truncated
		}
		l := int64(binary.BigEndian.Uint32(raw[pos:]))
		typ := raw[pos+4 : pos+8]
		for _, c := range typ {
			if !(c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z') {
				return fail("image_bad", "PNG chunk type is invalid")
			}
		}
		chunks++
		if chunks > maxPNGChunks {
			return fail("image_format", "PNG has more than %d chunks", maxPNGChunks)
		}
		if typ[0] >= 'a' {
			anc++
			if anc > maxPNGAncillary {
				return fail("image_format", "PNG has more than %d ancillary chunks", maxPNGAncillary)
			}
		}
		t := string(typ)
		if chunks == 1 && t != "IHDR" {
			return fail("image_bad", "PNG does not start with IHDR")
		}
		switch t {
		case "acTL", "fcTL", "fdAT":
			return fail("image_format", "animated PNG is not accepted")
		}
		if l > 0x7fffffff || int64(pos)+12+l > int64(n) {
			if t == "IHDR" && l != 13 {
				return fail("image_bad", "PNG IHDR length is invalid")
			}
			return nil // truncated inside a chunk
		}
		data := raw[pos+8 : pos+8+int(l)]
		pos += 12 + int(l)
		switch t {
		case "IHDR":
			if len(data) != 13 {
				return fail("image_bad", "PNG IHDR length is invalid")
			}
			if data[8] != 8 {
				return fail("image_format", "only 8-bit PNG is accepted (bit depth %d)", data[8])
			}
			switch data[9] {
			case 0, 2, 3, 4, 6:
			default:
				return fail("image_bad", "PNG colour type %d is invalid", data[9])
			}
			if data[10] != 0 || data[11] != 0 {
				return fail("image_bad", "PNG compression or filter method is invalid")
			}
			if data[12] != 0 {
				return fail("image_format", "interlaced PNG is not accepted")
			}
		case "IEND":
			if pos != n {
				return fail("image_format", "PNG has %d bytes after its end chunk", n-pos)
			}
			return nil
		}
	}
}
