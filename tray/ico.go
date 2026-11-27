package tray

import (
	"encoding/binary"
	"errors"
)

// selectIcoEntry reads the directory of an .ico file held in memory and picks the image to use
// for a square icon of want pixels: an exact width match, else the smallest larger image, else
// the largest one. It returns the image's offset and length inside ico. It is pure so it can be
// tested on any system.
func selectIcoEntry(ico []byte, want int) (off, size int, err error) {
	if len(ico) < 6 {
		return 0, 0, errors.New("tray: ico too short")
	}
	if binary.LittleEndian.Uint16(ico[0:]) != 0 || binary.LittleEndian.Uint16(ico[2:]) != 1 {
		return 0, 0, errors.New("tray: not an ico file")
	}
	n := int(binary.LittleEndian.Uint16(ico[4:]))
	if n == 0 || len(ico) < 6+16*n {
		return 0, 0, errors.New("tray: ico directory is empty or cut short")
	}
	bestW := -1 // width of the entry chosen so far; -1 means none
	for i := 0; i < n; i++ {
		e := ico[6+16*i:]
		w := int(e[0])
		if w == 0 {
			w = 256
		}
		sz := int(binary.LittleEndian.Uint32(e[8:]))
		o := int(binary.LittleEndian.Uint32(e[12:]))
		if sz <= 0 || o < 6+16*n || o > len(ico) || sz > len(ico)-o {
			continue // an entry that points outside the file is skipped
		}
		if better(w, bestW, want) {
			bestW, off, size = w, o, sz
		}
	}
	if bestW < 0 {
		return 0, 0, errors.New("tray: ico has no usable image")
	}
	return off, size, nil
}

// better reports whether width w beats the current best width cur for the wanted size.
func better(w, cur, want int) bool {
	switch {
	case cur < 0:
		return true
	case cur == want:
		return false
	case w == want:
		return true
	case cur > want: // the current choice is larger than wanted: take only a closer larger one
		return w > want && w < cur
	default: // the current choice is too small: any larger image, or a bigger small one, is better
		return w > cur
	}
}
