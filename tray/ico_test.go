package tray

import (
	"encoding/binary"
	"testing"
)

// buildIco makes an ico file with one image per width; image i holds i+1 bytes of filler.
func buildIco(widths ...int) []byte {
	n := len(widths)
	b := make([]byte, 6+16*n)
	binary.LittleEndian.PutUint16(b[2:], 1)
	binary.LittleEndian.PutUint16(b[4:], uint16(n))
	off := len(b)
	for i, w := range widths {
		e := b[6+16*i:]
		e[0], e[1] = byte(w), byte(w) // 256 wraps to 0, as in the format
		binary.LittleEndian.PutUint32(e[8:], uint32(i+1))
		binary.LittleEndian.PutUint32(e[12:], uint32(off))
		off += i + 1
	}
	return append(b, make([]byte, off-len(b))...)
}

func TestSelectIcoEntry(t *testing.T) {
	ico := buildIco(16, 32, 48, 256)
	dir := 6 + 16*4
	cases := []struct {
		want     int
		wantSize int // the size field identifies which entry was picked (index+1)
	}{
		{16, 1}, {32, 2}, {24, 2}, {17, 2}, {48, 3}, {64, 4}, {256, 4}, {512, 4}, {1, 1},
	}
	for _, c := range cases {
		off, size, err := selectIcoEntry(ico, c.want)
		if err != nil || size != c.wantSize {
			t.Errorf("want %d: got off=%d size=%d err=%v, expected size %d", c.want, off, size, err, c.wantSize)
		}
		if err == nil && off < dir {
			t.Errorf("want %d: offset %d is inside the directory", c.want, off)
		}
	}
}

func TestSelectIcoEntryUnordered(t *testing.T) {
	_, size, err := selectIcoEntry(buildIco(48, 16, 32), 16)
	if err != nil || size != 2 {
		t.Fatalf("got size=%d err=%v, expected the 16 px entry (size 2)", size, err)
	}
	_, size, err = selectIcoEntry(buildIco(48, 16, 32), 20)
	if err != nil || size != 3 {
		t.Fatalf("got size=%d err=%v, expected the 32 px entry (size 3)", size, err)
	}
}

func TestSelectIcoEntryBad(t *testing.T) {
	good := buildIco(16)
	bad := map[string][]byte{
		"empty":     nil,
		"short":     {0, 0, 1},
		"not ico":   append([]byte{1, 0, 1, 0, 1, 0}, make([]byte, 16)...),
		"no images": {0, 0, 1, 0, 0, 0},
		"cut dir":   {0, 0, 1, 0, 2, 0, 0, 0},
	}
	outside := append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(outside[6+12:], 1<<30)
	bad["offset outside"] = outside
	for name, b := range bad {
		if _, _, err := selectIcoEntry(b, 16); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
