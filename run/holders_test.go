package run

import "testing"

// Two words come before one scan: that scan serves both of them, and so both watches. A word that
// comes after it needs a scan of its own.
func TestHolderScanOneScanServesTwoWords(t *testing.T) {
	n := 0
	f := func() map[int]string { n++; return map[int]string{n: "x"} }
	old := holderScanHook.Swap(&f)
	t.Cleanup(func() { holderScanHook.Store(old) })
	h := &holderScan{}
	a, b := h.word(), h.word()
	ra := h.look(a, "")
	rb := h.look(b, "")
	if h.scans != 1 || len(ra) != 1 || len(rb) != 1 || ra[1] != "x" {
		t.Fatalf("scans %d, %v, %v", h.scans, ra, rb)
	}
	c := h.word()
	if rc := h.look(c, ""); h.scans != 2 || rc[2] != "x" {
		t.Fatalf("scans %d, %v", h.scans, rc)
	}
	if h.look(c, ""); h.scans != 2 {
		t.Fatalf("a repeat of the same word scanned again: %d", h.scans)
	}
}
