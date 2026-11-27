package fit

import (
	"errors"
	"slices"
	"testing"
)

// A list of cache types in any order is the same list best first (most bytes per element first):
// Choose answers the same shape or the same refusal, and the walk down ends at the same least
// shape, so the need a refusal is worded from (the last type's floor) is the same.
func TestKVTypesAnyOrder(t *testing.T) {
	ref, dense := seed(refInfo(), "q4_0"), seed(denseInfo(), "q4_0")
	best := []string{"f16", "q8_0", "q4_0"}
	orders := [][]string{{"q4_0", "f16"}, {"q4_0", "q8_0", "f16"}, {"q8_0", "q4_0", "f16"}, {"f16", "q4_0", "q8_0"}}
	walk := func(fp Footprint, r Room, w Wants) (Shape, error) {
		cur, err := Choose(fp, r, w)
		if err != nil {
			return Shape{}, err
		}
		for i := 0; i < 1000; i++ {
			next, err := StepDown(fp, cur, r, w)
			if errors.Is(err, ErrTooBig) {
				return cur, nil
			}
			if err != nil {
				return Shape{}, err
			}
			cur = next
		}
		t.Fatal("the walk did not end")
		return Shape{}, nil
	}
	for _, fp := range []Footprint{ref, dense} {
		for vram := 4096; vram <= 24576; vram += 2048 {
			for _, whole := range []bool{true, false} {
				r := room(vram, 1255, 64000)
				for _, list := range orders {
					sorted := slices.Clone(list)
					slices.SortStableFunc(sorted, func(a, b string) int { return slices.Index(best, a) - slices.Index(best, b) })
					w, ws := game, game
					w.Whole, w.KVTypes = whole, list
					ws.Whole, ws.KVTypes = whole, sorted
					s1, e1 := Choose(fp, r, w)
					s2, e2 := Choose(fp, r, ws)
					if s1 != s2 || (e1 == nil) != (e2 == nil) {
						t.Errorf("vram %d whole %v %v: Choose %v %v, best first %v: %v %v", vram, whole, list, s1, e1, sorted, s2, e2)
						continue
					}
					l1, e1 := walk(fp, r, w)
					l2, e2 := walk(fp, r, ws)
					if l1 != l2 || (e1 == nil) != (e2 == nil) {
						t.Errorf("vram %d whole %v %v: least %v %v, best first: %v %v", vram, whole, list, l1, e1, l2, e2)
					}
				}
			}
		}
	}
	if got := (Wants{KVTypes: []string{"q4_0", "", "f16", "q8_0", "f16"}}).Norm().KVTypes; !slices.Equal(got, best) {
		t.Errorf("Norm's list = %v, want %v", got, best)
	}
}
