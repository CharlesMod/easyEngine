package fit

import (
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
)

// feed is what one case hands an entry.
type feed struct {
	fp   Footprint
	s    Shape
	room Room
	w    Wants
	kv   string
	book *Book
}

// entry is one exported function or method of the package. takes says which of a room, wants, a
// shape, a cache type name and a footprint or a set of constants (fp) it is given; call answers the price it gave and the price at the
// shape's own type (both 0 where the case prices nothing). why, for an entry with no call, says
// why it prices no memory and reads no room.
type entry struct {
	room, wants, shape, name, fp bool
	call                         func(f feed) (got, want float64, err error)
	why                          string
}

// runs is the cache type a shape is run at: its own, or for "" the default Args renders.
func runs(s Shape) string {
	if s.KVType == "" {
		return "q4_0"
	}
	return s.KVType
}

// walked is a walk entry's answer priced as shapePrice does, and an error where it left the type
// the failed shape ran at: without a list, and with a list that does not hold that type, every
// step keeps it.
func walked(f feed, s Shape, err error) (float64, float64, error) {
	if err == nil && len(f.w.KVTypes) == 0 && runs(s) != runs(f.s) {
		return 0, 0, errors.New("the walk left the type " + runs(f.s) + " the failed shape ran at for " + runs(s))
	}
	return shapePrice(f.fp, s, err)
}

// shapePrice is a returned shape's predicted memory and that shape priced at its own type.
func shapePrice(fp Footprint, s Shape, err error) (float64, float64, error) {
	if err != nil {
		return 0, 0, err
	}
	ft, perr := fp.atType(runs(s))
	if perr != nil { // never the typed error: only the entry may answer that
		return 0, 0, errors.New("the entry answered a shape at a type it cannot price: " + s.KVType)
	}
	if s.CPUOnly {
		return float64(s.RAMMiB), math.Ceil(ft.need(s)), nil
	}
	return float64(s.VRAMMiB), math.Ceil(ft.need(s)), nil
}

// entries is every exported function and method of the package (TestEntriesTableIsComplete).
var entries = map[string]entry{
	"Choose": {room: true, wants: true, fp: true, call: func(f feed) (float64, float64, error) {
		s, err := Choose(f.fp, f.room, f.w)
		return shapePrice(f.fp, s, err)
	}},
	"MissRetry": {room: true, wants: true, shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		s, err := MissRetry(f.fp, f.s, 500, f.room, f.w)
		return walked(f, s, err)
	}},
	"StepDown": {room: true, wants: true, shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		s, err := StepDown(f.fp, f.s, f.room, f.w)
		return walked(f, s, err)
	}},
	"Grow": {room: true, wants: true, shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		s, _, err := Grow(f.fp, f.s, f.room, f.w)
		return shapePrice(f.fp, s, err)
	}},
	"Book.Calibrate": {shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		// A reading equal to the shape's own predicted memory is off by about nothing.
		c, err := f.book.Calibrate("k", f.fp, f.s, f.s.VRAMMiB, stamp)
		if err == nil && c.Verdict != Taken {
			err = errors.New("not taken: " + c.String())
		}
		return math.Round(c.OffMiB()), 0, err
	}},
	"Book.LowerCap": {room: true, shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		ctx := f.s.CtxTotal() / 2
		if err := f.book.LowerCap("k", f.fp, f.s, ctx, f.room, "b1", stamp); err != nil {
			return 0, 0, err
		}
		ft, _ := f.fp.atType(runs(f.s))
		miss := math.Ceil(ft.kvAt(f.s.Slots, f.s.CtxPerSlot) - ft.kvAt(f.s.Slots, ctx/f.s.Slots))
		return float64(f.book.Entries["k"].Lift.MissMiB), miss, nil
	}},
	"Book.LetGo": {room: true, call: func(f feed) (float64, float64, error) {
		_, _, err := f.book.LetGo("k", f.room, "b1")
		return 0, 0, err
	}},
	"Room.Budget": {room: true, call: func(f feed) (float64, float64, error) {
		_, _, err := f.room.Budget()
		return 0, 0, err
	}},
	"Footprint.Need": {shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		v, _, err := f.fp.Need(f.s)
		ft, _ := f.fp.atType(runs(f.s))
		return v, ft.vram(f.s.Slots, f.s.CtxPerSlot, f.s.NCpuMoe), err
	}},
	"Footprint.Largest": {shape: true, fp: true, call: func(f feed) (float64, float64, error) {
		v, err := f.fp.Largest(f.s)
		ft, _ := f.fp.atType(runs(f.s))
		return v, ft.largest(f.s.Slots, f.s.CtxPerSlot, f.s.NCpuMoe), err
	}},
	"Seed": {name: true, call: func(f feed) (float64, float64, error) { _, _, err := Seed(refInfo(), f.kv); return 0, 0, err }},
	"SeedFromBytes": {name: true, call: func(f feed) (float64, float64, error) {
		_, err := SeedFromBytes(10<<30, true, f.kv)
		return 0, 0, err
	}},
	"SeedWith": {name: true, fp: true, call: func(f feed) (float64, float64, error) {
		_, _, err := SeedWith(refInfo(), f.kv, setOf(f.fp))
		return 0, 0, err
	}},
	"SeedFromBytesWith": {name: true, fp: true, call: func(f feed) (float64, float64, error) {
		_, err := SeedFromBytesWith(10<<30, true, f.kv, setOf(f.fp))
		return 0, 0, err
	}},
	"KVBytes": {name: true, call: func(f feed) (float64, float64, error) { _, err := KVBytes(f.kv); return 0, 0, err }},

	"Args":               {why: "renders a shape's flags; prices nothing"},
	"BookKey":            {why: "joins three names"},
	"DefaultMargin":      {why: "a share of a card total it is handed, not a room"},
	"Wants.Norm":         {why: "fills defaults; every entry that prices resolves its wants in settle"},
	"Shape.CtxTotal":     {why: "slots times ctx"},
	"Shape.String":       {why: "prints a shape"},
	"Book.Footprint":     {why: "overlays the book's fixed part and cap on a seed; prices nothing"},
	"Book.Mark":          {why: "copies a reading taken of that very shape; prices nothing"},
	"Calibration.Taken":  {why: "reads a verdict"},
	"Calibration.OffMiB": {why: "subtracts two figures already worked out"},
	"Calibration.String": {why: "prints a verdict"},
	"TPSCurve.At":        {why: "a speed, not memory"},
	"UnreadError.Error":  {why: "an error's text"},
	"KVTypeError.Error":  {why: "an error's text"},
	"ConstError.Error":   {why: "an error's text"},
	"DefaultConsts":      {why: "the set written into the core; prices nothing"},
}

// setOf is the set of constants a footprint carries, or the default where it carries none: what a
// seed entry is given in a case.
func setOf(fp Footprint) Consts {
	if fp.Consts == nil {
		return DefaultConsts()
	}
	return *fp.Consts
}

// Every entry that prices memory or reads a room goes through settle, so each refuses alike: a
// room with a field marked unread is *UnreadError naming it, never a zero, and the book is left as
// it was (no cap let go or lowered, no room recorded); an unknown cache type in the wants, the
// list or a shape is *KVTypeError, never priced or stored; a shape at a type other than the
// seed's is priced at its own type, never the seed's; a footprint or a set of constants lacking one
// is *ConstError naming it, never priced as zero, and the book is left as it was.
func TestEveryEntryRefusesAlike(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	base := room(16303, 3000, 24000)
	unread := base
	unread.Others, unread.Unread = 0, []string{"others"} // taken as zero, the room would look larger
	w := Wants{Slots: 1, CtxMax: 8192, CtxFloor: 8192, KVType: "q4_0"}
	listed := w
	listed.KVTypes = []string{"f16", "q4_0"}
	f16, err := Choose(ref, base, listed)
	if err != nil || f16.KVType != "f16" {
		t.Fatalf("no f16 shape to feed: %v %v", f16, err)
	}
	q4 := f16
	q4.KVType = "q4_0"
	q4.VRAMMiB = int(math.Ceil(ref.vram(q4.Slots, q4.CtxPerSlot, q4.NCpuMoe)))
	untyped := q4
	untyped.KVType = ""
	book := func() *Book { // a cap that a larger room would let go
		return &Book{Entries: map[string]Calib{"k": {CapCtxTotal: 1 << 20, Lift: &CapLift{RoomMiB: capRoom(base), Build: "b1"}}}}
	}
	type kase struct {
		name        string
		applies     func(e entry) bool
		mod         func(f *feed)
		unreadField string    // the *UnreadError expected
		unknownType string    // the *KVTypeError expected
		lackingSet  ConstName // the *ConstError expected
		priced      bool      // got must equal want
		keepsBook   bool
	}
	cases := []kase{
		{name: "every field read, known types", applies: func(entry) bool { return true }, mod: func(*feed) {}},
		{name: "a room field unread", applies: func(e entry) bool { return e.room }, mod: func(f *feed) { f.room = unread }, unreadField: "others", keepsBook: true},
		{name: "unknown type in wants", applies: func(e entry) bool { return e.wants }, mod: func(f *feed) { f.w.KVType = "q3_k" }, unknownType: "q3_k"},
		{name: "unknown type in the list", applies: func(e entry) bool { return e.wants }, mod: func(f *feed) { f.w.KVTypes = []string{"q8_0", "q3_k"} }, unknownType: "q3_k"},
		{name: "unknown type in the shape", applies: func(e entry) bool { return e.shape }, mod: func(f *feed) { f.s.KVType = "q3_k" }, unknownType: "q3_k", keepsBook: true},
		{name: "a set lacking a constant", applies: func(e entry) bool { return e.fp }, mod: func(f *feed) {
			c := DefaultConsts()
			delete(c.Of, MaskMiBPer1k)
			f.fp.Consts = &c
		}, lackingSet: MaskMiBPer1k, keepsBook: true},
		{name: "unknown type name", applies: func(e entry) bool { return e.name }, mod: func(f *feed) { f.kv = "q3_k" }, unknownType: "q3_k"},
		{name: "a shape at a type not the seed's", applies: func(e entry) bool { return e.shape || e.wants }, mod: func(f *feed) { f.s = f16; f.w.KVType = "f16" }, priced: true},
		// A shape naming no type runs at the default (Args): every entry prices it there, with
		// wants at another type or a list that does not hold the default, or with none.
		{name: "a shape naming no type, wants at another", applies: func(e entry) bool { return e.shape || e.wants }, mod: func(f *feed) { f.s = untyped; f.w.KVType = "f16" }, priced: true},
		{name: "a shape naming no type, a list without the default", applies: func(e entry) bool { return e.shape || e.wants }, mod: func(f *feed) { f.s = untyped; f.w.KVTypes = []string{"f16", "q8_0"} }, priced: true},
	}
	names := make([]string, 0, len(entries))
	for n := range entries {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, c := range cases {
		for _, n := range names {
			e := entries[n]
			if e.call == nil || !c.applies(e) {
				continue
			}
			f := feed{fp: ref, s: q4, room: base, w: w, book: book()}
			c.mod(&f)
			before, _ := json.Marshal(f.book)
			got, want, err := e.call(f)
			var ue *UnreadError
			var ke *KVTypeError
			var ce *ConstError
			switch {
			case c.unreadField != "":
				if !errors.As(err, &ue) || ue.Field != c.unreadField {
					t.Errorf("%s: %s answered %v, want the unread %s", n, c.name, err, c.unreadField)
				}
			case c.lackingSet != "":
				if !errors.As(err, &ce) || ce.Name != c.lackingSet {
					t.Errorf("%s: %s answered %v, want the lacking %s", n, c.name, err, c.lackingSet)
				}
			case c.unknownType != "":
				if !errors.As(err, &ke) || ke.Name != c.unknownType {
					t.Errorf("%s: %s answered %v, want the unknown %s", n, c.name, err, c.unknownType)
				}
			case err != nil:
				t.Errorf("%s: %s answered %v", n, c.name, err)
			case c.priced && math.Abs(got-want) > 1:
				t.Errorf("%s: %s priced %.1f MiB, at its own type %.1f", n, c.name, got, want)
			}
			if after, _ := json.Marshal(f.book); c.keepsBook && string(after) != string(before) {
				t.Errorf("%s: %s changed the book:\n%s\n%s", n, c.name, before, after)
			}
		}
	}
}

// The table above is every exported function and method, read from the package's source, so an
// entry added without a row fails here. And the one checked place is the only one: the priced
// forms are built only in settle and price, and price and resolve are called only from settle
// (and fig.at), so no entry reaches a raw price or budget past it. A set of constants is read only
// by readConsts, which only settle calls.
func TestEntriesTableIsComplete(t *testing.T) {
	files, _ := filepath.Glob("*.go")
	fset := token.NewFileSet()
	found := map[string]bool{}
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(fset, name, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range file.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			key := fn.Name.Name
			if fn.Recv != nil {
				rt := fn.Recv.List[0].Type
				if st, ok := rt.(*ast.StarExpr); ok {
					rt = st.X
				}
				recv := rt.(*ast.Ident).Name
				key = recv + "." + key
				if !ast.IsExported(recv) {
					key = ""
				}
			}
			if key != "" && fn.Name.IsExported() {
				found[key] = true
			}
			where := fn.Name.Name
			if fn.Recv != nil {
				where = key
				if where == "" {
					where = "?." + fn.Name.Name
				}
			}
			ast.Inspect(fn, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.CompositeLit:
					if id, ok := x.Type.(*ast.Ident); ok && (id.Name == "fig" || id.Name == "roomFig") &&
						fn.Name.Name != "settle" && fn.Name.Name != "price" {
						t.Errorf("%s builds a %s itself: only settle and price may", where, id.Name)
					}
					if id, ok := x.Type.(*ast.Ident); ok && id.Name == "constFig" && fn.Name.Name != "readConsts" {
						t.Errorf("%s builds a constFig itself: only readConsts may", where)
					}
				case *ast.CallExpr:
					if id, ok := x.Fun.(*ast.Ident); ok && (id.Name == "price" || id.Name == "resolve") &&
						fn.Name.Name != "settle" && !(fn.Name.Name == "at" && id.Name == "price") {
						t.Errorf("%s calls %s past settle", where, id.Name)
					}
					if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "readConsts" && fn.Name.Name != "settle" {
						t.Errorf("%s reads a set of constants past settle", where)
					}
				}
				return true
			})
		}
	}
	for k := range found {
		if _, ok := entries[k]; !ok {
			t.Errorf("exported %s has no row in the entries table", k)
		}
	}
	for k, e := range entries {
		if !found[k] {
			t.Errorf("the entries table names %s, which the package does not export", k)
		}
		if (e.call == nil) == (e.why == "") {
			t.Errorf("%s: a row has a call or says why it needs none, not both", k)
		}
	}
}

// A shape naming no type runs at the default, so the walk down prices it there whatever type the
// wants name: no step answers a shape needing more memory, more slots or more context than it.
func TestUntypedShapeWalksAtTheDefault(t *testing.T) {
	ref := seed(refInfo(), "q4_0")
	rm := room(16303, 3000, 24000)
	cur := Shape{Slots: 1, CtxPerSlot: 16384}
	q4, _ := ref.atType("q4_0")
	need := q4.vram(cur.Slots, cur.CtxPerSlot, 0)
	for _, list := range [][]string{nil, {"f16", "q8_0"}} {
		w := Wants{Slots: 1, CtxMax: 16384, CtxFloor: 1024, KVType: "f16", KVTypes: list}
		if v, _, err := ref.Need(cur); err != nil || math.Abs(v-need) > 1 {
			t.Fatalf("Need: %.1f %v, at the default %.1f", v, err, need)
		}
		check := func(op string, s Shape, err error) {
			if err != nil {
				t.Errorf("%v %s: %v", list, op, err)
				return
			}
			ft, _ := ref.atType(runs(s))
			if n := ft.need(s); n > need || s.Slots > cur.Slots || s.CtxPerSlot > cur.CtxPerSlot || runs(s) != "q4_0" {
				t.Errorf("%v %s after a failed %v (%.0f MiB at q4_0): %v %s needing %.0f MiB", list, op, cur, need, s, s.KVType, n)
			}
		}
		s, err := StepDown(ref, cur, rm, w)
		check("StepDown", s, err)
		s, err = MissRetry(ref, cur, 500, rm, w)
		check("MissRetry", s, err)
	}
}

// The walk down from shapes naming no type or a named one, over a handful of rooms and lists,
// StepDown and MissRetry alike: each step lowers the measure (types left after the type the shape
// runs at, then its memory) and is no larger in slots or ctx, and the walk ends at ErrTooBig.
func TestWalkFromAnyTypeEnds(t *testing.T) {
	ref, dense := seed(refInfo(), "q4_0"), seed(denseInfo(), "q4_0")
	lists := [][]string{nil, {"f16", "q4_0"}, {"q4_0", "f16"}, {"f16", "q8_0"}}
	cases := []struct {
		fp    Footprint
		whole bool
	}{{ref, true}, {ref, false}, {dense, true}}
	for _, c := range cases {
		for vram := 8192; vram <= 24576; vram += 8192 {
			r := room(vram, 1255, 64000)
			for _, list := range lists {
				for _, wt := range []string{"q4_0", "f16"} {
					for _, st := range []string{"", "q4_0", "f16", "q8_0"} {
						w := game
						w.Whole, w.KVType, w.KVTypes = c.whole, wt, list
						types := kvTypes(list)
						measure := func(s Shape) (int, float64) {
							ft, _ := c.fp.atType(runs(s))
							left := 0
							if i := slices.Index(types, runs(s)); i >= 0 {
								left = len(types) - 1 - i
							}
							return left, ft.need(s)
						}
						for op, step := range map[string]func(Shape) (Shape, error){
							"StepDown":  func(s Shape) (Shape, error) { return StepDown(c.fp, s, r, w) },
							"MissRetry": func(s Shape) (Shape, error) { return MissRetry(c.fp, s, 500, r, w) },
						} {
							cur := Shape{Slots: 2, CtxPerSlot: 16384, KVType: st}
							for i := 0; ; i++ {
								next, err := step(cur)
								if errors.Is(err, ErrTooBig) {
									break
								}
								if err != nil || i > 2000 {
									t.Fatalf("%s %v %s from %q on %d: %v (step %d)", op, list, wt, st, vram, err, i)
								}
								l0, m0 := measure(cur)
								l1, m1 := measure(next)
								if l1 > l0 || (l1 == l0 && m1 >= m0) || next.Slots > cur.Slots || next.CtxPerSlot > cur.CtxPerSlot {
									t.Fatalf("%s %v wants %s, whole %v, on %d: %v %q -> %v %q does not lower the measure",
										op, list, wt, c.whole, vram, cur, cur.KVType, next, next.KVType)
								}
								cur = next
							}
						}
					}
				}
			}
		}
	}
}
