package fit

// The raw prices, for a test to state what an entry should answer. They are declared in a test
// file on purpose: the package's own code cannot call them, and prices only through settle.

func raw(fp Footprint) fig { return fig{Footprint: fp, src: fp} }

func (fp Footprint) kvAt(slots, ctx int) float64       { return raw(fp).kvAt(slots, ctx) }
func (fp Footprint) vram(slots, ctx, n int) float64    { return raw(fp).vram(slots, ctx, n) }
func (fp Footprint) largest(slots, ctx, n int) float64 { return raw(fp).largest(slots, ctx, n) }
func (fp Footprint) need(s Shape) float64              { return raw(fp).need(s) }
func (fp Footprint) shape(w Wants, slots, ctx, n int, cpuOnly bool) Shape {
	return raw(fp).shape(w, slots, ctx, n, cpuOnly)
}
func (fp Footprint) atType(t string) (Footprint, error) {
	f, err := price(fp, t)
	return f.Footprint, err
}

// capRoom and vramBudget are a room's figures, read as settle reads them.
func capRoom(r Room) int {
	st, err := settle(nil, nil, &r, nil)
	if err != nil {
		panic(err)
	}
	return st.room.capRoom()
}

func vramBudget(r Room) float64 {
	v, _, err := r.Budget()
	if err != nil {
		panic(err)
	}
	return v
}

// taken is whether a calibration was taken, an error counting as not.
func taken(c Calibration, err error) bool { return err == nil && c.Taken() }
