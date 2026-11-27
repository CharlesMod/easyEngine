package yield

import "sort"

// Use is one process on the engine's adapter: its memory there, and whether it holds a GPU
// device open (Linux /proc/<pid>/fd; a program must open the device before it can allocate).
type Use struct {
	PID    int
	Name   string
	MiB    int
	Opened bool
}

// Change is a process that is new since the baseline (it opened the GPU or holds memory on
// the engine's adapter) or a baseline process that grew past the noise floor.
type Change struct {
	PID  int
	Name string
	MiB  int
	Why  string // opened | new | grew
}

// Tracker compares what runs on the card now with a baseline taken when the engine last
// loaded or resumed. Processes the caller excludes (the engine's own, the host) never reach it.
// It is fed only readings that have held (host.Level reports a change once two reads in a row
// agree), so a newcomer counts on the first reading it is in: a short-lived probe never held.
type Tracker struct {
	Floor  int // MiB a baseline process may grow before it counts
	MinNew int // MiB a newcomer must hold to count by memory alone (0: any)

	base  map[int]Use
	known bool
}

// Reset takes cur as the new baseline.
func (t *Tracker) Reset(cur []Use) {
	t.base = map[int]Use{}
	for _, u := range cur {
		t.base[u.PID] = u
	}
	t.known = true
}

// Baselined says whether Reset has run.
func (t *Tracker) Baselined() bool { return t.known }

// Changes lists what differs from the baseline in a reading that has held. A newcomer whose name matches a baseline
// process that has gone, at no more memory than it held plus the floor, is a relaunch: it
// takes that place in the baseline and is not a change.
func (t *Tracker) Changes(cur []Use) []Change {
	if !t.known {
		return nil
	}
	live := map[int]bool{}
	for _, u := range cur {
		live[u.PID] = true
	}
	var out []Change
	for _, u := range cur {
		b, ok := t.base[u.PID]
		if !ok {
			if old, gone := t.relaunchOf(u, live); gone {
				delete(t.base, old.PID)
				u.Opened = u.Opened || old.Opened
				t.base[u.PID] = u
				continue
			}
			switch {
			case u.MiB > 0 && u.MiB >= t.MinNew:
				out = append(out, Change{PID: u.PID, Name: u.Name, MiB: u.MiB, Why: "new"})
			case u.Opened:
				out = append(out, Change{PID: u.PID, Name: u.Name, MiB: u.MiB, Why: "opened"})
			}
			continue
		}
		if u.MiB > b.MiB+t.Floor {
			out = append(out, Change{PID: u.PID, Name: u.Name, MiB: u.MiB, Why: "grew"})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].PID < out[j].PID })
	return out
}

func (t *Tracker) relaunchOf(u Use, live map[int]bool) (Use, bool) {
	if u.Name == "" {
		return Use{}, false
	}
	for pid, b := range t.base {
		if !live[pid] && b.Name == u.Name && u.MiB <= b.MiB+t.Floor {
			return b, true
		}
	}
	return Use{}, false
}
