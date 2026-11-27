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
type Tracker struct {
	Floor  int // MiB a baseline process may grow before it counts; held across two reads
	MinNew int // MiB a newcomer must hold to count by memory alone (0: any)

	base  map[int]Use
	over  map[int]int // reads in a row a baseline process stood above its floor
	seen  map[int]int // reads in a row a newcomer was there
	known bool
}

// Reset takes cur as the new baseline.
func (t *Tracker) Reset(cur []Use) {
	t.base = map[int]Use{}
	t.over = map[int]int{}
	t.seen = map[int]int{}
	for _, u := range cur {
		t.base[u.PID] = u
	}
	t.known = true
}

// Baselined says whether Reset has run.
func (t *Tracker) Baselined() bool { return t.known }

// Changes lists what differs from the baseline. A newcomer counts once it is there in two
// reads in a row, so a short-lived probe (nvidia-smi, vulkaninfo) never yields the card; the
// caller reads again soon after a device open to keep that fast. A newcomer whose name matches a baseline
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
	for pid := range t.seen {
		if !live[pid] {
			delete(t.seen, pid)
		}
	}
	var out []Change
	for _, u := range cur {
		b, ok := t.base[u.PID]
		if !ok {
			if old, gone := t.relaunchOf(u, live); gone {
				delete(t.base, old.PID)
				delete(t.over, old.PID)
				u.Opened = u.Opened || old.Opened
				t.base[u.PID] = u
				continue
			}
			why := ""
			switch {
			case u.MiB > 0 && u.MiB >= t.MinNew:
				why = "new"
			case u.Opened:
				why = "opened"
			}
			if why == "" {
				continue
			}
			t.seen[u.PID]++
			if t.seen[u.PID] >= 2 {
				out = append(out, Change{PID: u.PID, Name: u.Name, MiB: u.MiB, Why: why})
			}
			continue
		}
		if u.MiB > b.MiB+t.Floor {
			t.over[u.PID]++
		} else {
			t.over[u.PID] = 0
		}
		if t.over[u.PID] >= 2 {
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
