package gpusense

// STUB (worker A2): the lead deletes this file at merge; A1's package supplies Sense,
// FindAdapter and OnAdapter.

import (
	"context"
	"strings"

	"hermesengine/probe"
)

// Sense reads adapters and per-process use. ours/host pids only set the flags.
func Sense(ctx context.Context, s probe.Sources, ours []int, host int) Picture {
	return Picture{Sources: map[string]string{}}
}

// FindAdapter is the index in as of the adapter a host-supplied (vendor_id, name) describes: the
// same vendor id and a name that contains or is contained in it, else a name match alone, else
// the only adapter of that vendor id. -1 when none.
func FindAdapter(as []Adapter, vendorID int, name string) int {
	n := strings.ToLower(strings.TrimSpace(name))
	match := func(a Adapter) bool {
		an := strings.ToLower(a.Name)
		return n != "" && an != "" && (strings.Contains(an, n) || strings.Contains(n, an))
	}
	for _, a := range as {
		if vendorID != 0 && a.VendorID == vendorID && match(a) {
			return a.Index
		}
	}
	for _, a := range as {
		if match(a) {
			return a.Index
		}
	}
	idx, count := -1, 0
	for _, a := range as {
		if vendorID != 0 && a.VendorID == vendorID {
			idx = a.Index
			count++
		}
	}
	if count == 1 {
		return idx
	}
	return -1
}

// OnAdapter is the processes that hold memory on the adapter with Index idx.
func (p Picture) OnAdapter(idx int) []Process {
	var out []Process
	for _, pr := range p.Processes {
		if pr.Adapter == idx {
			out = append(out, pr)
		}
	}
	return out
}
