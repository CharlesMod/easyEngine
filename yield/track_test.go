package yield

import (
	"reflect"
	"testing"
)

func TestTrackerChangesBeforeReset(t *testing.T) {
	tr := &Tracker{}
	if tr.Baselined() {
		t.Fatal("baselined before Reset")
	}
	if got := tr.Changes([]Use{{PID: 1, MiB: 5000}}); got != nil {
		t.Fatalf("got %v, want nil", got)
	}
	tr.Reset(nil)
	if !tr.Baselined() {
		t.Fatal("not baselined after Reset")
	}
}

func TestTrackerBaselineIsQuiet(t *testing.T) {
	tr := &Tracker{Floor: 100}
	base := []Use{{PID: 1, Name: "a", MiB: 500}, {PID: 2, Name: "b", MiB: 10, Opened: true}}
	tr.Reset(base)
	for i := 0; i < 3; i++ {
		if got := tr.Changes(base); len(got) != 0 {
			t.Fatalf("read %d: %v", i, got)
		}
	}
}

func TestTrackerNewcomer(t *testing.T) {
	tr := &Tracker{Floor: 100}
	tr.Reset([]Use{{PID: 1, Name: "a", MiB: 500}})
	cur := []Use{{PID: 1, Name: "a", MiB: 500}, {PID: 9, Name: "g", MiB: 3000}}
	if got := tr.Changes(cur); len(got) != 0 {
		t.Fatalf("first read counted: %v", got)
	}
	want := []Change{{PID: 9, Name: "g", MiB: 3000, Why: "new"}}
	if got := tr.Changes(cur); !reflect.DeepEqual(got, want) {
		t.Fatalf("second read: got %v want %v", got, want)
	}
	if got := tr.Changes(cur); !reflect.DeepEqual(got, want) {
		t.Fatalf("third read: got %v want %v", got, want)
	}
}

func TestTrackerTransientNeverCounts(t *testing.T) {
	tr := &Tracker{}
	tr.Reset(nil)
	probe := []Use{{PID: 5, Name: "nvidia-smi", MiB: 300}}
	if got := tr.Changes(probe); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
	if got := tr.Changes(nil); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
	// A new process reusing the pid must start its count over.
	if got := tr.Changes(probe); len(got) != 0 {
		t.Fatalf("reappearance counted without two reads in a row: %v", got)
	}
	if got := tr.Changes(probe); len(got) != 1 {
		t.Fatalf("want 1 change on second consecutive read, got %v", got)
	}
}

func TestTrackerOpenedOnly(t *testing.T) {
	tr := &Tracker{MinNew: 200}
	tr.Reset(nil)
	cur := []Use{{PID: 7, Name: "vk", MiB: 0, Opened: true}}
	if got := tr.Changes(cur); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
	want := []Change{{PID: 7, Name: "vk", MiB: 0, Why: "opened"}}
	if got := tr.Changes(cur); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestTrackerMinNew(t *testing.T) {
	tr := &Tracker{MinNew: 200}
	tr.Reset(nil)
	small := []Use{{PID: 7, Name: "x", MiB: 150}}
	for i := 0; i < 3; i++ {
		if got := tr.Changes(small); len(got) != 0 {
			t.Fatalf("below MinNew counted: %v", got)
		}
	}
	atMin := []Use{{PID: 8, Name: "y", MiB: 200}}
	tr.Changes(atMin)
	if got := tr.Changes(atMin); len(got) != 1 || got[0].Why != "new" {
		t.Fatalf("at MinNew: %v", got)
	}
	// Below MinNew but opened: counts as opened.
	op := []Use{{PID: 9, Name: "z", MiB: 50, Opened: true}}
	tr.Changes(op)
	if got := tr.Changes(op); len(got) != 1 || got[0].Why != "opened" {
		t.Fatalf("opened small: %v", got)
	}
}

func TestTrackerGrowth(t *testing.T) {
	mk := func(mib int) []Use { return []Use{{PID: 1, Name: "a", MiB: mib}} }
	t.Run("two reads over floor", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset(mk(1000))
		if got := tr.Changes(mk(1200)); len(got) != 0 {
			t.Fatalf("first: %v", got)
		}
		want := []Change{{PID: 1, Name: "a", MiB: 1200, Why: "grew"}}
		if got := tr.Changes(mk(1200)); !reflect.DeepEqual(got, want) {
			t.Fatalf("second: %v", got)
		}
	})
	t.Run("at floor never", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset(mk(1000))
		for i := 0; i < 5; i++ {
			if got := tr.Changes(mk(1100)); len(got) != 0 {
				t.Fatalf("read %d: %v", i, got)
			}
		}
	})
	t.Run("dip resets", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset(mk(1000))
		tr.Changes(mk(1500))
		if got := tr.Changes(mk(1000)); len(got) != 0 {
			t.Fatalf("dip: %v", got)
		}
		if got := tr.Changes(mk(1500)); len(got) != 0 {
			t.Fatalf("after dip first read counted: %v", got)
		}
		if got := tr.Changes(mk(1500)); len(got) != 1 {
			t.Fatalf("after dip second read: %v", got)
		}
	})
	t.Run("reset rebaselines", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset(mk(1000))
		tr.Changes(mk(1500))
		tr.Changes(mk(1500))
		tr.Reset(mk(1500))
		if got := tr.Changes(mk(1500)); len(got) != 0 {
			t.Fatalf("got %v", got)
		}
	})
}

func TestTrackerRelaunch(t *testing.T) {
	t.Run("adopted silently", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset([]Use{{PID: 1, Name: "browser", MiB: 800}})
		re := []Use{{PID: 2, Name: "browser", MiB: 850}}
		for i := 0; i < 4; i++ {
			if got := tr.Changes(re); len(got) != 0 {
				t.Fatalf("read %d: %v", i, got)
			}
		}
		// The adopted pid is now baseline: growth past floor counts like any baseline process.
		big := []Use{{PID: 2, Name: "browser", MiB: 1500}}
		tr.Changes(big)
		if got := tr.Changes(big); len(got) != 1 || got[0].Why != "grew" {
			t.Fatalf("adopted growth: %v", got)
		}
	})
	t.Run("above floor counts", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset([]Use{{PID: 1, Name: "browser", MiB: 800}})
		cur := []Use{{PID: 2, Name: "browser", MiB: 901}}
		tr.Changes(cur)
		got := tr.Changes(cur)
		want := []Change{{PID: 2, Name: "browser", MiB: 901, Why: "new"}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("different name counts", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset([]Use{{PID: 1, Name: "browser", MiB: 800}})
		cur := []Use{{PID: 2, Name: "other", MiB: 800}}
		tr.Changes(cur)
		if got := tr.Changes(cur); len(got) != 1 {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("old pid still live is not a relaunch", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset([]Use{{PID: 1, Name: "worker", MiB: 800}})
		cur := []Use{{PID: 1, Name: "worker", MiB: 800}, {PID: 2, Name: "worker", MiB: 800}}
		tr.Changes(cur)
		if got := tr.Changes(cur); len(got) != 1 || got[0].PID != 2 {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("empty name never relaunch", func(t *testing.T) {
		tr := &Tracker{Floor: 100}
		tr.Reset([]Use{{PID: 1, Name: "", MiB: 800}})
		cur := []Use{{PID: 2, Name: "", MiB: 800}}
		tr.Changes(cur)
		if got := tr.Changes(cur); len(got) != 1 {
			t.Fatalf("got %v", got)
		}
	})
}

func TestTrackerChangesSortedByPID(t *testing.T) {
	tr := &Tracker{}
	tr.Reset(nil)
	cur := []Use{{PID: 30, Name: "c", MiB: 5}, {PID: 10, Name: "a", MiB: 5}, {PID: 20, Name: "b", MiB: 5}}
	tr.Changes(cur)
	got := tr.Changes(cur)
	if len(got) != 3 || got[0].PID != 10 || got[1].PID != 20 || got[2].PID != 30 {
		t.Fatalf("got %v", got)
	}
}
