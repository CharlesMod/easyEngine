package yield

import (
	"strings"
	"testing"
	"time"
)

var t0 = time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)

const (
	grace  = 60 * time.Second
	steady = 30 * time.Second
)

func newTestMachine(strict bool) *Machine {
	return NewMachine(Config{Grace: grace, Steady: steady, Floor: 100, Strict: strict})
}

var game = Launch{Kind: "steam", AppID: "1", Exe: "SomeGame.exe"}

func chg(pid, mib int) Change { return Change{PID: pid, Name: "p", MiB: mib, Why: "new"} }

func TestMachineIdle(t *testing.T) {
	m := newTestMachine(false)
	for i := 0; i < 5; i++ {
		if d := m.Step(Input{Now: t0.Add(time.Duration(i) * time.Hour)}); d.Action != None {
			t.Fatalf("idle: %+v", d)
		}
	}
	if m.Yielded() {
		t.Fatal("yielded while idle")
	}
}

func TestMachineYieldGame(t *testing.T) {
	m := newTestMachine(false)
	d := m.Step(Input{Now: t0, Games: []Launch{game}, Changes: []Change{chg(1, 10)}})
	if d.Action != Yield || d.Kind != "game" || d.Reason != "AppId 1 (SomeGame.exe)" {
		t.Fatalf("got %+v", d)
	}
	if !m.Yielded() {
		t.Fatal("not yielded")
	}
}

func TestMachineYieldGPU(t *testing.T) {
	tests := []struct {
		c    Change
		want string
	}{
		{Change{PID: 4, Name: "foo", MiB: 2000, Why: "new"}, "foo (2000 MiB)"},
		{Change{PID: 4, Name: "foo", Why: "opened"}, "foo (opened the GPU)"},
		{Change{PID: 4, Name: "foo", MiB: 3000, Why: "grew"}, "foo (grew to 3000 MiB)"},
		{Change{PID: 4, MiB: 5, Why: "new"}, "pid 4 (5 MiB)"},
	}
	for _, tc := range tests {
		m := newTestMachine(false)
		d := m.Step(Input{Now: t0, Changes: []Change{tc.c}})
		if d.Action != Yield || d.Kind != "gpu" || d.Reason != tc.want {
			t.Errorf("%+v: got %+v, want reason %q", tc.c, d, tc.want)
		}
	}
}

func TestMachineGameGrace(t *testing.T) {
	m := newTestMachine(false)
	m.Step(Input{Now: t0, Games: []Launch{game}})
	// game still running: stays.
	for _, off := range []time.Duration{10 * time.Second, 5 * time.Minute} {
		if d := m.Step(Input{Now: t0.Add(off), Games: []Launch{game}}); d.Action != None {
			t.Fatalf("with game at +%v: %+v", off, d)
		}
	}
	last := t0.Add(5 * time.Minute)
	if d := m.Step(Input{Now: last.Add(grace - time.Second)}); d.Action != None {
		t.Fatalf("inside grace: %+v", d)
	}
	// Game returns during grace: grace restarts from this sighting.
	back := last.Add(grace - time.Second)
	m.Step(Input{Now: back, Games: []Launch{game}})
	gone := back.Add(grace - time.Second) // the step that finds the game gone is its last sighting
	if d := m.Step(Input{Now: gone}); d.Action != None {
		t.Fatalf("game restarts during grace: %+v", d)
	}
	if left := m.Due(gone.Add(time.Second)); left != grace-time.Second {
		t.Fatalf("grace left %v", left)
	}
	if d := m.Step(Input{Now: gone.Add(grace - time.Second)}); d.Action != None {
		t.Fatalf("inside grace: %+v", d)
	}
	d := m.Step(Input{Now: gone.Add(grace)})
	if d.Action != Resume || d.Beside {
		t.Fatalf("after grace: %+v", d)
	}
}

func TestMachineGraceBlocksChangesToo(t *testing.T) {
	m := newTestMachine(false)
	m.Step(Input{Now: t0, Games: []Launch{game}})
	d := m.Step(Input{Now: t0.Add(grace + time.Hour), Changes: []Change{chg(1, 10)}})
	// Past grace, with a newcomer: steady wait begins, not an immediate resume.
	if d.Action != None {
		t.Fatalf("got %+v", d)
	}
}

func TestMachineResumeWhenChangesGone(t *testing.T) {
	m := newTestMachine(false)
	m.Step(Input{Now: t0, Changes: []Change{chg(1, 500)}})
	d := m.Step(Input{Now: t0.Add(time.Second)})
	if d.Action != Resume || d.Beside {
		t.Fatalf("got %+v", d)
	}
}

func TestMachineSteadyBeside(t *testing.T) {
	m := newTestMachine(false)
	m.Step(Input{Now: t0, Changes: []Change{chg(1, 500)}})
	ch := []Change{chg(1, 500)}
	if d := m.Step(Input{Now: t0.Add(time.Second), Changes: ch}); d.Action != None {
		t.Fatalf("got %+v", d)
	}
	// small wobble within floor stays steady
	ch2 := []Change{chg(1, 590)}
	if d := m.Step(Input{Now: t0.Add(steady), Changes: ch2}); d.Action != None {
		t.Fatalf("not yet steady (since +1s): %+v", d)
	}
	d := m.Step(Input{Now: t0.Add(time.Second + steady), Changes: ch2})
	if d.Action != Resume || !d.Beside {
		t.Fatalf("got %+v", d)
	}
}

func TestMachineSteadyRestarts(t *testing.T) {
	tests := []struct {
		name  string
		first []Change
		later []Change
	}{
		{"change over floor", []Change{chg(1, 500)}, []Change{chg(1, 700)}},
		{"dropped over floor", []Change{chg(1, 500)}, []Change{chg(1, 300)}},
		{"pid set grows", []Change{chg(1, 500)}, []Change{chg(1, 500), chg(2, 500)}},
		{"pid replaced", []Change{chg(1, 500)}, []Change{chg(2, 500)}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := newTestMachine(false)
			m.Step(Input{Now: t0, Changes: tc.first})
			m.Step(Input{Now: t0.Add(time.Second), Changes: tc.first})
			mid := t0.Add(steady - time.Second)
			if d := m.Step(Input{Now: mid, Changes: tc.later}); d.Action != None {
				t.Fatalf("at change: %+v", d)
			}
			// Original window would have elapsed, but the wait restarted at mid.
			if d := m.Step(Input{Now: t0.Add(time.Second + steady), Changes: tc.later}); d.Action != None {
				t.Fatalf("resumed on the old clock: %+v", d)
			}
			d := m.Step(Input{Now: mid.Add(steady), Changes: tc.later})
			if d.Action != Resume || !d.Beside {
				t.Fatalf("after restarted window: %+v", d)
			}
		})
	}
}

func TestMachineStrict(t *testing.T) {
	m := newTestMachine(true)
	m.Step(Input{Now: t0, Changes: []Change{chg(1, 500)}})
	for i := 1; i <= 10; i++ {
		d := m.Step(Input{Now: t0.Add(time.Duration(i) * time.Minute), Changes: []Change{chg(1, 500)}})
		if d.Action != None {
			t.Fatalf("strict resumed beside: %+v", d)
		}
	}
	d := m.Step(Input{Now: t0.Add(time.Hour)})
	if d.Action != Resume || d.Beside {
		t.Fatalf("got %+v", d)
	}
}

func TestMachineHold(t *testing.T) {
	m := newTestMachine(false)
	ch := []Change{chg(1, 500)}
	m.Step(Input{Now: t0, Changes: ch})
	m.Step(Input{Now: t0.Add(time.Second), Changes: ch})
	if d := m.Step(Input{Now: t0.Add(time.Second + steady), Changes: ch}); d.Action != Resume || !d.Beside {
		t.Fatalf("precondition: %+v", d)
	}
	hold := t0.Add(time.Minute)
	m.Hold(hold)
	if d := m.Step(Input{Now: hold.Add(steady - time.Second), Changes: ch}); d.Action != None {
		t.Fatalf("after Hold: %+v", d)
	}
	if d := m.Step(Input{Now: hold.Add(steady), Changes: ch}); d.Action != Resume || !d.Beside {
		t.Fatalf("after Hold+Steady: %+v", d)
	}
}

func TestMachineReset(t *testing.T) {
	m := newTestMachine(false)
	m.Step(Input{Now: t0, Games: []Launch{game}})
	m.Reset()
	if m.Yielded() {
		t.Fatal("still yielded")
	}
	// lastGame was cleared: a fresh yield for a gpu change must resume without grace delay.
	m.Step(Input{Now: t0.Add(time.Second), Changes: []Change{chg(1, 1)}})
	if d := m.Step(Input{Now: t0.Add(2 * time.Second)}); d.Action != Resume {
		t.Fatalf("grace survived Reset: %+v", d)
	}
	// Config survives Reset.
	m.Reset()
	m.Step(Input{Now: t0, Games: []Launch{game}})
	if d := m.Step(Input{Now: t0.Add(grace - time.Second)}); d.Action != None {
		t.Fatalf("config lost: %+v", d)
	}
}

func TestDescribeNoGameWords(t *testing.T) {
	if s := describe(Change{PID: 1}); !strings.HasPrefix(s, "pid 1") {
		t.Fatal(s)
	}
}

func TestMachineDue(t *testing.T) {
	m := newTestMachine(false)
	if m.Due(t0) != 0 {
		t.Fatal("due while active")
	}
	m.Step(Input{Now: t0, Games: []Launch{game}})
	if m.Due(t0) != 0 {
		t.Fatal("a game running waits on no time")
	}
	m.Step(Input{Now: t0.Add(time.Second)}) // the game is gone
	if got := m.Due(t0.Add(2 * time.Second)); got != grace-time.Second {
		t.Fatalf("grace left %v", got)
	}
	// past the grace with a newcomer beside: the steady wait runs from its first sighting
	at := t0.Add(time.Second + grace)
	m.Step(Input{Now: at, Changes: []Change{chg(1, 500)}})
	if got := m.Due(at.Add(5 * time.Second)); got != steady-5*time.Second {
		t.Fatalf("steady left %v", got)
	}
	if got := m.Due(at.Add(steady)); got != 0 {
		t.Fatalf("steady over: %v", got)
	}
	// strict never resumes beside: no time to wait for
	st := newTestMachine(true)
	st.Step(Input{Now: t0, Changes: []Change{chg(1, 500)}})
	st.Step(Input{Now: t0.Add(time.Second), Changes: []Change{chg(1, 500)}})
	if st.Due(t0.Add(2*time.Second)) != 0 {
		t.Fatal("strict waits on no time")
	}
	// no newcomers left: resume is due at once
	m.Step(Input{Now: at.Add(time.Second)})
	if m.Due(at.Add(time.Second)) != 0 {
		t.Fatal("nothing beside: due now")
	}
}
