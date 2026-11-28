//go:build !windows

package app_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/run"
	"github.com/CharlesMod/easyEngine/tray"
)

func isSleeping(e event.Event) bool { return phaseIs(e, "sleeping") }
func isEnd(e event.Event) bool      { return requestIs(e, "end") }

// idleAfter sets the idle rule's quiet for callers that only send requests (--idle-sleep).
func idleAfter(d time.Duration) func(*app.Options) {
	return func(o *app.Options) { o.IdleSleep, o.IdleSleepSet = d, true }
}

// A remote client with no parent: its request wakes the sleeping model and is held until it is
// up; once its requests stop, the idle rule's quiet (idle_sleep_s, not the linger: it never
// attached) puts the model to sleep and the engine stays.
func TestRemoteRequestWakesThenIdleSleeps(t *testing.T) {
	const quiet = 600 * time.Millisecond
	r := startFakeModelWith(t, 0, nil, nil, idleAfter(quiet)) // linger 0: were it used, the sleep would come at once
	s := r.follow(t)
	if code := r.post(t); code != 200 {
		t.Fatalf("the request was answered %d", code)
	}
	evs := s.until(t, "the model asleep after the request ended", func(evs []event.Event) bool {
		end := indexOf(evs, 0, isEnd)
		return end >= 0 && indexOf(evs, end, isSleeping) >= 0
	})
	end := indexOf(evs, 0, isEnd)
	if indexOf(evs, 0, func(e event.Event) bool { return phaseIs(e, "ready") }) < 0 {
		t.Fatal("the request was answered without the model ever being ready")
	}
	slept := evs[indexOf(evs, end, isSleeping)]
	var ph struct {
		Message     string
		SleepReason string `json:"sleep_reason"`
	}
	body(slept, &ph)
	if ph.Message != run.IdleMessage || ph.SleepReason != "idle" {
		t.Fatalf("sleeping %+v", ph)
	}
	if d := slept.At.Sub(evs[end].At); d < quiet {
		t.Fatalf("the model slept %v after the request ended, idle_sleep is %v", d, quiet)
	}
	if _, err := os.Stat(filepath.Join(r.dir, "engine.lock")); err != nil {
		t.Fatal("no lock:", err)
	}
	if st := r.status(t); st.Phase != "sleeping" { // the engine answers: it stayed
		t.Fatalf("phase %q", st.Phase)
	}
	select {
	case <-r.res:
		t.Fatal("the engine ended")
	default:
	}
}

// A request during the idle rule's moment starts it over: the model sleeps no sooner than the
// quiet after the second request's end.
func TestRequestDuringIdleMomentCancelsIt(t *testing.T) {
	const linger = 1500 * time.Millisecond // the quiet, here idle_sleep
	r := startFakeModelWith(t, 0, nil, map[string]string{"FAKELLAMA_CHAT_DELAY": "300ms"}, idleAfter(linger))
	s := r.follow(t)
	if code := r.post(t); code != 200 {
		t.Fatalf("first request: %d", code)
	}
	s.until(t, "the first request ended", func(evs []event.Event) bool {
		return indexOf(evs, 0, isEnd) >= 0
	})
	// the model is up and the linger runs; the second request comes inside it
	if code := r.post(t); code != 200 {
		t.Fatalf("second request: %d", code)
	}
	evs := s.until(t, "the model asleep after the second request", func(evs []event.Event) bool {
		first := indexOf(evs, 0, isEnd)
		second := indexOf(evs, first+1, isEnd)
		return second >= 0 && indexOf(evs, second, isSleeping) >= 0
	})
	first := indexOf(evs, 0, isEnd)
	second := indexOf(evs, first+1, isEnd)
	if i := indexOf(evs, first, isSleeping); i < second {
		t.Fatalf("the model slept (event %d) before the second request ended (event %d)", i, second)
	}
	slept := evs[indexOf(evs, second, isSleeping)]
	if d := slept.At.Sub(evs[second].At); d < linger {
		t.Fatalf("the model slept %v after the second request ended, linger is %v", d, linger)
	}
}

// A request held while the model loads keeps it up: with a quiet of 0 the model is still not
// put to sleep before the held request is answered.
func TestHeldRequestKeepsTheModelUp(t *testing.T) {
	r := startFakeModelWith(t, 0, nil, map[string]string{"FAKELLAMA_HEALTH_DELAY": "1s"}, idleAfter(0))

	s := r.follow(t)
	if code := r.post(t); code != 200 {
		t.Fatalf("the held request was answered %d", code)
	}
	evs := s.until(t, "the model asleep after the request ended", func(evs []event.Event) bool {
		end := indexOf(evs, 0, isEnd)
		return end >= 0 && indexOf(evs, end, isSleeping) >= 0
	})
	// The stream was opened before the request, so everything from event 0 is the held request's
	// time (its begin is said when it passes the door, after the hold).
	end := indexOf(evs, 0, isEnd)
	if i := indexOf(evs, 0, isSleeping); i >= 0 && i < end {
		t.Fatalf("the model slept (event %d) while the request was held (end %d)", i, end)
	}
	if indexOf(evs, 0, func(e event.Event) bool {
		var d struct{ Held int }
		body(e, &d)
		return e.Name == "door" && d.Held > 0
	}) < 0 {
		t.Fatal("the door never held the request")
	}
	if ready := indexOf(evs, 0, func(e event.Event) bool { return phaseIs(e, "ready") }); ready < 0 || ready > end {
		t.Fatalf("the request ended (%d) before the model was ready (%d)", end, ready)
	}
}

// With no click at all the tooltip follows the hub: it says no model is loaded, and then that
// one is ready once a request woke it.
func TestTrayTooltipFollowsTheHub(t *testing.T) {
	ft := newFakeTray()
	r := startFakeModelWith(t, time.Hour, nil, nil, func(o *app.Options) { o.Tray = ft })
	ft.until(t, "no model loaded", func(_ []tray.Item, tip string) bool { return tip == "easy-engine: no model loaded" })
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	ft.until(t, "ready", func(_ []tray.Item, tip string) bool { return strings.HasPrefix(tip, "easy-engine: ready, ") })
}

// After an idle sleep the tooltip says the card is free.
func TestTrayTooltipSaysIdle(t *testing.T) {
	ft := newFakeTray()
	r := startFakeModelWith(t, time.Hour, nil, nil, func(o *app.Options) { o.Tray = ft; idleAfter(0)(o) })
	if code := r.post(t); code != 200 {
		t.Fatalf("request: %d", code)
	}
	ft.until(t, "idle", func(_ []tray.Item, tip string) bool {
		return tip == "easy-engine: no model loaded (idle, the card is free)"
	})
}
