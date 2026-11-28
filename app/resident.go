package app

import (
	"context"
	"time"

	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/run"
)

// wakeForClient loads the model for a client that has just attached, when it was put to sleep
// (not when a person stopped it: that stays stopped). A request at the proxy wakes it too.
func wakeForClient(sup *run.Supervisor) {
	if st := sup.Status(); st.Phase == "sleeping" && st.Message != "Stopped" {
		go sup.Wake()
	}
}

// up is a phase in which a model is loaded or on its way up: the linger may put it to sleep.
func up(phase string) bool {
	switch phase {
	case "ready", "resizing", "starting", "loading":
		return true
	}
	return false
}

// idleRule is the idle rule's settings as the loop reads them each time it looks.
type idleRule struct {
	on     bool          // idle_sleep (and no --no-idle-sleep)
	quiet  time.Duration // idle_sleep_s: the quiet for callers that only send requests
	linger time.Duration // linger_s: the quiet after the last attachment left
}

// idleLoop is the idle rule, the same in the app and in serve (docs/ENGINE.md "Idle sleep",
// docs/EVENTS.md section 8). The model sleeps when no attachment stands (clients: the app's
// attached parents and the clients' attachments, held or by pid; serve's --parent-pid parents
// are not clients, they keep the engine's process, not its model), no request is in flight or
// held, a model is up, and the quiet has lasted its delay. The delay is a named moment armed
// when the quiet begins and cancelled by any request begun or any attach: linger_s when the
// quiet began with the last attachment leaving (no request begun since), else idle_sleep_s. A
// change of the settings arms the same moment again for the new delay, counted from the start
// of the quiet. When it comes with still nobody there, the model is put to sleep (SleepIdle);
// the engine stays.
//
// It waits on words only: an attach, a parent's end (the Watcher, nil in serve), the hub's
// request, door, phase, attachment and idle facts, the moment, ctx.
func idleLoop(ctx context.Context, w *lifecycle.Watcher, sup *run.Supervisor, attached <-chan struct{},
	clients func() int, rule func() idleRule, after func()) {
	hub := sup.Hub()
	wk, err := hub.Wake("request", "door", "phase", "attachment", "idle")
	if err != nil {
		return
	}
	defer wk.Close()
	// parents says when every attached parent has gone; again starts its next watch after an attach.
	parents, again := make(chan struct{}, 1), make(chan struct{}, 1)
	if w != nil {
		go func() {
			for {
				if !w.Run(ctx) {
					return
				}
				poke(parents)
				select {
				case <-again:
				case <-ctx.Done():
					return
				}
			}
		}()
	}
	var (
		fired      chan struct{} // the moment, while armed
		cancel     func() bool
		armedAt    uint64 // the door's count of requests begun when armed: one begun since cancels it
		armedWhy   string
		armedFor   time.Duration
		quietAt    time.Time // when the quiet began (the moment counts from here)
		lastBegun  = sup.Begun()
		hadClients bool // an attachment stood when last looked
		leftAttach bool // the last attachment left and no request has begun since
	)
	disarm := func() {
		if cancel != nil {
			cancel()
		}
		cancel, fired = nil, nil
		sup.SetIdleMoment(false, "", 0)
	}
	arm := func(why string, delay, left time.Duration) {
		f := make(chan struct{})
		fired, armedAt, armedWhy, armedFor = f, sup.Begun(), why, delay
		sup.SetIdleMoment(true, why, delay)             // written before the moment can come
		cancel = host.Moment(left, func() { close(f) }) // named moment: the idle rule's quiet
	}
	defer func() {
		if cancel != nil {
			cancel()
		}
	}()
	for {
		if b := sup.Begun(); b != lastBegun {
			lastBegun, leftAttach = b, false
			if fired != nil {
				disarm() // a request came: the quiet starts again at the door's next empty
			}
		}
		n := clients()
		switch {
		case n > 0:
			hadClients = true
		case hadClients:
			hadClients, leftAttach = false, true
		}
		r := rule()
		idle := r.on && n == 0 && sup.InFlight()+sup.Held() == 0 && up(sup.Phase())
		why, delay := "idle_sleep", r.quiet
		if leftAttach {
			why, delay = "linger", r.linger
		}
		switch {
		case !idle:
			if fired != nil {
				disarm()
			}
		case fired == nil:
			quietAt = time.Now()
			arm(why, delay, delay)
		case why != armedWhy || delay != armedFor:
			// a setting changed while armed: the same moment, armed again for what is left
			cancel()
			left := delay - time.Since(quietAt)
			if left < 0 {
				left = 0
			}
			arm(why, delay, left)
		}
		select {
		case <-attached:
			poke(again)
			if fired != nil {
				disarm() // an attach cancels the moment
			}
		case <-parents:
		case <-wk.C():
		case <-fired:
			fired, cancel = nil, nil
			sup.SetIdleMoment(false, "", 0)
			if sup.Begun() != armedAt || clients() > 0 || sup.InFlight()+sup.Held() > 0 {
				continue // somebody came: looked at again at the top
			}
			if up(sup.Phase()) {
				sup.SleepIdle()
				after()
			}
		case <-ctx.Done():
			return
		}
	}
}

func poke(c chan struct{}) {
	select {
	case c <- struct{}{}:
	default:
	}
}
