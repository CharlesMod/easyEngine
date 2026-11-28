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

// loaded is a phase in which a model is loaded: the idle rule's quiet counts from here (a load
// in progress is not quiet yet; once it is ready, the quiet begins).
func loaded(phase string) bool { return phase == "ready" || phase == "resizing" }

// traffic is what the idle rule compares between looks: the requests begun (Begun) and the
// requests that came to the door, held or not, answered or refused, and the attachments that
// left, each with how many were remote (run.Supervisor.IdleSeen).
type traffic struct {
	begun, arrivals, remoteArrivals, leaves, remoteLeaves uint64
}

func trafficOf(sup *run.Supervisor) traffic {
	var t traffic
	t.begun = sup.Begun()
	t.arrivals, t.remoteArrivals, t.leaves, t.remoteLeaves = sup.IdleSeen()
	return t
}

// requests is one number that moves with any request at the door.
func (t traffic) requests() uint64 { return t.begun + t.arrivals }

// idleDelay is the moment's delay for what the quiet follows, and its why (the delay's name and
// the cause: linger/local_leave, idle_sleep/remote_leave, ...). A local leave takes linger_s; a
// remote leave or remote requests take idle_sleep_s but never less than run.RemoteIdleFloor (no
// remote identity can load and unload the model faster: docs/ENGINE.md section 6c); local
// requests and the engine's own load take idle_sleep_s exactly.
func idleDelay(cause string, r idleRule) (string, time.Duration) {
	switch cause {
	case "local_leave":
		return "linger/" + cause, r.linger
	case "remote_leave", "remote_requests":
		if r.quiet < run.RemoteIdleFloor {
			return "remote_floor/" + cause, run.RemoteIdleFloor
		}
	}
	return "idle_sleep/" + cause, r.quiet
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
		armedAt    traffic // the door's traffic when armed: any since cancels it
		armedWhy   string
		armedFor   time.Duration
		quietAt    time.Time // when the quiet began (the moment counts from here)
		seen       = trafficOf(sup)
		hadClients bool // an attachment stood when last looked
		// cause is what the quiet follows: load (nothing since the model was loaded, or only the
		// engine's own), local_requests | remote_requests (the last requests at the door; remote
		// when any of them came from another machine), local_leave | remote_leave (the last
		// attachment left and no request came since; remote when a remote attachment left)
		cause = "load"
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
		fired, armedAt, armedWhy, armedFor = f, trafficOf(sup), why, delay
		sup.SetIdleMoment(true, why, delay)             // written before the moment can come
		cancel = host.Moment(left, func() { close(f) }) // named moment: the idle rule's quiet
	}
	defer func() {
		if cancel != nil {
			cancel()
		}
	}()
	for {
		now := trafficOf(sup)
		if now.requests() != seen.requests() {
			cause = "local_requests"
			if now.remoteArrivals != seen.remoteArrivals {
				cause = "remote_requests"
			}
		}
		n := clients()
		switch {
		case n > 0:
			hadClients = true
		case hadClients:
			hadClients, cause = false, "local_leave" // a parent's exit, a local attachment's leave
			if now.remoteLeaves != seen.remoteLeaves {
				cause = "remote_leave"
			}
		}
		seen = now
		r := rule()
		idle := r.on && n == 0 && sup.InFlight()+sup.Held() == 0 && loaded(sup.Phase())
		why, delay := idleDelay(cause, r)
		switch {
		case !idle:
			if fired != nil {
				disarm()
			}
		case fired == nil:
			quietAt = time.Now()
			arm(why, delay, delay)
		case why != armedWhy || delay != armedFor:
			// a setting changed while armed: the same moment, armed again for what is left (from
			// now when a request the loop did not see in flight came since it was armed)
			cancel()
			if trafficOf(sup).requests() != armedAt.requests() {
				quietAt = time.Now()
			}
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
			if trafficOf(sup).requests() != armedAt.requests() || clients() > 0 || sup.InFlight()+sup.Held() > 0 {
				// somebody came since it was armed (a request the loop never saw in flight, or
				// one it did): the quiet starts again from here, looked at again at the top
				continue
			}
			if up(sup.Phase()) {
				sup.SleepIdle()
				cause = "load" // what wakes it next says its own cause
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
