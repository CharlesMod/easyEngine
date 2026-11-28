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
	if sup.Phase() == "sleeping" && !sup.StoppedByPerson() { // by the stop itself, never its words
		go sup.Wake()
	}
}

// traffic is what the idle rule compares between looks: the requests begun (Begun) and what
// the supervisor counts (run.IdleTraffic: arrivals at the door, held or not, answered or
// refused; attachments' leaves; each with how many were remote; wakes; the last traffic's time).
type traffic struct {
	begun uint64
	run.IdleTraffic
}

func trafficOf(sup *run.Supervisor) traffic {
	return traffic{begun: sup.Begun(), IdleTraffic: sup.IdleSeen()}
}

// moved says any request or leave since o.
func (t traffic) moved(o traffic) bool {
	return t.begun != o.begun || t.Arrivals != o.Arrivals || t.Leaves != o.Leaves
}

// stretch is what the busy stretch before a quiet held: kind is the last thing seen (load:
// nothing since the model was loaded; requests; leave: the last attachment left; wake: a wake
// somebody asked for), remote is sticky (any remote request or leave in the stretch, whatever
// came after it: remote_requests or remote_leave, whichever remote act was seen last), woke is
// sticky (a wake was asked in the stretch).
type stretch struct {
	kind, remote string
	woke         bool
}

// idleDelay is the moment's delay for the stretch the quiet follows, and its why
// (<delay>/<cause>). Remote traffic anywhere in the stretch: idle_sleep_s, never less than the
// floor (run.RemoteIdleFloor): no remote identity loads and unloads the model faster (docs/
// ENGINE.md section 6c). A wake asked for (POST wake, an attach that woke the model) and then
// nothing but a local leave or nothing at all: the delay, never less than the floor. Else the
// exact settings: linger_s after a local leave, idle_sleep_s after local requests or a load.
func idleDelay(st stretch, r idleRule, floor time.Duration) (string, time.Duration) {
	base, name := r.quiet, "idle_sleep"
	if st.kind == "leave" {
		base, name = r.linger, "linger"
	}
	floored := func(cause string) (string, time.Duration) {
		if base < floor {
			return "floor/" + cause, floor
		}
		return name + "/" + cause, base
	}
	switch {
	case st.remote != "":
		base, name = r.quiet, "idle_sleep" // remote traffic never takes the linger
		return floored(st.remote)
	case st.woke && st.kind != "requests":
		return floored("wake")
	case st.kind == "leave":
		return name + "/local_leave", base
	case st.kind == "requests":
		return name + "/local_requests", base
	}
	return name + "/load", base
}

// idleRule is the idle rule's settings as the loop reads them each time it looks.
type idleRule struct {
	on     bool          // idle_sleep, under --no-idle-sleep
	quiet  time.Duration // idle_sleep_s (or --idle-sleep): the quiet after requests or a load
	linger time.Duration // linger_s: the quiet after the last local attachment left
}

// idleLoop is the idle rule, the same in the app and in serve (docs/ENGINE.md section 6c,
// docs/EVENTS.md section 8). The model sleeps when no attachment stands (clients: the app's
// attached parents and the clients' attachments, held or by pid; serve's --parent-pid parents
// are not clients, they keep the engine's process, not its model), no request is in flight or
// held, the engine is at rest with a model loaded (run.Supervisor.IdleSleepable: phase ready,
// no swap, job, reshape, drain or yield), and the quiet has lasted its delay (idleDelay). The
// delay is a named moment armed when the quiet begins; anything that ends the rest (a request,
// an attach, a phase away from ready) disarms it, and the next quiet begins anew. A request the
// loop never saw in flight is found when the moment comes: the quiet then counts from that
// request's time. A change of the settings arms the same moment again for what is left of the
// new delay. The sleep itself is SleepIdle, one decision with the door; a request that wins the
// race makes it decline, and the quiet starts again.
//
// It waits on words only: an attach, a parent's end (the Watcher, nil in serve), the hub's
// request, door, phase, attachment, idle, job, upgrade, yield and nudge words, the moment, ctx.
func idleLoop(ctx context.Context, w *lifecycle.Watcher, sup *run.Supervisor, attached <-chan struct{},
	clients func() int, rule func() idleRule, after func()) {
	hub := sup.Hub()
	wk, err := hub.Wake("request", "door", "phase", "attachment", "idle", "job", "upgrade", "yield", "nudge")
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
		armedAt    traffic // the traffic when armed: any request or leave since is found at the fire
		armedWhy   string
		armedFor   time.Duration
		armedSt    stretch   // the stretch the armed quiet follows
		armedRest  uint64    // the phase's changes when the quiet began (run.RestSeq)
		quietAt    time.Time // when the quiet began (the moment counts from here)
		requietAt  time.Time // set at a fire that found traffic: the next quiet counts from it
		seen       = trafficOf(sup)
		hadClients bool // an attachment stood when last looked
		st         = stretch{kind: "load"}
	)
	disarm := func() {
		if cancel != nil {
			cancel()
		}
		cancel, fired = nil, nil
		sup.SetIdleMoment(false, "", 0, 0)
	}
	arm := func(why string, delay, left time.Duration) {
		if left < 0 {
			left = 0
		}
		f := make(chan struct{})
		fired, armedAt, armedWhy, armedFor = f, trafficOf(sup), why, delay
		sup.SetIdleMoment(true, why, delay, left)       // written before the moment can come
		cancel = host.Moment(left, func() { close(f) }) // named moment: the idle rule's quiet
	}
	defer func() {
		if cancel != nil {
			cancel()
		}
	}()
	for {
		now := trafficOf(sup)
		// remote traffic is sticky for the stretch, whatever the loop saw or missed in between
		if now.RemoteArrivals != seen.RemoteArrivals {
			st.remote = "remote_requests"
		}
		if now.RemoteLeaves != seen.RemoteLeaves {
			st.remote = "remote_leave"
		}
		if now.Wakes != seen.Wakes {
			st.woke = true
			if now.Arrivals == seen.Arrivals {
				st.kind = "wake"
			}
		}
		if now.Arrivals != seen.Arrivals {
			st.kind = "requests"
		}
		n := clients()
		switch {
		case n > 0:
			hadClients = true
		case hadClients || now.Leaves != seen.Leaves:
			hadClients, st.kind = false, "leave" // the last attachment left (seen or not)
		}
		seen = now
		r := rule()
		idle := r.on && n == 0 && sup.InFlight()+sup.Held() == 0 && sup.IdleSleepable()
		switch {
		case !idle:
			if fired != nil {
				disarm()
			}
		case fired == nil:
			// a new quiet: the stretch it follows is what came since the last one began
			armedSt, st.remote, st.woke = st, "", false
			armedRest = sup.RestSeq()
			why, delay := idleDelay(armedSt, r, sup.IdleFloor())
			quietAt = time.Now()
			if !requietAt.IsZero() {
				quietAt, requietAt = requietAt, time.Time{}
			}
			arm(why, delay, delay-time.Since(quietAt))
		case !trafficOf(sup).moved(armedAt): // traffic since it was armed is the fire's to find
			if why, delay := idleDelay(armedSt, r, sup.IdleFloor()); why != armedWhy || delay != armedFor {
				// a setting changed while armed: the same moment, armed again for what is left
				cancel()
				arm(why, delay, delay-time.Since(quietAt))
			}
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
			sup.SetIdleMoment(false, "", 0, 0)
			if t := trafficOf(sup); t.moved(armedAt) {
				// traffic the loop never saw in flight came while armed: the next quiet counts
				// from its time, looked at again at the top
				requietAt = t.Last
				if requietAt.IsZero() || requietAt.After(time.Now()) {
					requietAt = time.Now()
				}
				continue
			}
			if clients() > 0 || sup.InFlight()+sup.Held() > 0 {
				continue
			}
			if sup.SleepIdleSince(armedRest) == nil {
				st = stretch{kind: "load"} // what wakes it next says its own stretch
				after()
			}
			// a request won the race with the decision: the quiet starts again at the top
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
