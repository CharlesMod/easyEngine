package app

import (
	"context"
	"encoding/json"
	"time"

	"github.com/CharlesMod/easyEngine/event"
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

// residentLoop is the app's idea of "the clients are gone" (docs/EVENTS.md section 8). The
// clients are the attached parents (by pid, their exit is the word) and the requests at the door
// (a remote client that never attached a pid: the door's count of requests in flight and held).
// Once a client has come since the model last went to sleep, and then no parent is attached and
// the door is empty while a model is up, the keeper's linger (a named moment) is armed; a request
// or an attach cancels it, and it is armed again when the door is empty again. When it comes
// with still nobody there, the model is put to sleep; the engine stays.
//
// It waits on words only: an attach, a parent's end (the Watcher), the hub's request, door and
// phase facts, the moment, ctx.
func residentLoop(ctx context.Context, w *lifecycle.Watcher, sup *run.Supervisor, attached <-chan struct{},
	linger func() time.Duration, after func()) {
	hub := sup.Hub()
	wk, err := hub.Wake("request", "door", "phase")
	if err != nil {
		return
	}
	defer wk.Close()
	// parents says when every attached parent has gone; again starts its next watch after an attach.
	parents, again := make(chan struct{}, 1), make(chan struct{}, 1)
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
	var (
		connected bool          // a client came since the model last went to sleep
		fired     chan struct{} // the linger's moment, while armed
		cancel    func() bool
		armedAt   uint64 // the hub's seq when the linger was armed: a request begun since cancels it
	)
	disarm := func() {
		if cancel != nil {
			cancel()
		}
		cancel, fired = nil, nil
	}
	defer disarm()
	for {
		door := sup.InFlight() + sup.Held()
		pids := len(w.PIDs())
		if door > 0 || pids > 0 {
			connected = true
		}
		if fired != nil && begunSince(hub.Since(armedAt, "request")) {
			disarm() // a request came during the linger: it runs again from the door's next empty
		}
		idle := connected && pids == 0 && door == 0 && up(sup.Phase())
		switch {
		case idle && fired == nil:
			f := make(chan struct{})
			fired, armedAt = f, hub.Seq()
			cancel = host.Moment(linger(), func() { close(f) }) // named moment: the keeper's linger
		case !idle && fired != nil:
			disarm()
		}
		select {
		case <-attached:
			connected = true
			poke(again)
			disarm() // an attach cancels the linger
		case <-parents:
		case <-wk.C():
		case <-fired:
			fired, cancel = nil, nil
			if begunSince(hub.Since(armedAt, "request")) || len(w.PIDs()) > 0 || sup.InFlight()+sup.Held() > 0 {
				continue // somebody came: looked at again at the top
			}
			if up(sup.Phase()) {
				sup.Sleep()
				after()
			}
			connected = false
		case <-ctx.Done():
			return
		}
	}
}

// begunSince is true when a request began among evs, or the history let go of them (gap).
func begunSince(evs []event.Event, gap bool) bool {
	if gap {
		return true
	}
	for _, e := range evs {
		var b struct {
			What string `json:"what"`
		}
		json.Unmarshal(e.Body, &b)
		if b.What == "begin" {
			return true
		}
	}
	return false
}

func poke(c chan struct{}) {
	select {
	case c <- struct{}{}:
	default:
	}
}
