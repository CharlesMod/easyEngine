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

// residentLoop is the app's idea of "the clients are gone": it waits for a connection (an
// attached parent), watches until the last parent is gone, runs the keeper's linger, and then,
// if no parent came back and nothing is in flight, puts the model to sleep. The engine stays.
// A client that never attached a pid (a remote client) is not counted yet.
func residentLoop(ctx context.Context, w *lifecycle.Watcher, sup *run.Supervisor, attached <-chan struct{},
	linger func() time.Duration, after func()) {
	for {
		select {
		case <-attached:
		case <-ctx.Done():
			return
		}
		for {
			if !w.Run(ctx) { // every parent that was attached has gone
				return
			}
			drain(attached)
			if len(w.PIDs()) > 0 {
				continue
			}
			fired := make(chan struct{})
			cancel := host.Moment(linger(), func() { close(fired) }) // named moment
			select {
			case <-fired:
			case <-attached: // a client came back during the linger
				cancel()
				continue
			case <-ctx.Done():
				cancel()
				return
			}
			if len(w.PIDs()) > 0 {
				continue
			}
			if sup.InFlight() > 0 {
				// A reply is still being written: the linger counts from its end, so it runs again.
				drain(attached)
				continue
			}
			sup.Sleep()
			after()
			break
		}
	}
}

func drain(c <-chan struct{}) {
	for {
		select {
		case <-c:
		default:
			return
		}
	}
}
