//go:build !windows

package app_test

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// gated makes the fake's streamed replies wait for <ctl>/go.N before each next chunk.
func gated(o *app.Options) {
	prev := o.Configure
	o.Configure = func(c *run.Config) {
		prev(c)
		c.ExtraEnv = append(c.ExtraEnv, "FAKELLAMA_STREAM_GATE=1")
	}
}

// streamFrom starts a streamed request with headers; got closes when its first chunk is read,
// done when it has ended. release lets the fake send the rest.
func (r *residentRun) streamFrom(t *testing.T, e *testkit.Env, hdr map[string]string) (got, done chan struct{}, release func()) {
	t.Helper()
	got, done = make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		req, _ := http.NewRequest("POST", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/v1/chat/completions",
			bytes.NewReader([]byte(`{"stream":true}`)))
		req.Header.Set("Content-Type", "application/json")
		for k, v := range hdr {
			req.Header.Set(k, v)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Error(err)
			close(got)
			return
		}
		defer resp.Body.Close()
		buf := make([]byte, 1)
		resp.Body.Read(buf)
		close(got)
		io.Copy(io.Discard, resp.Body)
	}()
	return got, done, func() {
		for i := 1; i <= 4; i++ {
			os.WriteFile(filepath.Join(e.Ctl, "go."+strconv.Itoa(i)), nil, 0o644)
		}
	}
}

// armedAfter is the first armed idle fact from from on.
func armedAfter(t *testing.T, s *stream, from int) idleFact {
	t.Helper()
	evs := s.until(t, "armed", func(evs []event.Event) bool { return indexOf(evs, from, anyArmed) >= 0 })
	var f idleFact
	body(evs[indexOf(evs, from, anyArmed)], &f)
	return f
}

// A (the reviewer's case): a remote streaming request in flight, a local request that comes and
// ends during it: the remote one ended last, but remote is sticky for the stretch: the floor.
func TestRemoteInFlightThenLocalRequestKeepsTheFloor(t *testing.T) {
	r, e := startEngine(t, false, nil, gated)
	s := r.follow(t)
	r.put(t, `{"idle_sleep_s": 0}`)
	s.until(t, "asleep after the load", func(evs []event.Event) bool { return indexOf(evs, 0, sleptFor("idle")) >= 0 })
	got, done, release := r.streamFrom(t, e, fwd)
	<-got
	waitPhase(t, r, "ready")
	if code := r.post(t); code != 200 { // a local request, during the remote one
		t.Fatalf("local: %d", code)
	}
	from := len(s.events())
	release()
	<-done
	if f := armedAfter(t, s, from); f.Why != "floor/remote_requests" || f.DelayMS != run.RemoteIdleFloor.Milliseconds() {
		t.Fatalf("after the stretch with a remote request: %+v", f)
	}
}

// A, the leave: a remote request in flight on a model loaded at the start (no wake), a local
// attachment that comes and leaves during it (linger_s 0): the floor, not the linger.
func TestRemoteInFlightThenLocalLeaveKeepsTheFloor(t *testing.T) {
	r, e := startEngine(t, false, nil, gated)
	s := r.follow(t)
	waitPhase(t, r, "ready")
	got, done, release := r.streamFrom(t, e, fwd)
	<-got
	r.put(t, `{"idle_sleep_s": 0}`)
	r.holdFrom(t, s, nil)() // a local attachment comes and leaves while the remote request runs
	from := len(s.events())
	release()
	<-done
	if f := armedAfter(t, s, from); f.Why != "floor/remote_requests" || f.DelayMS != run.RemoteIdleFloor.Milliseconds() {
		t.Fatalf("after a local leave in a stretch with a remote request: %+v", f)
	}
}

// D: a remote attach and leave that happen while the loop is inside SleepIdle (between its
// decision and the stop) are still a remote leave: the next quiet takes the floor.
func TestRemoteAttachAndLeaveDuringTheSleepIsRemote(t *testing.T) {
	decided, proceed := make(chan struct{}), make(chan struct{})
	r, _ := startEngine(t, false, nil, func(o *app.Options) {
		prev := o.Configure
		o.Configure = func(c *run.Config) {
			prev(c)
			first := true
			c.IdleDecided = func() {
				if first {
					first = false
					close(decided)
					<-proceed
				}
			}
		}
	})
	s := r.follow(t)
	waitPhase(t, r, "ready")
	r.put(t, `{"idle_sleep_s": 0}`)
	<-decided
	r.holdFrom(t, s, fwd)() // attaches (and asks a wake) and leaves inside the sleep
	from := len(s.events())
	close(proceed)
	if f := armedAfter(t, s, from); f.Why != "floor/remote_leave" || f.DelayMS != run.RemoteIdleFloor.Milliseconds() {
		t.Fatalf("after a remote attach and leave the loop never saw: %+v", f)
	}
}

// G: a POST wake with idle_sleep_s 0 is not undone at once: the wake keeps the floor.
func TestAskedWakeKeepsTheFloor(t *testing.T) {
	const floor = 400 * time.Millisecond
	r, _ := startEngine(t, false, nil, shortFloor(floor))
	s := r.follow(t)
	r.put(t, `{"idle_sleep_s": 0}`)
	s.until(t, "asleep after the load", func(evs []event.Event) bool { return indexOf(evs, 0, sleptFor("idle")) >= 0 })
	from := len(s.events())
	if code := r.postControl(t, "wake"); code != 202 {
		t.Fatalf("wake: %d", code)
	}
	evs := s.until(t, "woken, then asleep", func(evs []event.Event) bool {
		rd := indexOf(evs, from, isReady)
		return rd >= 0 && indexOf(evs, rd+1, sleptFor("idle")) > 0
	})
	ready := indexOf(evs, from, isReady)
	if indexOf(evs, ready, armed("floor/wake", floor)) < 0 {
		t.Fatal("the quiet after an asked wake was not the floor")
	}
	if d := evs[indexOf(evs, ready+1, sleptFor("idle"))].At.Sub(evs[ready].At); d < floor {
		t.Fatalf("the asked wake was undone %v after ready, the floor is %v", d, floor)
	}
}

// F: --idle-sleep forces the quiet: status and the idle fact say the effective value and that it
// is forced; a PUT of another idle_sleep_s is refused (409), the effective one is accepted.
func TestIdleSleepFlagForcesTheQuiet(t *testing.T) {
	r, _ := startEngine(t, false, nil, idleAfter(7*time.Second))
	var st idleStatus
	r.get(t, "/engine/v1/status", &st)
	if !st.IdleSleep || st.IdleSleepS != 7 || !st.IdleSleepForced {
		t.Fatalf("status %+v", st)
	}
	if f := r.idleNow(t); f.IdleSleepS != 7 || !f.IdleSleepForced {
		t.Fatalf("idle fact %+v", f)
	}
	if code := r.putCode(t, `{"idle_sleep_s": 30}`); code != 409 {
		t.Fatalf("PUT idle_sleep_s under --idle-sleep: %d", code)
	}
	r.put(t, `{"idle_sleep_s": 7}`)
}
