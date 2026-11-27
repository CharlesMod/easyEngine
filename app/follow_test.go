//go:build !windows

package app_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/event"
)

// stream follows the engine's own event stream (GET /engine/v1/facts, then
// GET /engine/v1/events from that seq) and keeps every event it hears, so a test waits on what
// the engine says and orders what came before what by the stream, never by a clock.
type stream struct {
	mu      sync.Mutex
	evs     []event.Event
	changed chan struct{} // one signal for each line heard
	stop    context.CancelFunc
}

// follow starts following r's engine; it returns once the stream is open (the facts read first,
// the events from that seq), so nothing the engine says after this call is missed.
func (r *residentRun) follow(t *testing.T) *stream {
	t.Helper()
	base := "http://127.0.0.1:" + strconv.Itoa(r.port) + "/engine/v1/"
	var f event.Standing
	r.get(t, "/engine/v1/facts", &f)
	ctx, cancel := context.WithCancel(context.Background())
	s := &stream{changed: make(chan struct{}, 1), stop: cancel}
	t.Cleanup(cancel)
	req, _ := http.NewRequestWithContext(ctx, "GET", base+"events?life="+f.Life+"&since="+strconv.FormatUint(f.Seq, 10), nil)
	resp, err := http.DefaultClient.Do(req) // answers its headers once the stream is registered
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		t.Fatalf("events: %d", resp.StatusCode)
	}
	go func() {
		defer resp.Body.Close()
		dec := json.NewDecoder(resp.Body)
		for {
			var e event.Event
			if dec.Decode(&e) != nil {
				return
			}
			s.mu.Lock()
			if e.Name != "" { // the last line of a stream the ring let go of is {"gap":true}
				s.evs = append(s.evs, e)
			}
			s.mu.Unlock()
			select {
			case s.changed <- struct{}{}:
			default:
			}
		}
	}()
	return s
}

// events is everything heard so far, oldest first.
func (s *stream) events() []event.Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]event.Event(nil), s.evs...)
}

// until waits for ok to hold of the events heard, re-checking it at every line the engine says.
// The one timer is a watchdog for silence: the engine said nothing that made ok true.
func (s *stream) until(t *testing.T, what string, ok func([]event.Event) bool) []event.Event {
	t.Helper()
	dog := time.NewTimer(20 * time.Second) // watchdog for silence: the engine said nothing that satisfied the wait
	defer dog.Stop()
	for {
		if evs := s.events(); ok(evs) {
			return evs
		}
		select {
		case <-s.changed:
		case <-dog.C: // watchdog for silence
			t.Fatal("the engine said nothing that satisfied: ", what)
		}
	}
}

// await waits until cond holds of the engine, re-checking at every event the engine says.
func (r *residentRun) await(t *testing.T, what string, cond func() bool) {
	t.Helper()
	s := r.follow(t) // listen first, then look
	defer s.stop()
	s.until(t, what, func([]event.Event) bool { return cond() })
}

// body decodes an event's body into v.
func body(e event.Event, v any) { json.Unmarshal(e.Body, v) }

// phaseIs says e is the phase fact with this phase.
func phaseIs(e event.Event, phase string) bool {
	var b struct{ Phase string }
	body(e, &b)
	return e.Name == "phase" && !e.Gone && b.Phase == phase
}

// requestIs says e is a request happening of this kind (begin, end or idle).
func requestIs(e event.Event, what string) bool {
	var b struct{ What string }
	body(e, &b)
	return e.Name == "request" && b.What == what
}

// indexOf is the index of the first event from i on that pred says yes to, or -1.
func indexOf(evs []event.Event, from int, pred func(event.Event) bool) int {
	for i := from; i < len(evs); i++ {
		if pred(evs[i]) {
			return i
		}
	}
	return -1
}

// post sends a chat completion to the engine's door and answers the status code.
func (r *residentRun) post(t *testing.T) int {
	t.Helper()
	resp, err := http.Post("http://127.0.0.1:"+strconv.Itoa(r.port)+"/v1/chat/completions", "application/json", bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}
