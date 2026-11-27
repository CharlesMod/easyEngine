package event

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
	"sync"
	"time"
)

// under is where a fact or a reading stands.
type under struct{ name, key string }

// Hub holds what one life of the engine has said. Its zero value is not
// usable; make one with New.
type Hub struct {
	book map[string]Decl
	now  func() time.Time
	life string

	mu       sync.Mutex
	seq      uint64            // the last seq given
	standing map[under]Event   // every standing fact and reading
	changed  map[string]uint64 // the seq each name last changed at (SayIf)
	ring     []Event           // history: every event said, by seq
	wakers   map[*Waker]struct{}
	closed   bool
	done     chan struct{}
}

// New makes a hub that may say the names of book. now gives each event its
// At (time.Now when nil); At is data, never a trigger.
func New(book []Decl, now func() time.Time) (*Hub, error) {
	m, err := checkBook(book)
	if err != nil {
		return nil, err
	}
	if now == nil {
		now = time.Now
	}
	var id [8]byte
	if _, err := rand.Read(id[:]); err != nil {
		return nil, Refused{"the hub", "the system gave no random bytes for a life's id: " + err.Error()}
	}
	return &Hub{book: m, now: now, life: hex.EncodeToString(id[:]),
		standing: map[under]Event{}, changed: map[string]uint64{},
		wakers: map[*Waker]struct{}{}, done: make(chan struct{})}, nil
}

// Life is this hub's life: one run of the engine process.
func (h *Hub) Life() string { return h.life }

// Seq is the last seq this hub gave (0 before anything is said).
func (h *Hub) Seq() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.seq
}

// Say puts one event on the hub. The sayer gives its Name, Key and Gone
// (and may give Kind, which must then be the book's; empty takes the
// book's); the hub gives Seq, At and Life. body, when not nil, is written as
// JSON and becomes the event's body (at most BodyMax bytes).
//
// A fact or a reading replaces what stands under its (Name, Key); said Gone
// it removes it. A fact or reading whose body is byte-identical to what
// stands is said already: nothing is said, no seq is taken, no one is woken,
// and the standing event is answered, so "said on change" holds for any
// sayer. Gone for something that does not stand is likewise nothing, and
// answers an event with Seq 0. A happening only enters the history ring.
func (h *Hub) Say(e Event, body any) (Event, error) {
	e, _, err := h.SayIf(e, body, nil)
	return e, err
}

// SayIf is Say only if no name in since has changed after the seq given for
// it: what was decided from those facts is said only while they stand as
// they were read, and otherwise nothing is said (an empty Event, seq 0 and a
// nil error). The change that refused it already woke whoever reads that
// name, who decides again.
//
// It answers the seq its saying landed at (for a no-op, the hub's seq as it
// stands), so a sayer of several acts decided from one reading passes that
// seq on as the place it read at, without reading again what its own saying
// changed.
func (h *Hub) SayIf(e Event, body any, since map[string]uint64) (Event, uint64, error) {
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return Event{}, 0, Refused{e.Name, "its body cannot be written: " + err.Error()}
		}
		e.Body = raw
	}
	d, ok := h.book[e.Name]
	if e.Kind == "" && ok {
		e.Kind = d.Kind
	}
	if err := e.check(d, ok); err != nil {
		return Event{}, 0, err
	}
	if e.Gone {
		e.Body = nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return Event{}, 0, ErrClosed
	}
	for name, at := range since {
		if h.changed[name] > at {
			return Event{}, 0, nil
		}
	}
	k := under{e.Name, e.Key}
	if e.Kind != Happening {
		old, stands := h.standing[k]
		if e.Gone && !stands {
			e.Seq, e.Life = 0, h.life
			return e, h.seq, nil // nothing stood: nothing is said
		}
		if !e.Gone && stands && bytes.Equal(old.Body, e.Body) {
			return old, h.seq, nil // said already
		}
	}
	h.seq++
	e.Seq, e.At, e.Life = h.seq, h.now(), h.life
	switch {
	case e.Kind == Happening:
	case e.Gone:
		delete(h.standing, k)
	default:
		h.standing[k] = e
	}
	h.changed[e.Name] = e.Seq
	h.keep(e)
	h.wake(e.Name)
	return e, e.Seq, nil
}

// keep puts an event into the ring, letting the oldest go. Every event said
// is kept, a fact's and a reading's too, so a follower hears each change in
// order. What stands is never let go by count. The lock is held.
func (h *Hub) keep(e Event) {
	if len(h.ring) < RingKeep {
		h.ring = append(h.ring, e)
		return
	}
	h.ring[(e.Seq-1)%RingKeep] = e
}

// Fact is one standing fact.
func (h *Hub) Fact(name, key string) (Event, bool) {
	return h.stands(Fact, name, key)
}

// Reading is one standing reading.
func (h *Hub) Reading(name, key string) (Event, bool) {
	return h.stands(Reading, name, key)
}

func (h *Hub) stands(k Kind, name, key string) (Event, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	e, ok := h.standing[under{name, key}]
	return e, ok && e.Kind == k
}

// Facts is every standing fact whose name begins with prefix, sorted by
// name, then key.
func (h *Hub) Facts(prefix string) []Event { return h.all(Fact, prefix) }

// Readings is every standing reading whose name begins with prefix, sorted
// by name, then key.
func (h *Hub) Readings(prefix string) []Event { return h.all(Reading, prefix) }

func (h *Hub) all(k Kind, prefix string) []Event {
	h.mu.Lock()
	out := h.allLocked(k, prefix)
	h.mu.Unlock()
	return sorted(out)
}

// allLocked is all, unsorted. The lock is held.
func (h *Hub) allLocked(k Kind, prefix string) []Event {
	out := make([]Event, 0, len(h.standing))
	for _, e := range h.standing {
		if e.Kind == k && strings.HasPrefix(e.Name, prefix) {
			out = append(out, e)
		}
	}
	return out
}

func sorted(out []Event) []Event {
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].Key < out[j].Key
	})
	return out
}

// Since is history after seq from: the events said after it whose name
// begins with prefix, oldest first, and whether the ring had let go of some
// the reader never saw (gap). With a gap, the events are those the ring
// still holds; the reader reads what stands whole instead. History is for
// telling what happened, never for deciding.
func (h *Hub) Since(from uint64, prefix string) (events []Event, gap bool) {
	events, _, gap = h.since(from, prefix)
	return events, gap
}

// since is Since and the seq to ask from next.
func (h *Hub) since(from uint64, prefix string) (events []Event, next uint64, gap bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if from > h.seq {
		from = h.seq
	}
	if h.seq > RingKeep && from < h.seq-RingKeep {
		from, gap = h.seq-RingKeep, true
	}
	for s := from + 1; s <= h.seq; s++ {
		if e := h.ring[(s-1)%RingKeep]; strings.HasPrefix(e.Name, prefix) {
			events = append(events, e)
		}
	}
	return events, h.seq, gap
}

// Close ends the hub: nothing more is said, every waker is woken, and every
// held stream ends, which is how a client hears the engine is gone.
func (h *Hub) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return
	}
	h.closed = true
	close(h.done)
	for w := range h.wakers {
		w.signal()
	}
}

// Done is closed when the hub is closed.
func (h *Hub) Done() <-chan struct{} { return h.done }

// Waker wakes one reader when something it listens for changes. The reader
// then reads what stands; several changes may be one wake.
type Waker struct {
	h        *Hub
	prefixes []string
	c        chan struct{}
}

// Wake gives a reader its waker for every name that begins with one of
// prefixes; "" is everything. A prefix no name in the book begins with could
// never wake anyone, so it is refused, and so is a waker for nothing.
func (h *Hub) Wake(prefixes ...string) (*Waker, error) {
	if len(prefixes) == 0 {
		return nil, Refused{"a reader", "it names nothing to be woken for"}
	}
	for _, p := range prefixes {
		if !h.declaredUnder(p) {
			return nil, Refused{p, "no name in the book begins with it, so this reader would never wake"}
		}
	}
	w := &Waker{h: h, prefixes: prefixes, c: make(chan struct{}, 1)}
	h.mu.Lock()
	h.wakers[w] = struct{}{}
	h.mu.Unlock()
	return w, nil
}

// C is signalled when something the reader listens for changed, and when
// the hub is closed. It holds one signal: signals coalesce, and a sayer
// never blocks on it.
func (w *Waker) C() <-chan struct{} { return w.c }

// Close ends the waker. Its channel is never closed.
func (w *Waker) Close() {
	w.h.mu.Lock()
	delete(w.h.wakers, w)
	w.h.mu.Unlock()
}

func (w *Waker) signal() {
	select {
	case w.c <- struct{}{}:
	default: // it is already owed a wake
	}
}

// wake signals each reader that listens for name. The lock is held.
func (h *Hub) wake(name string) {
	for w := range h.wakers {
		for _, p := range w.prefixes {
			if strings.HasPrefix(name, p) {
				w.signal()
				break
			}
		}
	}
}
