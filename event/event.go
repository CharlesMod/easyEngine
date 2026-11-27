// Package event is the engine's hub: what it says, held in memory, and the
// book of every name it may say (docs/EVENTS.md section 2). It is the hive's
// hum in one process: one shape of event, a fact stands until it is replaced
// or said gone, a reading is a level said when it changes, a happening is
// history.
//
// A reader is never handed something to drain in order: it is woken, and
// reads what stands. There is no subscriber queue to overflow and no slow
// reader case. Nothing here is written to disk, and nothing here waits on a
// clock: each start of a hub is a new life, named by a random id, and a
// client that meets a new life reads the facts whole again.
//
// The package imports only the standard library.
package event

import (
	"encoding/json"
	"strconv"
	"time"
)

// Kind is which of the three an event is.
type Kind string

const (
	// Fact stands until it is replaced or said gone.
	Fact Kind = "fact"
	// Reading is a level a sense says when it changes; it stands like a fact.
	Reading Kind = "reading"
	// Happening is history: it enters the ring and stands nowhere. No choice
	// may depend on the ring, only on what stands.
	Happening Kind = "happening"
)

// Event is one thing said on the hub. The envelope is typed and cheap to
// read; the body is the name's own fields and is decoded only by a reader
// that wants it.
type Event struct {
	Seq  uint64          `json:"seq"`            // per life, from 1
	At   time.Time       `json:"at"`             // data only: no choice depends on it
	Life string          `json:"life"`           // one run of the engine process
	Name string          `json:"name"`           // declared in the book, e.g. "phase", "mind.shape"
	Kind Kind            `json:"kind"`           // "fact" | "reading" | "happening"
	Key  string          `json:"key,omitempty"`  // which one, where a name has several (a client, a slot)
	Gone bool            `json:"gone,omitempty"` // this fact or reading stands no longer
	Body json.RawMessage `json:"body,omitempty"` // the name's own fields
}

// BodyMax is the most bytes an event's body may hold, as JSON (64 KiB, the
// hive's bound). A body is a fact's fields, not a payload: anything larger
// belongs in a file the body names.
const BodyMax = 64 << 10

// RingKeep is how many events the history ring holds.
const RingKeep = 4096

// Refused is the hub's no: the name it was about and why, in words.
type Refused struct {
	Name string
	Why  string
}

func (r Refused) Error() string { return "event: " + r.Name + " is refused: " + r.Why }

// ErrClosed is said by a hub that has been closed.
var ErrClosed = Refused{"the hub", "it is closed"}

// check says what is wrong with an event's envelope against its declaration.
func (e Event) check(d Decl, ok bool) error {
	switch {
	case e.Name == "":
		return Refused{"an event", "it has no name"}
	case !ok:
		return Refused{e.Name, "the book does not declare it"}
	case e.Kind != d.Kind:
		return Refused{e.Name, "the book declares it a " + string(d.Kind) + ", not a " + string(e.Kind)}
	case !d.Keyed && e.Key != "":
		return Refused{e.Name, "it carries a key and the book gives it none"}
	case d.Keyed && e.Key == "":
		return Refused{e.Name, "it carries no key and the book says it is keyed"}
	case e.Gone && e.Kind == Happening:
		return Refused{e.Name, "a happening is history and cannot be said gone"}
	case len(e.Body) > BodyMax:
		return Refused{e.Name, "its body is " + strconv.Itoa(len(e.Body)) + " bytes, over " + strconv.Itoa(BodyMax)}
	}
	return nil
}
