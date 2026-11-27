package event

import (
	"encoding/json"
	"net/http"
	"strconv"
)

// Standing is everything that stands, whole, with the life and seq it stands
// at: the body of GET /engine/v1/facts. A client that follows the stream
// asks events from Seq of this Life.
type Standing struct {
	Life     string  `json:"life"`
	Seq      uint64  `json:"seq"`
	Facts    []Event `json:"facts"`
	Readings []Event `json:"readings"`
}

// Snapshot is everything that stands, read at one seq.
func (h *Hub) Snapshot() Standing {
	h.mu.Lock()
	s := Standing{Life: h.life, Seq: h.seq, Facts: h.allLocked(Fact, ""), Readings: h.allLocked(Reading, "")}
	h.mu.Unlock()
	s.Facts, s.Readings = sorted(s.Facts), sorted(s.Readings)
	return s
}

// ServeFacts answers GET facts: everything that stands, with life and seq.
func (h *Hub) ServeFacts(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h.Snapshot())
}

// ServeBook answers GET book: every declaration, sorted by name.
func (h *Hub) ServeBook(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h.Book())
}

// lifeRefused is the body of a 409: the life the caller asked for is not
// this one, or it claims a seq this life never gave.
type lifeRefused struct {
	Error string `json:"error"`
	Life  string `json:"life"`
	Seq   uint64 `json:"seq"`
}

// gapLine is the last line of a stream the ring let go of.
type gapLine struct {
	Gap bool `json:"gap"`
}

// ServeEvents answers GET events?life=L&since=N[&prefix=P]: what this hub
// said after seq N of life L, one JSON event a line (application/x-ndjson),
// oldest first, flushed as said, for as long as the caller holds the
// connection.
//
// It answers 409, with the present life and seq, when the life asked for is
// not this one or since is past this life's seq; the caller reads the facts
// whole and follows again. When the ring has let go of something the caller
// never had, the stream ends with a last line {"gap":true}, and the caller
// does the same. The stream ends when the request's context ends and when
// the hub is closed, which is how a client hears the engine is gone.
//
// The stream waits on a Waker and the request's context, never on a clock.
// The server must set no write deadline on it.
func (h *Hub) ServeEvents(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	since, err := strconv.ParseUint(q.Get("since"), 10, 64)
	if err != nil {
		http.Error(w, "event: since is not a seq", http.StatusBadRequest)
		return
	}
	prefix := q.Get("prefix")
	wk, err := h.Wake(prefix) // before the first read, so nothing said meanwhile is missed
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer wk.Close()
	if seq := h.Seq(); q.Get("life") != h.life || since > seq {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(lifeRefused{"not this life's seq: read the facts whole", h.life, seq})
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	out, enc := http.NewResponseController(w), json.NewEncoder(w)
	ended := false
	for {
		events, next, gap := h.since(since, prefix)
		if gap {
			enc.Encode(gapLine{true})
			out.Flush()
			return
		}
		for _, e := range events {
			if enc.Encode(e) != nil {
				return
			}
		}
		since = next
		if out.Flush() != nil || ended {
			return
		}
		select {
		case <-wk.C():
		case <-r.Context().Done():
			return
		case <-h.done:
			ended = true // write what was said before the close, then end
		}
	}
}
