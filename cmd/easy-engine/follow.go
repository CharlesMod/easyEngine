package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/event"
)

// errEngineEnded is the answer when the engine's event stream ends: the engine is gone.
var errEngineEnded = errors.New("the engine ended")

// standing is what stands on the engine's stream: facts and readings by name and key.
type standing map[string]event.Event

func standKey(e event.Event) string { return e.Name + "\x00" + e.Key }

func (s standing) put(e event.Event) {
	if e.Gone {
		delete(s, standKey(e))
		return
	}
	s[standKey(e)] = e
}

// get is the standing event of a name (key ""), if any.
func (s standing) get(name string) (event.Event, bool) {
	e, ok := s[name+"\x00"]
	return e, ok
}

// follow reads what stands (GET facts) and then follows the engine's held event stream, calling
// step with what stands: once at the start with ev nil, then once per event. It returns nil when
// step says it is done, errEngineEnded when the stream ends, ctx's error when ctx ends. A 409 (the
// engine is another life now) or a gap line (the ring let go of what was missed) reads the facts
// again and follows again. Nothing here is on a clock: the stream is held open until the engine
// says something or ends.
func follow(ctx context.Context, tg app.Target, prefix string, step func(st standing, ev *event.Event) (done bool)) error {
	c := &http.Client{} // no overall timeout: the stream is held for as long as the job takes
	for {
		st, life, seq, err := readFacts(ctx, c, tg)
		if err != nil {
			return err
		}
		if step(st, nil) {
			return nil
		}
		again, done, err := followOnce(ctx, c, tg, prefix, st, life, seq, step)
		if err != nil || done {
			return err
		}
		if !again {
			return errEngineEnded
		}
	}
}

func readFacts(ctx context.Context, c *http.Client, tg app.Target) (standing, string, uint64, error) {
	req, err := tg.NewRequest("GET", "/engine/v1/facts", nil)
	if err != nil {
		return nil, "", 0, err
	}
	resp, err := c.Do(req.WithContext(ctx))
	if err != nil {
		return nil, "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, "", 0, fmt.Errorf("facts: %s", resp.Status)
	}
	var body struct {
		Life     string        `json:"life"`
		Seq      uint64        `json:"seq"`
		Facts    []event.Event `json:"facts"`
		Readings []event.Event `json:"readings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, "", 0, fmt.Errorf("facts: %v", err)
	}
	st := standing{}
	for _, e := range append(body.Facts, body.Readings...) {
		st.put(e)
	}
	return st, body.Life, body.Seq, nil
}

// followOnce follows one stream; again says the facts must be read anew (409 or a gap).
func followOnce(ctx context.Context, c *http.Client, tg app.Target, prefix string, st standing, life string, seq uint64, step func(standing, *event.Event) bool) (again, done bool, err error) {
	q := url.Values{"life": {life}, "since": {strconv.FormatUint(seq, 10)}}
	if prefix != "" {
		q.Set("prefix", prefix)
	}
	req, err := tg.NewRequest("GET", "/engine/v1/events?"+q.Encode(), nil)
	if err != nil {
		return false, false, err
	}
	resp, err := c.Do(req.WithContext(ctx))
	if err != nil {
		return false, false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusConflict {
		return true, false, nil
	}
	if resp.StatusCode != 200 {
		return false, false, fmt.Errorf("events: %s", resp.Status)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		var line struct {
			event.Event
			Gap bool `json:"gap"`
		}
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			return false, false, fmt.Errorf("events: %v", err)
		}
		if line.Gap {
			return true, false, nil
		}
		ev := line.Event
		st.put(ev)
		if step(st, &ev) {
			return false, true, nil
		}
	}
	if err := ctx.Err(); err != nil {
		return false, false, err
	}
	return false, false, nil // the stream ended: the engine is gone
}
