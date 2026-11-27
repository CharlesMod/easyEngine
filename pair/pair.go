// Package pair holds the pairing requests of clients on the network (docs/ENGINE.md "Pairing").
//
// A client asks with its name and gets an unguessable request id and a six-digit code. It shows
// the code; the engine's side (the setup page, the tray, `easy-engine pair`) shows the same code
// and the name; the person compares them and answers Allow or Deny. Nothing is typed from one
// device into another. The code is a comparison code: it grants nothing, and nothing accepts it
// from the network.
//
// There is no clock here. A request waits until it is answered, withdrawn by its client, pushed
// out by the cap (the oldest waiting one goes when a new one arrives over it), or the engine ends.
// The token is issued when the client collects an Allow, and shown that once.
package pair

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"math/big"
	"strings"
	"sync"
	"unicode"

	"github.com/CharlesMod/easyEngine/event"
)

// DefaultMax is the number of requests that may wait at once.
const DefaultMax = 4

// NameMax caps a client's name, in runes.
const NameMax = 40

// FactName is the hub fact a waiting request is said as, keyed by its id.
const FactName = "pairing"

// Errors the callers turn into statuses.
var (
	ErrNoRequest = errors.New("no such pairing request")                 // 404 to the answerer, 410 to the client
	ErrAmbiguous = errors.New("that code matches more than one request") // refused: answer by id
	ErrClosed    = errors.New("the engine is ending")
)

// Issuer makes a token for a named client (internal/tokens.Store).
type Issuer interface {
	Issue(name string) (id, token string, err error)
}

// Request is what a person is shown: never a token.
// Key is public: the fact is keyed by it and an answer names it. The client's request id is a
// secret apart (whoever holds it collects the token) and is never said on the hub.
type Request struct {
	Key  string `json:"key"`
	Name string `json:"name"`
	Code string `json:"code"`           // six digits, e.g. "493201"
	From string `json:"from,omitempty"` // the address it came from, as the engine saw it
	// Quiet: the person has denied a request from this address in this life, so this one is
	// listed but raises no notification.
	Quiet bool `json:"quiet,omitempty"`
}

// Outcome is how a request ended for its client.
type Outcome int

const (
	Allowed Outcome = iota + 1
	Denied
	Gone // withdrawn, pushed out, already collected, never known, or the engine ended
)

// Result is what the client collects.
type Result struct {
	Outcome  Outcome
	Token    string // Allowed only: shown this once
	ClientID string // Allowed only
}

type state int

const (
	waiting state = iota
	allowed
	denied
	gone
)

type req struct {
	Request
	id    string // the client's secret request id
	n     uint64 // order of arrival
	state state
	done  chan struct{} // closed when the state leaves waiting
}

// Pairer is the set of pairing requests of one engine life. Safe for concurrent use.
type Pairer struct {
	hub *event.Hub // may be nil
	iss Issuer
	max int

	mu     sync.Mutex
	n      uint64
	reqs   map[string]*req
	denied map[string]bool // source addresses the person denied in this life
	closed bool
	end    chan struct{}
}

// New makes a Pairer that says its requests on hub (nil says nothing) and issues tokens with iss.
// max <= 0 is DefaultMax.
func New(hub *event.Hub, iss Issuer, max int) *Pairer {
	if max <= 0 {
		max = DefaultMax
	}
	return &Pairer{hub: hub, iss: iss, max: max, reqs: map[string]*req{}, denied: map[string]bool{}, end: make(chan struct{})}
}

// CleanName makes a client's name fit to show: control and format characters dropped, runs of
// space made one, trimmed, capped at NameMax runes; an empty name is "unnamed client".
func CleanName(s string) string {
	var b strings.Builder
	space := false
	n := 0
	for _, r := range s {
		if unicode.IsSpace(r) {
			space = b.Len() > 0
			continue
		}
		if !unicode.IsPrint(r) {
			continue
		}
		if space {
			if n+1 >= NameMax {
				break
			}
			b.WriteRune(' ')
			n++
			space = false
		}
		if n >= NameMax {
			break
		}
		b.WriteRune(r)
		n++
	}
	if b.Len() == 0 {
		return "unnamed client"
	}
	return b.String()
}

// SpacedCode shows a code as people read it: "493 201".
func SpacedCode(c string) string {
	if len(c) != 6 {
		return c
	}
	return c[:3] + " " + c[3:]
}

func randomID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func randomCode() (string, error) {
	n, err := rand.Int(rand.Reader, big.NewInt(1000000))
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%06d", n.Int64()), nil
}

// Ask opens a request for a client that calls itself name, from the address from, and returns
// what a person is shown and the client's secret request id. One address has at most one
// waiting request: a new ask from it replaces the one it had (that client is told Gone) and keeps
// its public key, so the person is not notified again. When max requests already wait, the
// oldest is pushed out (its client is told Gone). All bounds are counts.
func (p *Pairer) Ask(name, from string) (Request, string, error) {
	id, err := randomID()
	if err != nil {
		return Request{}, "", err
	}
	key, err := randomID()
	if err != nil {
		return Request{}, "", err
	}
	key = key[:12]
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return Request{}, "", ErrClosed
	}
	code := ""
	for code == "" { // a code no other waiting request shows, so a person can tell them apart
		c, err := randomCode()
		if err != nil {
			return Request{}, "", err
		}
		if len(p.byCodeLocked(c)) == 0 {
			code = c
		}
	}
	// One waiting request per source: a new one replaces it, under the same public key.
	if old := p.waitingFromLocked(from); old != nil {
		key = old.Key
		old.state = gone
		close(old.done)
		delete(p.reqs, old.id) // its fact is said again below, under the same key: not gone
	}
	// Count bounds, not clocks: at most max waiting, and at most max answered but not collected.
	p.trimLocked(func(r *req) bool { return r.state == waiting }, p.max-1)
	p.trimLocked(func(r *req) bool { return r.state != waiting }, p.max)
	p.n++
	r := &req{Request: Request{Key: key, Name: CleanName(name), Code: code, From: from, Quiet: from != "" && p.denied[from]}, id: id, n: p.n, done: make(chan struct{})}
	p.reqs[id] = r
	p.say(r.Request)
	return r.Request, id, nil
}

// waitingFromLocked is the waiting request from the address from (none for an unknown address).
func (p *Pairer) waitingFromLocked(from string) *req {
	if from == "" {
		return nil
	}
	for _, r := range p.reqs {
		if r.state == waiting && r.From == from {
			return r
		}
	}
	return nil
}

func (p *Pairer) byKeyLocked(key string) *req {
	for _, r := range p.reqs {
		if r.Key == key {
			return r
		}
	}
	return nil
}

// trimLocked pushes out the oldest requests matching sel until at most keep remain.
func (p *Pairer) trimLocked(sel func(*req) bool, keep int) {
	for {
		var oldest *req
		count := 0
		for _, r := range p.reqs {
			if !sel(r) {
				continue
			}
			count++
			if oldest == nil || r.n < oldest.n {
				oldest = r
			}
		}
		if count <= keep || oldest == nil {
			return
		}
		p.endLocked(oldest, gone)
		delete(p.reqs, oldest.id)
	}
}

// endLocked moves a waiting request to st, wakes its client and takes the fact down.
func (p *Pairer) endLocked(r *req, st state) {
	if r.state != waiting {
		return
	}
	r.state = st
	close(r.done)
	p.sayGone(r.Key)
}

func (p *Pairer) byCodeLocked(code string) []*req {
	var out []*req
	for _, r := range p.reqs {
		if r.state == waiting && r.Code == code {
			out = append(out, r)
		}
	}
	return out
}

// Wait holds until the request is answered and returns what its client collects: Allowed with
// a fresh token (issued now, shown this once), Denied, or Gone. A request that was collected
// already, or was never known, is Gone. When ctx ends first the request keeps waiting (the client
// may come back for it) and ctx's error is returned.
func (p *Pairer) Wait(ctx context.Context, id string) (Result, error) {
	p.mu.Lock()
	r, ok := p.reqs[id]
	p.mu.Unlock()
	if !ok {
		return Result{Outcome: Gone}, nil
	}
	select {
	case <-r.done:
	case <-p.end:
		return Result{Outcome: Gone}, nil
	case <-ctx.Done():
		return Result{}, ctx.Err()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.reqs[id] != r { // collected by another call, or pushed out
		return Result{Outcome: Gone}, nil
	}
	switch r.state {
	case allowed:
		cid, tok, err := p.iss.Issue(r.Name)
		if err != nil {
			return Result{}, err // the Allow stands: the client may ask again
		}
		delete(p.reqs, id)
		return Result{Outcome: Allowed, Token: tok, ClientID: cid}, nil
	case denied:
		delete(p.reqs, id)
		return Result{Outcome: Denied}, nil
	}
	delete(p.reqs, id)
	return Result{Outcome: Gone}, nil
}

// Withdraw is the client going away for good: its request ends and its fact goes.
func (p *Pairer) Withdraw(id string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	r, ok := p.reqs[id]
	if !ok {
		return false
	}
	p.endLocked(r, gone)
	delete(p.reqs, id)
	return true
}

// Answer is a person's Allow (allow true) or Deny for the waiting request with this public key.
func (p *Pairer) Answer(key string, allow bool) (Request, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	r := p.byKeyLocked(key)
	if key == "" || r == nil || r.state != waiting {
		return Request{}, ErrNoRequest
	}
	st := denied
	if allow {
		st = allowed
	}
	if !allow && r.From != "" {
		p.denied[r.From] = true // its later requests are listed, never notified
	}
	p.endLocked(r, st)
	return r.Request, nil
}

// ByCode finds the one waiting request that shows code (spaces ignored). A code two requests
// show is refused with ErrAmbiguous.
func (p *Pairer) ByCode(code string) (Request, error) {
	code = strings.ReplaceAll(code, " ", "")
	p.mu.Lock()
	defer p.mu.Unlock()
	rs := p.byCodeLocked(code)
	switch len(rs) {
	case 0:
		return Request{}, ErrNoRequest
	case 1:
		return rs[0].Request, nil
	}
	return Request{}, ErrAmbiguous
}

// Waiting lists the requests that wait for an answer, oldest first.
func (p *Pairer) Waiting() []Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	var rs []*req
	for _, r := range p.reqs {
		if r.state == waiting {
			rs = append(rs, r)
		}
	}
	for i := 1; i < len(rs); i++ {
		for j := i; j > 0 && rs[j].n < rs[j-1].n; j-- {
			rs[j], rs[j-1] = rs[j-1], rs[j]
		}
	}
	out := make([]Request, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.Request)
	}
	return out
}

// Close is the engine ending: every waiting request ends Gone for its client (a held GET is
// answered 410), the facts go, and no request is taken after. Answered requests not yet
// collected are dropped: no token is issued for them.
func (p *Pairer) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return
	}
	p.closed = true
	for id, r := range p.reqs {
		p.endLocked(r, gone)
		delete(p.reqs, id)
	}
	close(p.end)
}

func (p *Pairer) say(r Request) {
	if p.hub == nil {
		return
	}
	_, _ = p.hub.Say(event.Event{Name: FactName, Key: r.Key}, map[string]any{"name": r.Name, "code": r.Code, "from": r.From, "quiet": r.Quiet})
}

func (p *Pairer) sayGone(id string) {
	if p.hub == nil {
		return
	}
	_, _ = p.hub.Say(event.Event{Name: FactName, Key: id, Gone: true}, nil)
}
