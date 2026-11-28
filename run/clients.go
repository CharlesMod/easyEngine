package run

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/lifecycle"
	"github.com/CharlesMod/easyEngine/plan"
)

// Clients: one engine, several programs (docs/EVENTS.md section 5, docs/ENGINE.md "Clients").
//
// A client is who connects (a paired token's client id, or the local machine: "loopback"); an
// attachment is one use of the engine by a client, with its profile. The --profile file is the
// base attachment, first in every order, for the engine's life. The engine's wants are the
// attachments' merged; the running mind's slot map is set at its start and changes at no other
// time: an attachment that leaves leaves its slots vacant in the running mind, and an attach of
// the same key (identity, name) takes them back with the cache in them. Nothing here runs on a
// clock: an attachment leaves when its pid ends (host.ExitsOf), when its held connection ends,
// or when it detaches.

// ServerOpts is a profile's launch options: what the launch needs that sizing must know, and
// extra llama-server arguments. Each is checked against the build's --help before it is used.
type ServerOpts struct {
	Ubatch    int      `json:"ubatch,omitempty"`
	Reasoning string   `json:"reasoning,omitempty"`
	ExtraArgs []string `json:"extra_args,omitempty"`
	// Required names options (ubatch, reasoning, agent, model, or an extra flag such as
	// --spec-type) whose refusal refuses the attach (409) instead of telling.
	Required []string `json:"required,omitempty"`
	// MMProj and Speculative are not offered: a projector comes from the catalog per model in a
	// later version, and a draft model is not sized. They are read only to refuse them plainly.
	MMProj      json.RawMessage `json:"mmproj,omitempty"`
	Speculative json.RawMessage `json:"speculative,omitempty"`
}

// CatalogOverlay is the models a local client adds to the catalog (catalog.Catalog.WithModels).
type CatalogOverlay struct {
	Models []catalog.Model `json:"models"`
}

// overlay is the catalog with a profile's overlay added, or the refusal (400 catalog_invalid);
// nil, nil when the profile brings none (or nothing new).
func (s *Supervisor) overlay(p Profile) (*catalog.Catalog, *ClientError) {
	if p.Catalog == nil || len(p.Catalog.Models) == 0 {
		return nil, nil
	}
	cat, err := s.catalog()
	if err != nil {
		return nil, cerr(503, "catalog_unavailable", "the engine's catalog could not be read: %v", err)
	}
	next, err := cat.WithModels(p.Catalog.Models)
	if err != nil {
		return nil, cerr(400, "catalog_invalid", "%v", err)
	}
	return next, nil
}

// takeOverlay puts a profile's overlay in the engine's catalog (checked again against the
// catalog that stands now: another attach may have added since). It stays for the engine's life.
func (s *Supervisor) takeOverlay(p Profile) *ClientError {
	if p.Catalog == nil || len(p.Catalog.Models) == 0 {
		return nil
	}
	if _, ce := s.overlay(p); ce != nil {
		return ce
	}
	s.mu.Lock()
	next, err := s.cat.WithModels(p.Catalog.Models) // s.cat is set: overlay read it
	if err == nil {
		s.cat = next
	}
	s.mu.Unlock()
	if err != nil {
		return cerr(400, "catalog_invalid", "%v", err)
	}
	return nil
}

// ModelPrefs is advice: the models a client can use and prefers. The selection stays the
// person's; a client is told, never a model swapped under another.
type ModelPrefs struct {
	Prefer []string `json:"prefer,omitempty"`
	CanUse []string `json:"can_use,omitempty"`
}

// ClientError is a refused attach, detach or request: the status, a code and the words.
type ClientError struct {
	Status int
	Code   string
	Msg    string
	Extra  map[string]any
}

func (e *ClientError) Error() string { return e.Msg }

func cerr(status int, code, format string, a ...any) *ClientError {
	return &ClientError{Status: status, Code: code, Msg: fmt.Sprintf(format, a...)}
}

// Bounds on what an attachment may ask (a remote profile is input from the network).
const (
	maxSlotsPerClient = 16
	maxSlotsAll       = 64
	maxAttachments    = 32
	maxPerIdentity    = 4
	minCtxPerSlot     = 1024
	maxCtxPerSlot     = 1 << 20
	baseID            = "profile"
	baseKey           = "profile"
	identityLoopback  = "loopback"
	// maxRemoteRestarts is how many restarts of the model server one remote identity may cause in
	// the engine's life (an attach that grows, a replace with other wants, a leave that reshapes,
	// an option change); after that its changes are accepted and wait (whyBudget).
	maxRemoteRestarts = 4
	// the remote clients' share: one identity's attachments want at most maxSlotsPerRemote slots
	// together; all remote attachments at most maxRemoteSlots slots and maxRemoteAttachments
	// attachments. Remote attachments never count against a local caller's caps.
	maxSlotsPerRemote    = 16
	maxRemoteSlots       = 32
	maxRemoteAttachments = 16
)

// The words an attachment's options and pending slots are told with.
const (
	whyNoHelp      = "no build to check against"
	whyDidNotStart = "the model server did not start with them"
	// whyIdle: a remote client's grow waits for the door's next empty moment (it never drains)
	whyIdle = "waiting_for_idle"
	// whyBudget: the client has caused maxRemoteRestarts restarts; its change waits for a start
	// that comes for another reason
	whyBudget = "restart_budget_spent"
)

// option is one launch option an attachment asks for.
type option struct {
	Name  string   // ubatch | reasoning | agent | --flag (extra); a later option adds a case to optionsOf
	Value string   // compared between attachments
	Args  []string // what goes on the command line
	Flags []string // the spellings the build's help must know (none: not a flag)
}

// NotTaken is an option an attachment asked for that the merge did not take.
type NotTaken struct {
	Option   string `json:"option"`
	Asked    string `json:"asked"`
	Standing string `json:"standing,omitempty"`
	HeldBy   string `json:"held_by,omitempty"`
	Why      string `json:"why"`
}

// OptReport is what became of an attachment's options in the merge.
type OptReport struct {
	Taken    []string   `json:"taken"`
	NotTaken []NotTaken `json:"not_taken"`
	Checked  bool       `json:"checked"`
}

type attachment struct {
	id, name, identity, key string
	local, base             bool
	pid                     int
	prof                    Profile
	wanting                 bool // its profile states slots
	seq                     uint64

	pidGen  uint64
	pidStop func()
	holdGen uint64
	hasHold bool
	holdEnd chan string // the current held stream's end word (detached | replaced | engine_ending)
	notify  chan struct{}

	slots    []int  // engine slots in the running mind; nil: none
	unserved string // why it has none in the running mind although it wants some
	short    bool   // unserved because the context given is below its floor: never a reshape's cause
	why      string // why its slots are pending (whyIdle, whyBudget), "" none said
	report   OptReport
	// charge: the budget key its own attach or change is charged to ("" none: a local one, the
	// base, the master key, or a return under its key with the profile that key last had)
	charge string
	// cred: the hash of the master token a master@<ip> attachment came with (a new master ends it)
	cred string
}

func (a *attachment) poke() {
	select {
	case a.notify <- struct{}{}:
	default:
	}
}

// runMap is the running mind's slot map, set at its start.
type runMap struct {
	owner []string // engine slot -> attachment id, "" vacant or anonymous
	keys  []string // engine slot -> the key it was given to ("" none)
	want  []int    // engine slot -> the slot count its key wanted when given
	ctx   int
	opts  map[string]string
	args  [][]string // the options' args as launched, in order
	help  *Help      // the build's help the options were checked against
	base  bool       // the base attachment has a range
	// ramCache: the server runs with llama-server's RAM prompt cache (one owner at its start, and
	// no listener for other machines); a second owner is then never served from vacant slots
	ramCache bool
	// tags: engine slot -> whose prompt it holds (the last one the door sent there, or laid back
	// at the start). A slot changes hands only through an erase (TakeSlot says when).
	tags []SlotTag
}

// SlotTag is whose prompt a slot holds: an attachment's key, "commons" (this machine's callers
// and the owner's, who trust each other), or "anon:<identity>" (a remote caller with no
// attachment); Trusted: this machine or the master key. Who "" is a slot nobody has used.
type SlotTag struct {
	Who     string
	Trusted bool
}

// needsErase: a slot that holds cur's prompt must be erased before tag's request runs in it. The
// trusted share among themselves (this machine's callers and the owner's key); everyone else
// only what is their own.
func needsErase(cur, tag SlotTag) bool {
	return cur.Who != "" && cur.Who != tag.Who && !(cur.Trusted && tag.Trusted)
}

// trustedIdentity: this machine, the engine itself, or the master key from another machine.
func trustedIdentity(identity string, local bool) bool {
	return local || identity == "engine" || strings.HasPrefix(identity, "master@")
}

// clientReg is the attachments and the running map, on the Supervisor (no package state).
type clientReg struct {
	mu   sync.Mutex
	list []*attachment
	seq  uint64
	excl map[string]string // left out of the merge until the next start: id -> why
	run  *runMap
	busy map[int]int // engine slot -> requests the door sent to it now
	// line: engine slot -> the requests waiting in the engine for it, in arrival order, behind a
	// handover (the slot holds another owner's prompt: it is erased once nothing the door sent
	// there is still running, and only then labelled the new owner's). lineWake is closed and
	// made again at every change a waiter may be waiting for (a release, a handover's end, a
	// restart): an event, never a poll.
	line     map[int][]*slotWaiter
	lineWake chan struct{}
	used     map[int]uint64 // engine slot -> the door's count when a request last went to it (LRU)
	tick     uint64
	// reclient: 0 nothing, 1 reshape when idle (a leave that gives someone something), 2 grow
	// (an attach the running mind cannot serve). A grow drains only when a local caller asked for
	// it (reLocal); a remote one's waits for idle. Read and cleared by the reactor.
	reclient int
	reLocal  bool
	plans    map[string]reshapePlan // whose change a reshape is asked for: attachment key -> plan
	restarts map[string]int         // budget key -> restarts its own changes caused in this life
	revokes  uint64                 // revokes begun (DetachRevoked): an attach's token read before one is read again
	// commons: a caller of this machine asked for the commons (rule 3-4) and the map gave it none:
	// every merge since gives it the defaults' slots (with no base), for the engine's life.
	// commonsShort: a start made with it still gave the commons nothing (no room): this machine's
	// anonymous requests are answered 503 until the next start, never held for a reshape that
	// cannot give it room.
	commons, commonsShort bool
	// commonsMerged: the last merge taken (mergedWants, for the start that follows it) gave the
	// commons the defaults' slots. A start whose merge was taken before the commons was asked for
	// says nothing of its room: the commons is asked again (a grow), never answered 503.
	commonsMerged bool
	// stale: the last client with wants left; the shape it grew stands (its range vacant) until
	// it returns, the model stops for another reason, or an under-served caller's request comes
	stale bool
	gen   uint64 // the running map's generation: a slot mark of an earlier map is not this one's
	// badExtra: the clients' extra_args a server exited with while starting (runShape): launched
	// without them while the extra_args standing are the same
	badExtra map[string]string
	agentRun string // the attachment agent that runs (agentChanged), "" none
}

func newID() string {
	var b [8]byte
	rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// clientsInit puts the base attachment in place when a profile was given.
func (s *Supervisor) clientsInit() {
	r := &s.clients
	r.excl = map[string]string{}
	r.busy = map[int]int{}
	r.line = map[int][]*slotWaiter{}
	r.lineWake = make(chan struct{})
	r.used = map[int]uint64{}
	r.plans = map[string]reshapePlan{}
	r.restarts = map[string]int{}
	if !s.cfg.ProfileGiven {
		return
	}
	name := s.cfg.Profile.Game
	if name == "" {
		name = baseKey
	}
	if ce := s.takeOverlay(s.cfg.Profile); ce != nil {
		s.logf("profile: the catalog overlay was not taken: %s", ce.Msg)
	}
	b := &attachment{id: baseID, name: name, identity: "engine", key: baseKey, local: true, base: true,
		prof: s.cfg.Profile, wanting: true, notify: make(chan struct{}, 1)}
	r.list = append(r.list, b)
}

// ---- options and the merge ---------------------------------------------------------------

// optionsOf lists a profile's launch options in a fixed order.
func optionsOf(p Profile) []option {
	var out []option
	if sv := p.Server; sv != nil {
		if sv.Ubatch > 0 {
			n := strconv.Itoa(sv.Ubatch)
			out = append(out, option{Name: "ubatch", Value: n, Args: []string{"-ub", n}, Flags: []string{"-ub", "--ubatch-size"}})
		}
		if sv.Reasoning != "" {
			out = append(out, option{Name: "reasoning", Value: sv.Reasoning, Args: []string{"--reasoning", sv.Reasoning}, Flags: []string{"--reasoning", "-rea"}})
		}
		for _, g := range extraGroups(sv.ExtraArgs) {
			name := normFlag(flagName(g[0]))
			out = append(out, option{Name: name, Value: strings.Join(g, " "), Args: g, Flags: []string{name}})
		}
	}
	if a := p.Agent; a != nil && a.Wheel != "" {
		out = append(out, option{Name: "agent", Value: a.ID + "@" + a.Wheel})
	}
	return out
}

// flagName is a token's flag name ("--x=v" is "--x").
func flagName(t string) string {
	if k := strings.Index(t, "="); k >= 0 {
		return t[:k]
	}
	return t
}

// isFlagToken says whether an extra_args token is a flag (a negative number is a value).
func isFlagToken(t string) bool {
	if !strings.HasPrefix(t, "-") || t == "-" {
		return false
	}
	_, err := strconv.ParseFloat(t, 64)
	return err != nil
}

// extraGroups splits extra_args into a flag and the values that follow it.
func extraGroups(args []string) [][]string {
	var out [][]string
	for _, t := range args {
		if isFlagToken(t) || len(out) == 0 {
			out = append(out, []string{t})
			continue
		}
		out[len(out)-1] = append(out[len(out)-1], t)
	}
	return out
}

type holder struct {
	value string
	by    *attachment
	args  []string
}

// mergeOptions walks the attachments in order: the first to state an option holds it, a later
// one with the same value shares it, a later one with another value is told (not taken, held by
// the first). An option whose flag the build lacks is never passed: not taken, said why. With no
// help (no build installed yet) nothing can be checked and the report says so.
func mergeOptions(list []*attachment, h *Help) (taken []string, held map[string]holder, reports map[string]OptReport) {
	held = map[string]holder{}
	reports = map[string]OptReport{}
	for _, a := range list {
		rep := OptReport{Taken: []string{}, NotTaken: []NotTaken{}, Checked: h != nil}
		for _, o := range optionsOf(a.prof) {
			if h == nil && strings.HasPrefix(o.Name, "-") {
				// extra_args are passed only as checked against the build's own --help
				rep.NotTaken = append(rep.NotTaken, NotTaken{Option: o.Name, Asked: o.Value, Why: whyNoHelp})
				continue
			}
			if h != nil && len(o.Flags) > 0 && !h.Has(o.Flags...) {
				rep.NotTaken = append(rep.NotTaken, NotTaken{Option: o.Name, Asked: o.Value, Why: "this build has no such flag"})
				continue
			}
			if h != nil && strings.HasPrefix(o.Name, "-") && ownedFlag(o.Name, h) {
				rep.NotTaken = append(rep.NotTaken, NotTaken{Option: o.Name, Asked: o.Value, Why: "the engine sets this flag"})
				continue
			}
			hd, ok := held[o.Name]
			switch {
			case !ok:
				held[o.Name] = holder{value: o.Value, by: a, args: o.Args}
				taken = append(taken, o.Name)
				rep.Taken = append(rep.Taken, o.Name)
			case hd.value == o.Value:
				rep.Taken = append(rep.Taken, o.Name)
			default:
				rep.NotTaken = append(rep.NotTaken, NotTaken{Option: o.Name, Asked: o.Value, Standing: hd.value, HeldBy: hd.by.name, Why: "conflict"})
			}
		}
		reports[a.id] = rep
	}
	return taken, held, reports
}

// launchOpts is the merged options of the standing attachments as llama-server args (the agent
// is not an argument), and the map compared to tell whether an attach changes them.
func launchOpts(list []*attachment, h *Help) (args [][]string, vals map[string]string, reports map[string]OptReport) {
	taken, held, reports := mergeOptions(list, h)
	vals = map[string]string{}
	for _, n := range taken {
		hd := held[n]
		if n == "agent" {
			continue
		}
		vals[n] = hd.value
		args = append(args, hd.args)
	}
	return args, vals, reports
}

// wantingLocked is the attachments whose wants count, in order, less those left out for room.
func (r *clientReg) wantingLocked() []*attachment {
	var out []*attachment
	for _, a := range r.list {
		if a.wanting && r.excl[a.id] == "" {
			out = append(out, a)
		}
	}
	return out
}

// mergedWants is the engine's wants: the attachments' merged (slots summed, min slots summed,
// the largest ctx and floor: llama-server gives every slot one ctx), or the profile (the base, or
// the defaults) when no attachment states any.
func (s *Supervisor) mergedWants() (w fit.Wants, from []string) {
	r := &s.clients
	r.mu.Lock()
	ws := r.wantingLocked()
	commons := r.commons
	r.commonsMerged = commons
	r.mu.Unlock()
	if len(ws) == 0 {
		p := s.cfg.Profile
		return fit.Wants{Slots: p.Slots, MinSlots: p.MinSlots, CtxMax: p.CtxPerSlot.Max, CtxFloor: p.CtxPerSlot.Floor}, nil
	}
	defer func() {
		// this machine's anonymous callers asked for the commons and no base gives it a range:
		// the defaults' slots beside the attachments' (left unassigned at the start: the commons)
		hasBase := false
		for _, a := range ws {
			hasBase = hasBase || a.base
		}
		if commons && !hasBase {
			p := s.cfg.Profile
			w.Slots += max(1, p.Slots)
			w.MinSlots += max(1, p.MinSlots)
			w.CtxMax = max(w.CtxMax, p.CtxPerSlot.Max)
			w.CtxFloor = max(w.CtxFloor, p.CtxPerSlot.Floor)
		}
	}()
	// a remote client's floor never raises the others': it is clamped to the local attachments'
	// and the base's merged floor (the defaults' when there are none)
	localFloor, anyLocal := 0, false
	localMax := s.cfg.Profile.CtxPerSlot.Max // a remote max is clamped to the owner's ceiling
	for _, a := range ws {
		if a.local || a.base {
			localFloor, anyLocal = max(localFloor, a.prof.CtxPerSlot.Floor), true
			localMax = max(localMax, a.prof.CtxPerSlot.Max)
		}
	}
	if !anyLocal {
		localFloor = s.cfg.Profile.CtxPerSlot.Floor
	}
	for _, a := range ws {
		p := a.prof
		floor, ctxMax := p.CtxPerSlot.Floor, p.CtxPerSlot.Max
		if !a.local && !a.base {
			floor, ctxMax = min(floor, localFloor), min(ctxMax, localMax)
		}
		w.Slots += p.Slots
		w.MinSlots += p.MinSlots
		w.CtxMax = max(w.CtxMax, ctxMax)
		w.CtxFloor = max(w.CtxFloor, floor)
		from = append(from, a.id)
	}
	return w, from
}

// floorShort leaves out of the merge every remote attachment whose floor is above the context
// per slot a shape gives (its floor was clamped and cannot raise the shape): unserved, said why,
// never a reshape's cause. True when one was left out (the shape is chosen again without it).
func (s *Supervisor) floorShort(ctx int) bool {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	out := false
	for _, a := range r.wantingLocked() {
		if !a.local && !a.base && a.prof.CtxPerSlot.Floor > ctx {
			r.excl[a.id] = floorWhy(ctx, a.prof.CtxPerSlot.Floor)
			out = true
		}
	}
	return out
}

func floorWhy(given, floor int) string {
	return fmt.Sprintf("the context per slot given (%d) is below its floor (%d)", given, floor)
}

// dropLatest leaves one wanting attachment out of the merge (the fit refused the merge even at
// its floor): the latest remote one, else the latest local one, never the base; false when none
// is left to drop, which is then the engine's own refusal.
func (s *Supervisor) dropLatest() (string, bool) {
	r := &s.clients
	r.mu.Lock()
	ws := r.wantingLocked()
	if len(ws) < 2 {
		r.mu.Unlock()
		return "", false
	}
	// remote before local, later before earlier, never the base
	var last *attachment
	for _, remote := range []bool{true, false} {
		for i := len(ws) - 1; i >= 0 && last == nil; i-- {
			if a := ws[i]; !a.base && a.local != remote {
				last = a
			}
		}
	}
	if last == nil {
		r.mu.Unlock()
		return "", false
	}
	r.excl[last.id] = "the card has no room for it beside the attachments before it"
	r.mu.Unlock()
	s.logf("clients: the merged wants do not fit; %s (%s) is left out until the next start", last.name, last.id)
	return last.id, true
}

// clearExcl lets every attachment back into the merge (a fresh start chooses again).
func (s *Supervisor) clearExcl() {
	s.clients.mu.Lock()
	clear(s.clients.excl)
	s.clients.mu.Unlock()
}

// grantSlots shares total slots among the wanting attachments in order: each its want when they
// all fit, else each its min slots, then the rest one at a time in order. Contiguous ranges.
func grantSlots(ws []*attachment, total int) [][]int {
	n := make([]int, len(ws))
	sum := 0
	for _, a := range ws {
		sum += a.prof.Slots
	}
	left := total
	if total >= sum {
		for i, a := range ws {
			n[i] = a.prof.Slots
			left -= n[i]
		}
	} else {
		for i, a := range ws {
			n[i] = min(a.prof.MinSlots, left)
			left -= n[i]
		}
		for left > 0 {
			moved := false
			for i, a := range ws {
				if left > 0 && n[i] < a.prof.Slots {
					n[i]++
					left--
					moved = true
				}
			}
			if !moved {
				break
			}
		}
	}
	out := make([][]int, len(ws))
	next := 0
	for i := range ws {
		for j := 0; j < n[i]; j++ {
			out[i] = append(out[i], next)
			next++
		}
	}
	return out
}

// launchSet is the clients' options a start launches with (clientOpts), carried to startMap so
// the map records what was launched, not what stands later.
type launchSet struct {
	args    [][]string
	vals    map[string]string
	reports map[string]OptReport
	help    *Help
	seq     uint64 // the attachments up to this seq were in the merge
	extra   bool   // it carries clients' extra_args
	// ramOff: the slot map will have more than one owner: the launch passes --cache-ram 0.
	// ramCache: the server runs with its RAM prompt cache (the build has it and it is not off).
	ramOff, ramCache bool
}

// extraOf is the extra_args part of merged option values.
func extraOf(vals map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range vals {
		if strings.HasPrefix(k, "-") {
			out[k] = v
		}
	}
	return out
}

// clientOpts is the merged launch options of the standing attachments now, less the extra_args
// a server would not start with (badExtra): those are told not taken.
func (s *Supervisor) clientOpts(h *Help, shape fit.Shape, known map[string]bool) launchSet {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	ls := s.clientOptsLocked(h)
	// llama-server's RAM prompt cache (--cache-ram, default 8192 MiB) keeps states from every
	// slot and loads the best match into whichever slot asks: with more than one owner of the
	// slots (the base and a client are two; the commons count as one) it is turned off
	if has(known, "--cache-ram", "-cram") {
		// the RAM prompt cache is not tied to a slot (server-slot-sched.cpp:167-206): on only when
		// no other machine can reach the engine and the map has one owner
		ls.ramOff = r.ownersLocked(shape) > 1 || s.cfg.RemoteListener
		ls.ramCache = !ls.ramOff
	}
	return ls
}

// ownersLocked is how many owners a start at shape would give the slots: each attachment granted
// slots, and the commons when slots are left over.
func (r *clientReg) ownersLocked(shape fit.Shape) int {
	var ws []*attachment
	for _, a := range r.wantingLocked() {
		if a.local || a.base || a.prof.CtxPerSlot.Floor <= shape.CtxPerSlot {
			ws = append(ws, a)
		}
	}
	n, sum := 0, 0
	for _, g := range grantSlots(ws, shape.Slots) {
		if len(g) > 0 {
			n++
			sum += len(g)
		}
	}
	if sum < shape.Slots {
		n++
	}
	return n
}

func (s *Supervisor) clientOptsLocked(h *Help) launchSet {
	r := &s.clients
	args, vals, reports := launchOpts(r.list, h)
	ex := extraOf(vals)
	if len(ex) > 0 && r.badExtra != nil && mapsEqual(ex, r.badExtra) {
		var keep [][]string
		for _, g := range args {
			if !strings.HasPrefix(g[0], "-") || ex[normFlag(flagName(g[0]))] == "" {
				keep = append(keep, g)
			}
		}
		args = keep
		for k := range ex {
			delete(vals, k)
		}
		for id, rep := range reports {
			var taken []string
			for _, n := range rep.Taken {
				if strings.HasPrefix(n, "-") {
					rep.NotTaken = append(rep.NotTaken, NotTaken{Option: n, Asked: ex[n], Why: whyDidNotStart})
				} else {
					taken = append(taken, n)
				}
			}
			if taken == nil {
				taken = []string{}
			}
			rep.Taken = taken
			reports[id] = rep
		}
		ex = nil
	}
	return launchSet{args: args, vals: vals, reports: reports, help: h, seq: r.seq, extra: len(ex) > 0}
}

// startFailedWith marks the extra_args of ls as ones the server did not start with: the next
// launch goes without them (true), unless they were already left out (false: no retry for that).
func (s *Supervisor) startFailedWith(ls launchSet) bool {
	if !ls.extra {
		return false
	}
	r := &s.clients
	bad := extraOf(ls.vals)
	r.mu.Lock()
	r.badExtra = bad
	r.mu.Unlock()
	s.logf("clients: the model server did not start with the clients' extra_args %v: starting without them", bad)
	return true
}

// withOptions puts the clients' option args before --host, each in place of the engine's own
// setting of the same flag (reasoning is off unless a client says otherwise).
func withOptions(args []string, opts [][]string, h *Help) []string {
	if len(opts) == 0 {
		return args
	}
	same := func(a, b string) bool {
		if a == b {
			return true
		}
		if h != nil {
			for _, x := range h.Groups[a] {
				if x == b {
					return true
				}
			}
		}
		return false
	}
	var out []string
	for _, g := range group(args) {
		drop := false
		for _, o := range opts {
			if same(g[0], o[0]) {
				drop = true
			}
		}
		if !drop {
			out = append(out, g...)
		}
	}
	var add []string
	for _, o := range opts {
		add = append(add, o...)
	}
	return withBefore(out, "--host", add)
}

// startMap sets the running map for a server that started at shape with the options of ls.
// Every standing attachment's slots, unserved word and report change here. An attachment that
// came after the merge the shape was chosen from waits for a grow.
func (s *Supervisor) startMap(shape fit.Shape, ls launchSet) {
	model := s.selectedID() // the model this start serves (a later start may step down a tier)
	r := &s.clients
	r.mu.Lock()
	var ws, late []*attachment
	for _, a := range r.wantingLocked() {
		switch {
		case a.seq > ls.seq:
			late = append(late, a)
		case requiresModel(a.prof) && !slices.Contains(a.prof.Models.CanUse, model):
			// it required a model it can use (judged at its attach); this start serves another
			// (a step down a tier): unserved, never given slots of a model it said it cannot use
			r.excl[a.id] = fmt.Sprintf("the model that serves (%s) is not one it can use", model)
		case !a.local && !a.base && a.prof.CtxPerSlot.Floor > shape.CtxPerSlot:
			// a step down gave less than its floor: unserved, not the cause of a reshape
			r.excl[a.id] = floorWhy(shape.CtxPerSlot, a.prof.CtxPerSlot.Floor)
		default:
			ws = append(ws, a)
		}
	}
	m := &runMap{owner: make([]string, shape.Slots), keys: make([]string, shape.Slots), want: make([]int, shape.Slots),
		ctx: shape.CtxPerSlot, opts: ls.vals, args: ls.args, help: ls.help}
	if m.opts == nil {
		m.opts = map[string]string{}
	}
	for _, a := range r.list {
		a.slots, a.unserved, a.short, a.why = nil, "", false, ""
		if why := r.excl[a.id]; a.wanting && why != "" {
			a.unserved = why
			a.short = strings.HasPrefix(why, "the context per slot given")
		}
	}
	if len(ws) > 0 {
		grants := grantSlots(ws, shape.Slots)
		for i, a := range ws {
			a.slots = grants[i]
			if len(a.slots) == 0 {
				a.unserved = "the card has no room for its slots beside the others"
			}
			for _, e := range a.slots {
				m.owner[e], m.keys[e], m.want[e] = a.id, a.key, a.prof.Slots
			}
			if a.base && len(a.slots) > 0 {
				m.base = true
			}
		}
	}
	for _, a := range r.list {
		if rep, ok := ls.reports[a.id]; ok {
			a.report = rep
		}
		a.poke()
	}
	m.ramCache = ls.ramCache
	// whose prompt each slot holds at the start: the server is new, so nothing but what the
	// lay-back puts back, and it labels each slot it fills with the kept prompt's owner
	// (context.go; an untrusted caller's prompt or an unclean slot is never kept). An empty slot is
	// labelled as its range's: an attachment's key, else the commons (a trusted caller not
	// confined may run in any of them while nothing is claimed).
	m.tags = make([]SlotTag, shape.Slots)
	commonsRange := false
	for e := range m.tags {
		m.tags[e] = SlotTag{Who: commonsWho, Trusted: true}
		if m.keys[e] == "" {
			commonsRange = true
		}
	}
	for _, a := range r.list {
		if a.base && len(a.slots) > 0 {
			commonsRange = true
		}
		if a.base {
			continue
		}
		for _, e := range a.slots {
			m.tags[e] = SlotTag{Who: a.key, Trusted: trustedIdentity(a.identity, a.local)}
		}
	}
	r.commonsShort = r.commons && r.commonsMerged && !commonsRange
	r.run = m
	r.gen++
	r.stale = false
	clear(r.plans)
	r.replanLocked()
	asked := false
	if r.commons && !r.commonsMerged && !commonsRange {
		// asked for after this start's merge was taken: the next look grows for it
		asked = r.planLocked(2, commonsKey, "", true, nil) || asked
	}
	for _, a := range late { // came during the start: the next look grows for it
		asked = r.planLocked(2, a.key, a.charge, a.local, a) || asked
	}
	clear(r.busy)
	clear(r.line)
	r.wakeLineLocked() // waiters see the new generation and are answered
	clear(r.used)
	r.mu.Unlock()
	s.sayClients()
	if asked {
		s.wakeMind(func(c *causes) { c.clients = true })
	}
}

// reshapePlan is a reshape asked for on someone's behalf: level 1 when idle, 2 a grow; who is the
// budget key charged when it happens ("" none); local: a local caller's (its grow drains).
type reshapePlan struct {
	level int
	who   string
	local bool
}

// commonsKey is the plan key of a reshape for a caller with no attachment (an under-served one).
const commonsKey = "~commons"

// budgetKey is who a remote attachment's restarts are counted against: "" none (a local caller,
// the base, the master key from another machine: the owner); every --open caller shares one.
func budgetKey(identity string, local bool) string {
	switch {
	case local || identity == "engine" || strings.HasPrefix(identity, "master@"):
		return ""
	case strings.HasPrefix(identity, "open@"):
		return "open"
	}
	return identity
}

// capKey is whose share of the remote caps an attachment counts in ("" a local caller's).
func capKey(identity string, local bool) string {
	if local {
		return ""
	}
	if strings.HasPrefix(identity, "open@") {
		return "open"
	}
	return identity
}

// replanLocked sets reclient and reLocal from the plans.
func (r *clientReg) replanLocked() {
	r.reclient, r.reLocal = 0, false
	for _, p := range r.plans {
		r.reclient = max(r.reclient, p.level)
		if p.level == 2 && p.local {
			r.reLocal = true
		}
	}
}

// planLocked asks the reactor for a reshape on behalf of key (an attachment's, or commonsKey):
// level 1 when idle, 2 a grow. who is the budget key the restart is charged to ("" none); local:
// a local caller's grow drains, a remote one's waits for idle (a.why says so). A charged change
// whose budget is spent asks for nothing (a.why: restart_budget_spent): its requests are then
// answered at once, never held. False when nothing was asked.
func (r *clientReg) planLocked(level int, key, who string, local bool, a *attachment) bool {
	if r.run == nil {
		return false
	}
	if who != "" && r.restarts[who] >= maxRemoteRestarts {
		if a != nil {
			a.why = whyBudget
		}
		return false
	}
	p := r.plans[key]
	p.level = max(p.level, level)
	if who != "" {
		p.who = who
	}
	p.local = p.local || local
	r.plans[key] = p
	if a != nil && level == 2 && !local {
		a.why = whyIdle
	}
	r.replanLocked()
	return true
}

// unplanLocked cancels the reshape asked on key's behalf (its key returned to its range).
func (r *clientReg) unplanLocked(key string) {
	delete(r.plans, key)
	r.replanLocked()
}

// refillBudgets ends a waking period: the model server stopped for the engine's own reason or the
// person's (a linger unload or a sleep asked from this machine, the person's stop, a yield of the
// card), so every remote client's restart budget is whole again. The stop itself is the event; no
// clock. Never called for a stop a remote client could cause: a crash, an out-of-memory walk or a
// failed start after a reshape, a reshape's own stop, or a sleep or stop a remote caller asked.
func (s *Supervisor) refillBudgets(why string) {
	r := &s.clients
	r.mu.Lock()
	spent := len(r.restarts) > 0
	clear(r.restarts)
	r.mu.Unlock()
	if spent {
		s.logf("clients: %s: every restart budget is whole again", why)
		s.sayClients()
	}
}

// countRestartCauses is a reshape for the clients going ahead: each budget key a plan names
// spends one of its restarts; the plans are spent.
func (s *Supervisor) countRestartCauses() {
	r := &s.clients
	r.mu.Lock()
	charged := map[string]bool{} // one restart costs each budget key one, however many of its plans asked
	for _, p := range r.plans {
		if p.who != "" && !charged[p.who] {
			charged[p.who] = true
			r.restarts[p.who]++
		}
	}
	clear(r.plans)
	r.replanLocked()
	r.mu.Unlock()
}

// stopMap: no mind runs; nobody has slots in one.
func (s *Supervisor) stopMap() {
	r := &s.clients
	r.mu.Lock()
	if r.run == nil {
		r.mu.Unlock()
		return
	}
	r.run = nil
	r.gen++
	r.stale = false
	clear(r.plans)
	r.replanLocked()
	for _, a := range r.list {
		a.slots = nil
		a.poke()
	}
	clear(r.busy)
	clear(r.line)
	r.wakeLineLocked() // waiters see the new generation and are answered
	clear(r.used)
	r.mu.Unlock()
	s.sayClients()
}

// ---- attach, leave ---------------------------------------------------------------------

// AttachRequest is one POST attach, after the API has read the caller (local or not, who).
type AttachRequest struct {
	Pid      int
	Name     string
	Hold     bool
	Profile  *Profile // nil: the old wire ({pid} alone), no wants
	Identity string   // "loopback" for a local caller, else the paired client id
	Local    bool
	// Valid, when set, says whether the caller's credential still stands; it is asked under the
	// clients lock as the attachment registers (a token revoked after the gate let it in).
	Valid func() bool
	// Cred is the hash of the master token a master@<ip> caller came with.
	Cred string
}

// AttachHold is a held attachment's connection: the API writes a line on Notify and the last
// line on End, and calls LeaveHold with Gen when the connection ends.
type AttachHold struct {
	ID     string
	Gen    uint64
	End    <-chan string
	Notify <-chan struct{}
}

// AttachView is the attach answer and the attachment fact.
type AttachView struct {
	Status     string     `json:"status,omitempty"`
	Pid        int        `json:"pid,omitempty"`
	Attachment string     `json:"attachment"`
	Name       string     `json:"name"`
	Key        string     `json:"key"`
	ClientID   string     `json:"client_id"`
	Local      bool       `json:"local"`
	Presence   string     `json:"presence"`
	Resumed    bool       `json:"resumed,omitempty"`
	Replaced   bool       `json:"replaced,omitempty"`
	Slots      SlotsView  `json:"slots"`
	Ctx        CtxView    `json:"ctx_per_slot"`
	Server     OptReport  `json:"server"`
	Model      *ModelView `json:"model,omitempty"`
	Agent      string     `json:"agent,omitempty"`
	// RestartsLeft is how many more restarts a remote client's own changes may cause (absent: not
	// budgeted: a local caller, the master key)
	RestartsLeft *int `json:"restarts_left,omitempty"`
}

// SlotsView is an attachment's slots: what it wants, the least it takes, what the running mind
// gives it (its first engine slot is said for a reader of the logs; a client numbers its own).
type SlotsView struct {
	Want     int    `json:"want"`
	Min      int    `json:"min"`
	Given    int    `json:"given"`
	First    *int   `json:"first,omitempty"`
	Pending  bool   `json:"pending"`
	Why      string `json:"why,omitempty"` // why it is pending: waiting_for_idle, restart_budget_spent
	Unserved string `json:"unserved,omitempty"`
}

// CtxView is the context per slot asked and the one the running mind gives (0: no mind runs).
type CtxView struct {
	Asked int `json:"asked"`
	Given int `json:"given"`
}

// ModelView is the selection against a client's advice.
type ModelView struct {
	Selected string `json:"selected"`
	Usable   bool   `json:"usable"`
	Advice   string `json:"advice,omitempty"`
}

func (a *attachment) presence() string {
	switch {
	case a.base:
		return "engine"
	case a.pid > 0 && a.hasHold:
		return "pid+hold"
	case a.pid > 0:
		return "pid"
	case a.hasHold:
		return "hold"
	}
	return "none"
}

// viewLocked is an attachment as said. Caller holds r.mu.
func (r *clientReg) viewLocked(a *attachment, model string) AttachView {
	v := AttachView{Attachment: a.id, Name: a.name, Key: a.key, ClientID: a.identity, Local: a.local, Presence: a.presence(),
		Pid: a.pid, Server: a.report}
	if v.Server.Taken == nil {
		v.Server = OptReport{Taken: []string{}, NotTaken: []NotTaken{}}
	}
	if a.wanting {
		v.Slots = SlotsView{Want: a.prof.Slots, Min: a.prof.MinSlots, Given: len(a.slots), Unserved: a.unserved}
		if len(a.slots) > 0 {
			f := a.slots[0]
			v.Slots.First = &f
		}
		v.Ctx.Asked = a.prof.CtxPerSlot.Max
		if r.run != nil {
			v.Ctx.Given = r.run.ctx
			v.Slots.Pending = len(a.slots) == 0 && a.unserved == "" && r.plans[a.key].level > 0
			if a.why == whyBudget || (len(a.slots) == 0 && a.unserved == "") {
				v.Slots.Why = a.why // a change its budget cannot pay for is said even while it keeps its slots
			}
		}
	}
	if mp := a.prof.Models; mp != nil && model != "" {
		mv := &ModelView{Selected: model, Usable: len(mp.CanUse) == 0 || slices.Contains(mp.CanUse, model)}
		if len(mp.Prefer) > 0 && !slices.Contains(mp.Prefer, model) {
			mv.Advice = "prefers " + strings.Join(mp.Prefer, ", ")
		}
		if !mv.Usable {
			mv.Advice = strings.TrimPrefix(mv.Advice+"; the model that serves is not one it can use", "; ")
		}
		v.Model = mv
	}
	if a.prof.Agent != nil && a.prof.Agent.Wheel != "" {
		v.Agent = a.prof.Agent.ID
	}
	if b := budgetKey(a.identity, a.local); b != "" && !a.base {
		left := max(0, maxRemoteRestarts-r.restarts[b])
		v.RestartsLeft = &left
	}
	return v
}

// requiresModel: the profile names model among its required options and lists the models it can
// use (an attach whose model is not among them is refused, 409 model_unsupported).
func requiresModel(p Profile) bool {
	if p.Models == nil || len(p.Models.CanUse) == 0 || p.Server == nil {
		return false
	}
	for _, need := range p.Server.Required {
		if strings.TrimSpace(need) == "model" {
			return true
		}
	}
	return false
}

// selectedID is the model a client's can_use is judged against: the one that serves (the
// selection, or the tier it stepped down to: tiers.go), never a selection that is not loaded.
func (s *Supervisor) selectedID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.st.Mode == "external" {
		return s.st.External.Model
	}
	return s.servedIDLocked()
}

// AttachClient is POST attach. A profile is checked first (checkProfile); then, under one lock,
// the key decides: a standing attachment of the key has its presence replaced (same id, same
// slots); a vacant range of the key in the running mind is taken back (resumed); vacant slots
// serve it when they suffice and its options change nothing; else the mind grows by drain.
func (s *Supervisor) AttachClient(req AttachRequest) (AttachView, *AttachHold, *ClientError) {
	if req.Pid > 0 && !req.Local {
		return AttachView{}, nil, cerr(400, "pid_remote", "a pid names a process on this machine; a client on another machine holds its attach instead (hold: true)")
	}
	if req.Pid <= 0 && !req.Hold {
		return AttachView{}, nil, cerr(400, "bad_request", "an attachment needs a pid or hold: true")
	}
	if req.Pid > 0 && !lifecycle.PidAlive(req.Pid) {
		return AttachView{}, nil, cerr(400, "bad_request", "no such process")
	}
	name := cleanName(req.Name)
	if name == "" {
		if req.Profile != nil && req.Profile.Game != "" {
			name = cleanName(req.Profile.Game)
		}
		if name == "" && req.Pid > 0 {
			name = "pid-" + strconv.Itoa(req.Pid)
		}
		if name == "" {
			name = "client"
		}
	}
	var prof Profile
	wanting := false
	// the build's --help is read only for an attach that brings extra_args; otherwise the help
	// already read (if any) checks the rest, and the launch checks again
	h := s.cachedHelp()
	if req.Profile != nil && req.Profile.Server != nil && len(req.Profile.Server.ExtraArgs) > 0 {
		h = s.buildHelp()
	}
	if req.Profile != nil {
		prof = *req.Profile
		if e := checkProfile(&prof, req.Local, h); e != nil {
			return AttachView{}, nil, e
		}
		wanting = req.Profile.Slots > 0
		prof.defaults()
		if _, ce := s.overlay(prof); ce != nil {
			return AttachView{}, nil, ce
		}
	}
	key := req.Identity + "/" + name
	oldWire := req.Profile == nil && !req.Hold
	var ended <-chan struct{}
	var stopWatch func()
	if req.Pid > 0 {
		st, err := host.StartOf(req.Pid)
		if err != nil && !errors.Is(err, host.ErrNoEvent) {
			return AttachView{}, nil, cerr(400, "bad_request", "no such process")
		}
		ended, stopWatch, err = s.cfg.ExitsOf(req.Pid, st)
		switch {
		case errors.Is(err, host.ErrNotThatProcess):
			return AttachView{}, nil, cerr(400, "bad_request", "no such process")
		case err != nil:
			// no word of its end on this system: an attachment would stand for the engine's life.
			// The old wire's {pid} is then a parent alone (the API attaches it), as before clients,
			// and never meets a cap; an attachment with wants or a hold stands as asked.
			ended, stopWatch = nil, nil
			if oldWire {
				return AttachView{Status: "attached", Pid: req.Pid}, nil, nil
			}
		}
	}
	watching := false
	defer func() {
		if stopWatch != nil && !watching {
			stopWatch()
		}
	}()
	model := s.selectedID()
	r := &s.clients
	// the token is read outside the clients lock (it may read the token file): a revoke that came
	// between the read and the registration bumps r.revokes, and the token is read again
	for {
		r.mu.Lock()
		g := r.revokes
		r.mu.Unlock()
		valid := req.Valid == nil || req.Valid()
		r.mu.Lock()
		if r.revokes != g {
			r.mu.Unlock()
			continue
		}
		if !valid {
			r.mu.Unlock()
			return AttachView{}, nil, cerr(401, "token_revoked", "the token this attach came with was revoked")
		}
		break
	}
	var old *attachment
	// caps: a local caller's are counted over local attachments only; a remote caller's over its
	// own share (one --open share for every address) and over all remote attachments
	ck := capKey(req.Identity, req.Local)
	count, mine, slotsMine, slotsSide := 0, 0, 0, 0
	for _, a := range r.list {
		if a.key == key && !a.base {
			old = a
			continue
		}
		if a.base {
			if ck == "" && a.wanting {
				slotsSide += a.prof.Slots
			}
			continue
		}
		akey := capKey(a.identity, a.local)
		if (akey == "") != (ck == "") {
			continue // the other side's: never counted against this caller
		}
		count++
		if a.wanting {
			slotsSide += a.prof.Slots
		}
		if ck != "" && akey == ck {
			mine++
			if a.wanting {
				slotsMine += a.prof.Slots
			}
		}
	}
	refuse := func(status int, code, format string, args ...any) (AttachView, *AttachHold, *ClientError) {
		r.mu.Unlock()
		return AttachView{}, nil, cerr(status, code, format, args...)
	}
	if !oldWire {
		switch {
		case old == nil && ck == "" && count >= maxAttachments:
			return refuse(429, "too_many_attachments", "this machine's programs hold %d attachments already", maxAttachments)
		case old == nil && ck != "" && count >= maxRemoteAttachments:
			return refuse(429, "too_many_attachments", "clients on other machines hold %d attachments already", maxRemoteAttachments)
		case old == nil && ck != "" && mine >= maxPerIdentity:
			return refuse(429, "too_many_attachments", "this client holds %d attachments already", maxPerIdentity)
		}
	}
	if wanting {
		switch {
		case ck == "" && slotsSide+prof.Slots > maxSlotsAll:
			return refuse(400, "bad_request", "this machine's attachments would want %d slots; at most %d in all", slotsSide+prof.Slots, maxSlotsAll)
		case ck != "" && slotsMine+prof.Slots > maxSlotsPerRemote:
			return refuse(429, "too_many_slots", "this client's attachments would want %d slots; at most %d", slotsMine+prof.Slots, maxSlotsPerRemote)
		case ck != "" && slotsSide+prof.Slots > maxRemoteSlots:
			return refuse(429, "too_many_slots", "clients on other machines would want %d slots; at most %d in all", slotsSide+prof.Slots, maxRemoteSlots)
		}
	}
	// charged for any restart its attach causes; a return served from its own standing range
	// causes none and so costs nothing, a return that needs a restart is charged as an attach is
	charge := budgetKey(req.Identity, req.Local)
	cand := &attachment{id: newID(), name: name, identity: req.Identity, key: key, local: req.Local, prof: prof, wanting: wanting,
		notify: make(chan struct{}, 1), charge: charge, cred: req.Cred}
	if old != nil {
		cand.id = old.id
	}
	// the merge with the candidate in place of the old one: a required option it does not get
	// refuses the attach, and nothing is registered
	trial := make([]*attachment, 0, len(r.list)+1)
	for _, a := range r.list {
		if a != old {
			trial = append(trial, a)
		}
	}
	trial = append(trial, cand)
	_, _, reports := mergeOptions(trial, h)
	rep := reports[cand.id]
	if sv := prof.Server; sv != nil {
		for _, need := range sv.Required {
			need = strings.TrimSpace(need)
			if need == "model" {
				if mp := prof.Models; mp != nil && len(mp.CanUse) > 0 && !slices.Contains(mp.CanUse, model) {
					r.mu.Unlock()
					e := cerr(409, "model_unsupported", "the model that serves (%q) is not one this client can use; the selection is the person's", model)
					e.Extra = map[string]any{"selected": model, "can_use": mp.CanUse}
					return AttachView{}, nil, e
				}
				continue
			}
			for _, nt := range rep.NotTaken {
				if nt.Option == normFlag(need) || nt.Option == need {
					r.mu.Unlock()
					e := cerr(409, "conflict", "required option %s not taken: %s", nt.Option, nt.Why)
					e.Extra = map[string]any{"option": nt.Option, "asked": nt.Asked, "standing": nt.Standing, "held_by": nt.HeldBy, "why": nt.Why}
					return AttachView{}, nil, e
				}
			}
		}
	}
	cand.report = rep
	var a *attachment
	resumed, replaced := false, false
	var endOld chan string
	var stopOld func()
	if old != nil {
		// the same key stands: its presence is replaced; the same id and the same slots
		a, replaced = old, true
		sameWants := old.wanting == wanting && (!wanting || (old.prof.Slots == prof.Slots && old.prof.CtxPerSlot == prof.CtxPerSlot && old.prof.MinSlots == prof.MinSlots))
		old.prof, old.wanting, old.local, old.report, old.charge, old.cred = prof, wanting, req.Local, rep, charge, req.Cred
		if old.hasHold {
			endOld = old.holdEnd
			old.hasHold, old.holdEnd = false, nil
		}
		if old.pidStop != nil {
			stopOld = old.pidStop
			old.pidStop = nil
		}
		old.pid = 0
		if !sameWants || !r.optsSameLocked(h) {
			// a change: it comes after every merge made before it, so a start that is loading
			// now (its merge taken before the change) grows for it once the model is up, charged
			// as an attach in a loading window is; with no model running it is folded into the
			// next start, free
			r.seq++
			old.seq = r.seq
			r.planLocked(2, a.key, charge, req.Local, a)
		}
	} else {
		cand.seq = r.seq + 1
		r.seq++
		r.list = append(r.list, cand)
		a = cand
		if wanting && r.run != nil {
			resumed = r.serveFromVacantLocked(a, h)
		}
	}
	if wanting {
		r.stale = false // a client with wants stands again
	}
	var hold *AttachHold
	if req.Hold {
		a.holdGen++
		a.hasHold = true
		a.holdEnd = make(chan string, 1)
		a.notify = make(chan struct{}, 1) // each held stream its own: a replaced one hears nothing more
		hold = &AttachHold{ID: a.id, Gen: a.holdGen, End: a.holdEnd, Notify: a.notify}
	}
	if req.Pid > 0 {
		a.pid = req.Pid
		a.pidGen++
		if ended != nil {
			// the watch is the attachment's under the same lock that registers it: a replace
			// that races in later stops this one (stopOld above), never leaves it running
			a.pidStop, watching = s.watchPid(a.id, a.pidGen, ended, stopWatch), true
		}
	}
	v := r.viewLocked(a, model)
	r.mu.Unlock()
	if endOld != nil {
		endOld <- "replaced"
	}
	if stopOld != nil {
		stopOld()
	}
	if ce := s.takeOverlay(prof); ce != nil {
		// another attach added a different model of the same id since the check: said, not taken
		s.logf("clients: %s's catalog overlay was not taken: %s", name, ce.Msg)
	}
	v.Status, v.Resumed, v.Replaced = "attached", resumed, replaced
	s.sayClients()
	s.agentChanged()
	s.wakeMind(func(c *causes) { c.clients = true })
	return v, hold, nil
}

// optsSameLocked: the standing attachments' options merge to what the running mind runs with.
func (r *clientReg) optsSameLocked(h *Help) bool {
	if r.run == nil {
		return true
	}
	_, vals, _ := launchOpts(r.list, h)
	return mapsEqual(vals, r.run.opts)
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; !ok || w != v {
			return false
		}
	}
	return true
}

// serveFromVacantLocked gives a new attachment slots of the running mind when that changes
// nothing for anyone: its own key's vacant range of the same count first (resumed: the cache in
// it is its own), else any vacant slots, lowest first, when they are enough, the running ctx is
// at least its floor and its options change nothing. Otherwise the mind grows by drain.
func (r *clientReg) serveFromVacantLocked(a *attachment, h *Help) (resumed bool) {
	m := r.run
	var own, free []int
	for e, o := range m.owner {
		if o != "" {
			continue
		}
		if m.keys[e] == "" && m.base {
			continue // the base's anonymous room is not vacant
		}
		if m.keys[e] == a.key {
			own = append(own, e)
		}
		free = append(free, e)
	}
	optsOK := r.optsSameLocked(h)
	ctxOK := m.ctx >= a.prof.CtxPerSlot.Floor
	// with the RAM prompt cache on (one owner at the start), only the key's own return is served
	// from vacant slots: a second owner's states would mix with the first's in that cache
	ownReturn := len(own) == a.prof.Slots && len(own) > 0 && m.want[own[0]] == a.prof.Slots
	if m.ramCache && !ownReturn {
		r.planLocked(2, a.key, a.charge, a.local, a)
		return false
	}
	if !ctxOK && optsOK && !a.local && !a.base {
		// a remote floor cannot raise the others': unserved until a start gives it room
		a.unserved, a.short = floorWhy(m.ctx, a.prof.CtxPerSlot.Floor), true
		return false
	}
	switch {
	case optsOK && ctxOK && ownReturn:
		a.slots, resumed = own, true
		r.unplanLocked(a.key) // a reshape its leave asked for is not needed: it is back
	case optsOK && ctxOK && len(free) >= a.prof.Slots:
		pick := append([]int{}, own...)
		for _, e := range free {
			if len(pick) == a.prof.Slots {
				break
			}
			if !slices.Contains(pick, e) {
				pick = append(pick, e)
			}
		}
		pick = pick[:a.prof.Slots]
		sort.Ints(pick)
		a.slots = pick
	default:
		r.planLocked(2, a.key, a.charge, a.local, a)
		return false
	}
	for _, e := range a.slots {
		m.owner[e], m.keys[e], m.want[e] = a.id, a.key, a.prof.Slots
	}
	return resumed
}

// watchPid follows an attachment's pid: its end is the attachment's leave (gen: a later attach
// of the same key that brought another pid makes this watch's word stale). The answer stops the
// watch and ends its goroutine (a replace, a detach, a leave): host's stop does not close ended.
func (s *Supervisor) watchPid(id string, gen uint64, ended <-chan struct{}, stop func()) func() {
	done := make(chan struct{})
	var once sync.Once
	stopAll := func() {
		once.Do(func() {
			close(done)
			stop()
		})
	}
	go func() {
		select {
		case <-ended:
			s.leavePid(id, gen)
		case <-s.root.Done():
		case <-done:
		}
		stopAll()
	}()
	return stopAll
}

func (s *Supervisor) leavePid(id string, gen uint64) {
	s.leaveIf(id, func(a *attachment) bool {
		if a.pidGen != gen || a.pid == 0 {
			return false
		}
		a.pid = 0
		return true
	}, "")
}

// LeaveHold is a held attachment's connection ending (gen: the hold it was).
func (s *Supervisor) LeaveHold(id string, gen uint64) {
	s.leaveIf(id, func(a *attachment) bool {
		if !a.hasHold || a.holdGen != gen {
			return false
		}
		a.hasHold, a.holdEnd = false, nil
		return true
	}, "")
}

// Detach is DELETE attach/{id}: identity must own it (loopback may detach any).
func (s *Supervisor) Detach(id, identity string) *ClientError {
	r := &s.clients
	r.mu.Lock()
	var a *attachment
	for _, x := range r.list {
		if x.id == id && !x.base {
			a = x
		}
	}
	if a == nil {
		r.mu.Unlock()
		return cerr(404, "attachment_unknown", "no attachment %s", id)
	}
	if identity != identityLoopback && a.identity != identity {
		r.mu.Unlock()
		return cerr(403, "not_yours", "attachment %s is another client's", id)
	}
	r.mu.Unlock()
	s.leaveIf(id, func(*attachment) bool { return true }, "detached")
	return nil
}

// DetachRevoked ends at once every attachment of a remote identity whose token no longer stands
// (valid says false; cred is the master token's hash a master@<ip> attachment came with, so a
// new master token ends those): its held stream ends with {"end":"revoked"}, its slots go vacant, its fact
// is said gone. The person's act: no restart is counted against anyone.
func (s *Supervisor) DetachRevoked(valid func(identity, cred string) bool) {
	r := &s.clients
	type who struct{ id, identity, cred string }
	r.mu.Lock()
	r.revokes++ // an attach whose token was read before this reads it again
	var list []who
	for _, a := range r.list {
		if !a.base && !a.local {
			list = append(list, who{a.id, a.identity, a.cred})
		}
	}
	r.mu.Unlock()
	for _, w := range list { // the token store is read outside the clients lock
		if !valid(w.identity, w.cred) {
			s.leaveIf(w.id, func(a *attachment) bool { return a.cred == w.cred }, "revoked")
		}
	}
}

// leaveIf removes the attachment when ok says the word that came ends its presence (with pid and
// hold both, the first to end ends it). Its slots stay in the running mind, vacant. A leave is
// never charged against a budget.
func (s *Supervisor) leaveIf(id string, ok func(*attachment) bool, endWord string) {
	r := &s.clients
	r.mu.Lock()
	idx := -1
	for i, a := range r.list {
		if a.id == id && !a.base {
			idx = i
		}
	}
	if idx < 0 {
		r.mu.Unlock()
		return
	}
	a := r.list[idx]
	if !ok(a) {
		r.mu.Unlock()
		return
	}
	r.list = slices.Delete(r.list, idx, idx+1)
	delete(r.excl, a.id)
	if m := r.run; m != nil {
		for _, e := range a.slots {
			if e < len(m.owner) && m.owner[e] == a.id {
				m.owner[e] = ""
			}
		}
	}
	a.slots = nil
	end, stop := a.holdEnd, a.pidStop
	a.holdEnd, a.pidStop = nil, nil
	if r.run != nil {
		last := a.wanting
		for _, x := range r.list {
			if x.wanting && !x.base {
				last = false
			}
		}
		switch {
		case last:
			// the last client with wants: nothing reshapes. Its range stays vacant for its return;
			// the model stopping for another reason, or an under-served caller's request, reshapes
			r.stale = true
		case a.local && r.givesLocked():
			// a local caller's leave is the machine's own word: a reshape that gives the others
			// something, when idle, charged to no one
			r.planLocked(1, a.key, "", true, nil)
		default:
			// a remote leave never reshapes by itself (no remote act restarts the model uncharged):
			// an attachment that waits for room is folded in by its own charged plan, else at the
			// next start made for another reason
			r.planWaitersLocked()
		}
	}
	r.mu.Unlock()
	if end != nil && endWord != "" {
		end <- endWord
	}
	if stop != nil {
		stop()
	}
	s.logf("clients: %s (%s) left", a.name, a.id)
	s.sayGone("attachment", a.id)
	s.sayClients()
	s.SayClientAttachments(a.identity)
	s.agentChanged()
	s.wakeMind(func(c *causes) { c.clients = true })
}

// planWaitersLocked asks a reshape, when idle, for each attachment left out or waiting for room
// (not one below its floor), on its own behalf and charged to its own budget: a waiter whose
// budget is spent waits for the next start made for another reason.
func (r *clientReg) planWaitersLocked() {
	for _, a := range r.list {
		if a.wanting && !a.base && len(a.slots) == 0 && !a.short && r.plans[a.key].level == 0 {
			r.planLocked(1, a.key, budgetKey(a.identity, a.local), a.local, a)
		}
	}
}

// givesLocked: a reshape now would give someone something (an attachment left out or waiting
// served, an option handed over, more context to those below their max). Otherwise the vacant
// slots stay until the next start for another reason.
func (r *clientReg) givesLocked() bool {
	m := r.run
	for _, a := range r.list {
		if a.wanting && len(a.slots) == 0 && !a.short {
			return true // left out, or waiting for a grow (one below its floor is not served by one)
		}
	}
	_, vals, _ := launchOpts(r.list, m.help)
	if !mapsEqual(vals, m.opts) {
		return true
	}
	for _, a := range r.list {
		if a.wanting && !a.short && m.ctx < a.prof.CtxPerSlot.Max {
			return true
		}
	}
	return false
}

// EndHolds ends every held attachment stream (the engine is ending).
func (s *Supervisor) EndHolds() {
	r := &s.clients
	r.mu.Lock()
	var ends []chan string
	for _, a := range r.list {
		if a.holdEnd != nil {
			ends = append(ends, a.holdEnd)
			a.holdEnd = nil
		}
	}
	r.mu.Unlock()
	for _, e := range ends {
		select {
		case e <- "engine_ending":
		default:
		}
	}
}

// ClientView is an attachment's view now (the held stream's next line).
func (s *Supervisor) ClientView(id string) (AttachView, bool) {
	model := s.selectedID()
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, a := range r.list {
		if a.id == id {
			return r.viewLocked(a, model), true
		}
	}
	return AttachView{}, false
}

// Attached is the count of standing attachments (the base not counted): the resident app's
// clients beside the pids and the door.
func (s *Supervisor) Attached() int {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, a := range r.list {
		if !a.base {
			n++
		}
	}
	return n
}

// ---- the door's side ---------------------------------------------------------------------

// Touch is what a request does with the model server's slots (the API reads it from the
// request's path, api/paths.go).
type Touch int

const (
	// TouchNone: no slot (health, models, tokenize, GET /slots).
	TouchNone Touch = iota
	// TouchSlot: one slot, named by the body's id_slot or the /slots/{id} path: confinable.
	TouchSlot
	// TouchUnconfinable: a slot the server picks itself, every slot, or a body the door cannot
	// rewrite; and any POST the table does not know.
	TouchUnconfinable
)

// DoorAsk is a request at the door: its id, the attachment it names (X-Easy-Client, "" none),
// who sent it, whether from this machine, and what it touches.
type DoorAsk struct {
	ReqID, Att, Identity string
	Local                bool
	Touch                Touch
}

// ClientDoor is what the door confines a request by. Slots nil: not confined (passed as it is).
// Own: Slots is an attachment's range in the client's numbering (index i is its slot i); else
// it is a range in the engine's numbering (the commons) and a slot outside it is refused.
type ClientDoor struct {
	ID    string
	Key   string // the attachment's key (its name space for conversation ids), "" none
	Slots []int
	Own   bool
	Err   *ClientError
	// Tag is whose prompt a confined request leaves in its slot (TakeSlot erases a slot that
	// holds another's first).
	Tag SlotTag
}

// clientDoorLocked decides who a request belongs to (docs/ENGINE.md section 7g, "Whose slots"):
//  1. X-Easy-Client names an attachment of the caller's identity: that attachment (else 403
//     not_yours; a local caller may name the base).
//  2. no header, a remote caller, one attachment of its own with slots: that one; more: 400
//     client_required.
//  3. none of its own with slots: the commons, the base's range when the base has one.
//  4. no base: the commons is the slots never given to an attachment.
//  5. no commons: a caller of this machine is held while the engine reshapes to give the
//     commons a range; any other gets 409 attach_required.
//
// The invariant (docs/ENGINE.md §7g): a request runs only in slots of its own range, its
// attachment's or the commons; while the map holds any attachment's range, standing or vacant,
// no caller is run unconfined; a slot changes hands only through an erase (TakeSlot). Nothing is
// confined only while the map holds no attachment's range and no untrusted caller's prompt, and
// only for a trusted caller (this machine, the master key). A request that touches no slot is
// only attributed; one that cannot be confined is refused while the map holds an attachment's
// range or an untrusted prompt, and always to an untrusted caller. Caller holds s.mu (lock order: s.mu, then
// clients.mu). pending: it wants slots the running mind has not given it yet (a grow comes).
func (s *Supervisor) clientDoorLocked(q DoorAsk) (d ClientDoor, pending bool) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	var a *attachment
	if q.Att != "" {
		for _, x := range r.list {
			if x.id == q.Att {
				a = x
			}
		}
		if a == nil {
			return ClientDoor{Err: cerr(404, "attachment_unknown", "no attachment %s (it left, or the engine restarted): attach again", q.Att)}, false
		}
		if a.identity != q.Identity && q.Identity != "engine" && !(a.base && q.Local) {
			return ClientDoor{Err: cerr(403, "not_yours", "attachment %s is another client's", q.Att)}, false
		}
		d.ID, d.Key = a.id, a.key
	}
	m := r.run
	trusted := trustedIdentity(q.Identity, q.Local)
	switch q.Touch {
	case TouchNone:
		return d, false
	case TouchUnconfinable:
		// the server picks the slot (or takes every slot): it cannot be kept to a range, nor
		// can its slot be erased first. Run only for a trusted caller while the map holds no
		// attachment's range and no untrusted caller's prompt; every slot is then the commons'.
		switch {
		case m != nil && r.claimedLocked(m):
			d.Err = cerr(403, "path_not_offered", "this path lets the model server pick the slot, and attached clients hold slots: it cannot be kept to this caller's own slots")
		case m != nil && !trusted:
			d.Err = cerr(403, "path_not_offered", "this path lets the model server pick the slot, which may hold another caller's prompt: it is offered to this machine's callers only")
		case m != nil:
			for e := range m.tags {
				m.tags[e] = SlotTag{Who: "commons", Trusted: true}
			}
		}
		return d, false
	}
	own := func(a *attachment) (ClientDoor, bool) {
		d.ID, d.Key = a.id, a.key
		d.Tag = SlotTag{Who: a.key, Trusted: trustedIdentity(a.identity, a.local)}
		switch {
		case len(a.slots) > 0:
			d.Slots, d.Own = append([]int{}, a.slots...), true
			return d, false
		case a.unserved != "":
			d.Err = cerr(503, "no_room", "%s", a.unserved)
			return d, false
		case m != nil && r.plans[a.key].level > 0:
			// a reshape for it is asked: the start that serves it (or the stop) ends the hold
			return d, true
		}
		// nothing is coming for it: answered at once, never held without an event to end it
		why := a.why
		if why == "" {
			why = "no restart is coming for it"
		}
		d.Err = cerr(503, "no_room", "this client has no slots in the running model: %s", why)
		d.Err.Extra = map[string]any{"why": why}
		return d, false
	}
	if a != nil && a.wanting && !a.base {
		return own(a)
	}
	// the last client left a shape grown for it (stale): a local caller with no slots of its own
	// that the shape under-serves is held while the engine reshapes, as a local grow is
	if m != nil && r.stale && q.Local && q.Touch != TouchNone && s.underServedLocked() {
		r.planLocked(2, commonsKey, "", true, nil)
		s.wakeMind(func(c *causes) { c.clients = true })
		return d, true
	}
	// rule 2, before anything else is looked at: a remote caller's own attachment with wants is
	// its, held or answered as one (it is never served beside the others as if anonymous)
	if q.Att == "" && !q.Local && m != nil {
		var mine []*attachment
		for _, x := range r.list {
			if !x.base && x.wanting && x.identity == q.Identity {
				mine = append(mine, x)
			}
		}
		switch {
		case len(mine) == 1:
			return own(mine[0])
		case len(mine) > 1:
			d.Err = cerr(400, "client_required", "this client holds %d attachments with slots: say which in %s", len(mine), "X-Easy-Client")
			return d, false
		}
	}
	if m == nil {
		return d, false // no model runs: the route holds or answers it
	}
	if trusted && !r.claimedLocked(m) {
		return d, false // nothing is confined: as before clients
	}
	d.Tag = SlotTag{Who: "commons", Trusted: true}
	if !trusted {
		d.Tag = SlotTag{Who: "anon:" + q.Identity}
		if m.ramCache {
			// the RAM prompt cache is one for all slots and an erase keeps it (only a request
			// forwarded to the loopback listener gets here): an untrusted caller is served only
			// in its own attachment's range, whose attach restarts the server without the cache
			d.Err = cerr(409, "attach_required", "this caller has no slots of its own in the running model: attach to be given slots")
			return d, false
		}
	}
	baseKey := ""
	for _, x := range r.list {
		if x.base {
			baseKey = x.key
			if len(x.slots) > 0 {
				d.Slots = append([]int{}, x.slots...) // the commons
				return d, false
			}
		}
	}
	for e := range m.owner {
		if m.keys[e] == "" || m.keys[e] == baseKey {
			d.Slots = append(d.Slots, e) // never given to an attachment: the commons
		}
	}
	switch {
	case d.Slots != nil:
		return d, false
	case q.Local && r.commonsShort:
		d.Err = cerr(503, "no_room", "the model has no room for this machine's requests beside the attached clients")
		return d, false
	case q.Local:
		// the commons has no range: held while the engine reshapes to give it one, as a local
		// grow is (never run in an attachment's slots)
		r.commons = true
		r.planLocked(2, commonsKey, "", true, nil)
		s.wakeMind(func(c *causes) { c.clients = true })
		return d, true
	}
	d.Err = cerr(409, "attach_required", "this caller has no slots of its own in the running model: attach to be given slots")
	return d, false
}

// claimedLocked: the map holds an attachment's range (standing or vacant, the base's aside) or a
// slot that holds an untrusted caller's prompt: then every request is confined.
func (r *clientReg) claimedLocked(m *runMap) bool {
	baseKey := ""
	for _, x := range r.list {
		if x.base {
			baseKey = x.key
		}
	}
	for e := range m.keys {
		if (m.keys[e] != "" && m.keys[e] != baseKey) || (e < len(m.tags) && !m.tags[e].Trusted && m.tags[e].Who != "") {
			return true
		}
	}
	return false
}

// underServedLocked: with no client standing, the running shape gives the base (or, with no base,
// the defaults) less than the wants that stand would: the base has no range in it, or the shape
// has more slots than those wants and less context per slot than their max.
func (s *Supervisor) underServedLocked() bool {
	r := &s.clients
	m := r.run
	def := s.cfg.Profile
	for _, a := range r.list {
		if a.base {
			if len(a.slots) == 0 {
				return true
			}
			def = a.prof
		}
	}
	return len(m.owner) > def.Slots && m.ctx < def.CtxPerSlot.Max
}

// EnterAs is Enter for a request whose attachment and touch the door has read. A request whose
// attachment waits for its slots is held like one the model is on its way up for.
func (s *Supervisor) EnterAs(q DoorAsk) (Route, ClientDoor, func(*RequestEnd)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rt := s.routeLocked()
	if rt.Hold {
		return rt, ClientDoor{}, nil
	}
	var d ClientDoor
	if rt.Kind == "local" {
		var pending bool
		d, pending = s.clientDoorLocked(q)
		if pending {
			return Route{Kind: "unavailable", Hold: true, Message: "Making room for this client"}, d, nil
		}
		if d.Err != nil {
			return rt, d, nil
		}
	} else if q.Att != "" {
		q.Touch = TouchNone // no slots of ours elsewhere: only whose it is
		d, _ = s.clientDoorLocked(q)
		if d.Err != nil {
			return rt, d, nil
		}
	}
	return rt, d, s.beginAs(q.ReqID, d.ID)
}

// slotWaiter is one request waiting in a slot's line.
type slotWaiter struct{}

// wakeLineLocked tells every waiter in a slot's line to look again. Caller holds r.mu.
func (r *clientReg) wakeLineLocked() {
	if r.lineWake != nil {
		close(r.lineWake)
	}
	r.lineWake = make(chan struct{})
}

// TakeSlot is TakeSlotFor with no owner named and nothing to erase (tests of the pick alone).
func (s *Supervisor) TakeSlot(slots []int, want int) (e int, release func()) {
	e, release, _ = s.TakeSlotFor(context.Background(), slots, want, SlotTag{}, nil, nil)
	return e, release
}

// TakeSlotFor is the engine slot a confined request goes to, marked busy until release: want (an
// engine slot) when it names one, else, of slots, a free one with no line, holding tag's own
// prompt first, then one it may use without an erase, then the least recently used; else a busy
// one by the same order (the model server queues it there).
//
// The engine orders every handover (docs/ENGINE.md §7g, "The invariant"; never llama-server,
// whose erase and completion are separate tasks it may reorder: server-queue.cpp:90-112 pops a
// deferred task naming the released slot before an erase, whose id_slot is -1):
//   - a slot that holds another owner's prompt is handed over only when no request the door sent
//     to it is still running; the newcomer waits in the engine for that (each release is the
//     event), then erase(e) is sent and answered, and only on its success is the slot labelled
//     tag's;
//   - while a handover is waiting or under way on a slot, every other request for that slot,
//     whoever sends it, waits behind it in arrival order;
//   - held (non-nil: a remote caller) counts a request that must wait against its client's share
//     of held requests; a request whose context ends while waiting leaves the line.
//
// Errors: 502 slot_not_cleared (the erase failed: the slot is labelled anyone's, "?", so every
// next owner erases it), 503 slot_lost (the model server restarted while it waited), 499
// client_gone, or held's refusal.
func (s *Supervisor) TakeSlotFor(ctx context.Context, slots []int, want int, tag SlotTag, erase func(e int) bool, held func() (func(), *ClientError)) (e int, release func(), ce *ClientError) {
	r := &s.clients
	r.mu.Lock()
	gen := r.gen
	m := r.run
	cur := func(x int) SlotTag {
		if m == nil || x < 0 || x >= len(m.tags) {
			return SlotTag{}
		}
		return m.tags[x]
	}
	hand := func(x int) bool { return tag.Who != "" && needsErase(cur(x), tag) }
	rank := func(x int) int { // lower is better
		c := cur(x)
		switch {
		case len(r.line[x]) > 0:
			return 3
		case tag.Who != "" && c.Who == tag.Who:
			return 0
		case !hand(x):
			return 1
		}
		return 2
	}
	e = want
	if e < 0 {
		best, bestFree, bestRank := -1, false, 0
		for _, x := range slots {
			free, rk := r.busy[x] == 0, rank(x)
			switch {
			case best < 0,
				free && !bestFree && rk < 3,
				free == bestFree && rk < bestRank,
				free == bestFree && rk == bestRank && r.used[x] < r.used[best]:
				best, bestFree, bestRank = x, free, rk
			}
		}
		e = best
	}
	take := func() func() { // caller holds r.mu
		if tag.Who != "" && m != nil && e >= 0 && e < len(m.tags) {
			m.tags[e] = tag
		}
		r.tick++
		r.used[e] = r.tick
		r.busy[e]++
		var once sync.Once
		return func() {
			once.Do(func() {
				r.mu.Lock()
				if r.gen == gen && r.busy[e] > 0 {
					r.busy[e]--
					r.wakeLineLocked()
				}
				r.mu.Unlock()
			})
		}
	}
	if len(r.line[e]) == 0 && !hand(e) {
		release = take()
		r.mu.Unlock()
		return e, release, nil
	}
	// it waits in the engine: behind the line, for the slot's requests to end, for the erase
	w := &slotWaiter{}
	r.line[e] = append(r.line[e], w)
	s.sayLineLocked(e)
	leave := func() { // caller holds r.mu
		if r.gen != gen {
			return // the line is gone with its map
		}
		l := r.line[e]
		for i, x := range l {
			if x == w {
				r.line[e] = append(l[:i:i], l[i+1:]...)
				break
			}
		}
		if len(r.line[e]) == 0 {
			delete(r.line, e)
		}
		r.wakeLineLocked()
		s.sayLineLocked(e)
	}
	var unheld func()
	if held != nil {
		r.mu.Unlock()
		u, he := held()
		r.mu.Lock()
		if he != nil {
			leave()
			r.mu.Unlock()
			return 0, nil, he
		}
		unheld = u
	}
	defer func() {
		if unheld != nil {
			unheld()
		}
	}()
	for {
		switch {
		case r.gen != gen || r.run == nil:
			r.mu.Unlock()
			return 0, nil, cerr503Lost()
		case r.line[e][0] == w && !hand(e):
			release = take()
			leave()
			r.mu.Unlock()
			return e, release, nil
		case r.line[e][0] == w && r.busy[e] == 0:
			// nothing the door sent there runs: erase, then label (never before the erase answered)
			r.busy[e]++ // the erase is in flight on it
			r.mu.Unlock()
			ok := erase != nil && erase(e)
			r.mu.Lock()
			if r.gen != gen || r.run == nil {
				r.mu.Unlock()
				return 0, nil, cerr503Lost()
			}
			r.busy[e]--
			if !ok {
				if e < len(m.tags) {
					m.tags[e] = SlotTag{Who: "?"} // anyone's: every next owner erases it first
				}
				leave()
				r.mu.Unlock()
				return 0, nil, cerr(502, "slot_not_cleared", "the slot this request was given holds another caller's prompt and the model server did not clear it: send again")
			}
			release = take()
			leave()
			r.mu.Unlock()
			return e, release, nil
		}
		wake := r.lineWake
		r.mu.Unlock()
		select {
		case <-wake:
		case <-ctx.Done():
			r.mu.Lock()
			leave()
			r.mu.Unlock()
			return 0, nil, cerr(499, "client_gone", "the client went away while its request waited for its slot")
		}
		r.mu.Lock()
	}
}

// sayLineLocked says how many requests wait for slot e now (the request happening, what
// slot_wait). Caller holds r.mu; the hub takes no lock of ours.
func (s *Supervisor) sayLineLocked(e int) {
	s.say("request", "", map[string]any{"what": "slot_wait", "slot": e, "waiting": len(s.clients.line[e])})
}

// SlotsWaiting is how many requests wait in the engine for a slot's handover, all slots.
func (s *Supervisor) SlotsWaiting() int {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, l := range r.line {
		n += len(l)
	}
	return n
}

// SlotsShownTo is what a caller may read of each slot in the model server's GET /slots: own(e)
// true for a slot whose tag is the caller's own (its attachments' keys; the commons for a trusted
// caller, who trust each other; its anon: tag for a remote caller with none), or one that holds
// no one's prompt. While every slot holding a prompt has one owner, nothing is hidden. A slot's
// prompt and generated text and its token counts are otherwise another's (docs/ENGINE.md §7g).
func (s *Supervisor) SlotsShownTo(identity string, local bool) (own func(e int) bool) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.run
	if m == nil {
		return func(int) bool { return true }
	}
	tags := append([]SlotTag{}, m.tags...)
	trusted := trustedIdentity(identity, local)
	mine := map[string]bool{"anon:" + identity: true}
	if trusted {
		mine[commonsWho] = true
	}
	for _, a := range r.list {
		if a.identity == identity || (local && a.local) || (a.base && trusted) {
			mine[a.key] = true
		}
	}
	owners := map[SlotTag]bool{}
	for _, t := range tags {
		if t.Who != "" {
			owners[t] = true
		}
	}
	one := len(owners) <= 1 && !owners[SlotTag{Who: "?"}]
	return func(e int) bool {
		if one || e < 0 || e >= len(tags) {
			return one
		}
		t := tags[e]
		return t.Who == "" || mine[t.Who]
	}
}

func cerr503Lost() *ClientError {
	return cerr(503, "slot_lost", "the model server restarted while this request waited for its slot: send it again")
}

// ClientSamplers is an attachment's own sampler profiles (looked up before the base's and the
// model's).
func (s *Supervisor) clientProfile(id string) (Profile, bool) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, a := range r.list {
		if a.id == id {
			return a.prof, true
		}
	}
	return Profile{}, false
}

// ---- facts -----------------------------------------------------------------------------

// WantsInfo is the wants fact and status wants.
type WantsInfo struct {
	Slots    int               `json:"slots"`
	MinSlots int               `json:"min_slots"`
	CtxMax   int               `json:"ctx_max"`
	CtxFloor int               `json:"ctx_floor"`
	From     []string          `json:"from"`
	Unserved []UnservedInfo    `json:"unserved"`
	Vacant   []VacantInfo      `json:"vacant"`
	Server   map[string]string `json:"server"`
	// RAMCache: the running server keeps llama-server's RAM prompt cache (one owner of the slots)
	RAMCache bool `json:"ram_cache"`
}

// UnservedInfo is an attachment that wants slots and has none in the running mind, and why.
type UnservedInfo struct {
	Attachment string `json:"attachment"`
	Why        string `json:"why"`
}

// VacantInfo is a range of the running mind whose attachment left (key: who had it).
type VacantInfo struct {
	First int    `json:"first"`
	Count int    `json:"count"`
	Key   string `json:"key"`
}

func (s *Supervisor) wantsInfo() WantsInfo {
	w, from := s.mergedWants()
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	wi := WantsInfo{Slots: w.Slots, MinSlots: w.MinSlots, CtxMax: w.CtxMax, CtxFloor: w.CtxFloor, From: from,
		Unserved: []UnservedInfo{}, Vacant: []VacantInfo{}, Server: map[string]string{}}
	if wi.From == nil {
		wi.From = []string{}
	}
	for _, a := range r.list {
		if !a.wanting || len(a.slots) > 0 {
			continue
		}
		switch {
		case a.unserved != "":
			wi.Unserved = append(wi.Unserved, UnservedInfo{a.id, a.unserved})
		case r.run != nil && r.plans[a.key].level > 0:
			wi.Unserved = append(wi.Unserved, UnservedInfo{a.id, "waiting for the model to make room (a reshape when the replies in flight end)"})
		case r.run != nil:
			why := a.why
			if why == "" {
				why = "no restart is coming for it"
			}
			wi.Unserved = append(wi.Unserved, UnservedInfo{a.id, why})
		}
	}
	if m := r.run; m != nil {
		for e := 0; e < len(m.owner); e++ {
			if m.owner[e] != "" || m.keys[e] == "" {
				continue
			}
			if n := len(wi.Vacant); n > 0 && wi.Vacant[n-1].Key == m.keys[e] && wi.Vacant[n-1].First+wi.Vacant[n-1].Count == e {
				wi.Vacant[n-1].Count++
				continue
			}
			wi.Vacant = append(wi.Vacant, VacantInfo{First: e, Count: 1, Key: m.keys[e]})
		}
		for k, v := range m.opts {
			wi.Server[k] = v
		}
		wi.RAMCache = m.ramCache
	}
	return wi
}

// Attachments is status attachments.
func (s *Supervisor) attachmentsInfo() []AttachView {
	model := s.selectedID()
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	out := []AttachView{}
	for _, a := range r.list {
		out = append(out, r.viewLocked(a, model))
	}
	return out
}

// sayClients says the wants fact and every attachment's fact (the hub says only what changed).
func (s *Supervisor) sayClients() {
	if s.hub == nil {
		return
	}
	s.say("wants", "", s.wantsInfo())
	for _, v := range s.attachmentsInfo() {
		s.say("attachment", v.Attachment, v)
	}
}

// sayClientAttachments says a paired client's attachments in its client fact, where that fact
// stands (the gate says it the first time it accepts the client's token in a life).
func (s *Supervisor) SayClientAttachments(identity string) {
	if s.hub == nil || identity == identityLoopback || identity == "" {
		return
	}
	f, ok := s.hub.Fact("client", identity)
	if !ok {
		return
	}
	body := map[string]any{}
	_ = json.Unmarshal(f.Body, &body)
	ids := []string{}
	s.clients.mu.Lock()
	for _, a := range s.clients.list {
		if a.identity == identity {
			ids = append(ids, a.id)
		}
	}
	s.clients.mu.Unlock()
	body["attachments"] = ids
	s.say("client", identity, body)
}

// ClientAttachments lists a client identity's standing attachments (the client fact).
func (s *Supervisor) ClientAttachments(identity string) []string {
	ids := []string{}
	s.clients.mu.Lock()
	defer s.clients.mu.Unlock()
	for _, a := range s.clients.list {
		if a.identity == identity {
			ids = append(ids, a.id)
		}
	}
	return ids
}

// cleanName is a client's name as shown: control and format characters dropped, spaces folded,
// at most 40 characters (as pairing names are).
func cleanName(s string) string {
	var b strings.Builder
	space := false
	n := 0
	for _, r := range s {
		if r < 0x20 || r == 0x7f || (r >= 0x80 && r < 0xa0) || r == 0x200b || r == 0x200e || r == 0x200f || (r >= 0x202a && r <= 0x202e) || (r >= 0x2066 && r <= 0x2069) || r == 0xfeff {
			continue
		}
		if r == ' ' || r == '\t' {
			space = true
			continue
		}
		if r == '/' {
			r = '-' // the key is identity/name
		}
		if space && b.Len() > 0 {
			b.WriteByte(' ')
			n++
		}
		space = false
		if n >= 40 {
			break
		}
		b.WriteRune(r)
		n++
	}
	return b.String()
}

// ---- the one agent -------------------------------------------------------------------------

// agentSpec is the agent the engine runs: the profile's (the base, as before), else the first
// attachment in order that brings one (the conflict rule: one agent; a later one is not taken).
func (s *Supervisor) agentSpec() *AgentSpec {
	if a := s.cfg.Profile.Agent; a != nil {
		return a
	}
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	if h := r.agentHolderLocked(); h != nil {
		return h.prof.Agent
	}
	return nil
}

func (r *clientReg) agentHolderLocked() *attachment {
	for _, a := range r.list {
		if a.prof.Agent != nil && a.prof.Agent.Wheel != "" {
			return a
		}
	}
	return nil
}

// agentKey is the agent fact's key: the profile's game name (else "agent"), as before, or the
// name of the attachment whose agent runs.
func (s *Supervisor) agentKey() string {
	if s.cfg.Profile.Agent != nil || s.cfg.ProfileGiven {
		if s.cfg.Profile.Game != "" {
			return s.cfg.Profile.Game
		}
		return "agent"
	}
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	if h := r.agentHolderLocked(); h != nil {
		return h.name
	}
	return "agent"
}

// agentChanged restarts the agent when the attachment whose agent runs changed (it came, or it
// left and the next one's is taken). The profile's own agent never changes this way.
func (s *Supervisor) agentChanged() {
	if s.cfg.Profile.Agent != nil {
		return
	}
	r := &s.clients
	r.mu.Lock()
	cur := ""
	if h := r.agentHolderLocked(); h != nil {
		cur = h.id + "|" + h.prof.Agent.ID + "@" + h.prof.Agent.Wheel
	}
	changed := cur != r.agentRun
	r.agentRun = cur
	r.mu.Unlock()
	if !changed || s.cfg.NoAutoStart {
		return
	}
	s.agent.mu.Lock()
	running := s.agent.cancel != nil
	s.agent.mu.Unlock()
	if !running && cur == "" {
		return
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.restartAgentInstalled()
	}()
}

// cachedHelp is the installed build's help when it has been read already; it never runs the
// binary (nil: not read yet).
func (s *Supervisor) cachedHelp() *Help {
	in, ok := s.resolveInstall()
	if !ok || in.bin == "" {
		return nil
	}
	if v, ok := helpCache.Load(in.bin); ok {
		h, _ := v.(*Help)
		return h
	}
	return nil
}

// buildHelp is the installed llama-server's --help (nil: no build installed, or it could not be
// read): never anything a client names is run.
func (s *Supervisor) buildHelp() *Help {
	in, ok := s.resolveInstall()
	if !ok || in.bin == "" {
		return nil
	}
	return helpOf(in.bin, s.cfg.ExtraEnv)
}

// ---- contexts (run/context.go) -----------------------------------------------------------

// slotOwner is the key an engine slot of the running mind was given to and its number in that
// key's own numbering ("" and nil: no client owns it).
func (s *Supervisor) slotOwner(e int) (string, *int) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.run
	if m == nil || e < 0 || e >= len(m.keys) || m.keys[e] == "" {
		return "", nil
	}
	k := m.keys[e]
	n := 0
	for i := 0; i < e; i++ {
		if m.keys[i] == k {
			n++
		}
	}
	return k, &n
}

// commonsWho is the tag's owner of a slot that holds the trusted callers' prompt (this
// machine's and the master key's, Trusted: they trust each other).
const commonsWho = "commons"

// slotTag is whose prompt engine slot e of the running mind holds (zero: none, or no map).
func (s *Supervisor) slotTag(e int) SlotTag {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	if m := r.run; m != nil && e >= 0 && e < len(m.tags) {
		return m.tags[e]
	}
	return SlotTag{}
}

// keptTag: a slot with this tag may be saved across a restart: an attachment's prompt, or the
// commons' (trusted). Never an untrusted caller's in the commons ("anon:"), a slot whose erase
// failed ("?"), or an empty one.
func keptTag(t SlotTag) bool {
	switch {
	case t.Who == "", t.Who == "?", strings.HasPrefix(t.Who, "anon:"):
		return false
	case t.Who == commonsWho:
		return t.Trusted
	}
	return true
}

// tagWords says a tag for the log.
func tagWords(t SlotTag) string {
	switch {
	case t.Who == "":
		return "no one's prompt"
	case t.Who == "?":
		return "a prompt whose erase failed"
	case strings.HasPrefix(t.Who, "anon:"):
		return "the prompt of " + strings.TrimPrefix(t.Who, "anon:") + ", a caller from another machine with no attachment"
	}
	return t.Who + "'s prompt"
}

// labelSlot records whose prompt engine slot e holds (the lay-back's restore of a kept slot).
func (s *Supervisor) labelSlot(e int, t SlotTag) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	if m := r.run; m != nil && e >= 0 && e < len(m.tags) {
		m.tags[e] = t
	}
}

// isBaseKey: key is the base attachment's (the --profile file's; its range is the commons).
func (s *Supervisor) isBaseKey(key string) bool {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, a := range r.list {
		if a.base {
			return a.key == key
		}
	}
	return false
}

// unownedSlot: engine slot e of the running mind was given to no client (true with no map).
func (s *Supervisor) unownedSlot(e int) bool {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.run
	return m == nil || e >= len(m.keys) || (m.keys[e] == "" && m.owner[e] == "")
}

// slotFor is the engine slot of the running mind that the standing attachment of key has under
// its own number n; why says, when there is none, why its saved slot is let go.
func (s *Supervisor) slotFor(key string, n int) (int, string) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, a := range r.list {
		if a.key != key {
			continue
		}
		if n < len(a.slots) {
			return a.slots[n], ""
		}
		return 0, fmt.Sprintf("the client has %d slot(s) now", len(a.slots))
	}
	return 0, "the client has left"
}

// clientHost is the first local attachment in order with a host pid (its profile's host_pid,
// else its own pid): the host app the reserve is measured for when nothing else names one.
// Other programs' use of the card is measured as others by the sense.
func (s *Supervisor) clientHost() (int, *plan.HostAdapter, int, bool) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, a := range r.list {
		if !a.local || a.base {
			continue
		}
		pid := a.prof.HostPID
		if pid == 0 {
			pid = a.pid
		}
		if pid > 0 {
			return pid, a.prof.HostAdapter, a.prof.GameReserveMiB, true
		}
	}
	return 0, nil, 0, false
}

// SamplerFor is Sampler for a request of an attachment: its own named profiles are looked up
// first, then the base's and the model's.
func (s *Supervisor) SamplerFor(att string) SamplerInfo {
	info := s.Sampler()
	if att == "" {
		return info
	}
	p, ok := s.clientProfile(att)
	if !ok || len(p.Samplers) == 0 {
		return info
	}
	prof := map[string]catalog.SamplerProfile{}
	for k, v := range info.Profiles {
		prof[k] = v
	}
	for k, v := range p.Samplers {
		prof[k] = v
	}
	info.Profiles = prof
	info.Active = len(prof) > 0
	if info.Active {
		info.Note = ""
	}
	return info
}
