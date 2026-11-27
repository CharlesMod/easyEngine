package run

import (
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
	report   OptReport
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
}

// clientReg is the attachments and the running map, on the Supervisor (no package state).
type clientReg struct {
	mu   sync.Mutex
	list []*attachment
	seq  uint64
	excl map[string]bool // left out of the merge for want of room, until the next start
	run  *runMap
	busy map[int]int // engine slot -> requests the door sent to it now
	// reclient: 0 nothing, 1 reshape when idle (a leave that gives someone something), 2 grow by
	// drain (an attach the running mind cannot serve). Read and cleared by the reactor.
	reclient int
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
	r.excl = map[string]bool{}
	r.busy = map[int]int{}
	if !s.cfg.ProfileGiven {
		return
	}
	name := s.cfg.Profile.Game
	if name == "" {
		name = baseKey
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
		if a.wanting && !r.excl[a.id] {
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
	r.mu.Unlock()
	if len(ws) == 0 {
		p := s.cfg.Profile
		return fit.Wants{Slots: p.Slots, MinSlots: p.MinSlots, CtxMax: p.CtxPerSlot.Max, CtxFloor: p.CtxPerSlot.Floor}, nil
	}
	for _, a := range ws {
		p := a.prof
		w.Slots += p.Slots
		w.MinSlots += p.MinSlots
		w.CtxMax = max(w.CtxMax, p.CtxPerSlot.Max)
		w.CtxFloor = max(w.CtxFloor, p.CtxPerSlot.Floor)
		from = append(from, a.id)
	}
	return w, from
}

// dropLatest leaves the latest wanting attachment out of the merge (the fit refused the merge
// even at its floor); false when one or none is left, which is then the engine's own refusal.
func (s *Supervisor) dropLatest() (string, bool) {
	r := &s.clients
	r.mu.Lock()
	ws := r.wantingLocked()
	if len(ws) < 2 {
		r.mu.Unlock()
		return "", false
	}
	last := ws[len(ws)-1]
	if last.base {
		r.mu.Unlock()
		return "", false
	}
	r.excl[last.id] = true
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
}

// clientOpts is the merged launch options of the standing attachments now.
func (s *Supervisor) clientOpts(h *Help) launchSet {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	args, vals, reports := launchOpts(r.list, h)
	return launchSet{args: args, vals: vals, reports: reports, help: h, seq: r.seq}
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
	r := &s.clients
	r.mu.Lock()
	var ws, late []*attachment
	for _, a := range r.wantingLocked() {
		if a.seq <= ls.seq {
			ws = append(ws, a)
		} else {
			late = append(late, a)
		}
	}
	m := &runMap{owner: make([]string, shape.Slots), keys: make([]string, shape.Slots), want: make([]int, shape.Slots),
		ctx: shape.CtxPerSlot, opts: ls.vals, args: ls.args, help: ls.help}
	if m.opts == nil {
		m.opts = map[string]string{}
	}
	for _, a := range r.list {
		a.slots, a.unserved = nil, ""
		if a.wanting && r.excl[a.id] {
			a.unserved = "the card has no room for it beside the attachments before it"
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
	r.run = m
	r.reclient = 0
	if len(late) > 0 {
		r.reclient = 2 // came during the start: the next look grows for it
	}
	clear(r.busy)
	r.mu.Unlock()
	s.sayClients()
	if len(late) > 0 {
		s.wakeMind(func(c *causes) { c.clients = true })
	}
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
	for _, a := range r.list {
		a.slots = nil
		a.poke()
	}
	clear(r.busy)
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
}

// SlotsView is an attachment's slots: what it wants, the least it takes, what the running mind
// gives it (its first engine slot is said for a reader of the logs; a client numbers its own).
type SlotsView struct {
	Want     int    `json:"want"`
	Min      int    `json:"min"`
	Given    int    `json:"given"`
	First    *int   `json:"first,omitempty"`
	Pending  bool   `json:"pending"`
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
			v.Slots.Pending = len(a.slots) == 0 && a.unserved == ""
		}
	}
	if mp := a.prof.Models; mp != nil && model != "" {
		mv := &ModelView{Selected: model, Usable: len(mp.CanUse) == 0 || slices.Contains(mp.CanUse, model)}
		if len(mp.Prefer) > 0 && !slices.Contains(mp.Prefer, model) {
			mv.Advice = "prefers " + strings.Join(mp.Prefer, ", ")
		}
		if !mv.Usable {
			mv.Advice = strings.TrimPrefix(mv.Advice+"; the selected model is not one it can use", "; ")
		}
		v.Model = mv
	}
	if a.prof.Agent != nil && a.prof.Agent.Wheel != "" {
		v.Agent = a.prof.Agent.ID
	}
	return v
}

func (s *Supervisor) selectedID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.st.Mode == "external" {
		return s.st.External.Model
	}
	return s.st.Model
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
	if req.Profile != nil {
		prof = *req.Profile
		h := s.buildHelp()
		if e := checkProfile(&prof, req.Local, h); e != nil {
			return AttachView{}, nil, e
		}
		wanting = req.Profile.Slots > 0
		prof.defaults()
	}
	key := req.Identity + "/" + name
	var st string
	if req.Pid > 0 {
		var err error
		if st, err = host.StartOf(req.Pid); err != nil && !errors.Is(err, host.ErrNoEvent) {
			return AttachView{}, nil, cerr(400, "bad_request", "no such process")
		}
	}
	h := s.buildHelp()
	model := s.selectedID()
	r := &s.clients
	r.mu.Lock()
	var old *attachment
	count, mine := 0, 0
	for _, a := range r.list {
		if a.key == key && !a.base {
			old = a
		}
		if !a.base {
			count++
			if a.identity == req.Identity {
				mine++
			}
		}
	}
	if old == nil && count >= maxAttachments {
		r.mu.Unlock()
		return AttachView{}, nil, cerr(429, "too_many_attachments", "this engine holds %d attachments already", maxAttachments)
	}
	if old == nil && !req.Local && mine >= maxPerIdentity {
		r.mu.Unlock()
		return AttachView{}, nil, cerr(429, "too_many_attachments", "this client holds %d attachments already", maxPerIdentity)
	}
	if wanting {
		sum := prof.Slots
		for _, a := range r.list {
			if a.wanting && a != old {
				sum += a.prof.Slots
			}
		}
		if sum > maxSlotsAll {
			r.mu.Unlock()
			return AttachView{}, nil, cerr(400, "bad_request", "the attachments would want %d slots; at most %d in all", sum, maxSlotsAll)
		}
	}
	cand := &attachment{id: newID(), name: name, identity: req.Identity, key: key, local: req.Local, prof: prof, wanting: wanting,
		notify: make(chan struct{}, 1)}
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
					e := cerr(409, "model_unsupported", "the selected model %q is not one this client can use; the selection is the person's", model)
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
		old.prof, old.wanting, old.local, old.report = prof, wanting, req.Local, rep
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
			s.planGrowLocked(a)
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
	var hold *AttachHold
	if req.Hold {
		a.holdGen++
		a.hasHold = true
		a.holdEnd = make(chan string, 1)
		hold = &AttachHold{ID: a.id, Gen: a.holdGen, End: a.holdEnd, Notify: a.notify}
	}
	if req.Pid > 0 {
		a.pid = req.Pid
		a.pidGen++
	}
	pidGen, id := a.pidGen, a.id
	v := r.viewLocked(a, model)
	r.mu.Unlock()
	if endOld != nil {
		endOld <- "replaced"
	}
	if stopOld != nil {
		stopOld()
	}
	if req.Pid > 0 {
		s.watchPid(id, req.Pid, st, pidGen)
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
		free = append(free, e)
		if m.keys[e] == a.key {
			own = append(own, e)
		}
	}
	optsOK := r.optsSameLocked(h)
	ctxOK := m.ctx >= a.prof.CtxPerSlot.Floor
	switch {
	case optsOK && ctxOK && len(own) == a.prof.Slots && m.want[own[0]] == a.prof.Slots:
		a.slots, resumed = own, true
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
		r.planGrowLockedR()
		return false
	}
	for _, e := range a.slots {
		m.owner[e], m.keys[e], m.want[e] = a.id, a.key, a.prof.Slots
	}
	return resumed
}

func (s *Supervisor) planGrowLocked(*attachment) { s.clients.planGrowLockedR() }

func (r *clientReg) planGrowLockedR() {
	if r.run != nil {
		r.reclient = 2
	}
}

// watchPid follows an attachment's pid: its end is the attachment's leave (gen: a later attach
// of the same key that brought another pid makes this watch's word stale).
func (s *Supervisor) watchPid(id string, pid int, start string, gen uint64) {
	ended, stop, err := host.ExitsOf(pid, start)
	switch {
	case errors.Is(err, host.ErrNotThatProcess):
		s.leavePid(id, gen)
		return
	case err != nil: // no word of its end on this system: it stays for the engine's life
		return
	}
	s.clients.mu.Lock()
	for _, a := range s.clients.list {
		if a.id == id && a.pidGen == gen {
			a.pidStop = stop
		}
	}
	s.clients.mu.Unlock()
	go func() {
		select {
		case <-ended:
			s.leavePid(id, gen)
		case <-s.root.Done():
		}
		stop()
	}()
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

// leaveIf removes the attachment when ok says the word that came ends its presence (with pid and
// hold both, the first to end ends it). Its slots stay in the running mind, vacant.
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
	if r.run != nil && r.reclient == 0 && r.givesLocked() {
		r.reclient = 1
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

// givesLocked: a reshape now would give someone something (an attachment left out or waiting
// served, an option handed over, more context to those below their max). Otherwise the vacant
// slots stay until the next start for another reason.
func (r *clientReg) givesLocked() bool {
	m := r.run
	for _, a := range r.list {
		if a.wanting && len(a.slots) == 0 {
			return true // left out, or waiting for a grow
		}
	}
	_, vals, _ := launchOpts(r.list, m.help)
	if !mapsEqual(vals, m.opts) {
		return true
	}
	for _, a := range r.list {
		if a.wanting && m.ctx < a.prof.CtxPerSlot.Max {
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

// ClientDoor is what the door maps a request by: the engine slots of the attachment it is
// for (in the client's numbering: index i is its slot i), or none (anonymous: passed through).
type ClientDoor struct {
	ID    string
	Slots []int
	Err   *ClientError
}

// clientDoorLocked resolves a request's attachment. Caller holds s.mu (lock order: s.mu, then
// clients.mu). pending: it wants slots the running mind has not given it yet (a grow comes).
func (s *Supervisor) clientDoorLocked(id, identity string) (d ClientDoor, pending bool) {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	if id == "" {
		// anonymous: the base's own when another attachment holds slots in the running mind
		m := r.run
		if m == nil || !m.base {
			return ClientDoor{}, false
		}
		var base *attachment
		others := false
		for _, a := range r.list {
			if a.base {
				base = a
			} else if len(a.slots) > 0 {
				others = true
			}
		}
		if base == nil || !others || len(base.slots) == 0 {
			return ClientDoor{}, false
		}
		return ClientDoor{ID: base.id, Slots: append([]int{}, base.slots...)}, false
	}
	var a *attachment
	for _, x := range r.list {
		if x.id == id {
			a = x
		}
	}
	if a == nil {
		return ClientDoor{Err: cerr(404, "attachment_unknown", "no attachment %s (it left, or the engine restarted): attach again", id)}, false
	}
	if identity != identityLoopback && identity != "engine" && a.identity != identity {
		return ClientDoor{Err: cerr(403, "not_yours", "attachment %s is another client's", id)}, false
	}
	if !a.wanting {
		return ClientDoor{ID: a.id}, false // states no slots: anonymous slots, as before
	}
	if len(a.slots) > 0 {
		return ClientDoor{ID: a.id, Slots: append([]int{}, a.slots...)}, false
	}
	if a.unserved != "" {
		e := cerr(503, "no_room", "%s", a.unserved)
		return ClientDoor{ID: a.id, Err: e}, false
	}
	return ClientDoor{ID: a.id}, r.run != nil
}

// EnterAs is Enter for a request that names its attachment (X-Easy-Client; "" none). A request
// whose attachment waits for its slots is held like one the model is on its way up for.
func (s *Supervisor) EnterAs(reqID, att, identity string) (Route, ClientDoor, func(*RequestEnd)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rt := s.routeLocked()
	if rt.Hold {
		return rt, ClientDoor{}, nil
	}
	var d ClientDoor
	if rt.Kind == "local" {
		var pending bool
		d, pending = s.clientDoorLocked(att, identity)
		if pending {
			return Route{Kind: "unavailable", Hold: true, Message: "Making room for this client"}, d, nil
		}
		if d.Err != nil {
			return rt, d, nil
		}
	} else if att != "" {
		d, _ = s.clientDoorLocked(att, identity)
		d.Slots = nil // no slots of ours elsewhere
		if d.Err != nil && d.Err.Code != "attachment_unknown" && d.Err.Code != "not_yours" {
			d.Err = nil
		}
		if d.Err != nil {
			return rt, d, nil
		}
	}
	return rt, d, s.beginAs(reqID, d.ID)
}

// PickSlot is the slot a request of an attachment that named none (or -1) goes to: its lowest
// slot the door has no request on, else its first. Release when the request ends.
func (s *Supervisor) PickSlot(slots []int) int {
	r := &s.clients
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, e := range slots {
		if r.busy[e] == 0 {
			return e
		}
	}
	return slots[0]
}

// Busy marks an engine slot with a request on it; the result unmarks it.
func (s *Supervisor) Busy(e int) func() {
	r := &s.clients
	r.mu.Lock()
	r.busy[e]++
	r.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			if r.busy[e] > 0 {
				r.busy[e]--
			}
			r.mu.Unlock()
		})
	}
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
		case r.run != nil:
			wi.Unserved = append(wi.Unserved, UnservedInfo{a.id, "waiting for the model to make room (a reshape when the replies in flight end)"})
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
