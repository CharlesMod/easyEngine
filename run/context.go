package run

// The prompt cache across a reshape (docs/EVENTS.md section 6, docs/ENGINE.md "The prompt cache
// across a reshape"). Every planned stop of a healthy model server saves each slot that holds
// tokens (llama-server's POST /slots/{i}?action=save, into --slot-save-path, the engine's own
// <data>/contexts/), after the drain and before the stop; the next start lays each one back
// (action=restore) once the server is up and before the phase says ready, where the context key
// (model file, llama build, cache type) is the same, the slot exists and its tokens fit the new
// context per slot. Everything else is let go: its file removed, by the name the engine gave it,
// from the engine's own index. A crash, a hang or an out-of-memory restart saves nothing.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/plan"
)

const (
	contextsDirName  = "contexts"
	contextIndexName = "index.json"
	slotSaveFlag     = "--slot-save-path"
)

// ctxKey is what a saved slot may be laid back into: the model file (its catalog sha256, or the
// path, size and time of a custom file that has none), the llama build (buildKey: the backend
// id, the release's full tag, b12651-easy1, and the sha256 of the archive installed under it,
// which names its patch set with it) and the cache type.
type ctxKey struct {
	Model string `json:"model"`
	Build string `json:"build"`
	KV    string `json:"kv"`
}

func (k ctxKey) String() string { return k.Model + "|" + k.Build + "|" + k.KV }

// ctxKeyOf is the key a server started from lc at shape saves under.
func ctxKeyOf(lc loadCtx, shape fit.Shape) ctxKey {
	m := lc.in.model.SHA256
	if m == "" {
		if st, err := os.Stat(lc.in.modelPath); err == nil {
			m = fmt.Sprintf("file:%s:%d:%d", lc.in.modelPath, st.Size(), st.ModTime().UnixNano())
		} else {
			m = "file:" + lc.in.modelPath
		}
	}
	kv := shape.KVType
	if kv == "" {
		kv = "q4_0" // the cache type a shape without one runs at (core/fit runKV)
	}
	return ctxKey{Model: m, Build: lc.in.build, KV: kv}
}

// keptCtx is one saved slot.
type keptCtx struct {
	Slot   int    `json:"slot"`
	Tokens int    `json:"tokens"`
	Bytes  int64  `json:"bytes"`
	Client string `json:"client,omitempty"` // the key of the attachment the slot served (run/clients.go); "" none
	// ClientSlot is the slot in that client's own numbering: it is laid back into the slot the
	// client has under that number after a start that renumbered the engine's slots.
	ClientSlot *int   `json:"client_slot,omitempty"`
	File       string `json:"file"`
	Key        ctxKey `json:"key"`
}

// KeptContext is one saved slot as status shows it.
type KeptContext struct {
	Slot   int    `json:"slot"`
	Tokens int    `json:"tokens"`
	Bytes  int64  `json:"bytes"`
	Client string `json:"client,omitempty"`
	// ClientSlot is the slot in the client's own numbering (with Client).
	ClientSlot *int `json:"client_slot,omitempty"`
}

// ContextsInfo is status "contexts": the slots kept while no model server holds them.
type ContextsInfo struct {
	Slots  int           `json:"slots"`
	Tokens int           `json:"tokens"`
	Bytes  int64         `json:"bytes"`
	Kept   []KeptContext `json:"kept"`
}

// contextSaid is the body of the "context" happening.
type contextSaid struct {
	What   string `json:"what"` // saved | laid_back | let_go | not_saved
	Slot   int    `json:"slot"`
	Tokens int    `json:"tokens,omitempty"`
	Bytes  int64  `json:"bytes,omitempty"`
	Why    string `json:"why,omitempty"`
	// Client and ClientSlot: the attachment's key and its own slot number, for a slot a client owns.
	Client     string `json:"client,omitempty"`
	ClientSlot *int   `json:"client_slot,omitempty"`
}

// keptContexts is the engine's record of its saved slots, in memory and in contexts/index.json.
type keptContexts struct {
	mu   sync.Mutex
	list []keptCtx
	seq  int
}

func (k *keptContexts) info() *ContextsInfo {
	k.mu.Lock()
	defer k.mu.Unlock()
	if len(k.list) == 0 {
		return nil
	}
	in := &ContextsInfo{Kept: []KeptContext{}}
	for _, c := range k.list {
		in.Slots++
		in.Tokens += c.Tokens
		in.Bytes += c.Bytes
		in.Kept = append(in.Kept, KeptContext{Slot: c.Slot, Tokens: c.Tokens, Bytes: c.Bytes, Client: c.Client, ClientSlot: c.ClientSlot})
	}
	return in
}

func (k *keptContexts) bytes() int64 {
	k.mu.Lock()
	defer k.mu.Unlock()
	var n int64
	for _, c := range k.list {
		n += c.Bytes
	}
	return n
}

// take empties the record and returns what it held.
func (k *keptContexts) take() []keptCtx {
	k.mu.Lock()
	defer k.mu.Unlock()
	l := k.list
	k.list = nil
	return l
}

// slotFileRe is the only shape of name the engine gives a saved slot; nothing else is removed.
var slotFileRe = regexp.MustCompile(`^slot[0-9]+-[0-9]+-[0-9]+\.bin$`)

func (s *Supervisor) contextsDir() string { return filepath.Join(s.dir, contextsDirName) }

// slotSaveArgs is --slot-save-path for a launch, when the build knows the flag (known nil: a
// build whose --help could not be read keeps every flag).
func (s *Supervisor) slotSaveArgs(known map[string]bool) []string {
	if known != nil && !known[slotSaveFlag] {
		s.logf("llama-server does not know %s: no prompt is kept across a restart", slotSaveFlag)
		return nil
	}
	return []string{slotSaveFlag, s.contextsDir()}
}

// withBefore puts extra into args just before flag (at the end when flag is absent).
func withBefore(args []string, flag string, extra []string) []string {
	for i, a := range args {
		if a == flag {
			return append(append(append([]string{}, args[:i]...), extra...), args[i:]...)
		}
	}
	return append(args, extra...)
}

// contextsInit makes contexts/ (0700) and lets go of whatever a dead engine left: the files its
// index names. A Sleep followed by an engine restart is not laid back.
func (s *Supervisor) contextsInit() {
	dir := s.contextsDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		s.logf("contexts: %v", err)
		return
	}
	os.Chmod(dir, 0o700)
	b, err := os.ReadFile(filepath.Join(dir, contextIndexName))
	if err != nil {
		return
	}
	var l []keptCtx
	if err := json.Unmarshal(b, &l); err != nil {
		s.logf("contexts: index unreadable (%v): removed", err)
	}
	for _, c := range l {
		s.dropFile(c, "the engine started (left by an earlier run)")
	}
	os.Remove(filepath.Join(dir, contextIndexName))
}

// writeIndex writes the record beside the files (a temp file renamed over), or removes it when
// nothing is kept.
func (s *Supervisor) writeIndex() {
	s.kept.mu.Lock()
	l := append([]keptCtx(nil), s.kept.list...)
	s.kept.mu.Unlock()
	p := filepath.Join(s.contextsDir(), contextIndexName)
	if len(l) == 0 {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			s.logf("contexts: %v", err)
		}
		return
	}
	b, _ := json.MarshalIndent(l, "", "  ")
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		s.logf("contexts: index not written: %v", err)
		return
	}
	if err := os.Rename(tmp, p); err != nil {
		s.logf("contexts: index not written: %v", err)
	}
}

// dropFile removes one saved slot's file (only a name the engine gives, only in its own
// directory) and says it let go.
func (s *Supervisor) dropFile(c keptCtx, why string) {
	if slotFileRe.MatchString(c.File) {
		if err := os.Remove(filepath.Join(s.contextsDir(), c.File)); err != nil && !os.IsNotExist(err) {
			s.logf("contexts: %v", err)
		}
	}
	s.logf("context: slot %d (%d tokens, %d bytes) let go: %s", c.Slot, c.Tokens, c.Bytes, why)
	s.say("context", strconv.Itoa(c.Slot), contextSaid{What: "let_go", Slot: c.Slot, Tokens: c.Tokens, Bytes: c.Bytes, Why: why, Client: c.Client, ClientSlot: c.ClientSlot})
}

// letGoContexts removes every kept slot.
func (s *Supervisor) letGoContexts(why string) {
	l := s.kept.take()
	if len(l) == 0 {
		return
	}
	for _, c := range l {
		s.dropFile(c, why)
	}
	s.writeIndex()
}

// slotState is one entry of llama-server's GET /slots (b12680): n_prompt_tokens and next_token
// appear only on a slot that has served a task since the server started.
type slotState struct {
	ID       int  `json:"id"`
	NCtx     int  `json:"n_ctx"`
	NPrompt  *int `json:"n_prompt_tokens"`
	NextToks []struct {
		NDecoded int `json:"n_decoded"`
	} `json:"next_token"`
}

// slotCall is one POST /slots/{id}?action=... with {"filename":...}: the answer's JSON, or an error
// with the server's message.
func (s *Supervisor) slotCall(base string, id int, action, file string, out any) error {
	// network timeout: SlotTimeout, one save or restore of a large file
	ctx, cancel := context.WithTimeout(s.root, s.cfg.SlotTimeout)
	defer cancel()
	body, _ := json.Marshal(map[string]string{"filename": file})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/slots/%d?action=%s", base, id, action), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	cl := *s.cfg.HTTP
	cl.Timeout = 0 // bounded by the context above, not by the client's short default
	resp, err := cl.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != 200 {
		var e struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		msg := strings.TrimSpace(string(b))
		if json.Unmarshal(b, &e) == nil && e.Error.Message != "" {
			msg = e.Error.Message
		}
		return fmt.Errorf("%d: %s", resp.StatusCode, msg)
	}
	return json.Unmarshal(b, out)
}

// slotTokens is the tokens each slot holds as the server says (GET /slots), with what was laid
// back into a slot that has not served since (the server says nothing of those).
func (s *Supervisor) slotTokens(srv *server, base string) map[int]int {
	held := map[int]int{}
	for id, n := range srv.laidBack {
		held[id] = n
	}
	// network timeout: one GET /slots
	ctx, cancel := context.WithTimeout(s.root, s.cfg.SlotTimeout)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, base+"/slots", nil)
	resp, err := s.cfg.HTTP.Do(req)
	if err != nil {
		s.logf("context: GET /slots: %v (only slots laid back are known)", err)
		return held
	}
	defer resp.Body.Close()
	var l []slotState
	if resp.StatusCode != 200 || json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&l) != nil {
		s.logf("context: GET /slots answered %d or nothing readable (only slots laid back are known)", resp.StatusCode)
		return held
	}
	for _, st := range l {
		if st.NPrompt == nil {
			continue
		}
		n := *st.NPrompt
		for _, t := range st.NextToks {
			n += t.NDecoded
		}
		if n > 0 {
			held[st.ID] = n
		}
	}
	return held
}

// estimateBytes is what a saved slot of tokens takes on disk, priced by the sizing core's
// footprint of the running model: the full-attention cache per token, the sliding-window cache
// up to its window three times over (the window and the checkpoints the save carries), the
// recurrent state per slot, and 1 MiB of header. Measured on llama.cpp b12680 (docs/ENGINE.md):
// a 31 961-token slot of Gemma 4 26B-A4B at q4_0 is 362 MB.
func estimateBytes(srv *server, tokens int) int64 {
	fp := srv.fp
	swa := tokens
	if fp.SWAWindow > 0 && fp.SWAWindow < swa {
		swa = fp.SWAWindow
	}
	mib := fp.PerTokenMiB*float64(tokens) + fp.SWAPerTokenMiB*float64(swa)*3 + fp.StatePerSlotMiB + 1
	return int64(mib * (1 << 20))
}

// keepContexts saves the slots of the running server before a planned stop (opMu held; the
// caller stops the server next). Nothing is saved with keep_context off, from a server not up
// or with requests in flight, or with too little disk; a yield with keep_context on saves only
// what fits YieldSaveMaxMiB (a program waits for the card meanwhile). A save that fails is said
// and that slot is not kept; the stop goes on whatever happens here.
func (s *Supervisor) keepContexts(why string, yielding bool) {
	srv := s.srv
	s.mu.Lock()
	base := s.llamaURL
	s.mu.Unlock()
	if srv == nil || base == "" || srv.proc.Exited() {
		return
	}
	if srv.oomShrink {
		s.notSaved(-1, 0, 0, why+" after out-of-memory while serving: a memory failure keeps nothing")
		return
	}
	mode := s.KeepContext()
	if mode == "off" {
		s.logf("context: keep_context is off: nothing saved before %s", why)
		return
	}
	if n := s.inflightNow(); n > 0 {
		s.notSaved(-1, 0, 0, fmt.Sprintf("%d request(s) in flight at %s", n, why))
		return
	}
	if next := s.stagedBuild(); next != "" {
		// the context key carries the build: the next start (promoteNext) could lay nothing back
		s.notSaved(-1, 0, 0, fmt.Sprintf("%s: the next start runs another llama build (%s)", why, next))
		return
	}
	held := s.slotTokens(srv, base)
	var ids []int
	var total int64
	est := map[int]int64{}
	for id := 0; id < srv.shape.Slots; id++ {
		if n := held[id]; n > 0 {
			ids = append(ids, id)
			est[id] = estimateBytes(srv, n)
			total += est[id]
		}
	}
	if len(ids) == 0 {
		return
	}
	if yielding && mode == "on" && total > int64(s.cfg.YieldSaveMaxMiB)<<20 {
		s.notSaved(-1, 0, total, fmt.Sprintf("%s: about %s to write, more than a yield waits for (yield_save_max %d MiB; keep_context \"always\" saves anyway)", why, plan.GB(total), s.cfg.YieldSaveMaxMiB))
		return
	}
	if free, err := s.cfg.DiskFree(s.dir); err != nil {
		s.logf("context: free disk unreadable (%v): saving without a space check", err)
	} else {
		need := plan.NeedOf([]plan.DiskStep{{What: "kept contexts", Peak: total, Keep: total}}, s.logRoom(), s.hostFiles())
		if fb := int64(free) << 20; need.Short(fb) {
			s.notSaved(-1, 0, total, fmt.Sprintf("%s: too little disk (%s, %s free)", why, need.Explain(), plan.GB(fb)))
			return
		}
	}
	s.letGoContexts("replaced by the save before " + why) // what an earlier stop left and no start took
	var got []keptCtx
	for _, id := range ids {
		s.kept.mu.Lock()
		s.kept.seq++
		name := fmt.Sprintf("slot%d-%d-%d.bin", id, os.Getpid(), s.kept.seq)
		s.kept.mu.Unlock()
		var ans struct {
			NSaved   int   `json:"n_saved"`
			NWritten int64 `json:"n_written"`
		}
		err := s.slotCall(base, id, "save", name, &ans)
		c := keptCtx{Slot: id, Tokens: held[id], File: name, Key: srv.ctxKey}
		c.Client, c.ClientSlot = s.slotOwner(id)
		switch {
		case err != nil:
			s.addError("context: saving slot %d before %s failed: %v (not kept)", id, why, err)
			s.dropFileQuiet(name)
			s.notSaved(id, held[id], est[id], "save failed: "+err.Error())
			continue
		case ans.NSaved <= 0:
			s.dropFileQuiet(name)
			continue
		}
		c.Tokens, c.Bytes = ans.NSaved, ans.NWritten
		got = append(got, c)
		s.logf("context: slot %d saved before %s: %d tokens, %d bytes", id, why, c.Tokens, c.Bytes)
		s.say("context", strconv.Itoa(id), contextSaid{What: "saved", Slot: id, Tokens: c.Tokens, Bytes: c.Bytes, Why: why, Client: c.Client, ClientSlot: c.ClientSlot})
	}
	if len(got) == 0 {
		return
	}
	s.kept.mu.Lock()
	s.kept.list = append(s.kept.list, got...)
	s.kept.mu.Unlock()
	s.writeIndex()
}

func (s *Supervisor) dropFileQuiet(name string) {
	if slotFileRe.MatchString(name) {
		os.Remove(filepath.Join(s.contextsDir(), name))
	}
}

// notSaved says a save that did not happen (slot -1: none of them).
func (s *Supervisor) notSaved(slot, tokens int, bytes int64, why string) {
	s.logf("context: nothing saved (slot %d): %s", slot, why)
	key := "all"
	if slot >= 0 {
		key = strconv.Itoa(slot)
	}
	s.say("context", key, contextSaid{What: "not_saved", Slot: slot, Tokens: tokens, Bytes: bytes, Why: why})
}

// layBack gives the new server the slots kept for it (opMu held; the server is up, the phase
// not yet ready, so the door still holds requests). A restore that fails is said and let go; it
// never fails the start.
func (s *Supervisor) layBack(srv *server, base string) {
	srv.laidBack = map[int]int{}
	l := s.kept.take()
	if len(l) == 0 {
		return
	}
	for _, c := range l {
		if c.Client != "" && c.ClientSlot != nil {
			// a client's slot goes back into the slot that client has under its own number now
			e, why := s.slotFor(c.Client, *c.ClientSlot)
			if why != "" {
				s.dropFile(c, why)
				continue
			}
			c.Slot = e
		} else if !s.unownedSlot(c.Slot) {
			// a slot no client owned goes back only into the same number when no client owns that
			// one now: never into a client's slot
			s.dropFile(c, "that slot belongs to a client now")
			continue
		}
		switch {
		case c.Key != srv.ctxKey:
			s.dropFile(c, layBackMismatch(c.Key, srv.ctxKey))
			continue
		case c.Slot >= srv.shape.Slots:
			s.dropFile(c, fmt.Sprintf("the new server has %d slot(s)", srv.shape.Slots))
			continue
		case c.Tokens > srv.shape.CtxPerSlot:
			s.dropFile(c, fmt.Sprintf("%d tokens do not fit the new context of %d per slot", c.Tokens, srv.shape.CtxPerSlot))
			continue
		}
		var ans struct {
			NRestored int   `json:"n_restored"`
			NRead     int64 `json:"n_read"`
		}
		if err := s.slotCall(base, c.Slot, "restore", c.File, &ans); err != nil {
			s.addError("context: laying slot %d back failed: %v (let go)", c.Slot, err)
			s.dropFile(c, "restore failed: "+err.Error())
			continue
		}
		srv.laidBack[c.Slot] = ans.NRestored
		s.dropFileQuiet(c.File)
		s.logf("context: slot %d laid back: %d tokens, %d bytes", c.Slot, ans.NRestored, ans.NRead)
		s.say("context", strconv.Itoa(c.Slot), contextSaid{What: "laid_back", Slot: c.Slot, Tokens: ans.NRestored, Bytes: ans.NRead, Why: "the start after the stop it was saved at", Client: c.Client, ClientSlot: c.ClientSlot})
	}
	s.writeIndex()
}

func layBackMismatch(was, now ctxKey) string {
	switch {
	case was.Model != now.Model:
		return "the model changed"
	case was.Build != now.Build:
		return "the llama build changed"
	default:
		return fmt.Sprintf("the cache type changed (%s to %s)", was.KV, now.KV)
	}
}
