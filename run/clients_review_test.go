//go:build !windows

package run_test

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// Where the system gives no word of a pid's end, the old wire's {pid} makes no attachment: a
// parent alone (the API attaches it), which never meets a cap.
func TestClientOldWireWithNoExitWordIsAParentAlone(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.ExitsOf = func(int, string) (<-chan struct{}, func(), error) { return nil, nil, host.ErrNoEvent }
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	k := newKid(t)
	for i := 0; i < 40; i++ { // past every cap
		v, h := clAttach(t, s, clReq("", k, false, nil))
		if h != nil || v.Attachment != "" || v.Status != "attached" || v.Pid != k.pid() {
			t.Fatalf("view = %+v", v)
		}
	}
	if s.Attached() != 0 {
		t.Fatalf("attached = %d", s.Attached())
	}
	// an attachment with wants still stands as asked (for the engine's life)
	v, _ := clAttach(t, s, clReq("w", k, false, clProf(1, 8192, 4096)))
	if v.Attachment == "" || s.Attached() != 1 {
		t.Fatalf("with wants: %+v", v)
	}
}

// A replace that brings another pid stops the old pid's watch: the old pid's end is not the
// attachment's leave; the new one's is.
func TestClientReplaceStopsTheOldPidWatch(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	k1, k2 := newKid(t), newKid(t)
	v1, _ := clAttach(t, s, clReq("x", k1, false, nil))
	v2, _ := clAttach(t, s, clReq("x", k2, false, nil))
	if !v2.Replaced || v2.Attachment != v1.Attachment {
		t.Fatalf("not a replace: %+v", v2)
	}
	k1.end()
	clQuiet(t, e, s, 1)
	if clFactGone(s, v1.Attachment) {
		t.Fatal("the old pid's end ended the replaced attachment")
	}
	k2.end()
	testkit.WaitFor(t, 15*time.Second, "the attachment gone with its new pid", func() bool { return clFactGone(s, v1.Attachment) })
}

// Each held stream has its own notify channel: a replaced stream is not woken for the new one.
func TestClientEachHoldItsOwnNotify(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	_, h1 := clAttach(t, s, clReq("h", nil, true, nil))
	_, h2 := clAttach(t, s, clReq("h", nil, true, nil))
	if h1 == nil || h2 == nil || h1.Notify == h2.Notify {
		t.Fatal("the two holds share a notify channel")
	}
	if w := <-h1.End; w != "replaced" {
		t.Fatalf("first end %q", w)
	}
}

// A context saved from a slot no client owned is laid back only into the same number when no
// client owns it in the new map; a client's slot never gets it.
func TestClientUnownedContextNeverIntoAClientsSlot(t *testing.T) {
	e := testkit.New(t)
	e.WatchFile(filepath.Join(e.Ctl, "slots.log"))
	s := ready(t, e)  // three anonymous slots
	ask(t, s, 2, 400) // an anonymous prompt in slot 2
	from := testkit.Mark(t, s, "before the client")
	b, _ := clAttach(t, s, clReq("b", nil, true, clProf(4, 65536, 4096))) // 4 slots: a grow, b gets 0..3
	clStarted(t, e, s, 2)
	testkit.WaitFor(t, 15*time.Second, "b served", func() bool { return clView(t, s, b.Attachment).Slots.Given == 4 })
	var saved, laid, letGo bool
	for _, ev := range testkit.Said(t, s, from, 0, "context") {
		var c clCtxSaid
		if err := json.Unmarshal(ev.Body, &c); err != nil {
			t.Fatal(err)
		}
		switch {
		case c.What == "saved" && c.Slot == 2 && c.Client == "":
			saved = true
		case c.What == "laid_back" && c.Slot == 2:
			laid = true
		case c.What == "let_go" && c.Slot == 2 && c.Client == "" && strings.Contains(c.Why, "belongs to a client"):
			letGo = true
		}
	}
	if !saved || laid || !letGo {
		t.Fatalf("saved %v laid back %v let go %v", saved, laid, letGo)
	}
}

// A local client's catalog overlay adds its models (kept once taken); a bad one, one that
// would replace a built-in, and any from a remote client are refused with nothing registered.
func TestClientCatalogOverlay(t *testing.T) {
	e := testkit.New(t)
	s := ready(t, e)
	n := len(s.CatalogNow().Models)
	builtin := s.CatalogNow().Models[0]
	p := clProf(0, 0, 0)
	p.Slots = 0
	p.Catalog = &run.CatalogOverlay{Models: []catalog.Model{{ID: "local-extra", Name: "Extra", File: "local-extra.gguf",
		URL: "https://example.com/local-extra.gguf", SHA256: strings.Repeat("cd", 32), Size: 123456}}}
	clAttach(t, s, clReq("o", nil, true, p))
	if m, ok := s.CatalogNow().Model("local-extra"); !ok || m.Size != 123456 || len(s.CatalogNow().Models) != n+1 {
		t.Fatalf("overlay not taken: %d models", len(s.CatalogNow().Models))
	}
	clAttach(t, s, clReq("o", nil, true, p)) // brought again: the same, nothing added
	if len(s.CatalogNow().Models) != n+1 {
		t.Fatal("added twice")
	}
	replace := clProf(0, 0, 0)
	b2 := builtin
	b2.URL = "https://example.com/other.gguf"
	replace.Catalog = &run.CatalogOverlay{Models: []catalog.Model{b2}}
	before := s.Attached()
	if e := clRefused(t, s, clReq("r", nil, true, replace)); e.Code != "catalog_invalid" || e.Status != 400 {
		t.Fatalf("replace: %+v", e)
	}
	remote := clReq("x", nil, true, p)
	remote.Identity, remote.Local = "client-1", false
	if e := clRefused(t, s, remote); e.Code != "catalog_not_allowed" {
		t.Fatalf("remote: %+v", e)
	}
	if s.Attached() != before {
		t.Fatal("a refused attach was registered")
	}
	if m, _ := s.CatalogNow().Model(builtin.ID); m.URL != builtin.URL {
		t.Fatal("a built-in was replaced")
	}
}

// Each pid watch ends with its attachment (a detach, a replace): its stop is called, its
// goroutine does not wait for the pid.
func TestClientPidWatchEndsWithItsAttachment(t *testing.T) {
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	var mu sync.Mutex
	stops := 0
	cfg.ExitsOf = func(int, string) (<-chan struct{}, func(), error) {
		return make(chan struct{}), func() { mu.Lock(); stops++; mu.Unlock() }, nil
	}
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	k := newKid(t)
	v, _ := clAttach(t, s, clReq("x", k, false, nil))
	clAttach(t, s, clReq("x", k, false, nil)) // a replace: the first watch stops
	if err := s.Detach(v.Attachment, "loopback"); err != nil {
		t.Fatal(err)
	}
	testkit.WaitFor(t, 10*time.Second, "both watches stopped", func() bool { mu.Lock(); defer mu.Unlock(); return stops == 2 })
}
