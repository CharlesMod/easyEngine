//go:build !windows

package run_test

import (
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/core/fit"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/run"
)

// A client's can_use is judged against the model that is loaded: with a tier step in force (Bonfire
// selected, too big; Campfire serving), a client that can use only Campfire is usable and its
// required model taken; one that can use only Bonfire is told the loaded model is not one it can
// use, and its required model is 409 model_unsupported.
func TestClientCanUseFollowsTheServingTier(t *testing.T) {
	_, s, _ := stepped(t)
	prof := func(canUse string, required bool) *run.Profile {
		p := clProf(1, 8192, 4096)
		p.Models = &run.ModelPrefs{CanUse: []string{canUse}}
		if required {
			p.Server = &run.ServerOpts{Required: []string{"model"}}
		}
		return p
	}
	v, _ := clAttach(t, s, clReq("serving", nil, true, prof("camp", false)))
	if v.Model == nil || v.Model.Selected != "camp" || !v.Model.Usable {
		t.Fatalf("can_use camp: %+v", v.Model)
	}
	clAttach(t, s, clReq("serving-req", nil, true, prof("camp", true)))
	v, _ = clAttach(t, s, clReq("selected", nil, true, prof("bon", false)))
	if v.Model == nil || v.Model.Usable {
		t.Fatalf("can_use bon (selected, not loaded): %+v", v.Model)
	}
	if e := clRefused(t, s, clReq("selected-req", nil, true, prof("bon", true))); e.Status != 409 || e.Code != "model_unsupported" {
		t.Fatalf("required bon: %d %s", e.Status, e.Code)
	}
}

// A client that required a model it can use (Bonfire), attached while Bonfire served: a later
// start steps down to Campfire (Bonfire no longer fits). The client is judged again at that
// start: unserved with the reason, never given slots of a model it said it cannot use; its view
// says the serving model is not usable. Back on Bonfire it is served again.
func TestRequiredModelJudgedAgainAtALaterStep(t *testing.T) {
	e, _ := tierEnv(t)
	cfg := e.Config()
	var bonMiB atomic.Int64
	bonMiB.Store(8000)
	cfg.Seed = func(m catalog.Model, path, kv string) (fit.Footprint, error) {
		return tierSeed(float64(bonMiB.Load()))(m, path, kv)
	}
	s := e.StartSup(cfg)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if st := s.Status(); st.Model != "bon" {
		t.Fatalf("serving %q, want bon", st.Model)
	}
	p := clProf(1, 8192, 4096)
	p.Models = &run.ModelPrefs{CanUse: []string{"bon"}}
	p.Server = &run.ServerOpts{Required: []string{"model"}}
	v, _ := clAttach(t, s, clReq("needs-bon", nil, true, p))
	settle(t, s)
	if v, _ = s.ClientView(v.Attachment); v.Slots.Given != 1 {
		t.Fatalf("on Bonfire: %+v", v.Slots)
	}
	restart := func(want string) run.AttachView {
		t.Helper()
		if err := s.Sleep(); err != nil {
			t.Fatal(err)
		}
		testkit.WaitPhase(t, s, "sleeping", 15*time.Second)
		go s.Wake()
		testkit.WaitFor(t, 15*time.Second, "serving "+want, func() bool { return s.Phase() == "ready" && s.Status().Model == want })
		got, _ := s.ClientView(v.Attachment)
		return got
	}
	bonMiB.Store(20000) // another start: Bonfire no longer fits, Campfire serves
	got := restart("camp")
	if got.Slots.Given != 0 || !strings.Contains(got.Slots.Unserved, "not one it can use") || got.Model == nil || got.Model.Usable {
		t.Fatalf("after the step: slots %+v model %+v", got.Slots, got.Model)
	}
	bonMiB.Store(8000)
	if got = restart("bon"); got.Slots.Given != 1 || got.Slots.Unserved != "" {
		t.Fatalf("back on Bonfire: %+v", got.Slots)
	}
}
