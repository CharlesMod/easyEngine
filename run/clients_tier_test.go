//go:build !windows

package run_test

import (
	"testing"

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
