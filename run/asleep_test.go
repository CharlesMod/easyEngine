//go:build !windows

package run_test

import (
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/internal/testkit"
)

// An engine started asleep (the app waiting for a connection) loads no model and says so; the
// first Wake loads it, and after a Sleep a request at the door loads it again.
func TestStartSleepingLoadsOnWakeAndOnARequest(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.StartSleeping = true
	s := e.StartSup(cfg) // Run has said the phase when it returns
	if st := s.Status(); st.Phase != "sleeping" || st.Message != "Model not loaded" || len(e.Starts()) != 0 {
		t.Fatalf("started asleep: %q %q, starts %v", st.Phase, st.Message, e.Starts())
	}
	if err := s.Wake(); err != nil { // an attach in the app wakes it this way
		t.Fatal(err)
	}
	if s.Phase() != "ready" || len(e.Starts()) != 1 {
		t.Fatalf("after Wake: %s %v", s.Phase(), e.Starts())
	}
	if err := s.Sleep(); err != nil {
		t.Fatal(err)
	}
	s.Route() // a request at the door
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if n := len(e.Starts()); n != 2 {
		t.Fatalf("%d starts, want 2", n)
	}
}

// The first request at the door loads an engine that started asleep.
func TestStartSleepingLoadsOnTheFirstRequest(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	cfg := e.Config()
	cfg.StartSleeping = true
	s := e.StartSup(cfg)
	if r := s.Route(); r.Kind != "unavailable" {
		t.Fatalf("route asleep = %+v", r)
	}
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	if r := s.Route(); r.Kind != "local" {
		t.Fatalf("route = %+v", r)
	}
}
