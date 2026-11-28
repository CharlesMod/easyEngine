package api_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"github.com/CharlesMod/easyEngine/internal/tokens"
)

func TestHelloAndStatusCarryModelCodename(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	e.Cat.Models[0].Codename = "Lantern"
	s := e.StartSup(e.Config())
	tk := tokens.Open(t.TempDir())
	master, err := tk.Master()
	if err != nil {
		t.Fatal(err)
	}
	a := &api.API{S: s, AnyHost: true, Tokens: tk}
	a.Name = func() string { return "test engine" }
	srv := httptest.NewServer(a.Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)
	st := s.Status()
	if st.ModelCodename != "Lantern" {
		t.Fatalf("status model_codename %q", st.ModelCodename)
	}
	c, body, _ := do(t, "GET", srv.URL+"/engine/v1/hello", "", nil)
	if c != 200 || !strings.Contains(body, `"model_codename":"Lantern"`) {
		t.Fatalf("hello: %d %s", c, body)
	}
	var h api.Hello
	decodeJSON(t, body, &h)
	if h.ModelCodename != st.ModelCodename {
		t.Errorf("hello %q status %q", h.ModelCodename, st.ModelCodename)
	}
	if c, body, _ := do(t, "GET", srv.URL+"/engine/v1/status", "", bearer(master)); c != 200 || !strings.Contains(body, `"model_codename":"Lantern"`) {
		t.Errorf("status: %d %s", c, body)
	}
	ps := api.PairedStatusOf(st)
	if ps.ModelCodename != "Lantern" {
		t.Errorf("paired status codename %q", ps.ModelCodename)
	}
	if b, _ := json.Marshal(ps); !strings.Contains(string(b), `"model_codename":"Lantern"`) {
		t.Errorf("paired status JSON: %s", b)
	}
}
