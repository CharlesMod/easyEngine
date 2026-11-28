package run_test

import (
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
	"github.com/CharlesMod/easyEngine/run"
)

func TestOptInModelIsNeverTheStarter(t *testing.T) {
	s := newStage(t)
	tiny := []byte(strings.Repeat("tiny", 100))
	s.cat.Models = append(s.cat.Models, catalog.Model{ID: "tiny", Name: "Tiny", File: "tiny.gguf", URL: s.hs.URL + "/tiny.gguf",
		SHA256: sum(tiny), Size: int64(len(tiny)), Quality: 10, OptIn: true})
	s.e.Cat = s.cat
	sup := s.start(t)
	if err := sup.Install("target", ""); err != nil {
		t.Fatal(err)
	}
	waitStatus(t, sup, "the non-opt-in starter serving", func(st run.Status) bool {
		return st.Phase == "ready" && st.Model == "starter" && st.ModelTier == "starter"
	})
	s.release()
}
