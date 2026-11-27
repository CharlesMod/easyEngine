package api_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/tokens"
	"github.com/CharlesMod/easyEngine/pair"
)

type failingIssuer struct{}

func (failingIssuer) Issue(string) (string, string, error) {
	return "", "", errors.New("open /home/someone/.local/share/easy-engine/tokens.json.tmp: permission denied")
}

// A keyless path never says an internal error's text: a token file error names the data dir,
// and with it the user.
func TestKeylessBodiesSayNoInternalError(t *testing.T) {
	st := tokens.Open(t.TempDir())
	master, _ := st.Master()
	var pr *pair.Pairer
	_, _, srv := lanServer(t, func(a *api.API) {
		a.Tokens = st
		pr = pair.New(a.S.Hub(), failingIssuer{}, 2)
		a.Pair = pr
	})
	pe := pairEnv{st: st, master: master, srv: srv, pr: pr}
	id, code := ask(t, pe, "mod")
	got := collect(t, pe, id)
	if c, b := answer(t, pe, bearer(master), `{"code":"`+code+`","allow":true}`); c != 200 {
		t.Fatalf("allow: %d %s", c, b)
	}
	r := waitCollected(t, got)
	if r.code != 500 || !strings.Contains(r.body, `"code":"issue_failed"`) || strings.Contains(r.body, "/home") || strings.Contains(r.body, "tokens.json") || strings.Contains(r.body, "permission") {
		t.Fatalf("issue failure body: %d %s", r.code, r.body)
	}
	c, b, _ := do(t, "POST", srv.URL+"/engine/v1/pair", `{"name": 12`, csrf)
	if c != 400 || !strings.Contains(b, `"code":"bad_request"`) || strings.Contains(b, "unexpected") {
		t.Fatalf("bad JSON body: %d %s", c, b)
	}
}
