package api_test

import (
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/web"
)

// The setup page takes the token from the address fragment, keeps it for the tab, clears the
// address bar and sends it on every call.
func TestSetupPageHandlesTheTokenFragment(t *testing.T) {
	p := string(web.Index)
	for _, want := range []string{"#token=", "token=([^&]+)", "sessionStorage", "history.replaceState", "'Authorization'", "easy-engine token"} {
		if !strings.Contains(p, want) {
			t.Errorf("setup page lacks %q", want)
		}
	}
}
