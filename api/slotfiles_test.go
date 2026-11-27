package api_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/internal/testkit"
	"net/http/httptest"
)

// Slot save and restore write and read files in the engine's own contexts directory. A client
// must not reach them through the proxy, by the plain path or through the chat page's prefix;
// it keeps GET /slots and erase.
func TestClientCannotSaveOrRestoreSlots(t *testing.T) {
	testkit.SkipWindows(t)
	e := testkit.New(t)
	e.InstallFake()
	s := e.StartSup(e.Config())
	srv := httptest.NewServer((&api.API{S: s}).Handler())
	t.Cleanup(srv.Close)
	testkit.WaitPhase(t, s, "ready", 15*time.Second)

	body := `{"filename":"mine.bin"}`
	hdr := map[string]string{"X-Easy-Engine": "1", "Content-Type": "application/json"}
	for _, u := range []string{
		"/slots/0?action=save",
		"/slots/0?action=restore",
		"/slots/0?action=erase&action=save",
		"/slots/0?act%69on=save",
		"/%73lots/0?action=save",
		"/x/../slots/0?action=save",
		"/chat/ui/slots/0?action=save",
		"/slots/0?action=SAVE",
	} {
		if c, b, _ := do(t, "POST", srv.URL+u, body, hdr); c != 403 || !strings.Contains(b, "engine's own") {
			t.Errorf("POST %s: %d %q, want 403 from the engine", u, c, b)
		}
	}
	// nothing reached the model server, and no file was written
	if b, err := os.ReadFile(filepath.Join(e.Ctl, "slots.log")); err == nil && len(b) > 0 {
		t.Fatalf("the model server saw slot calls: %s", b)
	}
	if m, _ := filepath.Glob(filepath.Join(e.Data, "contexts", "*")); len(m) != 0 {
		t.Fatalf("files in contexts/: %v", m)
	}
	// what stays a client's
	if c, b, _ := do(t, "GET", srv.URL+"/slots", "", nil); c != 200 {
		t.Fatalf("GET /slots: %d %q", c, b)
	}
	if c, b, _ := do(t, "POST", srv.URL+"/slots/0?action=erase", "", hdr); c == 403 {
		t.Fatalf("erase refused: %d %q", c, b)
	}
}
