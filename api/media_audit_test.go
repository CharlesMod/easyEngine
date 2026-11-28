package api

import (
	"testing"

	"github.com/CharlesMod/easyEngine/catalog"
)

// The media audit (docs/ENGINE.md section 7h, "Media from remote callers") was done against one
// llama.cpp build's source. When the catalog pins another, the audit is stale.
func TestMediaAuditMatchesPin(t *testing.T) {
	c, err := catalog.Load("")
	if err != nil {
		t.Fatal(err)
	}
	r := c.Rungs()
	if len(r) == 0 {
		t.Fatal("the embedded catalog pins no llama.cpp build")
	}
	if r[0].Tag != mediaAuditTag {
		t.Fatalf("the pinned llama.cpp build moved from %s to %s: redo the media audit (every way media enters a request, every projector/media flag and env var) against the new source, update the table in docs/ENGINE.md section 7h 'Media from remote callers', api/media.go, run/ownedflags.go and run/util.go mediaEnv, then move mediaAuditTag", mediaAuditTag, r[0].Tag)
	}
}

// The allow list covers both chat spellings, cleaned like the other path checks.
func TestTextOnlyChatPath(t *testing.T) {
	for p, want := range map[string]bool{
		"/v1/chat/completions": true, "/chat/completions": true, "/v1//chat/completions": true,
		"/v1/./chat/completions": true, "/completion": false, "/v1/messages": false,
	} {
		if textOnlyChatPath(p) != want {
			t.Errorf("%s: want %v", p, want)
		}
	}
}
