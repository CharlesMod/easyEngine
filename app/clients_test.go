//go:build !windows

package app_test

import (
	"bufio"
	"context"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A held attachment is a client of the app: its attach wakes the model, and while it holds the
// model is not put to sleep; when its connection ends, the linger runs and the model sleeps.
func TestResidentHeldAttachmentIsAClient(t *testing.T) {
	// linger 0, and the floor after a wake (the attach woke the model) made short
	r := startFakeModelWith(t, 0, nil, nil, shortFloor(200*time.Millisecond))
	waitPhase(t, r, "sleeping")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", "http://127.0.0.1:"+strconv.Itoa(r.port)+"/engine/v1/attach",
		strings.NewReader(`{"hold":true,"name":"remote-ish","profile":{"slots":1}}`))
	req.Header.Set("X-Easy-Engine", "1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	br := bufio.NewReader(resp.Body)
	line, err := br.ReadString('\n')
	if err != nil || !strings.Contains(line, `"presence":"hold"`) {
		t.Fatalf("first line %q (%v)", line, err)
	}
	waitPhase(t, r, "ready") // the attach was the connection: the model loaded
	if st := r.status(t); st.Phase != "ready" {
		t.Fatalf("phase %q while the attachment holds", st.Phase)
	}
	cancel() // the connection ends: the attachment leaves, the linger (0) runs, the model sleeps
	waitPhase(t, r, "sleeping")
}
