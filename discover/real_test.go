package discover

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

var errNoEventForTest = errors.New("no event")

// TestRealNetwork sends real multicast, so it runs only when asked:
// EASY_ENGINE_MDNS_TEST=1. It starts a responder on a high port and finds it.
func TestRealNetwork(t *testing.T) {
	if os.Getenv("EASY_ENGINE_MDNS_TEST") != "1" {
		t.Skip("set EASY_ENGINE_MDNS_TEST=1 to use the real network")
	}
	const port = 45353
	info := Info{ID: "feedfacecafebeef", Name: "Real Test", API: 1, Port: 4480}
	r, err := start(info, Options{}, port)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	got, err := find(context.Background(), 2*time.Second, port)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range got {
		if f.ID == info.ID && f.Name == "Real Test" && f.Port == 4480 {
			return
		}
	}
	t.Fatalf("not found among %+v", got)
}
