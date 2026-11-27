package app_test

import (
	"bufio"
	"encoding/json"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/CharlesMod/easyEngine/app"
	"github.com/CharlesMod/easyEngine/event"
	"github.com/CharlesMod/easyEngine/internal/tokens"
)

func getJSON(t *testing.T, url, tok string, v any) {
	t.Helper()
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatal(err)
	}
}

// A revoke by `easy-engine token --revoke` (a rewrite of tokens.json) takes the client's fact
// down at once: the app follows the file.
func TestRevokeTakesTheClientFactDown(t *testing.T) {
	lanOK(t)
	r := startResident(t, app.Options{Host: "127.0.0.2", Port: 47665})
	base := "http://127.0.0.2:" + itoa(r.port) + "/engine/v1/"
	st := tokens.Open(r.dir)
	master, _ := st.Existing()
	id, tok, err := st.Issue("den")
	if err != nil {
		t.Fatal(err)
	}
	if c := get(t, base+"status", tok); c != 200 {
		t.Fatalf("issued token: %d", c)
	}
	var f event.Standing
	getJSON(t, base+"facts", master, &f)
	found := false
	for _, e := range f.Facts {
		if e.Name == "client" && e.Key == id {
			found = true
		}
	}
	if !found {
		t.Fatalf("no client fact for %s", id)
	}

	// follow the stream from here, then revoke the way the CLI does: by the file
	req, _ := http.NewRequest("GET", base+"events?life="+f.Life+"&since="+strconv.FormatUint(f.Seq, 10)+"&prefix=client", nil)
	req.Header.Set("Authorization", "Bearer "+master)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	gone := make(chan bool, 1)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			var e event.Event
			if json.Unmarshal(sc.Bytes(), &e) == nil && e.Name == "client" && e.Key == id && e.Gone {
				gone <- true
				return
			}
		}
		gone <- false
	}()
	if err := tokens.Open(r.dir).Revoke(id); err != nil {
		t.Fatal(err)
	}
	select {
	case ok := <-gone:
		if !ok {
			t.Fatal("the stream ended before the client fact went")
		}
	case <-time.After(20 * time.Second): // watchdog for silence: the fact never went
		t.Fatal("the revoked client's fact still stands")
	}
}
