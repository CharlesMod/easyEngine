package api_test

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/web"
)

// The setup page follows the event stream. These tests hold the page to the
// endpoints it calls: each call path it names must exist in the page source
// and answer through the real handler.

func TestPageCallsAnswer(t *testing.T) {
	_, _, srv := readyEnv(t)
	page := string(web.Index)

	// every path the page builds with API+'...' or call(method,'...')
	var found = map[string]bool{}
	for _, m := range regexp.MustCompile(`API\+'([a-z]+)`).FindAllStringSubmatch(page, -1) {
		found[m[1]] = true
	}
	for _, m := range regexp.MustCompile(`call\('(?:GET|POST|PUT)','([a-z]+)`).FindAllStringSubmatch(page, -1) {
		found[m[1]] = true
	}
	for _, p := range []string{"facts", "events", "status", "plan", "hardware", "logs"} {
		if !found[p] {
			t.Errorf("the page never calls %q (found %v)", p, found)
		}
	}
	for p := range found {
		switch p {
		case "facts", "events", "status", "plan", "hardware", "logs", "test", "start", "stop", "sleep", "wake", "cancel", "import", "install", "settings":
		default:
			t.Errorf("the page calls %q, which this test does not know", p)
		}
	}

	get := func(path string) string {
		c, body, _ := do(t, "GET", srv.URL+"/engine/v1/"+path, "", nil)
		if c < 200 || c > 299 || strings.TrimSpace(body) == "" {
			t.Fatalf("GET %s: %d %q", path, c, body)
		}
		return body
	}
	var st map[string]any
	if err := json.Unmarshal([]byte(get("status")), &st); err != nil || st["phase"] != "ready" {
		t.Fatalf("status %v %v", st, err)
	}
	var f struct {
		Life  string           `json:"life"`
		Seq   uint64           `json:"seq"`
		Facts []map[string]any `json:"facts"`
	}
	if err := json.Unmarshal([]byte(get("facts")), &f); err != nil || f.Life == "" || len(f.Facts) == 0 {
		t.Fatalf("facts %+v %v", f, err)
	}
	var book any
	if err := json.Unmarshal([]byte(get("book")), &book); err != nil {
		t.Fatalf("book: %v", err)
	}
	for _, p := range []string{"plan", "hardware", "logs?name=engine&tail=10"} {
		get(p)
	}

	// events: the first line, then the caller lets go (a context cancel, no clock).
	// Nothing was said after f.Seq, so ask from one before it when there is one.
	since := f.Seq
	if since > 0 {
		since--
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", srv.URL+"/engine/v1/events?life="+f.Life+"&since="+itoa(since), nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(resp.Header.Get("Content-Type"), "ndjson") {
		t.Fatalf("events: %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatalf("events first line: %q %v", line, err)
	}
	var ev struct {
		Seq  uint64 `json:"seq"`
		Name string `json:"name"`
	}
	if json.Unmarshal([]byte(line), &ev) != nil || ev.Name == "" || ev.Seq <= since {
		t.Fatalf("events first line %q (since %d)", line, since)
	}
	cancel()

	// a life that is not this one is a 409, which the page answers by reading the facts again
	if c, _, _ := do(t, "GET", srv.URL+"/engine/v1/events?life=not-this-life&since=0", "", nil); c != 409 {
		t.Fatalf("events with a foreign life: %d", c)
	}
}

func itoa(n uint64) string {
	b, _ := json.Marshal(n)
	return string(b)
}

func TestPageHasNoClockAndNoTokenInAURL(t *testing.T) {
	page := string(web.Index)
	for _, bad := range []string{"setInterval(", "setTimeout("} {
		if strings.Contains(page, bad) {
			t.Errorf("the page holds %s", bad)
		}
	}
	// the token travels only as a header
	if strings.Contains(page, "token=") && !strings.Contains(page, "[#&]token=") {
		t.Errorf("the page uses token= beyond reading the address fragment")
	}
	for _, m := range regexp.MustCompile(`(?:fetch|call)\([^)]*token`).FindAllString(page, -1) {
		t.Errorf("a call mentions the token: %s", m)
	}
	if !strings.Contains(page, "'Authorization']='Bearer '+TOKEN") {
		t.Errorf("the page does not send the token as a Bearer header")
	}
	// facts and events carry it too: follow() sets the header on its own fetch
	if n := strings.Count(page, "Bearer '+TOKEN"); n < 2 {
		t.Errorf("Bearer header set %d times; call() and the events fetch both need it", n)
	}
}
