package event

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var testBook = []Decl{
	{Name: "phase", Kind: Fact, What: "what the engine is doing"},
	{Name: "mind.shape", Kind: Fact, What: "the shape the mind runs at"},
	{Name: "client", Kind: Fact, Keyed: true, What: "one attached client"},
	{Name: "install.progress", Kind: Reading, What: "bytes of the install so far"},
	{Name: "error", Kind: Happening, What: "something went wrong"},
	{Name: "request", Kind: Happening, Keyed: true, What: "a request ended"},
}

// at is a clock that only moves when a test says, so At is data the test
// can check.
func newHub(t *testing.T) *Hub {
	t.Helper()
	base := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)
	var mu sync.Mutex
	n := 0
	h, err := New(testBook, func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		n++
		return base.Add(time.Duration(n) * time.Second)
	})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func say(t *testing.T, h *Hub, e Event, body any) Event {
	t.Helper()
	got, err := h.Say(e, body)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestTheBookIsChecked(t *testing.T) {
	if _, err := New([]Decl{{Name: "a", Kind: Fact}, {Name: "a", Kind: Fact}}, nil); err == nil {
		t.Error("a name declared twice was taken")
	}
	if _, err := New([]Decl{{Name: "a", Kind: "level"}}, nil); err == nil {
		t.Error("an unknown kind was taken")
	}
	if _, err := New([]Decl{{Kind: Fact}}, nil); err == nil {
		t.Error("an empty name was taken")
	}
	h := newHub(t)
	b := h.Book()
	if len(b) != len(testBook) || b[0].Name != "client" || b[len(b)-1].Name != "request" {
		t.Errorf("book not sorted whole: %v", b)
	}
	for _, c := range []struct {
		e    Event
		what string
	}{
		{Event{Name: "nope"}, "undeclared"},
		{Event{Name: "phase", Key: "x"}, "key on an unkeyed name"},
		{Event{Name: "client"}, "no key on a keyed name"},
		{Event{Name: "phase", Kind: Happening}, "wrong kind"},
		{Event{Name: "error", Gone: true}, "a happening said gone"},
		{Event{}, "no name"},
	} {
		var r Refused
		if _, err := h.Say(c.e, nil); !errors.As(err, &r) {
			t.Errorf("%s: %v, want Refused", c.what, err)
		}
	}
	if _, err := h.Say(Event{Name: "phase"}, strings.Repeat("x", BodyMax)); err == nil {
		t.Error("a body over BodyMax was taken")
	}
	if _, err := h.Say(Event{Name: "phase"}, func() {}); err == nil {
		t.Error("a body that cannot be written was taken")
	}
	if h.Seq() != 0 {
		t.Errorf("refusals took seqs: %d", h.Seq())
	}
}

func TestSayOrdersAndStamps(t *testing.T) {
	h := newHub(t)
	a := say(t, h, Event{Name: "phase"}, "loading")
	b := say(t, h, Event{Name: "error"}, map[string]string{"why": "x"})
	c := say(t, h, Event{Name: "phase", Kind: Fact}, "ready")
	if a.Seq != 1 || b.Seq != 2 || c.Seq != 3 || h.Seq() != 3 {
		t.Fatalf("seqs %d %d %d (hub %d)", a.Seq, b.Seq, c.Seq, h.Seq())
	}
	if a.Life != h.Life() || len(h.Life()) != 16 || a.Kind != Fact || b.Kind != Happening {
		t.Errorf("envelope: %+v %+v", a, b)
	}
	if !a.At.Before(b.At) || !b.At.Before(c.At) {
		t.Errorf("At not from the injected clock: %v %v %v", a.At, b.At, c.At)
	}
	if string(c.Body) != `"ready"` {
		t.Errorf("body %s", c.Body)
	}
	if h2 := newHub(t); h2.Life() == h.Life() {
		t.Error("two hubs, one life")
	}
}

func TestAFactIsReplacedAndSaidGone(t *testing.T) {
	h := newHub(t)
	say(t, h, Event{Name: "client", Key: "a"}, 1)
	say(t, h, Event{Name: "client", Key: "b"}, 2)
	say(t, h, Event{Name: "client", Key: "a"}, 3)
	say(t, h, Event{Name: "error"}, "history")
	say(t, h, Event{Name: "install.progress"}, 10)
	fs := h.Facts("client")
	if len(fs) != 2 || fs[0].Key != "a" || string(fs[0].Body) != "3" || fs[1].Key != "b" {
		t.Fatalf("facts %+v", fs)
	}
	if len(h.Facts("")) != 2 || len(h.Readings("")) != 1 {
		t.Errorf("a happening stands, or a reading is a fact: %v %v", h.Facts(""), h.Readings(""))
	}
	if _, ok := h.Fact("install.progress", ""); ok {
		t.Error("a reading read as a fact")
	}
	if r, ok := h.Reading("install.progress", ""); !ok || string(r.Body) != "10" {
		t.Errorf("reading %+v %v", r, ok)
	}
	g := say(t, h, Event{Name: "client", Key: "a", Gone: true}, "ignored")
	if !g.Gone || g.Body != nil || g.Seq != 6 {
		t.Errorf("gone event %+v", g)
	}
	if _, ok := h.Fact("client", "a"); ok {
		t.Error("a fact said gone still stands")
	}
	// gone for what does not stand is nothing
	if n := say(t, h, Event{Name: "client", Key: "a", Gone: true}, nil); n.Seq != 0 || h.Seq() != 6 {
		t.Errorf("gone twice was said: %+v, seq %d", n, h.Seq())
	}
}

func TestTheSameBodyIsSaidOnce(t *testing.T) {
	h := newHub(t)
	w, err := h.Wake("phase")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	first := say(t, h, Event{Name: "phase"}, map[string]any{"phase": "ready", "port": 1})
	<-w.C()
	again := say(t, h, Event{Name: "phase"}, map[string]any{"port": 1, "phase": "ready"}) // same JSON: map keys are sorted
	if again.Seq != first.Seq || h.Seq() != 1 || !again.At.Equal(first.At) {
		t.Fatalf("an unchanged fact was said again: %+v vs %+v", again, first)
	}
	select {
	case <-w.C():
		t.Fatal("an unchanged fact woke a reader")
	default:
	}
	if ev, _ := h.Since(0, ""); len(ev) != 1 {
		t.Errorf("history took the unchanged fact: %v", ev)
	}
	if r := say(t, h, Event{Name: "install.progress"}, 5); r.Seq != 2 {
		t.Fatal(r)
	}
	if r := say(t, h, Event{Name: "install.progress"}, 5); r.Seq != 2 {
		t.Errorf("an unchanged reading was said again: %+v", r)
	}
	// a happening is said every time
	say(t, h, Event{Name: "error"}, "x")
	if e := say(t, h, Event{Name: "error"}, "x"); e.Seq != 4 {
		t.Errorf("a repeated happening: %+v", e)
	}
}

func TestWakesCoalesce(t *testing.T) {
	h := newHub(t)
	w, err := h.Wake("client", "phase")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	other, _ := h.Wake("error")
	defer other.Close()
	for i := 1; i <= 100; i++ {
		say(t, h, Event{Name: "client", Key: "a"}, i) // the sayer never blocks
	}
	<-w.C()
	select {
	case <-w.C():
		t.Fatal("100 says gave more than one wake")
	default:
	}
	if f, _ := h.Fact("client", "a"); string(f.Body) != "100" {
		t.Errorf("the reader reads %s, want the last", f.Body)
	}
	select {
	case <-other.C():
		t.Fatal("a reader of another name was woken")
	default:
	}
	w.Close()
	say(t, h, Event{Name: "phase"}, "x")
	select {
	case <-w.C():
		t.Fatal("a closed waker was woken")
	default:
	}
	if _, err := h.Wake("nothing."); err == nil {
		t.Error("a prefix no name begins with was taken")
	}
	if _, err := h.Wake(); err == nil {
		t.Error("a waker for nothing was taken")
	}
	if all, err := h.Wake(""); err != nil {
		t.Error(err)
	} else {
		all.Close()
	}
}

func TestSayIfComparesBeforeSaying(t *testing.T) {
	h := newHub(t)
	say(t, h, Event{Name: "phase"}, "loading")
	read := h.Seq()
	e, at, err := h.SayIf(Event{Name: "mind.shape"}, "4x8k", map[string]uint64{"phase": read})
	if err != nil || e.Seq != 2 || at != 2 {
		t.Fatalf("SayIf on what stood: %+v %d %v", e, at, err)
	}
	// a sayer of several acts passes on the place its own saying landed at
	if e, at, _ = h.SayIf(Event{Name: "mind.shape"}, "2x8k", map[string]uint64{"phase": read, "mind.shape": at}); e.Seq != 3 || at != 3 {
		t.Fatalf("second act: %+v %d", e, at)
	}
	say(t, h, Event{Name: "phase"}, "ready")
	e, at, err = h.SayIf(Event{Name: "mind.shape"}, "1x8k", map[string]uint64{"phase": read})
	if err != nil || e.Seq != 0 || at != 0 {
		t.Fatalf("SayIf over a changed fact said: %+v %d %v", e, at, err)
	}
	if f, _ := h.Fact("mind.shape", ""); string(f.Body) != `"2x8k"` {
		t.Errorf("shape %s", f.Body)
	}
}

func TestSinceAndTheGap(t *testing.T) {
	h := newHub(t)
	say(t, h, Event{Name: "phase"}, "a")
	say(t, h, Event{Name: "error"}, "b")
	say(t, h, Event{Name: "phase"}, "c")
	ev, gap := h.Since(1, "")
	if gap || len(ev) != 2 || ev[0].Seq != 2 || ev[1].Seq != 3 {
		t.Fatalf("since 1: %v %v", ev, gap)
	}
	if ev, _ := h.Since(0, "phase"); len(ev) != 2 || ev[1].Seq != 3 {
		t.Errorf("by prefix: %v", ev)
	}
	if ev, _ := h.Since(3, ""); len(ev) != 0 {
		t.Errorf("since now: %v", ev)
	}
	for i := 0; i < RingKeep; i++ {
		say(t, h, Event{Name: "error"}, i)
	}
	ev, gap = h.Since(2, "")
	if !gap || len(ev) != RingKeep || ev[0].Seq != h.Seq()-RingKeep+1 {
		t.Fatalf("since a let-go seq: gap %v, %d events from %d", gap, len(ev), ev[0].Seq)
	}
	if ev, gap = h.Since(h.Seq()-RingKeep, ""); gap || len(ev) != RingKeep {
		t.Errorf("the oldest kept place is no gap: %v %d", gap, len(ev))
	}
	if _, ok := h.Fact("phase", ""); !ok {
		t.Error("the ring let a standing fact go")
	}
}

func TestConcurrentSays(t *testing.T) {
	h := newHub(t)
	w, _ := h.Wake("")
	defer w.Close()
	var wg sync.WaitGroup
	const sayers, each = 8, 500
	for s := 0; s < sayers; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < each; i++ {
				if _, err := h.Say(Event{Name: "client", Key: strconv.Itoa(s)}, i); err != nil {
					t.Error(err)
				}
				if _, err := h.Say(Event{Name: "request", Key: "r"}, i); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	readers := make(chan struct{})
	go func() { // a reader reading while they say
		for range w.C() {
			h.Facts("")
			h.Since(0, "")
			if h.Seq() >= sayers*each*2 {
				close(readers)
				return
			}
		}
	}()
	wg.Wait()
	say(t, h, Event{Name: "phase"}, "done") // one more wake after the last
	<-readers
	ev, _ := h.Since(0, "")
	for i := 1; i < len(ev); i++ {
		if ev[i].Seq != ev[i-1].Seq+1 {
			t.Fatalf("history out of order at %d: %d after %d", i, ev[i].Seq, ev[i-1].Seq)
		}
	}
	if fs := h.Facts("client"); len(fs) != sayers || string(fs[0].Body) != strconv.Itoa(each-1) {
		t.Errorf("facts %v", fs)
	}
}

// follow opens a stream and answers a line reader.
func follow(t *testing.T, srv *httptest.Server, q string) (*http.Response, *bufio.Reader) {
	t.Helper()
	resp, err := http.Get(srv.URL + "/events?" + q)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp, bufio.NewReader(resp.Body)
}

func line(t *testing.T, r *bufio.Reader) map[string]any {
	t.Helper()
	l, err := r.ReadBytes('\n')
	if err != nil {
		t.Fatalf("reading a line: %v (%q)", err, l)
	}
	var m map[string]any
	if err := json.Unmarshal(l, &m); err != nil {
		t.Fatalf("line %q: %v", l, err)
	}
	return m
}

// serve mounts h on a test server; the server is closed after the streams'
// bodies (cleanups run last-made first), so its Close does not wait on them.
func serve(t *testing.T, h *Hub) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/facts", h.ServeFacts)
	mux.HandleFunc("/events", h.ServeEvents)
	mux.HandleFunc("/book", h.ServeBook)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestTheStreamSaysEachEventAsSaid(t *testing.T) {
	h := newHub(t)
	srv := serve(t, h)
	say(t, h, Event{Name: "phase"}, "loading")
	say(t, h, Event{Name: "error"}, "before")

	resp, err := http.Get(srv.URL + "/facts")
	if err != nil {
		t.Fatal(err)
	}
	var st Standing
	json.NewDecoder(resp.Body).Decode(&st)
	resp.Body.Close()
	if st.Life != h.Life() || st.Seq != 2 || len(st.Facts) != 1 || st.Facts[0].Name != "phase" {
		t.Fatalf("facts %+v", st)
	}

	res, r := follow(t, srv, "life="+h.Life()+"&since=1")
	if ct := res.Header.Get("Content-Type"); ct != "application/x-ndjson" {
		t.Errorf("content type %q", ct)
	}
	if m := line(t, r); m["seq"] != 2.0 || m["name"] != "error" {
		t.Fatalf("history first: %v", m)
	}
	for i := 0; i < 5; i++ {
		e := say(t, h, Event{Name: "client", Key: "k"}, i)
		if m := line(t, r); m["seq"] != float64(e.Seq) || m["key"] != "k" || m["body"] != float64(i) {
			t.Fatalf("line %d: %v", i, m)
		}
	}
	say(t, h, Event{Name: "client", Key: "k", Gone: true}, nil)
	if m := line(t, r); m["gone"] != true {
		t.Fatalf("gone line %v", m)
	}

	_, pr := follow(t, srv, "life="+h.Life()+"&since="+strconv.FormatUint(h.Seq(), 10)+"&prefix=phase")
	say(t, h, Event{Name: "error"}, "not for this one")
	say(t, h, Event{Name: "phase"}, "ready")
	if m := line(t, pr); m["name"] != "phase" || m["body"] != "ready" {
		t.Fatalf("prefix stream: %v", m)
	}

	resp, _ = http.Get(srv.URL + "/book")
	var book []Decl
	json.NewDecoder(resp.Body).Decode(&book)
	resp.Body.Close()
	if len(book) != len(testBook) || book[0].Name != "client" || !book[0].Keyed {
		t.Errorf("book %v", book)
	}
}

func TestTheStreamRefusesAnotherLife(t *testing.T) {
	h := newHub(t)
	srv := serve(t, h)
	say(t, h, Event{Name: "phase"}, "x")
	for _, q := range []string{"life=feedfacefeedface&since=0", "since=0", "life=" + h.Life() + "&since=9"} {
		resp, err := http.Get(srv.URL + "/events?" + q)
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Life string
			Seq  uint64
		}
		json.NewDecoder(resp.Body).Decode(&body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusConflict || body.Life != h.Life() || body.Seq != 1 {
			t.Errorf("%s: %d %+v, want 409 with the life", q, resp.StatusCode, body)
		}
	}
	for _, q := range []string{"life=" + h.Life() + "&since=x", "life=" + h.Life() + "&since=0&prefix=nothing"} {
		resp, _ := http.Get(srv.URL + "/events?" + q)
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: %d, want 400", q, resp.StatusCode)
		}
	}
}

func TestTheStreamEndsWithAGap(t *testing.T) {
	h := newHub(t)
	srv := serve(t, h)
	for i := 0; i < RingKeep+10; i++ {
		say(t, h, Event{Name: "error"}, i)
	}
	_, r := follow(t, srv, "life="+h.Life()+"&since=1")
	if m := line(t, r); m["gap"] != true || len(m) != 1 {
		t.Fatalf("want a gap line, got %v", m)
	}
	if _, err := r.ReadByte(); err != io.EOF {
		t.Fatalf("the stream went on after the gap: %v", err)
	}
}

func TestTheStreamEndsWhenTheHubCloses(t *testing.T) {
	h := newHub(t)
	srv := serve(t, h)
	_, r := follow(t, srv, "life="+h.Life()+"&since=0")
	say(t, h, Event{Name: "phase"}, "up")
	if m := line(t, r); m["body"] != "up" {
		t.Fatal(m)
	}
	h.Close()
	if _, err := r.ReadByte(); err != io.EOF {
		t.Fatalf("the stream did not end with the hub: %v", err)
	}
	if _, err := h.Say(Event{Name: "phase"}, "after"); !errors.Is(err, ErrClosed) {
		t.Errorf("a closed hub said: %v", err)
	}
	h.Close() // twice is nothing
	_, r2 := follow(t, srv, "life="+h.Life()+"&since=0")
	if m := line(t, r2); m["body"] != "up" {
		t.Fatalf("a stream on a closed hub gives what was said: %v", m)
	}
	if _, err := r2.ReadByte(); err != io.EOF {
		t.Fatalf("a stream on a closed hub went on: %v", err)
	}
}
