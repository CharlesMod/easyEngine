package api

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

func testToEngine(c int) (int, error) {
	if c > 3 {
		return 0, errors.New("slot out of range")
	}
	return c + 10, nil
}

func TestSlotRequestMapping(t *testing.T) {
	pick := func() int { return 7 }
	cases := []struct{ name, in, want string }{
		{"present", `{"id_slot":2,"prompt":"hi"}`, `{"id_slot":12,"prompt":"hi"}`},
		{"absent", `{"prompt":"hi"}`, `{"id_slot":7,"prompt":"hi"}`},
		{"minus one", `{"id_slot":-1,"prompt":"hi"}`, `{"id_slot":7,"prompt":"hi"}`},
		{"null", `{"id_slot":null}`, `{"id_slot":7}`},
	}
	for _, c := range cases {
		out, ch, err := mapRequestSlot([]byte(c.in), testToEngine, pick)
		if err != nil || !ch || string(out) != c.want {
			t.Errorf("%s: got %s %v %v", c.name, out, ch, err)
		}
	}
	for _, in := range []string{`{"id_slot":"1"}`, `{"id_slot":1.5}`, `{"id_slot":-2}`} {
		_, _, err := mapRequestSlot([]byte(in), testToEngine, pick)
		var se *SlotError
		if !errors.As(err, &se) || se.Msg != "id_slot must be an integer" {
			t.Errorf("%s: err %v", in, err)
		}
	}
	if _, _, err := mapRequestSlot([]byte(`{"id_slot":9}`), testToEngine, pick); err == nil || err.Error() != "slot out of range" {
		t.Errorf("out of range: %v", err)
	}
	for _, in := range []string{``, `nope`, `[1,2]`, `"x"`, `null`} {
		out, ch, err := mapRequestSlot([]byte(in), testToEngine, pick)
		if ch || err != nil || string(out) != in {
			t.Errorf("non-object %q: %s %v %v", in, out, ch, err)
		}
	}
}

func TestSlotRequestKeepsOtherValues(t *testing.T) {
	in := `{"messages":[{"role":"user","content":"aé  b"}],"temperature":0.70,"id_slot":1,"big":12345678901234567890}`
	out, _, err := mapRequestSlot([]byte(in), testToEngine, func() int { return 0 })
	if err != nil {
		t.Fatal(err)
	}
	for _, frag := range []string{`"messages":[{"role":"user","content":"aé  b"}]`, `"temperature":0.70`, `"big":12345678901234567890`, `"id_slot":11`} {
		if !strings.Contains(string(out), frag) {
			t.Errorf("missing %s in %s", frag, out)
		}
	}
}

func testToClient(e int) (int, bool) {
	if e >= 10 && e < 14 {
		return e - 10, true
	}
	return 0, false
}

func TestSlotReplyMapping(t *testing.T) {
	out, ch := mapReplySlots([]byte(`{"id_slot":12,"x":1}`), testToClient)
	if !ch || string(out) != `{"id_slot":2,"x":1}` {
		t.Errorf("object: %s %v", out, ch)
	}
	out, ch = mapReplySlots([]byte(`[{"id":0,"id_slot":10},{"id":1,"id_slot":99},{"id":2}]`), testToClient)
	if !ch || string(out) != `[{"id":0,"id_slot":0},{"id":1,"id_slot":99},{"id":2}]` {
		t.Errorf("array: %s %v", out, ch)
	}
	in := []byte(`{"a": 1,  "b":2}`)
	out, ch = mapReplySlots(in, testToClient)
	if ch || !bytes.Equal(out, in) {
		t.Errorf("no id_slot: %s %v", out, ch)
	}
	in = []byte(`{"id_slot":99}`)
	out, ch = mapReplySlots(in, testToClient)
	if ch || !bytes.Equal(out, in) {
		t.Errorf("toClient false: %s %v", out, ch)
	}
	in = []byte(`not json`)
	if out, ch = mapReplySlots(in, testToClient); ch || !bytes.Equal(out, in) {
		t.Errorf("not json: %s %v", out, ch)
	}
}

func TestSlotStreamBody(t *testing.T) {
	in := ": easy waiting\n\n" +
		"data: {\"id_slot\":11,\"n\":1}\n\n" +
		"data:{\"id_slot\":12,\"n\":2}\r\n\r\n" +
		"data: {\"n\":3}\n\n" +
		"data: {\"id_slot\":99}\n\n" +
		"data: [DONE]\n\n" +
		"data: {\"id_slot\":10}" // no final newline
	want := ": easy waiting\n\n" +
		"data: {\"id_slot\":1,\"n\":1}\n\n" +
		"data:{\"id_slot\":2,\"n\":2}\r\n\r\n" +
		"data: {\"n\":3}\n\n" +
		"data: {\"id_slot\":99}\n\n" +
		"data: [DONE]\n\n" +
		"data: {\"id_slot\":0}"
	rc := newSlotStream(io.NopCloser(strings.NewReader(in)), testToClient)
	got, err := io.ReadAll(rc)
	if err != nil || string(got) != want {
		t.Fatalf("got %q err %v", got, err)
	}
	if _, err := rc.Read(make([]byte, 4)); err != io.EOF {
		t.Errorf("read after EOF: %v", err)
	}
	if err := rc.Close(); err != nil {
		t.Error(err)
	}
}

func TestSlotStreamNoHoldBack(t *testing.T) {
	pr, pw := io.Pipe()
	rc := newSlotStream(pr, testToClient)
	got := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		pw.Write([]byte("data: {\"id_slot\":14,\"id_slot2\":1}\n\n")) // 14 stays: toClient false
		pw.Write([]byte("data: {\"id_slot\":13}\n\n"))
		<-got
		pw.Close()
	}()
	buf := make([]byte, 1024)
	n, err := rc.Read(buf)
	if err != nil || string(buf[:n]) != "data: {\"id_slot\":14,\"id_slot2\":1}\n" {
		t.Fatalf("first read %q %v", buf[:n], err)
	}
	rest, err := io.ReadAll(io.LimitReader(rc, int64(len("\ndata: {\"id_slot\":3}\n\n"))))
	if err != nil || string(rest) != "\ndata: {\"id_slot\":3}\n\n" {
		t.Fatalf("rest %q %v", rest, err)
	}
	close(got)
	<-done
}

func TestSlotStreamEventBeforeMoreWritten(t *testing.T) {
	pr, pw := io.Pipe()
	rc := newSlotStream(pr, testToClient)
	got := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		pw.Write([]byte("data: {\"id_slot\":14,\"a\":1}\n\n"))
		pw.Write([]byte("data: {\"id_slot\":11}\n\n"))
		<-got // the reader has the first event before we go on
		pw.Close()
	}()
	// the pipe write returns only once the reader consumed it, so the first event is
	// delivered while the writer is still in its second write
	want := "data: {\"id_slot\":14,\"a\":1}\n"
	buf := make([]byte, len(want))
	if _, err := io.ReadFull(rc, buf); err != nil || string(buf) != want {
		t.Fatalf("got %q %v", buf, err)
	}
	close(got)
	rest, _ := io.ReadAll(rc)
	if string(rest) != "\ndata: {\"id_slot\":1}\n\n" {
		t.Errorf("rest %q", rest)
	}
	<-done
}

func TestSlotStreamLongLine(t *testing.T) {
	big := strings.Repeat("x", 1<<20)
	in := "data: {\"id_slot\":12,\"t\":\"" + big + "\"}\n\n"
	rc := newSlotStream(io.NopCloser(strings.NewReader(in)), testToClient)
	got, err := io.ReadAll(rc)
	want := "data: {\"id_slot\":2,\"t\":\"" + big + "\"}\n\n"
	if err != nil || string(got) != want {
		t.Fatalf("len %d want %d err %v", len(got), len(want), err)
	}
}
