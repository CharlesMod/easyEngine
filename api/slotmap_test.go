package api

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestOneObjectFailsClosed(t *testing.T) {
	deep := strings.Repeat(`{"a":`, maxJSONDepth+1) + `1` + strings.Repeat(`}`, maxJSONDepth+1)
	for _, in := range []string{``, `  `, `nope`, `[1,2]`, `"x"`, `null`, `1`, `{"a":1} {"b":2}`, `{"a":1}x`,
		"\xEF\xBB\xBF{\"id_slot\":0}", deep, `{"a":`} {
		if obj, why := oneObject([]byte(in)); why == "" || obj != nil {
			t.Errorf("%.40q: taken (%v)", in, obj)
		}
	}
	ok := strings.Repeat(`[`, maxJSONDepth-1) + strings.Repeat(`]`, maxJSONDepth-1)
	for _, in := range []string{`{}`, ` {"id_slot":1} `, `{"s":"[[[[{{{{\"]]"}`, `{"x":` + ok + `}`} {
		if _, why := oneObject([]byte(in)); why != "" {
			t.Errorf("%.40q refused: %s", in, why)
		}
	}
}

func TestSlotDigits(t *testing.T) {
	for _, in := range []string{"", "1x", " 1", "+1", "-1", "1 ", "0x1", "１", "1234567"} {
		if _, ok := slotDigits(in); ok {
			t.Errorf("%q taken", in)
		}
	}
	for in, want := range map[string]int{"0": 0, "7": 7, "012": 12} {
		if n, ok := slotDigits(in); !ok || n != want {
			t.Errorf("%q: %d %v", in, n, ok)
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
