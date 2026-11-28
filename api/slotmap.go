package api

import (
	"bytes"
	"encoding/json"
	"io"
	"strconv"
)

func slotInt(raw json.RawMessage) (int, bool) {
	n, err := strconv.Atoi(string(bytes.TrimSpace(raw)))
	return n, err == nil
}

// mapSlotObject rewrites id_slot in one JSON object; false when nothing changed.
func mapSlotObject(raw json.RawMessage, toClient func(int) (int, bool)) (json.RawMessage, bool) {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil || obj == nil {
		return raw, false
	}
	v, ok := obj["id_slot"]
	if !ok {
		return raw, false
	}
	n, isInt := slotInt(v)
	if !isInt {
		return raw, false
	}
	c, ok := toClient(n)
	if !ok {
		return raw, false
	}
	obj["id_slot"] = json.RawMessage(strconv.Itoa(c))
	b, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}
	return b, true
}

// mapReplySlots rewrites a top-level "id_slot" integer of a JSON object, or of each object of a
// top-level array, from engine numbering to the client's. Unchanged bytes come back as given.
func mapReplySlots(body []byte, toClient func(engine int) (int, bool)) ([]byte, bool) {
	trim := bytes.TrimSpace(body)
	if len(trim) == 0 {
		return body, false
	}
	switch trim[0] {
	case '{':
		out, ch := mapSlotObject(trim, toClient)
		if !ch {
			return body, false
		}
		return out, true
	case '[':
		var arr []json.RawMessage
		if json.Unmarshal(trim, &arr) != nil {
			return body, false
		}
		any := false
		for i := range arr {
			if o, ch := mapSlotObject(arr[i], toClient); ch {
				arr[i] = o
				any = true
			}
		}
		if !any {
			return body, false
		}
		b, err := json.Marshal(arr)
		if err != nil {
			return body, false
		}
		return b, true
	}
	return body, false
}

type slotStream struct {
	rc       io.ReadCloser
	br       *bufioReader
	toClient func(int) (int, bool)
	out      []byte
	err      error
}

// bufioReader is a minimal line reader: it returns a whole line as soon as one has arrived.
type bufioReader struct {
	r   io.Reader
	buf []byte
}

// line returns the next "\n"-ended line (or the final unterminated rest with io.EOF).
func (b *bufioReader) line() ([]byte, error) {
	for {
		if i := bytes.IndexByte(b.buf, '\n'); i >= 0 {
			l := append([]byte(nil), b.buf[:i+1]...)
			b.buf = b.buf[i+1:]
			return l, nil
		}
		tmp := make([]byte, 32<<10)
		n, err := b.r.Read(tmp)
		b.buf = append(b.buf, tmp[:n]...)
		if err != nil {
			if len(b.buf) > 0 && bytes.IndexByte(b.buf, '\n') < 0 {
				l := b.buf
				b.buf = nil
				return l, err
			}
			if err == io.EOF && len(b.buf) == 0 {
				return nil, err
			}
			if bytes.IndexByte(b.buf, '\n') >= 0 {
				continue // deliver the complete lines first; the error returns after
			}
			return nil, err
		}
	}
}

// newSlotStream wraps an SSE body, mapping id_slot in "data:" lines to the client's numbering.
// Every other line passes unchanged, and a whole line is returned as soon as it has arrived.
func newSlotStream(rc io.ReadCloser, toClient func(engine int) (int, bool)) io.ReadCloser {
	return &slotStream{rc: rc, br: &bufioReader{r: rc}, toClient: toClient}
}

var dataPrefix = []byte("data:")
var slotKey = []byte(`"id_slot"`)

func (s *slotStream) rewrite(line []byte) []byte {
	if !bytes.HasPrefix(line, dataPrefix) || !bytes.Contains(line, slotKey) {
		return line
	}
	end := len(line)
	for end > 0 && (line[end-1] == '\n' || line[end-1] == '\r') {
		end--
	}
	ending := line[end:]
	rest := line[len(dataPrefix):end]
	sp := 0
	for sp < len(rest) && (rest[sp] == ' ' || rest[sp] == '\t') {
		sp++
	}
	mapped, ch := mapReplySlots(rest[sp:], s.toClient)
	if !ch {
		return line
	}
	out := make([]byte, 0, len(line)+8)
	out = append(out, dataPrefix...)
	out = append(out, rest[:sp]...)
	out = append(out, mapped...)
	return append(out, ending...)
}

func (s *slotStream) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for len(s.out) == 0 {
		if s.err != nil {
			return 0, s.err
		}
		l, err := s.br.line()
		if err != nil {
			s.err = err
		}
		if len(l) > 0 {
			s.out = s.rewrite(l)
		}
	}
	n := copy(p, s.out)
	s.out = s.out[n:]
	return n, nil
}

func (s *slotStream) Close() error { return s.rc.Close() }
