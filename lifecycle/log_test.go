package lifecycle

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestLogSinkOnLine(t *testing.T) {
	s, err := OpenLog(filepath.Join(t.TempDir(), "x.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var got []string
	s.Write([]byte("before the hook\n"))
	s.OnLine(func(l string) { got = append(got, l) })
	s.Write([]byte("one\ntw"))
	s.Write([]byte("o\r\nthree"))
	s.Write([]byte("\n"))
	if want := []string{"one", "two", "three"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("lines = %q, want %q", got, want)
	}
	if tail := s.Tail(); tail != "before the hook\none\ntwo\r\nthree\n" {
		t.Fatalf("tail = %q", tail)
	}
	s.Close()
	s.Write([]byte("after close\n")) // the ring and the hook still see a child's last words
	if got[len(got)-1] != "after close" {
		t.Fatalf("lines after close = %q", got)
	}
}
