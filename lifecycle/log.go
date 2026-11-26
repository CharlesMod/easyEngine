// Package lifecycle holds process supervision primitives: children that die
// with the engine, parent-pid watching, the single-instance lock and log files.
package lifecycle

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// MaxLogSize is the size at which a log file is rotated (two old copies are kept).
var MaxLogSize int64 = 10 << 20

const ringSize = 64 << 10

// LogSink is an io.WriteCloser that appends to a rotating log file and keeps the
// last 64 KiB in memory (used to read a child's final words, e.g. an OOM message).
type LogSink struct {
	mu   sync.Mutex
	path string
	f    *os.File
	size int64
	ring []byte
	hook func(line string) // called with each complete line, under mu (keep it short)
	part []byte            // the unfinished line the hook has not seen yet
}

// OnLine sets a function called with every complete line written from now on (without the
// newline). It runs under the sink's lock, so it must be quick and must not call the sink.
func (s *LogSink) OnLine(f func(line string)) {
	s.mu.Lock()
	s.hook, s.part = f, nil
	s.mu.Unlock()
}

// OpenLog opens (appending) a rotating log file. A nil path keeps only the ring.
func OpenLog(path string) (*LogSink, error) {
	s := &LogSink{path: path}
	if path == "" {
		return s, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := s.open(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *LogSink) open() error {
	if st, err := os.Stat(s.path); err == nil && st.Size() >= MaxLogSize {
		os.Remove(s.path + ".2")
		os.Rename(s.path+".1", s.path+".2")
		os.Rename(s.path, s.path+".1")
	}
	f, err := os.OpenFile(s.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	st, _ := f.Stat()
	s.f = f
	if st != nil {
		s.size = st.Size()
	} else {
		s.size = 0
	}
	return nil
}

func (s *LogSink) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.hook != nil {
		s.lines(p)
	}
	s.ring = append(s.ring, p...)
	if len(s.ring) > 2*ringSize {
		s.ring = append([]byte(nil), s.ring[len(s.ring)-ringSize:]...)
	}
	if s.f != nil {
		if s.size >= MaxLogSize {
			s.f.Close()
			if err := s.open(); err != nil {
				return len(p), nil
			}
		}
		n, _ := s.f.Write(p)
		s.size += int64(n)
	}
	return len(p), nil
}

func (s *LogSink) lines(p []byte) {
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			if len(s.part) < 4096 { // a line longer than this is not one the hook needs whole
				s.part = append(s.part, p...)
			}
			return
		}
		line := append(s.part, p[:i]...)
		s.part = nil
		s.hook(strings.TrimRight(string(line), "\r"))
		p = p[i+1:]
	}
}

// Tail returns the most recent output kept in memory.
func (s *LogSink) Tail() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.ring
	if len(r) > ringSize {
		r = r[len(r)-ringSize:]
	}
	return string(r)
}

func (s *LogSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.f != nil {
		err := s.f.Close()
		s.f = nil
		return err
	}
	return nil
}
