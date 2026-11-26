// Package lifecycle holds process supervision primitives: children that die
// with the engine, parent-pid watching, the single-instance lock and log files.
package lifecycle

import (
	"os"
	"path/filepath"
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

// StdioLogName is the file under logs/ that the engine's own fd 1 and 2 go to once detached
// from the host (app.detachStdio). Everything the engine says goes through engine.log's
// LogSink; only what bypasses it lands here: a Go crash trace and a last exit line. It is a
// separate file because a raw descriptor on engine.log would keep writing to the renamed copy
// after the sink rotates it, and hold its disk after the copy is deleted.
const StdioLogName = "engine-stdio.log"

// StdioLogCap: RotateStdioLog moves a stdio log at or past this to .1 (one old copy kept).
var StdioLogCap int64 = 1 << 20

// stdioRunMax bounds what one run writes to the stdio log: a crash trace (every goroutine's
// stack, the engine runs a few dozen) and one exit line, far under 1 MiB.
const stdioRunMax = 1 << 20

// StdioLogCeiling is the most the stdio log and its old copy take: each starts a run under the
// cap and gains at most one run's writes.
func StdioLogCeiling() int64 { return 2 * roundBlock(StdioLogCap+stdioRunMax) }

// RotateStdioLog readies the stdio log at path for this run: at or past the cap, it becomes
// path.1 (replacing the old copy).
func RotateStdioLog(path string) {
	if st, err := os.Stat(path); err == nil && st.Size() >= StdioLogCap {
		os.Rename(path, path+".1")
	}
}

func roundBlock(n int64) int64 { return (n + 4095) / 4096 * 4096 }
