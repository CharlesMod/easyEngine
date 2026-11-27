package app

import (
	"io"
	"net"
	"sync"
	"testing"

	"github.com/CharlesMod/easyEngine/discover"
)

// memNet is an in-memory discover.Transport: it records what would be sent, sends nothing.
type memNet struct {
	mu    sync.Mutex
	sent  [][]byte
	done  chan struct{}
	once  sync.Once
	close int
}

func (m *memNet) Read() (discover.Packet, error) { <-m.done; return discover.Packet{}, io.EOF }
func (m *memNet) Multicast(b []byte, _ int) error {
	m.mu.Lock()
	m.sent = append(m.sent, b)
	m.mu.Unlock()
	return nil
}
func (m *memNet) Unicast(b []byte, _ *net.UDPAddr) error { return m.Multicast(b, 0) }
func (m *memNet) Addrs(int) []net.IP                     { return []net.IP{net.IPv4(192, 0, 2, 9)} }
func (m *memNet) Refresh() error                         { return nil }
func (m *memNet) Close() error {
	m.once.Do(func() { close(m.done) })
	m.mu.Lock()
	m.close++
	m.mu.Unlock()
	return nil
}
func (m *memNet) names() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []string
	for _, b := range m.sent {
		for _, f := range discover.Parse(b) {
			out = append(out, f.Name)
		}
	}
	return out
}

func noChanges() (<-chan struct{}, func(), error) { return nil, func() {}, nil }

// The announcer follows the settings: on starts the responder (an announcement), a new name
// re-announces, off says goodbye and stops; Quit ends it for good.
func TestAnnouncerFollowsTheSettings(t *testing.T) {
	m := &memNet{done: make(chan struct{})}
	an := &announcer{eligible: true, opts: discover.Options{Transport: m, NetChanges: noChanges}}
	info := discover.Info{ID: "0123456789abcdef", Name: "den", API: 1, Port: 47480}
	an.apply(true, info)
	if !an.running() || len(m.names()) != 1 || m.names()[0] != "den" {
		t.Fatalf("start: %v %q", an.running(), m.names())
	}
	info.Name = "study"
	an.apply(true, info)
	if n := m.names(); n[len(n)-1] != "study" {
		t.Fatalf("rename: %q", n)
	}
	an.apply(false, info)
	if an.running() || m.close != 1 {
		t.Fatalf("off: running %v, closed %d", an.running(), m.close)
	}
	an.close()
	an.apply(true, info)
	if an.running() {
		t.Fatal("started after Quit")
	}

	// a listener that may not announce never opens the transport
	m2 := &memNet{done: make(chan struct{})}
	off := &announcer{eligible: false, opts: discover.Options{Transport: m2, NetChanges: noChanges}}
	off.apply(true, info)
	if off.running() || len(m2.sent) != 0 {
		t.Fatal("an ineligible listener announced")
	}
}
