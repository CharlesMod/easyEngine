package discover

import (
	"net"
	"sort"
	"sync"
	"testing"

	"golang.org/x/net/dns/dnsmessage"
)

// multiNet is the fake with an interface table: interface index -> its own addresses.
type multiNet struct {
	*fake
	mu    sync.Mutex
	table map[int][]net.IP
}

var (
	lanV4  = net.IPv4(192, 168, 1, 20)
	lanV6  = net.ParseIP("fd00::20")
	tsV4   = net.IPv4(100, 64, 0, 5)
	tsV6   = net.ParseIP("fd7a::5")
	peerLn = net.IPv4(192, 168, 1, 9)
	peerTs = net.IPv4(100, 64, 0, 9)
)

func newMulti() *multiNet {
	return &multiNet{fake: newFake(), table: map[int][]net.IP{1: {lanV4, lanV6}, 2: {tsV4, tsV6}}}
}

func (m *multiNet) set(table map[int][]net.IP) { m.mu.Lock(); m.table = table; m.mu.Unlock() }

func (m *multiNet) Interfaces() []int {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []int
	for i := range m.table {
		out = append(out, i)
	}
	sort.Ints(out)
	return out
}

func (m *multiNet) Addrs(i int) []net.IP {
	m.mu.Lock()
	defer m.mu.Unlock()
	if i != 0 {
		return m.table[i]
	}
	var all []net.IP
	for _, a := range m.table {
		all = append(all, a...)
	}
	return all
}

// Reach: the interfaces whose address shares the querier's first three (v4) bytes.
func (m *multiNet) Reach(from net.IP) []net.IP {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []net.IP
	for _, as := range m.table {
		for _, a := range as {
			if a.To4() != nil && from.To4() != nil && a.To4()[0] == from.To4()[0] && a.To4()[1] == from.To4()[1] && a.To4()[2] == from.To4()[2] {
				out = append(out, as...)
				break
			}
		}
	}
	return out
}

func startMulti(t *testing.T, m *multiNet, listen []net.IP) (*Responder, chan struct{}) {
	t.Helper()
	changes := make(chan struct{}, 1)
	r, err := Start(testInfo, Options{Transport: m, Listen: listen,
		NetChanges: func() (<-chan struct{}, func(), error) { return changes, func() {}, nil }})
	if err != nil {
		t.Fatal(err)
	}
	return r, changes
}

// has reports whether the addresses are exactly want.
func sameAddrs(got []net.IP, want ...net.IP) bool {
	if len(got) != len(want) {
		return false
	}
	for _, w := range want {
		ok := false
		for _, g := range got {
			ok = ok || g.Equal(w)
		}
		if !ok {
			return false
		}
	}
	return true
}

func TestOneListenedIPAnnouncesOnlyThatIPOnlyOnItsInterface(t *testing.T) {
	m := newMulti()
	r, _ := startMulti(t, m, []net.IP{lanV4})
	s := m.next(t)
	if s.kind != "multicast" || s.ifIndex != 1 {
		t.Fatalf("announcement %+v", s)
	}
	if ttl, f := ttlsOf(t, s.data); ttl == 0 || !sameAddrs(f.Addrs, lanV4) {
		t.Fatalf("announced %v", f.Addrs)
	}
	m.quiet(t) // interface 2 holds none of the set: nothing

	// a question on interface 2 is not answered; on interface 1 it carries the one IP
	m.feed(Packet{Data: Query(), From: &net.UDPAddr{IP: peerTs, Port: 5353}, IfIndex: 2})
	m.quiet(t)
	m.feed(Packet{Data: Query(), From: &net.UDPAddr{IP: peerLn, Port: 5353}, IfIndex: 1})
	s = m.next(t)
	if s.kind != "multicast" || s.ifIndex != 1 || len(Parse(s.data)) != 1 || !sameAddrs(Parse(s.data)[0].Addrs, lanV4) {
		t.Fatalf("answer %+v %v", s, Parse(s.data))
	}

	// the goodbye follows the same rule
	r.Close()
	s = m.next(t)
	if ttl, f := ttlsOf(t, s.data); s.ifIndex != 1 || ttl != 0 || !sameAddrs(f.Addrs, lanV4) {
		t.Fatalf("goodbye %+v ttl %d %v", s, ttl, f.Addrs)
	}
	if len(m.sends) != 0 { // Close has ended the reader: nothing is left to settle
		t.Fatal("a goodbye for the interface that holds none")
	}
}

func TestWildcardEachInterfaceCarriesItsOwn(t *testing.T) {
	m := newMulti()
	r, _ := startMulti(t, m, nil)
	want := map[int][]net.IP{1: {lanV4, lanV6}, 2: {tsV4, tsV6}}
	for range 2 {
		s := m.next(t)
		ttl, f := ttlsOf(t, s.data)
		if ttl == 0 || want[s.ifIndex] == nil || !sameAddrs(f.Addrs, want[s.ifIndex]...) {
			t.Fatalf("announcement on %d: %v", s.ifIndex, f.Addrs)
		}
		delete(want, s.ifIndex)
	}
	m.quiet(t)

	// a multicast answer on interface 2 has only interface 2's addresses
	m.feed(Packet{Data: Query(), From: &net.UDPAddr{IP: peerTs, Port: 5353}, IfIndex: 2})
	s := m.next(t)
	if s.ifIndex != 2 || !sameAddrs(Parse(s.data)[0].Addrs, tsV4, tsV6) {
		t.Fatalf("answer on 2: %+v %v", s, Parse(s.data))
	}

	// goodbyes: one per interface, each with its own
	r.Close()
	seen := map[int]bool{}
	for range 2 {
		s := m.next(t)
		ttl, f := ttlsOf(t, s.data)
		own := m.table[s.ifIndex]
		if ttl != 0 || seen[s.ifIndex] || !sameAddrs(f.Addrs, own...) {
			t.Fatalf("goodbye on %d: ttl %d %v", s.ifIndex, ttl, f.Addrs)
		}
		seen[s.ifIndex] = true
	}
}

func TestLegacyUnicastCarriesTheArrivalInterfacesAddresses(t *testing.T) {
	m := newMulti()
	startMulti(t, m, nil)
	m.next(t)
	m.next(t)
	q := question(t, 7, serviceName, dnsmessage.TypePTR, dnsmessage.ClassINET)

	// the arrival interface is known
	m.feed(Packet{Data: q, From: &net.UDPAddr{IP: peerTs, Port: 41000}, IfIndex: 2})
	s := m.next(t)
	if s.kind != "unicast" || !sameAddrs(Parse(s.data)[0].Addrs, tsV4, tsV6) {
		t.Fatalf("unicast on 2: %+v %v", s, Parse(s.data))
	}
	// it is not: the addresses that share a subnet with the querier
	m.feed(Packet{Data: q, From: &net.UDPAddr{IP: peerLn, Port: 41000}})
	s = m.next(t)
	if s.kind != "unicast" || !sameAddrs(Parse(s.data)[0].Addrs, lanV4, lanV6) {
		t.Fatalf("unicast by subnet: %+v %v", s, Parse(s.data))
	}
	// a querier on no shared subnet is not answered
	m.feed(Packet{Data: q, From: &net.UDPAddr{IP: net.IPv4(10, 9, 9, 9), Port: 41000}})
	m.quiet(t)
}

func TestListenSetFiltersLegacyByInterface(t *testing.T) {
	m := newMulti()
	startMulti(t, m, []net.IP{tsV4})
	m.next(t) // interface 2 only
	q := question(t, 7, serviceName, dnsmessage.TypePTR, dnsmessage.ClassINET)
	m.feed(Packet{Data: q, From: &net.UDPAddr{IP: peerLn, Port: 41000}, IfIndex: 1})
	m.quiet(t)
	m.feed(Packet{Data: q, From: &net.UDPAddr{IP: peerTs, Port: 41000}})
	s := m.next(t)
	if !sameAddrs(Parse(s.data)[0].Addrs, tsV4) {
		t.Fatalf("got %v", Parse(s.data)[0].Addrs)
	}
}

func TestNetworkChangeReannouncesPerInterface(t *testing.T) {
	m := newMulti()
	_, ch := startMulti(t, m, nil)
	m.next(t)
	m.next(t)
	other := net.IPv4(192, 168, 1, 77)
	m.set(map[int][]net.IP{1: {other}, 3: {net.IPv4(10, 0, 0, 2)}})
	ch <- struct{}{}
	got := map[int][]net.IP{}
	for range 2 {
		s := m.next(t)
		got[s.ifIndex] = Parse(s.data)[0].Addrs
	}
	if !sameAddrs(got[1], other) || !sameAddrs(got[3], net.IPv4(10, 0, 0, 2)) || got[2] != nil {
		t.Fatalf("after the change: %v", got)
	}
}

func TestEmptyListenSetSaysNothing(t *testing.T) {
	m := newMulti()
	startMulti(t, m, []net.IP{})
	m.quiet(t)
	m.feed(Packet{Data: Query(), From: &net.UDPAddr{IP: peerLn, Port: 5353}, IfIndex: 1})
	m.quiet(t)
}
