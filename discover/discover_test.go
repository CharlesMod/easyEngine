package discover

import (
	"net"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

var (
	testInfo  = Info{ID: "0123456789abcdef", Name: "Study PC", API: 3, Port: 4480}
	testAddr4 = net.IPv4(192, 168, 1, 20)
	testAddr6 = net.ParseIP("fd00::20")
	testAddrs = []net.IP{testAddr4, testAddr6}
)

func unpack(t *testing.T, b []byte) dnsmessage.Message {
	t.Helper()
	var m dnsmessage.Message
	if err := m.Unpack(b); err != nil {
		t.Fatalf("unpack: %v", err)
	}
	return m
}

// kinds counts the records of each type in a section.
func kinds(rs []dnsmessage.Resource) map[dnsmessage.Type]int {
	out := map[dnsmessage.Type]int{}
	for _, r := range rs {
		out[r.Header.Type]++
	}
	return out
}

func question(t *testing.T, id uint16, name string, typ dnsmessage.Type, class dnsmessage.Class) []byte {
	t.Helper()
	m := dnsmessage.Message{Header: dnsmessage.Header{ID: id},
		Questions: []dnsmessage.Question{{Name: dnsmessage.MustNewName(name), Type: typ, Class: class}}}
	b, err := m.Pack()
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestQueryAnswerParse(t *testing.T) {
	reply, ok := Answer(testInfo, Query(), testAddrs, false)
	if !ok {
		t.Fatal("no answer to Query")
	}
	m := unpack(t, reply)
	if m.ID != 0 || !m.Response {
		t.Errorf("header: id %d response %v", m.ID, m.Response)
	}
	all := append(append([]dnsmessage.Resource{}, m.Answers...), m.Additionals...)
	k := kinds(all)
	for _, typ := range []dnsmessage.Type{dnsmessage.TypePTR, dnsmessage.TypeSRV, dnsmessage.TypeTXT, dnsmessage.TypeA, dnsmessage.TypeAAAA} {
		if k[typ] != 1 {
			t.Errorf("record type %v: %d, want 1", typ, k[typ])
		}
	}
	if len(m.Answers) != 1 || m.Answers[0].Header.Type != dnsmessage.TypePTR {
		t.Errorf("answers should be the PTR alone, got %v", kinds(m.Answers))
	}
	for _, r := range all {
		flush := r.Header.Class&cacheFlush != 0
		if r.Header.Type == dnsmessage.TypePTR {
			if flush || r.Header.TTL != 4500 {
				t.Errorf("PTR flush %v ttl %d", flush, r.Header.TTL)
			}
			continue
		}
		if !flush {
			t.Errorf("%v lacks the cache-flush bit", r.Header.Type)
		}
		want := uint32(120)
		if r.Header.Type == dnsmessage.TypeTXT {
			want = 4500
		}
		if r.Header.TTL != want {
			t.Errorf("%v ttl %d, want %d", r.Header.Type, r.Header.TTL, want)
		}
	}
	got := Parse(reply)
	if len(got) != 1 {
		t.Fatalf("Parse gave %d engines", len(got))
	}
	f := got[0]
	if f.ID != testInfo.ID || f.Name != "Study PC" || f.API != 3 || f.Port != 4480 {
		t.Errorf("found %+v", f)
	}
	if f.Host != "easy-engine-01234567.local." {
		t.Errorf("host %q", f.Host)
	}
	if f.Instance != "Study PC [012345]._easy-engine._tcp.local." {
		t.Errorf("instance %q", f.Instance)
	}
	if len(f.Addrs) != 2 || !f.Addrs[0].Equal(testAddr4) || !f.Addrs[1].Equal(testAddr6) {
		t.Errorf("addrs %v", f.Addrs)
	}
}

func TestLegacyAnswer(t *testing.T) {
	q := question(t, 0x4242, serviceName, dnsmessage.TypePTR, dnsmessage.ClassINET)
	reply, ok := Answer(testInfo, q, testAddrs, true)
	if !ok {
		t.Fatal("no answer")
	}
	m := unpack(t, reply)
	if m.ID != 0x4242 {
		t.Errorf("id %#x not echoed", m.ID)
	}
	if len(m.Questions) != 1 || m.Questions[0].Name.String() != serviceName {
		t.Errorf("question not repeated: %v", m.Questions)
	}
	for _, r := range append(append([]dnsmessage.Resource{}, m.Answers...), m.Additionals...) {
		if r.Header.TTL > 10 {
			t.Errorf("%v ttl %d", r.Header.Type, r.Header.TTL)
		}
		if r.Header.Class&cacheFlush != 0 {
			t.Errorf("%v has the cache-flush bit", r.Header.Type)
		}
	}
	if len(Parse(reply)) != 1 {
		t.Error("legacy reply does not parse")
	}
}

func TestAnswerRefusals(t *testing.T) {
	other := question(t, 0, "_http._tcp.local.", dnsmessage.TypePTR, dnsmessage.ClassINET)
	if _, ok := Answer(testInfo, other, testAddrs, false); ok {
		t.Error("answered another service")
	}
	resp := Announcement(testInfo, testAddrs, 120)
	if _, ok := Answer(testInfo, resp, testAddrs, false); ok {
		t.Error("answered a response")
	}
	if _, ok := Answer(testInfo, []byte{1, 2, 3}, testAddrs, false); ok {
		t.Error("answered garbage")
	}
	wrongType := question(t, 0, serviceName, dnsmessage.TypeA, dnsmessage.ClassINET)
	if _, ok := Answer(testInfo, wrongType, testAddrs, false); ok {
		t.Error("answered an A question for the service name")
	}
}

func TestHostQueries(t *testing.T) {
	host := testInfo.host()
	reply, ok := Answer(testInfo, question(t, 0, host, dnsmessage.TypeA, dnsmessage.ClassINET), testAddrs, false)
	if !ok {
		t.Fatal("no answer to A")
	}
	m := unpack(t, reply)
	if len(m.Answers) != 1 || len(m.Additionals) != 0 {
		t.Fatalf("A reply: %d answers, %d additionals", len(m.Answers), len(m.Additionals))
	}
	a, isA := m.Answers[0].Body.(*dnsmessage.AResource)
	if !isA || net.IP(a.A[:]).String() != "192.168.1.20" {
		t.Errorf("A body %v", m.Answers[0].Body)
	}
	reply, ok = Answer(testInfo, question(t, 0, host, dnsmessage.TypeAAAA, dnsmessage.ClassINET), testAddrs, false)
	if !ok {
		t.Fatal("no answer to AAAA")
	}
	m = unpack(t, reply)
	if k := kinds(m.Answers); len(m.Answers) != 1 || k[dnsmessage.TypeAAAA] != 1 {
		t.Errorf("AAAA reply %v", k)
	}
	reply, _ = Answer(testInfo, question(t, 0, host, dnsmessage.TypeALL, dnsmessage.ClassINET), testAddrs, false)
	if k := kinds(unpack(t, reply).Answers); k[dnsmessage.TypeA] != 1 || k[dnsmessage.TypeAAAA] != 1 {
		t.Errorf("ANY reply %v", k)
	}
}

func TestInstanceAndMetaQueries(t *testing.T) {
	inst := testInfo.instance()
	for typ, want := range map[dnsmessage.Type]dnsmessage.Type{dnsmessage.TypeSRV: dnsmessage.TypeSRV, dnsmessage.TypeTXT: dnsmessage.TypeTXT} {
		reply, ok := Answer(testInfo, question(t, 0, inst, typ, dnsmessage.ClassINET), testAddrs, false)
		if !ok {
			t.Fatalf("no answer to %v", typ)
		}
		m := unpack(t, reply)
		if len(m.Answers) != 1 || m.Answers[0].Header.Type != want {
			t.Errorf("%v: answers %v", typ, kinds(m.Answers))
		}
	}
	reply, _ := Answer(testInfo, question(t, 0, inst, dnsmessage.TypeALL, dnsmessage.ClassINET), testAddrs, false)
	if k := kinds(unpack(t, reply).Answers); k[dnsmessage.TypeSRV] != 1 || k[dnsmessage.TypeTXT] != 1 {
		t.Errorf("ANY on instance: %v", k)
	}
	reply, ok := Answer(testInfo, question(t, 0, metaName, dnsmessage.TypePTR, dnsmessage.ClassINET), testAddrs, false)
	if !ok {
		t.Fatal("no answer to the meta query")
	}
	m := unpack(t, reply)
	p, isPTR := m.Answers[0].Body.(*dnsmessage.PTRResource)
	if !isPTR || p.PTR.String() != serviceName {
		t.Errorf("meta answer %v", m.Answers[0].Body)
	}
}

func TestTXTKeys(t *testing.T) {
	reply, _ := Answer(testInfo, Query(), testAddrs, false)
	m := unpack(t, reply)
	for _, r := range m.Additionals {
		tx, ok := r.Body.(*dnsmessage.TXTResource)
		if !ok {
			continue
		}
		var keys []string
		for _, kv := range tx.TXT {
			k, _, _ := strings.Cut(kv, "=")
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if strings.Join(keys, ",") != "api,id,name" {
			t.Errorf("TXT keys %v", keys)
		}
		return
	}
	t.Fatal("no TXT record")
}

func TestGoodbye(t *testing.T) {
	b := Announcement(testInfo, testAddrs, 0)
	m := unpack(t, b)
	if !m.Response || len(m.Answers) != 5 {
		t.Fatalf("goodbye has %d answers", len(m.Answers))
	}
	for _, r := range m.Answers {
		if r.Header.TTL != 0 {
			t.Errorf("%v ttl %d", r.Header.Type, r.Header.TTL)
		}
	}
	if f := Parse(b); len(f) != 1 || f[0].ID != testInfo.ID {
		t.Errorf("goodbye parse %+v", f)
	}
}

func TestNameSanitised(t *testing.T) {
	cases := map[string]string{
		"":                      "easy-engine",
		"a.b\x01c":              "abc",
		strings.Repeat("x", 80): strings.Repeat("x", 40),
		"...":                   "easy-engine",
		strings.Repeat("é", 30): strings.Repeat("é", 20),
	}
	for in, want := range cases {
		if got := sanitise(in); got != want {
			t.Errorf("sanitise(%q) = %q, want %q", in, got, want)
		}
	}
	long := Info{ID: testInfo.ID, Name: strings.Repeat("y", 90), Port: 1}
	if f := Parse(Announcement(long, nil, 120)); len(f) != 1 || len(f[0].Name) != 40 {
		t.Errorf("long name %+v", f)
	}
}

func TestParseSkipsNoID(t *testing.T) {
	m := dnsmessage.Message{Header: dnsmessage.Header{Response: true}, Answers: []dnsmessage.Resource{
		{Header: hdr(serviceName, dnsmessage.TypePTR, 120, false), Body: &dnsmessage.PTRResource{PTR: mustName("x._easy-engine._tcp.local.")}},
		{Header: hdr("x._easy-engine._tcp.local.", dnsmessage.TypeTXT, 120, false), Body: &dnsmessage.TXTResource{TXT: []string{"name=x"}}},
	}}
	b, _ := m.Pack()
	if f := Parse(b); len(f) != 0 {
		t.Errorf("parsed %+v", f)
	}
}

func TestStartRefusesBadInfo(t *testing.T) {
	ft := newFake()
	if _, err := Start(Info{ID: "abc", Port: 1}, Options{Transport: ft}); err == nil {
		t.Error("short id accepted")
	}
	if _, err := Start(Info{ID: testInfo.ID}, Options{Transport: ft}); err == nil {
		t.Error("port 0 accepted")
	}
	if len(ft.sends) != 0 {
		t.Error("sent something for a refused Start")
	}
}

// --- the in-memory transport ---

type send struct {
	kind    string // "multicast" or "unicast"
	data    []byte
	ifIndex int
	to      *net.UDPAddr
}

type fake struct {
	in      chan Packet
	sends   chan send
	entered chan int // the number of each Read call as it begins
	refresh chan struct{}

	calls   atomic.Int64
	fed     int
	closeMu sync.Once
	closed  chan struct{}
}

func newFake() *fake {
	return &fake{in: make(chan Packet, 16), sends: make(chan send, 64), entered: make(chan int, 1024),
		refresh: make(chan struct{}, 16), closed: make(chan struct{})}
}

func (f *fake) Read() (Packet, error) {
	f.entered <- int(f.calls.Add(1))
	select {
	case p := <-f.in:
		return p, nil
	case <-f.closed:
		return Packet{}, net.ErrClosed
	}
}
func (f *fake) Multicast(d []byte, i int) error {
	f.sends <- send{kind: "multicast", data: d, ifIndex: i}
	return nil
}
func (f *fake) Unicast(d []byte, to *net.UDPAddr) error {
	f.sends <- send{kind: "unicast", data: d, to: to}
	return nil
}
func (f *fake) Addrs(int) []net.IP { return testAddrs }
func (f *fake) Refresh() error     { f.refresh <- struct{}{}; return nil }
func (f *fake) Close() error       { f.closeMu.Do(func() { close(f.closed) }); return nil }

// feed hands the responder a packet.
func (f *fake) feed(p Packet) {
	f.fed++
	f.in <- p
}

// settle feeds a response (which gets no reply) and waits until the Read
// after it begins: by then every packet fed before has been handled.
func (f *fake) settle(t *testing.T) {
	t.Helper()
	f.feed(Packet{Data: Announcement(Info{ID: "ffffffffffff", Port: 1}, nil, 120), From: &net.UDPAddr{IP: testAddr4, Port: 5353}})
	for {
		select {
		case n := <-f.entered:
			if n > f.fed {
				return
			}
		case <-time.After(10 * time.Second): // watchdog for silence
			t.Fatal("responder stopped reading")
		}
	}
}

func (f *fake) next(t *testing.T) send {
	t.Helper()
	select {
	case s := <-f.sends:
		return s
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("expected a send, none came")
		return send{}
	}
}

func (f *fake) quiet(t *testing.T) {
	t.Helper()
	f.settle(t)
	select {
	case s := <-f.sends:
		t.Fatalf("unexpected %s send (%d bytes)", s.kind, len(s.data))
	default:
	}
}

func ttlsOf(t *testing.T, b []byte) (uint32, Found) {
	t.Helper()
	m := unpack(t, b)
	f := Parse(b)
	if len(f) != 1 {
		t.Fatalf("send parses to %d engines", len(f))
	}
	return m.Answers[0].Header.TTL, f[0]
}

func TestResponder(t *testing.T) {
	ft := newFake()
	changes := make(chan struct{}, 1)
	stopped := make(chan struct{})
	r, err := Start(testInfo, Options{Transport: ft, NetChanges: func() (<-chan struct{}, func(), error) {
		return changes, func() { close(stopped) }, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	closed := false
	defer func() {
		if !closed {
			r.Close()
		}
	}()

	// one announcement at start
	s := ft.next(t)
	if s.kind != "multicast" || s.ifIndex != 0 {
		t.Fatalf("start send %+v", s)
	}
	if ttl, f := ttlsOf(t, s.data); ttl == 0 || f.ID != testInfo.ID || len(f.Addrs) != 2 {
		t.Errorf("start announcement ttl %d %+v", ttl, f)
	}
	ft.quiet(t)

	// a query from 5353 is answered by multicast on its interface
	ft.feed(Packet{Data: Query(), From: &net.UDPAddr{IP: net.IPv4(192, 168, 1, 9), Port: 5353}, IfIndex: 7})
	s = ft.next(t)
	if s.kind != "multicast" || s.ifIndex != 7 {
		t.Fatalf("answer send %+v", s)
	}
	if m := unpack(t, s.data); m.ID != 0 || len(Parse(s.data)) != 1 {
		t.Error("multicast answer is not a plain answer")
	}

	// a query from another port is legacy: unicast to the source
	src := &net.UDPAddr{IP: net.IPv4(192, 168, 1, 9), Port: 40000}
	ft.feed(Packet{Data: question(t, 9, serviceName, dnsmessage.TypePTR, dnsmessage.ClassINET), From: src, IfIndex: 7})
	s = ft.next(t)
	if s.kind != "unicast" || s.to.Port != 40000 || !s.to.IP.Equal(src.IP) {
		t.Fatalf("legacy send %+v", s)
	}
	if m := unpack(t, s.data); m.ID != 9 {
		t.Errorf("legacy id %d", m.ID)
	}

	// the QU bit asks for unicast even from 5353
	ft.feed(Packet{Data: question(t, 0, serviceName, dnsmessage.TypePTR, dnsmessage.ClassINET|cacheFlush),
		From: &net.UDPAddr{IP: net.IPv4(192, 168, 1, 10), Port: 5353}, IfIndex: 7})
	if s = ft.next(t); s.kind != "unicast" || s.to.Port != 5353 {
		t.Fatalf("QU send %+v", s)
	}

	// a response, a stranger's query and garbage get nothing
	ft.feed(Packet{Data: Announcement(testInfo, testAddrs, 120), From: &net.UDPAddr{IP: testAddr4, Port: 5353}, IfIndex: 7})
	ft.feed(Packet{Data: question(t, 0, "_http._tcp.local.", dnsmessage.TypePTR, dnsmessage.ClassINET), From: src, IfIndex: 7})
	ft.feed(Packet{Data: []byte{0xff}, From: src})
	ft.quiet(t)

	// a network change: Refresh, then one announcement
	changes <- struct{}{}
	select {
	case <-ft.refresh:
	case <-time.After(10 * time.Second): // watchdog for silence
		t.Fatal("no Refresh after a network change")
	}
	if s = ft.next(t); s.kind != "multicast" {
		t.Fatalf("change send %+v", s)
	}
	if ttl, _ := ttlsOf(t, s.data); ttl == 0 {
		t.Error("change announcement is a goodbye")
	}
	ft.quiet(t)
	if len(ft.refresh) != 0 {
		t.Error("more than one Refresh")
	}

	// a changed name: goodbye for the old, announcement of the new
	renamed := testInfo
	renamed.Name = "Living Room"
	r.Update(renamed)
	s = ft.next(t)
	ttl, f := ttlsOf(t, s.data)
	if ttl != 0 || f.Name != "Study PC" {
		t.Errorf("update goodbye ttl %d %+v", ttl, f)
	}
	s = ft.next(t)
	ttl, f = ttlsOf(t, s.data)
	if ttl == 0 || f.Name != "Living Room" {
		t.Errorf("update announcement ttl %d %+v", ttl, f)
	}
	ft.quiet(t)

	// answers now carry the new name
	ft.feed(Packet{Data: Query(), From: &net.UDPAddr{IP: net.IPv4(192, 168, 1, 9), Port: 5353}, IfIndex: 2})
	if _, f = ttlsOf(t, ft.next(t).data); f.Name != "Living Room" {
		t.Errorf("answer after update names %q", f.Name)
	}
	ft.quiet(t)

	// close: a goodbye, the watch stopped, the read loop ended
	r.Close()
	closed = true
	s = ft.next(t)
	if ttl, f = ttlsOf(t, s.data); ttl != 0 || f.Name != "Living Room" {
		t.Errorf("close send ttl %d %+v", ttl, f)
	}
	select {
	case <-stopped:
	default:
		t.Error("network-change watch not stopped")
	}
	if _, err := ft.Read(); err == nil {
		t.Error("transport still reads after Close")
	}
	r.Close() // again: harmless
	r.Update(renamed)
	select {
	case s := <-ft.sends:
		t.Errorf("send after close: %+v", s)
	default:
	}
}

func TestStartWithoutNetworkEvents(t *testing.T) {
	ft := newFake()
	r, err := Start(testInfo, Options{Transport: ft, NetChanges: func() (<-chan struct{}, func(), error) {
		return nil, nil, errNoEventForTest
	}})
	if err != nil {
		t.Fatal(err)
	}
	ft.next(t)
	r.Close()
	if s := ft.next(t); s.kind != "multicast" {
		t.Errorf("goodbye %+v", s)
	}
}

func (f *fake) Reach(net.IP) []net.IP { return testAddrs }
