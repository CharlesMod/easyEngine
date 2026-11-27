package discover

import (
	"context"
	"errors"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

var errNoEventForTest = errors.New("no event")

// TestRealNetwork sends real multicast, so it runs only when asked: EASY_ENGINE_MDNS_TEST=1.
// It binds the real mDNS port beside whatever responder the system runs, announces a neutral
// instance ("easy-engine test", no host or user name), finds it with Find, and watches with a
// second socket on the port for the goodbye (TTL 0) that Close sends.
func TestRealNetwork(t *testing.T) {
	if os.Getenv("EASY_ENGINE_MDNS_TEST") != "1" {
		t.Skip("set EASY_ENGINE_MDNS_TEST=1 to use the real network")
	}
	info := Info{ID: "feedfacecafebeef0000000000000000", Name: "easy-engine test", API: 1, Port: 4480}
	watch, err := openReal(mdnsPort) // a listener only: it sends nothing
	if err != nil {
		t.Fatalf("a second socket on %d (the goodbye watch): %v", mdnsPort, err)
	}
	defer watch.Close()
	r, err := start(info, Options{}, mdnsPort)
	if err != nil {
		t.Fatalf("bind %d beside the system's responder: %v", mdnsPort, err)
	}
	t.Logf("responder up on port %d, interfaces %v", mdnsPort, r.t.(*realTransport).Interfaces())
	got, err := Find(context.Background(), 2*time.Second)
	if err != nil {
		r.Close()
		t.Fatalf("find: %v", err)
	}
	seen := false
	for _, f := range got {
		t.Logf("found: %q id=%s port=%d host=%s addrs=%v", f.Name, f.ID, f.Port, f.Host, f.Addrs)
		if f.ID == info.ID && f.Name == info.Name && f.Port == info.Port {
			seen = true
		}
	}
	bye := make(chan bool, 1)
	var wmu sync.Mutex
	var wrong []string
	go func() {
		for {
			p, err := watch.Read()
			if err != nil {
				return
			}
			// each announcement or answer carries the addresses of ONE interface and no other
			// (two NICs on one LAN hear each other's packets, so "heard on" can differ from "sent on")
			for _, f := range Parse(p.Data) {
				if f.ID != info.ID {
					continue
				}
				one := false
				for _, ifi := range eligible() {
					own := ifaceAddrs(ifi)
					all := len(f.Addrs) > 0
					for _, a := range f.Addrs {
						all = all && hasAddr(own, a)
					}
					one = one || all
				}
				name := "?"
				if ifi, err := net.InterfaceByIndex(p.IfIndex); err == nil {
					name = ifi.Name
				}
				t.Logf("heard on %s: %v (one interface's own: %v)", name, f.Addrs, one)
				if !one {
					wmu.Lock()
					wrong = append(wrong, "addresses of more than one interface, heard on "+name)
					wmu.Unlock()
				}
			}
			var m dnsmessage.Message
			if m.Unpack(p.Data) != nil || !m.Response {
				continue
			}
			for _, a := range m.Answers {
				if a.Header.Type == dnsmessage.TypeSRV && foldName(a.Header.Name.String()) == foldName(info.instance()) && a.Header.TTL == 0 {
					bye <- true
					return
				}
			}
		}
	}()
	r.Close()
	gotBye := false
	select {
	case gotBye = <-bye:
	case <-time.After(3 * time.Second): // network timeout: the goodbye is one packet on the wire
	}
	wmu.Lock()
	for _, w := range wrong {
		t.Errorf("address of another interface: %s", w)
	}
	wmu.Unlock()
	t.Logf("find saw the engine: %v; goodbye heard on the port: %v", seen, gotBye)
	if !seen {
		t.Errorf("find did not see the engine among %d answers", len(got))
	}
	if !gotBye {
		t.Error("no goodbye (TTL 0) was heard after Close")
	}
}

func hasAddr(in []net.IP, a net.IP) bool {
	for _, x := range in {
		if x.Equal(a) {
			return true
		}
	}
	return false
}
