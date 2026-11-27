package host

import (
	"errors"
	"net"
	"testing"
)

func addr(s string) net.Addr {
	ip, n, _ := net.ParseCIDR(s)
	n.IP = ip
	return n
}

func set(lines ...string) netSet { return netSet(lines) }

func TestNetTrackerSignalsOnlyOnChange(t *testing.T) {
	var cur netSet
	var err error
	read := func() (netSet, error) { return cur, err }
	var tr netTracker

	cur = set("eth0 10.0.0.2/24")
	if !tr.changed(read) {
		t.Fatal("the first read is the first set: a signal")
	}
	for i := 0; i < 3; i++ { // the same address announced again, a lifetime refreshed
		if tr.changed(read) {
			t.Fatal("same set: no signal")
		}
	}
	cur = set("eth0 10.0.0.2/24", "eth0 fe80::1/64") // address added
	if !tr.changed(read) {
		t.Fatal("address added: signal")
	}
	cur = set("eth0 10.0.0.2/24") // removed
	if !tr.changed(read) {
		t.Fatal("address removed: signal")
	}
	cur = set() // link down: its lines go
	if !tr.changed(read) {
		t.Fatal("link down: signal")
	}
	if tr.changed(read) {
		t.Fatal("still down: no signal")
	}
	cur = set("wlan0 192.168.1.5/24") // another link up
	if !tr.changed(read) {
		t.Fatal("link up: signal")
	}
	err = errors.New("no interfaces") // cannot read: something moved, unknown what
	if !tr.changed(read) {
		t.Fatal("a failed read is a change")
	}
	err = nil // and the same set as before the failure is no change
	if tr.changed(read) {
		t.Fatal("the failure did not move the remembered set")
	}
}

func TestLinkLines(t *testing.T) {
	if got := linkLines("eth0", false, []net.Addr{addr("10.0.0.2/24")}); len(got) != 0 {
		t.Errorf("down link: %v", got)
	}
	if got := linkLines("eth0", true, nil); len(got) != 1 || got[0] != "eth0 -" {
		t.Errorf("up, no address: %v", got)
	}
	got := linkLines("eth0", true, []net.Addr{addr("10.0.0.2/24"), addr("fe80::1/64")})
	if len(got) != 2 || got[0] != "eth0 10.0.0.2/24" {
		t.Errorf("up, two addresses: %v", got)
	}
}

func TestReadNetSetSorted(t *testing.T) {
	s, err := readNetSet()
	if err != nil {
		t.Skip(err)
	}
	for i := 1; i < len(s); i++ {
		if s[i-1] > s[i] {
			t.Fatalf("not sorted: %v", s)
		}
	}
	if !s.equal(s) {
		t.Fatal("a set equals itself")
	}
}
