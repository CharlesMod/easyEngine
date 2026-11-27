package app

import (
	"errors"
	"net"
	"testing"
)

func TestListenSet(t *testing.T) {
	old := lookupIP
	defer func() { lookupIP = old }()
	lookupIP = func(h string) ([]net.IP, error) {
		switch h {
		case "den":
			return []net.IP{net.ParseIP("fd00::1"), net.IPv4(192, 168, 1, 4), net.IPv4(192, 168, 1, 5)}, nil
		case "v6only":
			return []net.IP{net.ParseIP("fd00::2")}, nil
		}
		return nil, errors.New("no such host")
	}
	for _, h := range []string{"", "0.0.0.0", "::", "[::]"} {
		if s := ListenSet(h); s != nil {
			t.Errorf("wildcard %q gave %v", h, s)
		}
	}
	one := func(h string, want string) {
		t.Helper()
		s := ListenSet(h)
		if len(s) != 1 || !s[0].Equal(net.ParseIP(want)) {
			t.Errorf("ListenSet(%q) = %v, want [%s]", h, s, want)
		}
	}
	one("192.168.1.20", "192.168.1.20")
	one("[fd00::20]", "fd00::20")
	one("den", "192.168.1.4") // as net.Listen: the first IPv4 address
	one("v6only", "fd00::2")
	if s := ListenSet("nowhere"); s == nil || len(s) != 0 {
		t.Errorf("an unresolvable name gave %v, want an empty non-nil set", s)
	}
}
