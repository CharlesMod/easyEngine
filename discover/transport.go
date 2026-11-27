package discover

import (
	"context"
	"errors"
	"net"
	"strconv"
	"sync"

	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

var (
	group4 = net.IPv4(224, 0, 0, 251)
	group6 = net.ParseIP("ff02::fb")
)

// eligible lists the interfaces mDNS runs on: up, multicast-capable, not loopback.
func eligible() []net.Interface {
	all, err := net.Interfaces()
	if err != nil {
		return nil
	}
	var out []net.Interface
	for _, ifi := range all {
		if ifi.Flags&net.FlagUp != 0 && ifi.Flags&net.FlagMulticast != 0 && ifi.Flags&net.FlagLoopback == 0 {
			out = append(out, ifi)
		}
	}
	return out
}

// ifaceAddrs gives the addresses worth advertising on one interface.
func ifaceAddrs(ifi net.Interface) []net.IP {
	as, err := ifi.Addrs()
	if err != nil {
		return nil
	}
	var out []net.IP
	for _, a := range as {
		var ip net.IP
		switch v := a.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip == nil || ip.IsLoopback() || ip.IsUnspecified() || ip.IsMulticast() {
			continue
		}
		if ip.To4() == nil && ip.IsLinkLocalUnicast() {
			continue // a bare fe80:: address is no use to a peer without its zone
		}
		out = append(out, ip)
	}
	return out
}

// realTransport is the Transport on real UDP sockets, one per family.
type realTransport struct {
	port int
	c4   *ipv4.PacketConn
	c6   *ipv6.PacketConn

	mu      sync.Mutex // guards the joined sets
	joined4 map[int]bool
	joined6 map[int]bool
	send    sync.Mutex // one send at a time: the outgoing interface is socket state

	pkts chan Packet
	done chan struct{}
	once sync.Once
	wg   sync.WaitGroup
}

// openReal opens the sockets on port and joins the groups. If one family
// fails the other serves alone; if both fail the error comes back.
func openReal(port int) (*realTransport, error) {
	lc := net.ListenConfig{Control: reuse}
	t := &realTransport{port: port, joined4: map[int]bool{}, joined6: map[int]bool{},
		pkts: make(chan Packet), done: make(chan struct{})}
	addr := ":" + strconv.Itoa(port)
	pc4, err4 := lc.ListenPacket(context.Background(), "udp4", addr)
	if err4 == nil {
		t.c4 = ipv4.NewPacketConn(pc4)
		_ = t.c4.SetControlMessage(ipv4.FlagInterface, true) // may fail on windows: IfIndex stays 0
	}
	pc6, err6 := lc.ListenPacket(context.Background(), "udp6", addr)
	if err6 == nil {
		t.c6 = ipv6.NewPacketConn(pc6)
		_ = t.c6.SetControlMessage(ipv6.FlagInterface, true)
	}
	if t.c4 == nil && t.c6 == nil {
		return nil, errors.Join(err4, err6)
	}
	_ = t.Refresh()
	if t.c4 != nil {
		t.wg.Add(1)
		go t.read4()
	}
	if t.c6 != nil {
		t.wg.Add(1)
		go t.read6()
	}
	return t, nil
}

func (t *realTransport) deliver(p Packet) bool {
	select {
	case t.pkts <- p:
		return true
	case <-t.done:
		return false
	}
}

func (t *realTransport) read4() {
	defer t.wg.Done()
	buf := make([]byte, 9000)
	for {
		n, cm, src, err := t.c4.ReadFrom(buf)
		if err != nil {
			return
		}
		u, _ := src.(*net.UDPAddr)
		p := Packet{Data: append([]byte(nil), buf[:n]...), From: u}
		if cm != nil {
			p.IfIndex = cm.IfIndex
		}
		if !t.deliver(p) {
			return
		}
	}
}

func (t *realTransport) read6() {
	defer t.wg.Done()
	buf := make([]byte, 9000)
	for {
		n, cm, src, err := t.c6.ReadFrom(buf)
		if err != nil {
			return
		}
		u, _ := src.(*net.UDPAddr)
		p := Packet{Data: append([]byte(nil), buf[:n]...), From: u}
		if cm != nil {
			p.IfIndex = cm.IfIndex
		}
		if !t.deliver(p) {
			return
		}
	}
}

// Read implements Transport.
func (t *realTransport) Read() (Packet, error) {
	select {
	case p := <-t.pkts:
		return p, nil
	case <-t.done:
		return Packet{}, net.ErrClosed
	}
}

// Refresh implements Transport: join the groups on interfaces not yet joined
// and forget the ones that are gone.
func (t *realTransport) Refresh() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := map[int]bool{}
	for _, ifi := range eligible() {
		ifi := ifi
		now[ifi.Index] = true
		if t.c4 != nil && !t.joined4[ifi.Index] {
			if t.c4.JoinGroup(&ifi, &net.UDPAddr{IP: group4}) == nil {
				t.joined4[ifi.Index] = true
			}
		}
		if t.c6 != nil && !t.joined6[ifi.Index] {
			if t.c6.JoinGroup(&ifi, &net.UDPAddr{IP: group6}) == nil {
				t.joined6[ifi.Index] = true
			}
		}
	}
	for i := range t.joined4 {
		if !now[i] {
			delete(t.joined4, i)
		}
	}
	for i := range t.joined6 {
		if !now[i] {
			delete(t.joined6, i)
		}
	}
	return nil
}

// targets is the interfaces a send for ifIndex goes out on.
func targets(ifIndex int) []net.Interface {
	all := eligible()
	if ifIndex > 0 {
		for _, ifi := range all {
			if ifi.Index == ifIndex {
				return []net.Interface{ifi}
			}
		}
	}
	return all
}

// Multicast implements Transport.
func (t *realTransport) Multicast(data []byte, ifIndex int) error {
	t.send.Lock()
	defer t.send.Unlock()
	var first error
	note := func(err error) {
		if err != nil && first == nil {
			first = err
		}
	}
	for _, ifi := range targets(ifIndex) {
		ifi := ifi
		if t.c4 != nil {
			if err := t.c4.SetMulticastInterface(&ifi); err == nil {
				_, err = t.c4.WriteTo(data, nil, &net.UDPAddr{IP: group4, Port: t.port})
				note(err)
			}
		}
		if t.c6 != nil {
			if err := t.c6.SetMulticastInterface(&ifi); err == nil {
				_, err = t.c6.WriteTo(data, nil, &net.UDPAddr{IP: group6, Port: t.port})
				note(err)
			}
		}
	}
	return first
}

// Unicast implements Transport.
func (t *realTransport) Unicast(data []byte, to *net.UDPAddr) error {
	t.send.Lock()
	defer t.send.Unlock()
	if to.IP.To4() != nil && t.c4 != nil {
		_, err := t.c4.WriteTo(data, nil, to)
		return err
	}
	if t.c6 != nil && to.IP.To4() == nil {
		_, err := t.c6.WriteTo(data, nil, to)
		return err
	}
	return errors.New("discover: no socket for that address family")
}

// Addrs implements Transport.
func (t *realTransport) Addrs(ifIndex int) []net.IP {
	var out []net.IP
	if ifIndex > 0 {
		for _, ifi := range targets(ifIndex) {
			if ifi.Index == ifIndex {
				out = ifaceAddrs(ifi)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	for _, ifi := range eligible() {
		out = append(out, ifaceAddrs(ifi)...)
	}
	return out
}

// Interfaces lists the interfaces that have an address to announce.
func (t *realTransport) Interfaces() []int {
	var out []int
	for _, ifi := range eligible() {
		if len(ifaceAddrs(ifi)) > 0 {
			out = append(out, ifi.Index)
		}
	}
	return out
}

// Close implements Transport.
func (t *realTransport) Close() error {
	t.once.Do(func() {
		close(t.done)
		if t.c4 != nil {
			_ = t.c4.Close()
		}
		if t.c6 != nil {
			_ = t.c6.Close()
		}
		t.wg.Wait()
	})
	return nil
}
