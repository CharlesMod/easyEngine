package discover

import (
	"context"
	"errors"
	"net"
	"sync"
	"time"

	"golang.org/x/net/ipv4"
	"golang.org/x/net/ipv6"
)

// Find asks the local network which engines are there. It sends one query from
// an ephemeral port, so responders answer it by unicast (RFC 6762 6.7), and
// collects the answers until wait has passed or ctx ends. The result is
// de-duplicated by ID and sorted by name.
func Find(ctx context.Context, wait time.Duration) ([]Found, error) {
	return find(ctx, wait, mdnsPort)
}

func find(ctx context.Context, wait time.Duration, port int) ([]Found, error) {
	q := Query()
	var conns []net.PacketConn
	var send func()
	var errs []error

	var c4 *ipv4.PacketConn
	if pc, err := net.ListenPacket("udp4", ":0"); err == nil {
		conns = append(conns, pc)
		c4 = ipv4.NewPacketConn(pc)
	} else {
		errs = append(errs, err)
	}
	var c6 *ipv6.PacketConn
	if pc, err := net.ListenPacket("udp6", ":0"); err == nil {
		conns = append(conns, pc)
		c6 = ipv6.NewPacketConn(pc)
	} else {
		errs = append(errs, err)
	}
	if len(conns) == 0 {
		return nil, errors.Join(errs...)
	}
	defer func() {
		for _, c := range conns {
			_ = c.Close()
		}
	}()

	var sendErr error
	send = func() {
		sent := false
		for _, ifi := range eligible() {
			ifi := ifi
			if c4 != nil && c4.SetMulticastInterface(&ifi) == nil {
				if _, err := c4.WriteTo(q, nil, &net.UDPAddr{IP: group4, Port: port}); err == nil {
					sent = true
				} else {
					sendErr = err
				}
			}
			if c6 != nil && c6.SetMulticastInterface(&ifi) == nil {
				if _, err := c6.WriteTo(q, nil, &net.UDPAddr{IP: group6, Port: port}); err == nil {
					sent = true
				} else {
					sendErr = err
				}
			}
		}
		if !sent && sendErr == nil {
			sendErr = errors.New("discover: no multicast interface to ask on")
		}
		if sent {
			sendErr = nil
		}
	}
	send()
	if sendErr != nil {
		return nil, sendErr
	}

	deadline := time.Now().Add(wait)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	var set = map[string]*Found{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, c := range conns {
		c := c
		_ = c.SetReadDeadline(deadline) // network timeout
		stop := context.AfterFunc(ctx, func() { _ = c.SetReadDeadline(time.Now()) })
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer stop()
			buf := make([]byte, 9000)
			for {
				n, _, err := c.ReadFrom(buf)
				if err != nil {
					return
				}
				for _, f := range Parse(buf[:n]) {
					mu.Lock()
					merge(set, f)
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	return sorted(set), nil
}
