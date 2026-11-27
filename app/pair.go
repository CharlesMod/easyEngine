package app

import (
	"log"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/CharlesMod/easyEngine/discover"
	osevents "github.com/CharlesMod/easyEngine/host"
	"github.com/CharlesMod/easyEngine/run"
)

// Discoverable reports whether a listener on host may announce itself on the network: not a
// loopback name or address (127.0.0.0/8, ::1, localhost). A loopback listener announces nothing.
func Discoverable(host string) bool {
	h := strings.Trim(host, "[]")
	if IsLoopbackHost(h) || strings.EqualFold(h, "localhost") {
		return false
	}
	if ip := net.ParseIP(h); ip != nil && ip.IsLoopback() {
		return false
	}
	return true
}

// lookupIP resolves a host name (a seam for tests).
var lookupIP = net.LookupIP

// ListenSet is the addresses a listener on host answers on, for the announcer: nil for a wildcard
// host (empty, 0.0.0.0, ::: every address of every up multicast-capable interface); the one
// address for an IP; for a host name, what net.Listen does with it, resolved once here: the
// first IPv4 address it names, else the first address. A name that does not resolve gives an
// empty non-nil set (the listener could not have bound it either): nothing is announced.
func ListenSet(host string) []net.IP {
	h := strings.Trim(host, "[]")
	if i := strings.IndexByte(h, '%'); i >= 0 {
		h = h[:i] // an IPv6 zone is no part of the address
	}
	if h == "" {
		return nil
	}
	if ip := net.ParseIP(h); ip != nil {
		if ip.IsUnspecified() {
			return nil
		}
		return []net.IP{ip}
	}
	ips, err := lookupIP(h)
	if err != nil || len(ips) == 0 {
		return []net.IP{}
	}
	for _, ip := range ips {
		if ip.To4() != nil {
			return []net.IP{ip}
		}
	}
	return []net.IP{ips[0]}
}

// engineName is this engine's human name: the settings name, else the host name's first label.
func engineName(sup *run.Supervisor) string {
	if n := strings.TrimSpace(sup.EngineName()); n != "" {
		return n
	}
	h, err := os.Hostname()
	if err != nil || h == "" {
		return "easy-engine"
	}
	if i := strings.IndexByte(h, '.'); i > 0 {
		h = h[:i]
	}
	return h
}

// followFile calls changed on each change of the file at path, as the system says it
// (host.Changes watches the file's directory, so the file need not exist yet: its creation,
// each rewrite and each rename onto it are told). The func ends the watch and waits for no call
// to be running after it returns.
func followFile(path string, changed func()) (func(), error) {
	ch, stop, err := osevents.Changes(path)
	if err != nil {
		return nil, err
	}
	done, ended := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(ended)
		for {
			select {
			case <-done:
				return
			case <-ch:
				changed()
			}
		}
	}()
	return func() { close(done); <-ended; stop() }, nil
}

// announcer keeps the mDNS responder in step with the settings: started when discovery is on
// and the listener may announce, updated when the name changes, stopped when it is turned off
// and on Quit. It acts at start and on each settings reload (events), never on a clock.
type announcer struct {
	eligible bool
	opts     discover.Options

	mu   sync.Mutex
	r    *discover.Responder
	info discover.Info
	done bool
}

func (an *announcer) apply(on bool, info discover.Info) {
	an.mu.Lock()
	defer an.mu.Unlock()
	if an.done {
		return
	}
	switch {
	case !an.eligible || !on:
		if an.r != nil {
			an.r.Close()
			an.r = nil
		}
	case an.r == nil:
		r, err := discover.Start(info, an.opts)
		if err != nil {
			log.Printf("easy-engine: mDNS announce: %v (clients can still be given the address)", err)
			return
		}
		an.r, an.info = r, info
	case an.info != info:
		an.r.Update(info)
		an.info = info
	}
}

// close says goodbye on the network (Quit) and takes no more applies.
func (an *announcer) close() {
	an.mu.Lock()
	defer an.mu.Unlock()
	an.done = true
	if an.r != nil {
		an.r.Close()
		an.r = nil
	}
}

// running reports whether a responder is up (tests).
func (an *announcer) running() bool {
	an.mu.Lock()
	defer an.mu.Unlock()
	return an.r != nil
}
