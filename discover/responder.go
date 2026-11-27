package discover

import (
	"errors"
	"net"
	"sync"

	"github.com/CharlesMod/easyEngine/host"
)

// Packet is one datagram read from the network.
type Packet struct {
	Data    []byte
	From    *net.UDPAddr
	IfIndex int // interface it came in on; 0 when unknown
}

// Transport is the network under a Responder. The real one is the default;
// tests give an in-memory one.
type Transport interface {
	Read() (Packet, error)                    // blocks; returns an error after Close
	Multicast(data []byte, ifIndex int) error // to 224.0.0.251:5353 / [ff02::fb]:5353 on that interface (0 = all)
	Unicast(data []byte, to *net.UDPAddr) error
	// Addrs gives the non-loopback addresses to advertise for a packet that came
	// in on ifIndex (0 or unknown means all up non-loopback interfaces).
	Addrs(ifIndex int) []net.IP
	Refresh() error // re-read interfaces and join the groups on new ones
	Close() error
}

// interfacer is an optional Transport method: the interface indexes to
// announce on, one announcement each (each carries that interface's own
// addresses). A Transport without it is announced on once, as interface 0.
type interfacer interface {
	Interfaces() []int
}

// Options are for tests; the zero value is the real network.
type Options struct {
	// Transport, when non-nil, replaces the real sockets.
	Transport Transport
	// NetChanges, when non-nil, replaces host.NetworkChanges.
	NetChanges func() (<-chan struct{}, func(), error)
}

// Responder announces one engine and answers queries for it.
type Responder struct {
	t    Transport
	stop func() // stops the network-change watch

	mu     sync.Mutex // guards info, closed and every send
	info   Info
	closed bool

	done chan struct{}
	wg   sync.WaitGroup
	once sync.Once
}

// Start opens (or takes) the transport, announces once on every interface and
// then serves: it answers each query that asks for ours, and announces again
// when the network changes. Nothing repeats on a clock.
func Start(info Info, o Options) (*Responder, error) {
	return start(info, o, mdnsPort)
}

func start(info Info, o Options, port int) (*Responder, error) {
	if err := info.valid(); err != nil {
		return nil, err
	}
	t := o.Transport
	if t == nil {
		rt, err := openReal(port)
		if err != nil {
			return nil, err
		}
		t = rt
	}
	r := &Responder{t: t, info: info, done: make(chan struct{}), stop: func() {}}

	nc := o.NetChanges
	if nc == nil {
		nc = host.NetworkChanges
	}
	var changes <-chan struct{}
	if ch, stop, err := nc(); err == nil && ch != nil {
		changes = ch
		if stop != nil {
			r.stop = stop
		}
	} else if err != nil && !errors.Is(err, host.ErrNoEvent) {
		// no network-change word: serve without it; a restart re-announces
		changes = nil
	}

	r.mu.Lock()
	r.announceLocked(info, ttlHost)
	r.mu.Unlock()

	r.wg.Add(1)
	go r.serve()
	if changes != nil {
		r.wg.Add(1)
		go r.watch(changes)
	}
	return r, nil
}

// announceLocked sends info on every interface; ttl 0 is a goodbye.
func (r *Responder) announceLocked(info Info, ttl uint32) {
	ifs := []int{0}
	if it, ok := r.t.(interfacer); ok {
		if l := it.Interfaces(); len(l) > 0 {
			ifs = l
		}
	}
	for _, i := range ifs {
		_ = r.t.Multicast(Announcement(info, r.t.Addrs(i), ttl), i)
	}
}

// serve reads packets until the transport closes.
func (r *Responder) serve() {
	defer r.wg.Done()
	for {
		p, err := r.t.Read()
		if err != nil {
			return
		}
		r.handle(p)
	}
}

func (r *Responder) handle(p Packet) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || p.From == nil {
		return
	}
	legacy := p.From.Port != mdnsPort
	reply, ok, qu := answer(r.info, p.Data, r.t.Addrs(p.IfIndex), legacy)
	if !ok {
		return
	}
	if legacy || qu {
		_ = r.t.Unicast(reply, p.From)
	} else {
		_ = r.t.Multicast(reply, p.IfIndex)
	}
}

// watch re-announces when the network changes.
func (r *Responder) watch(ch <-chan struct{}) {
	defer r.wg.Done()
	for {
		select {
		case <-r.done:
			return
		case _, ok := <-ch:
			if !ok {
				return
			}
			r.mu.Lock()
			if !r.closed {
				_ = r.t.Refresh()
				r.announceLocked(r.info, ttlHost)
			}
			r.mu.Unlock()
		}
	}
}

// Update changes what is announced: a goodbye for the old instance, then an
// announcement of the new. The ID and port must stay valid; an invalid info is
// ignored.
func (r *Responder) Update(info Info) {
	if info.valid() != nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	old := r.info
	r.info = info
	if old.instance() != info.instance() || old.host() != info.host() {
		r.announceLocked(old, 0)
	}
	r.announceLocked(info, ttlHost)
}

// Close says goodbye on every interface, closes the transport and waits for
// the serving goroutines to end. It may be called again.
func (r *Responder) Close() {
	r.once.Do(func() {
		r.mu.Lock()
		r.announceLocked(r.info, 0)
		r.closed = true
		r.mu.Unlock()
		close(r.done)
		r.stop()
		_ = r.t.Close()
		r.wg.Wait()
	})
}
