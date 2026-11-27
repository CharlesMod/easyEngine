// Package discover finds easyEngine engines on the local network and lets an
// engine be found: a small mDNS / DNS-SD responder and browser in pure Go.
//
// The responder announces one service, _easy-engine._tcp, and answers the
// queries that ask for it. It is driven by events only: it announces when it
// starts, when the network changes, when its name changes and when it closes,
// and it answers a query when one arrives. It repeats nothing on a clock.
package discover

import (
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"golang.org/x/net/dns/dnsmessage"
)

// ServiceType is the DNS-SD service type, browsed as _easy-engine._tcp.local.
const ServiceType = "_easy-engine._tcp"

const (
	serviceName  = ServiceType + ".local."
	metaName     = "_services._dns-sd._udp.local."
	mdnsPort     = 5353
	ttlHost      = 120
	ttlOther     = 4500
	ttlLegacyMax = 10
	cacheFlush   = 1 << 15
	maxNameBytes = 40
	defaultName  = "easy-engine"
)

// Info describes the engine being announced.
type Info struct {
	ID   string // hex engine id, at least 8 characters
	Name string // human name
	API  int    // API version
	Port int    // TCP port of the engine's API
}

// Found is one engine seen on the network.
type Found struct {
	ID, Name       string
	API, Port      int
	Instance, Host string // DNS names of the service instance and its host
	Addrs          []net.IP
}

func (i Info) valid() error {
	if len(i.ID) < 8 {
		return fmt.Errorf("discover: engine id %q is shorter than 8 characters", i.ID)
	}
	if i.Port <= 0 || i.Port > 65535 {
		return fmt.Errorf("discover: bad port %d", i.Port)
	}
	return nil
}

// sanitise makes a name fit a DNS label: no dots, no control characters,
// at most 40 bytes, and never empty.
func sanitise(name string) string {
	var b strings.Builder
	for _, r := range name {
		if r == '.' || r < 0x20 || r == 0x7f || r == utf8.RuneError {
			continue
		}
		if b.Len()+utf8.RuneLen(r) > maxNameBytes {
			break
		}
		b.WriteRune(r)
	}
	s := strings.TrimSpace(b.String())
	if s == "" {
		return defaultName
	}
	return s
}

func prefix(id string, n int) string {
	if len(id) < n {
		return id
	}
	return id[:n]
}

func (i Info) instance() string {
	return sanitise(i.Name) + " [" + prefix(i.ID, 6) + "]." + serviceName
}

func (i Info) host() string { return "easy-engine-" + prefix(i.ID, 8) + ".local." }

func (i Info) txt() []string {
	return []string{"id=" + i.ID, "api=" + strconv.Itoa(i.API), "name=" + sanitise(i.Name)}
}

func mustName(s string) dnsmessage.Name {
	n, err := dnsmessage.NewName(s)
	if err != nil {
		// names are built from sanitised parts and are always short enough
		return dnsmessage.MustNewName(".")
	}
	return n
}

func hdr(name string, t dnsmessage.Type, ttl uint32, flush bool) dnsmessage.ResourceHeader {
	c := dnsmessage.ClassINET
	if flush {
		c |= cacheFlush
	}
	return dnsmessage.ResourceHeader{Name: mustName(name), Type: t, Class: c, TTL: ttl}
}

func (i Info) ptr(ttl uint32) dnsmessage.Resource {
	return dnsmessage.Resource{Header: hdr(serviceName, dnsmessage.TypePTR, ttl, false),
		Body: &dnsmessage.PTRResource{PTR: mustName(i.instance())}}
}

func (i Info) srv(ttl uint32, flush bool) dnsmessage.Resource {
	return dnsmessage.Resource{Header: hdr(i.instance(), dnsmessage.TypeSRV, ttl, flush),
		Body: &dnsmessage.SRVResource{Port: uint16(i.Port), Target: mustName(i.host())}}
}

func (i Info) txtRec(ttl uint32, flush bool) dnsmessage.Resource {
	return dnsmessage.Resource{Header: hdr(i.instance(), dnsmessage.TypeTXT, ttl, flush),
		Body: &dnsmessage.TXTResource{TXT: i.txt()}}
}

// addrRecs gives the A records (want4) or AAAA records (!want4) for addrs.
func (i Info) addrRecs(addrs []net.IP, want4 bool, ttl uint32, flush bool) []dnsmessage.Resource {
	var out []dnsmessage.Resource
	for _, ip := range addrs {
		if v4 := ip.To4(); v4 != nil {
			if !want4 {
				continue
			}
			var a [4]byte
			copy(a[:], v4)
			out = append(out, dnsmessage.Resource{Header: hdr(i.host(), dnsmessage.TypeA, ttl, flush), Body: &dnsmessage.AResource{A: a}})
		} else if v6 := ip.To16(); v6 != nil {
			if want4 {
				continue
			}
			var a [16]byte
			copy(a[:], v6)
			out = append(out, dnsmessage.Resource{Header: hdr(i.host(), dnsmessage.TypeAAAA, ttl, flush), Body: &dnsmessage.AAAAResource{AAAA: a}})
		}
	}
	return out
}

// Query is a PTR question for _easy-engine._tcp.local. (ID 0, no unicast-response bit).
func Query() []byte {
	m := dnsmessage.Message{Questions: []dnsmessage.Question{{
		Name: mustName(serviceName), Type: dnsmessage.TypePTR, Class: dnsmessage.ClassINET}}}
	b, _ := m.Pack()
	return b
}

// Announcement is an unsolicited response carrying the PTR, SRV, TXT and the
// A/AAAA records for info. ttl is the lifetime in seconds of every record;
// ttl 0 is a goodbye.
func Announcement(info Info, addrs []net.IP, ttl uint32) []byte {
	m := dnsmessage.Message{Header: dnsmessage.Header{Response: true, Authoritative: true}}
	m.Answers = append(m.Answers, info.ptr(ttl), info.srv(ttl, true), info.txtRec(ttl, true))
	m.Answers = append(m.Answers, info.addrRecs(addrs, true, ttl, true)...)
	m.Answers = append(m.Answers, info.addrRecs(addrs, false, ttl, true)...)
	b, _ := m.Pack()
	return b
}

// Answer builds the reply to an mDNS query that asks for something of ours.
// ok is false when query is a response, is not parseable, or asks nothing of
// ours. legacy says the querier's source port is not 5353 (RFC 6762 6.7): the
// reply then echoes the query ID, repeats the question, caps TTLs at 10 and
// sets no cache-flush bit.
func Answer(info Info, query []byte, addrs []net.IP, legacy bool) (reply []byte, ok bool) {
	reply, ok, _ = answer(info, query, addrs, legacy)
	return reply, ok
}

func sameRec(a, b dnsmessage.Resource) bool {
	return a.Header.Name == b.Header.Name && a.Header.Type == b.Header.Type && fmt.Sprint(a.Body) == fmt.Sprint(b.Body)
}

// answer is Answer, also saying whether any question carried the QU bit.
func answer(info Info, query []byte, addrs []net.IP, legacy bool) (reply []byte, ok, qu bool) {
	var q dnsmessage.Message
	if err := q.Unpack(query); err != nil || q.Response {
		return nil, false, false
	}
	ttlOf := func(t uint32) uint32 {
		if legacy && t > ttlLegacyMax {
			return ttlLegacyMax
		}
		return t
	}
	flush := !legacy
	var ans, add []dnsmessage.Resource
	put := func(dst *[]dnsmessage.Resource, rs ...dnsmessage.Resource) {
		for _, r := range rs {
			dup := false
			for _, o := range *dst {
				if sameRec(o, r) {
					dup = true
				}
			}
			if !dup {
				*dst = append(*dst, r)
			}
		}
	}
	allAddrs := func() {
		put(&add, info.addrRecs(addrs, true, ttlOf(ttlHost), flush)...)
		put(&add, info.addrRecs(addrs, false, ttlOf(ttlHost), flush)...)
	}
	inst, host := strings.ToLower(info.instance()), strings.ToLower(info.host())
	for _, qs := range q.Questions {
		name := strings.ToLower(qs.Name.String())
		if qs.Class&cacheFlush != 0 {
			qu = true
		}
		if c := qs.Class &^ cacheFlush; c != dnsmessage.ClassINET && c != dnsmessage.ClassANY {
			continue
		}
		t := qs.Type
		all := t == dnsmessage.TypeALL
		switch name {
		case serviceName:
			if t == dnsmessage.TypePTR || all {
				put(&ans, info.ptr(ttlOf(ttlOther)))
				put(&add, info.srv(ttlOf(ttlHost), flush), info.txtRec(ttlOf(ttlOther), flush))
				allAddrs()
			}
		case metaName:
			if t == dnsmessage.TypePTR || all {
				put(&ans, dnsmessage.Resource{Header: hdr(metaName, dnsmessage.TypePTR, ttlOf(ttlOther), false),
					Body: &dnsmessage.PTRResource{PTR: mustName(serviceName)}})
			}
		case inst:
			if t == dnsmessage.TypeSRV || all {
				put(&ans, info.srv(ttlOf(ttlHost), flush))
				allAddrs()
			}
			if t == dnsmessage.TypeTXT || all {
				put(&ans, info.txtRec(ttlOf(ttlOther), flush))
			}
		case host:
			if t == dnsmessage.TypeA || all {
				put(&ans, info.addrRecs(addrs, true, ttlOf(ttlHost), flush)...)
			}
			if t == dnsmessage.TypeAAAA || all {
				put(&ans, info.addrRecs(addrs, false, ttlOf(ttlHost), flush)...)
			}
		}
	}
	if len(ans) == 0 {
		return nil, false, qu
	}
	// what is already an answer is not repeated as an additional
	var extra []dnsmessage.Resource
	for _, r := range add {
		dup := false
		for _, a := range ans {
			if sameRec(a, r) {
				dup = true
			}
		}
		if !dup {
			extra = append(extra, r)
		}
	}
	m := dnsmessage.Message{Header: dnsmessage.Header{Response: true, Authoritative: true}, Answers: ans, Additionals: extra}
	if legacy {
		m.ID = q.ID
		m.Questions = q.Questions
	}
	b, err := m.Pack()
	if err != nil {
		return nil, false, qu
	}
	return b, true, qu
}

// Parse pulls engines out of a response (answers and additionals), joining
// PTR to SRV to TXT to A/AAAA by name. Entries without an id TXT are skipped.
// A goodbye (TTL 0) parses like any other response.
func Parse(msg []byte) []Found {
	var m dnsmessage.Message
	if err := m.Unpack(msg); err != nil || !m.Response {
		return nil
	}
	type srvRec struct {
		port   int
		target string
	}
	var ptrs []string
	srvs := map[string]srvRec{}
	txts := map[string][]string{}
	addrs := map[string][]net.IP{}
	low := strings.ToLower
	for _, r := range append(append([]dnsmessage.Resource{}, m.Answers...), m.Additionals...) {
		name := low(r.Header.Name.String())
		switch b := r.Body.(type) {
		case *dnsmessage.PTRResource:
			if name == serviceName {
				ptrs = append(ptrs, b.PTR.String())
			}
		case *dnsmessage.SRVResource:
			srvs[name] = srvRec{int(b.Port), low(b.Target.String())}
		case *dnsmessage.TXTResource:
			txts[name] = append(txts[name], b.TXT...)
		case *dnsmessage.AResource:
			addrs[name] = append(addrs[name], net.IP(append([]byte(nil), b.A[:]...)))
		case *dnsmessage.AAAAResource:
			addrs[name] = append(addrs[name], net.IP(append([]byte(nil), b.AAAA[:]...)))
		}
	}
	var out []Found
	for _, p := range ptrs {
		key := low(p)
		tx, ok := txts[key]
		if !ok {
			continue
		}
		f := Found{Instance: p}
		for _, kv := range tx {
			k, v, _ := strings.Cut(kv, "=")
			switch k {
			case "id":
				f.ID = v
			case "name":
				f.Name = v
			case "api":
				f.API, _ = strconv.Atoi(v)
			}
		}
		if f.ID == "" {
			continue
		}
		if s, ok := srvs[key]; ok {
			f.Port = s.port
			f.Host = s.target
			f.Addrs = addrs[s.target]
		}
		out = append(out, f)
	}
	return out
}

// merge adds f to the set by ID, joining address lists.
func merge(set map[string]*Found, f Found) {
	if cur, ok := set[f.ID]; ok {
		for _, ip := range f.Addrs {
			dup := false
			for _, o := range cur.Addrs {
				if o.Equal(ip) {
					dup = true
				}
			}
			if !dup {
				cur.Addrs = append(cur.Addrs, ip)
			}
		}
		return
	}
	c := f
	set[f.ID] = &c
}

func sorted(set map[string]*Found) []Found {
	out := make([]Found, 0, len(set))
	for _, f := range set {
		out = append(out, *f)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return out[i].ID < out[j].ID
	})
	return out
}
