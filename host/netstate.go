package host

// netstate.go — what the network stands as, for a watch that is told "something moved" far more
// often than anything that matters to a program waiting for the network did (an address's
// lifetime refreshed, the same address announced again, a VPN's route churn). The set of links
// that are up and the addresses they hold is the reachability; a signal is made only when that set
// differs from the one last signalled.

import (
	"net"
	"sort"
	"strings"
)

// netSet is the sorted lines "link addr" for every link that is up and not loopback, one per
// address (a link with no address is one line). Equal sets are equal slices.
type netSet []string

// readNetSet reads what stands now.
func readNetSet() (netSet, error) {
	ifs, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	var set netSet
	for _, ifc := range ifs {
		if ifc.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := ifc.Addrs()
		if err != nil {
			addrs = nil // the link vanished between the two calls: it has no addresses
		}
		set = append(set, linkLines(ifc.Name, ifc.Flags&net.FlagUp != 0, addrs)...)
	}
	sort.Strings(set)
	return set, nil
}

// linkLines is the lines one link contributes: none when it is down (a link that is down reaches
// nothing, and its going down is a change), else one per address.
func linkLines(name string, up bool, addrs []net.Addr) []string {
	if !up {
		return nil
	}
	var out []string
	for _, a := range addrs {
		out = append(out, name+" "+a.String())
	}
	if len(out) == 0 {
		out = append(out, name+" -")
	}
	return out
}

func (a netSet) equal(b netSet) bool {
	return strings.Join(a, "\n") == strings.Join(b, "\n")
}

// netTracker remembers the set last signalled.
type netTracker struct {
	last netSet
	have bool
}

// changed reads what stands via read and says whether it differs from the set last signalled
// (and remembers it). A read that fails counts as a change (something moved and cannot be told
// what); the last set is kept so the next good read is compared with the one before.
func (t *netTracker) changed(read func() (netSet, error)) bool {
	cur, err := read()
	if err != nil {
		return true
	}
	if t.have && t.last.equal(cur) {
		return false
	}
	t.last, t.have = cur, true
	return true
}
