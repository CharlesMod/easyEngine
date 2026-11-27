package yield

import "strings"

// OS is the platform's view of other programs; tests replace it.
type OS interface {
	// Launches lists the games launchers are running now.
	Launches() []Launch
	// Openers lists processes holding a GPU device open (pid -> name), skipping those skip
	// says; nil where the platform cannot tell.
	Openers(skip func(pid int) bool) map[int]string
	// Adapter narrows Openers to the device nodes of the adapter at this PCI address (Linux:
	// /dev/nvidia<minor>, its DRM card and render nodes); "" or an unknown address: every GPU node.
	Adapter(pci string)
	// Under says whether pid descends from anc (false where the platform cannot tell).
	Under(pid, anc int) bool
	Close()
}

// Words are what the system itself says about other programs, where it says anything
// (docs/EVENTS.md section 3). A nil channel is a word the system does not give: the caller
// reads that thing on the host watchdog instead.
type Words struct {
	Launches <-chan struct{} // a launcher's game may have started or ended (Steam's RunningAppID)
	Opens    <-chan struct{} // some process opened a GPU device node of the engine's adapter
	Stop     func()          // lets the watches go; nil-safe through Release
	Set      string          // what Opens watches (the nodes), to compare with Noder.NodeSet
}

// Release lets the watches go.
func (w Words) Release() {
	if w.Stop != nil {
		w.Stop()
	}
}

// Watcher is the optional OS piece that gives Words. An OS without it gives none: its launchers
// are read on the host watchdog with everything else.
type Watcher interface {
	Watch() Words
}

// Noder is the optional OS piece that names the set of nodes its Watch would watch now. The set
// is known only once Adapter has found the engine's nodes, which may be after a first Watch: the
// caller watches again when it differs from the Set of the Words it holds.
type Noder interface {
	NodeSet() string
}

// ParseRegistryVDF finds Steam's RunningAppID in a registry.vdf (macOS and Linux Steam keep
// it there): "0" or absent means no game.
func ParseRegistryVDF(b []byte) string {
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && strings.EqualFold(strings.Trim(f[0], `"`), "RunningAppID") {
			v := strings.Trim(f[1], `"`)
			if digits(v) && strings.Trim(v, "0") != "" {
				return v
			}
			return ""
		}
	}
	return ""
}
