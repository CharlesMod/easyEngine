package plan

import (
	"fmt"
	"strings"

	"hermesengine/catalog"
)

// DiskHeadroomMiB stays free after the whole setup has written everything it will write
// (every download and unpack at its peak, the logs at their caps, the host's declared files).
// Every byte the engine writes is counted in a DiskNeed, so the headroom covers only what no
// count can see: the filesystem's own metadata for new files and directories. The plan's disk
// verdict and every installer check use it through DiskNeed. Evidence for the figure is in
// docs/ENGINE.md section 7a (measured on a fresh install).
const DiskHeadroomMiB = 512

// diskBlock is the allocation unit the counts round each file up to (ext4 and NTFS default).
const diskBlock = 4096

// unpackFactor stands in for an unknown unpacked size: a catalog older than version 8 has no
// unpacked_size. The real ratios at b12651 are 1.3x to 2.9x; 4x keeps an old catalog safe.
const unpackFactor = 4

// DiskStep is one stage of a setup in install order. Peak is the extra disk above what was
// used before the stage, at the stage's highest point (an archive and its unpacked tree side
// by side); Keep is what stays once the stage is done.
type DiskStep struct {
	What string `json:"what"`
	Peak int64  `json:"peak_bytes"`
	Keep int64  `json:"keep_bytes"`
}

// DiskNeed is the free disk a setup needs from the current state: the running maximum over its
// steps in install order, plus room for the logs to reach their caps, plus the host's declared
// files, plus DiskHeadroomMiB.
type DiskNeed struct {
	Steps    []DiskStep `json:"steps,omitempty"`
	Peak     int64      `json:"peak_bytes"`     // the running maximum over Steps
	Logs     int64      `json:"logs_bytes"`     // room for the engine's logs to reach their caps
	Host     int64      `json:"host_bytes"`     // the host's own growing files (profile)
	Headroom int64      `json:"headroom_bytes"` // DiskHeadroomMiB
	Total    int64      `json:"total_bytes"`
}

// NeedOf adds up steps (in install order), the log room and the host's files. The steps' peaks
// do not coincide: while step i runs, everything earlier steps kept is on disk, so the need is
// max over i of (kept before i + peak of i), never the sum of the peaks.
func NeedOf(steps []DiskStep, logs, host int64) DiskNeed {
	n := DiskNeed{Logs: max(logs, 0), Host: max(host, 0), Headroom: DiskHeadroomMiB << 20}
	var base int64
	for _, st := range steps {
		if st.Peak <= 0 && st.Keep <= 0 {
			continue
		}
		n.Steps = append(n.Steps, st)
		n.Peak = max(n.Peak, base+st.Peak)
		base += st.Keep
	}
	n.Peak = max(n.Peak, base)
	n.Total = n.Peak + n.Logs + n.Host + n.Headroom
	return n
}

// Downloads reports whether the setup still writes anything beyond logs (a step left to run).
func (n DiskNeed) Downloads() bool { return len(n.Steps) > 0 }

// Short reports whether free (bytes) is less than the need. An unknown free (< 0) is not short.
func (n DiskNeed) Short(free int64) bool { return free >= 0 && free < n.Total }

// Explain names what needs the space: "needs 4.1 GB: model 3.4 GB, graphics runtime 1.1 GB,
// unpacking room 0.9 GB, logs 0.1 GB, headroom 0.1 GB". The parts add up to the total: what the
// steps keep, the extra their peak needs on top (temporary files), logs, host files, headroom.
func (n DiskNeed) Explain() string {
	var parts []string
	var kept int64
	for _, st := range n.Steps {
		kept += st.Keep
		if st.Keep > 0 {
			parts = append(parts, st.What+" "+GB(st.Keep))
		}
	}
	if t := n.Peak - kept; t > 0 {
		parts = append(parts, "temporary files while unpacking "+GB(t))
	}
	if n.Logs > 0 {
		parts = append(parts, "logs "+GB(n.Logs))
	}
	if n.Host > 0 {
		parts = append(parts, "the host's files "+GB(n.Host))
	}
	parts = append(parts, "headroom "+GB(n.Headroom))
	return "needs " + GB(n.Total) + ": " + strings.Join(parts, ", ")
}

// GB formats bytes as GB with one decimal (MB under 0.1 GB), as the setup page shows them.
func GB(b int64) string {
	if b < 100<<20 {
		return fmt.Sprintf("%d MB", (b+(1<<20)-1)>>20)
	}
	return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
}

func roundBlock(n int64) int64 { return (n + diskBlock - 1) / diskBlock * diskBlock }

func unpacked(a catalog.Asset) int64 {
	if a.UnpackedSize > 0 {
		return a.UnpackedSize
	}
	return unpackFactor * a.Size
}

// BackendSteps are the steps of installing a llama.cpp build and its extras, as
// run.installBackend does them: the archive is downloaded, unpacked beside it (both on disk),
// then deleted. An extra (cudart) is downloaded, unpacked to a temp dir, its archive deleted,
// its files copied into the server's dir, then the temp dir removed: archive + tree, then two
// trees.
func BackendSteps(a catalog.Asset, extras []catalog.Asset) []DiskStep {
	u := unpacked(a)
	steps := []DiskStep{{What: "model server", Peak: roundBlock(a.Size) + u, Keep: u}}
	for _, x := range extras {
		xu := unpacked(x)
		steps = append(steps, DiskStep{What: "graphics runtime", Peak: max(roundBlock(x.Size)+xu, 2*xu), Keep: xu})
	}
	return steps
}

// UVStep is installing uv (run.ensureUV): archive + tree, then the tree + the copied binary.
// What stays is counted as the whole tree (the binary is most of it).
func UVStep(a catalog.Asset) DiskStep {
	u := unpacked(a)
	return DiskStep{What: "Python installer", Peak: max(roundBlock(a.Size)+u, 2*u), Keep: u}
}

// ModelStep is downloading a model: the .part grows to the full size beside nowhere else and is
// renamed at the end, so the peak is what is left to fetch (the size minus a partial .part).
func ModelStep(what string, size, part int64) DiskStep {
	left := size
	if part > 0 && part < size {
		left = size - part
	}
	left = roundBlock(left)
	return DiskStep{What: what, Peak: left, Keep: left}
}

// AgentStep is the agent's runtime (managed Python, uv cache, venv) from a measured setup,
// less what is already on disk of it (have bytes).
func AgentStep(rt catalog.RuntimeSize, have int64) DiskStep {
	return DiskStep{What: "agent runtime", Peak: max(rt.PeakBytes-have, 0), Keep: max(rt.KeepBytes-have, 0)}
}

// ModelOnlyNeed is the need of downloading m with nothing else counted: the plan's default when
// no live state is given (tests, the CLI plan without a data dir).
func ModelOnlyNeed(m catalog.Model) DiskNeed {
	return NeedOf([]DiskStep{ModelStep("model", m.Size, 0)}, 0, 0)
}
