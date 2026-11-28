package run

import (
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// A profile sent with an attach is checked before anything is registered. A remote client's is
// input from the network: every list and string is bounded, and a path (a file on this machine)
// is a local client's privilege, as are a pid, an agent and extra llama-server arguments.

var (
	samplerNameRe = regexp.MustCompile(`^[a-z0-9_-]{1,32}$`)
	plainNameRe   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
)

// requiredNames are the options a profile may mark required (beside an extra flag's name).
var requiredNames = []string{"ubatch", "reasoning", "agent", "model"}

// checkProfile vets a client's profile; it lower-cases sampler names (a request's X-Easy-Sampler
// is looked up lower-cased, so a name with capitals could never match).
func checkProfile(p *Profile, local bool, h *Help) *ClientError {
	bad := func(format string, a ...any) *ClientError { return cerr(400, "bad_request", format, a...) }
	if len(p.Game) > 128 {
		return bad("game is longer than 128 bytes")
	}
	if p.Slots < 0 || p.Slots > maxSlotsPerClient {
		return bad("slots must be 1 to %d", maxSlotsPerClient)
	}
	if p.MinSlots < 0 || p.MinSlots > maxSlotsPerClient {
		return bad("min_slots must be 0 to %d", maxSlotsPerClient)
	}
	if c := p.CtxPerSlot; c.Max != 0 && (c.Max < minCtxPerSlot || c.Max > maxCtxPerSlot) {
		return bad("ctx_per_slot.max must be %d to %d", minCtxPerSlot, maxCtxPerSlot)
	}
	if c := p.CtxPerSlot; c.Floor != 0 && (c.Floor < minCtxPerSlot || c.Floor > maxCtxPerSlot) {
		return bad("ctx_per_slot.floor must be %d to %d", minCtxPerSlot, maxCtxPerSlot)
	}
	if p.GameReserveMiB < 0 || p.GameReserveMiB > 1<<20 {
		return bad("game_reserve_mib out of range")
	}
	if len(p.Samplers) > 16 {
		return bad("at most 16 samplers")
	}
	if p.Samplers != nil {
		low := map[string]bool{}
		for k := range p.Samplers {
			if !samplerNameRe.MatchString(strings.ToLower(k)) {
				return bad("sampler name %q: 1 to 32 of a-z, 0-9, _ and -", k)
			}
			if low[strings.ToLower(k)] {
				return bad("sampler name %q twice", k)
			}
			low[strings.ToLower(k)] = true
		}
		for k, v := range p.Samplers {
			if lk := strings.ToLower(k); lk != k {
				delete(p.Samplers, k)
				p.Samplers[lk] = v
			}
		}
	}
	if len(p.YieldExcept) > 16 {
		return bad("at most 16 yield_except entries")
	}
	for _, y := range p.YieldExcept {
		if len(y) > 64 {
			return bad("yield_except entries are at most 64 bytes")
		}
	}
	if a := p.HostAdapter; a != nil && len(a.Name) > 128 {
		return bad("host_adapter.name is longer than 128 bytes")
	}
	if mp := p.Models; mp != nil {
		if len(mp.Prefer) > 16 || len(mp.CanUse) > 16 {
			return bad("models lists hold at most 16 ids")
		}
		for _, id := range append(append([]string{}, mp.Prefer...), mp.CanUse...) {
			if !plainNameRe.MatchString(id) {
				return bad("model id %q is not a plain name", id)
			}
		}
	}
	if !local {
		switch {
		case p.HostPID != 0:
			return cerr(400, "pid_remote", "host_pid names a process on this machine; a remote client may not send it")
		case p.Agent != nil:
			return cerr(400, "agent_not_allowed", "a client on another machine may not bring an agent")
		case p.HostFilesMiB != 0:
			return cerr(400, "path_not_allowed", "host_files_mib is about this machine's disk; a remote client may not send it")
		case p.Catalog != nil:
			return cerr(400, "catalog_not_allowed", "a client on another machine may not bring models: this machine would download what its person did not choose")
		}
		if sv := p.Server; sv != nil {
			switch {
			case len(sv.ExtraArgs) > 0:
				return cerr(400, "extra_args_not_allowed", "a client on another machine may send ubatch and reasoning only, no extra_args")
			}
		}
	}
	if p.HostPID < 0 {
		return bad("host_pid must not be negative")
	}
	if a := p.Agent; a != nil {
		if !plainNameRe.MatchString(a.ID) {
			return bad("agent.id %q is not a plain name", a.ID)
		}
		if a.Wheel != "" && !filepath.IsAbs(a.Wheel) {
			return cerr(400, "path_not_allowed", "agent.wheel must be an absolute path")
		}
		if len(a.Args) > 32 || len(a.Env) > 64 {
			return bad("agent args at most 32, env at most 64")
		}
	}
	if sv := p.Server; sv != nil {
		if sv.Ubatch != 0 && (sv.Ubatch < 32 || sv.Ubatch > 512) {
			return bad("server.ubatch must be 32 to 512 (the sizing counts 512)")
		}
		switch sv.Reasoning {
		case "", "on", "off", "auto":
		default:
			return bad("server.reasoning must be on, off or auto")
		}
		if len(sv.MMProj) > 0 {
			return cerr(400, "not_offered", "server.mmproj is not offered: a vision projector will come from the catalog with its model")
		}
		if len(sv.Speculative) > 0 {
			return cerr(400, "not_offered", "server.speculative is not offered: a draft model is not sized; ngram speculation goes in a local client's extra_args")
		}
		if len(sv.ExtraArgs) > 0 {
			if _, ae := CheckExtraArgs(sv.ExtraArgs, h); ae != nil {
				e := cerr(400, ae.Code, "%s", ae.Msg)
				if ae.Flag != "" {
					e.Extra = map[string]any{"flag": ae.Flag}
				}
				return e
			}
			for _, g := range extraGroups(sv.ExtraArgs) {
				if !isFlagToken(g[0]) {
					return bad("extra_args must begin with a flag, not %q", g[0])
				}
			}
		}
		if len(sv.Required) > 8 {
			return bad("at most 8 required options")
		}
		for _, r := range sv.Required {
			if !slices.Contains(requiredNames, r) && !strings.HasPrefix(r, "-") {
				return bad("required option %q is not one of %s or an extra flag", r, strings.Join(requiredNames, ", "))
			}
		}
	}
	return nil
}
