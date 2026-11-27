// Package yield decides when the engine gives the card to another program: a program that
// opens the GPU or holds memory on the engine's adapter, or a game a launcher started.
// It holds the pure parts (matching, tracking, the state machine); run/ wires them to the
// supervisor and the OS.
package yield

import (
	"path/filepath"
	"strings"
)

// Launch is a game a launcher started; it lives exactly as long as the game.
type Launch struct {
	PID   int
	Kind  string // steam | lutris | umu | steam-registry
	AppID string // Steam app id, when known
	Exe   string // the game's executable or title, when known
}

// Label names the launch for a person: "AppId 123 (game.exe)", or the exe alone.
func (l Launch) Label() string {
	switch {
	case l.AppID != "" && l.Exe != "":
		return "AppId " + l.AppID + " (" + l.Exe + ")"
	case l.AppID != "":
		return "AppId " + l.AppID
	case l.Exe != "":
		return l.Exe
	}
	return l.Kind
}

// MatchArgv recognises a launcher's game process from its argv (Linux /proc/<pid>/cmdline):
//   - Steam: ".../reaper SteamLaunch AppId=<n> -- ..." (reaper lives as long as the game);
//   - Lutris: a process titled "lutris-wrapper: <name>";
//   - umu (Heroic and Lutris run Proton games through it): argv[0] or argv[1] is umu-run.
//
// Kernel threads (empty argv, such as oom_reaper) never match.
func MatchArgv(argv []string) (Launch, bool) {
	if len(argv) == 0 || argv[0] == "" {
		return Launch{}, false
	}
	a0 := filepath.Base(argv[0])
	if a0 == "reaper" {
		steam, id := false, ""
		for _, a := range argv[1:] {
			if a == "SteamLaunch" {
				steam = true
			}
			if v, ok := strings.CutPrefix(a, "AppId="); ok && digits(v) && strings.Trim(v, "0") != "" {
				id = v
			}
		}
		if steam && id != "" {
			return Launch{Kind: "steam", AppID: id, Exe: gameExe(argv)}, true
		}
		return Launch{}, false
	}
	if t, ok := strings.CutPrefix(argv[0], "lutris-wrapper: "); ok {
		return Launch{Kind: "lutris", Exe: strings.TrimSpace(t)}, true
	}
	if a0 == "umu-run" || (len(argv) > 1 && filepath.Base(argv[1]) == "umu-run" && strings.HasPrefix(a0, "python")) {
		return Launch{Kind: "umu", Exe: gameExe(argv)}, true
	}
	return Launch{}, false
}

// gameExe is the last argument that names a Windows executable, else the one after the last "--".
func gameExe(argv []string) string {
	for i := len(argv) - 1; i > 0; i-- {
		if strings.HasSuffix(strings.ToLower(argv[i]), ".exe") {
			return baseAny(argv[i])
		}
	}
	for i := len(argv) - 2; i > 0; i-- {
		if argv[i] == "--" {
			return baseAny(argv[i+1])
		}
	}
	return ""
}

// baseAny is the last element of a Unix or Windows path.
func baseAny(p string) string {
	if i := strings.LastIndexAny(p, `/\`); i >= 0 {
		return p[i+1:]
	}
	return p
}

func digits(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}
