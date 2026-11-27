package yield

import (
	"strings"
	"testing"
)

func TestMatchArgv(t *testing.T) {
	const steam = "/home/user/.local/share/Steam/"
	tests := []struct {
		name string
		argv []string
		want Launch
		ok   bool
	}{
		{"steam proton", strings.Fields(steam + "ubuntu12_32/reaper SteamLaunch AppId=1962700 -- " +
			steam + "ubuntu12_32/steam-launch-wrapper -- " +
			steam + "steamapps/common/SteamLinuxRuntime_sniper/_v2-entry-point --verb=waitforexitandrun -- " +
			steam + "steamapps/common/Proton/proton waitforexitandrun " +
			steam + "steamapps/common/SomeGame/SomeGame.exe"),
			Launch{Kind: "steam", AppID: "1962700", Exe: "SomeGame.exe"}, true},
		{"steam native", []string{steam + "ubuntu12_32/reaper", "SteamLaunch", "AppId=42", "--", steam + "ubuntu12_32/steam-launch-wrapper", "--", "/path/game.x86_64"},
			Launch{Kind: "steam", AppID: "42", Exe: "game.x86_64"}, true},
		{"steam windows path", []string{"reaper", "SteamLaunch", "AppId=7", "--", `C:\Games\SomeGame\SomeGame.EXE`},
			Launch{Kind: "steam", AppID: "7", Exe: "SomeGame.EXE"}, true},
		{"steam no exe", []string{"reaper", "SteamLaunch", "AppId=7"}, Launch{Kind: "steam", AppID: "7"}, true},
		{"reaper without SteamLaunch", []string{"/x/reaper", "AppId=1962700", "--", "/g/a.exe"}, Launch{}, false},
		{"AppId zero", []string{"reaper", "SteamLaunch", "AppId=0"}, Launch{}, false},
		{"AppId zeros", []string{"reaper", "SteamLaunch", "AppId=000"}, Launch{}, false},
		{"AppId abc", []string{"reaper", "SteamLaunch", "AppId=abc"}, Launch{}, false},
		{"AppId empty", []string{"reaper", "SteamLaunch", "AppId="}, Launch{}, false},
		{"SteamLaunch without id", []string{"reaper", "SteamLaunch", "--", "x"}, Launch{}, false},
		{"nil argv", nil, Launch{}, false},
		{"empty argv", []string{}, Launch{}, false},
		{"empty string argv", []string{""}, Launch{}, false},
		{"oom_reaper", []string{"oom_reaper"}, Launch{}, false},
		{"unrelated", []string{"/usr/bin/bash", "-c", "x"}, Launch{}, false},
		{"lutris", []string{"lutris-wrapper: Some Title"}, Launch{Kind: "lutris", Exe: "Some Title"}, true},
		{"lutris trimmed", []string{"lutris-wrapper:  Some Title  "}, Launch{Kind: "lutris", Exe: "Some Title"}, true},
		{"lutris-wrapper no separator", []string{"lutris-wrapper"}, Launch{}, false},
		{"umu direct", []string{"/usr/bin/umu-run", "/g/SomeGame.exe"}, Launch{Kind: "umu", Exe: "SomeGame.exe"}, true},
		{"umu via python", []string{"python3", "/x/umu-run", "/g/SomeGame.exe"}, Launch{Kind: "umu", Exe: "SomeGame.exe"}, true},
		{"umu via python3.12 path", []string{"/usr/bin/python3.12", "/x/umu-run", "-- ", "/g/SomeGame.exe"}, Launch{Kind: "umu", Exe: "SomeGame.exe"}, true},
		{"umu-run as argv1 of non-python", []string{"bash", "/x/umu-run"}, Launch{}, false},
		{"python other script", []string{"python3", "/x/other.py"}, Launch{}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := MatchArgv(tc.argv)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("MatchArgv(%q) = %+v, %v; want %+v, %v", tc.argv, got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestLaunchLabel(t *testing.T) {
	tests := []struct {
		l    Launch
		want string
	}{
		{Launch{Kind: "steam", AppID: "123", Exe: "game.exe"}, "AppId 123 (game.exe)"},
		{Launch{Kind: "steam", AppID: "123"}, "AppId 123"},
		{Launch{Kind: "lutris", Exe: "Some Title"}, "Some Title"},
		{Launch{Kind: "umu"}, "umu"},
		{Launch{}, ""},
	}
	for _, tc := range tests {
		if got := tc.l.Label(); got != tc.want {
			t.Errorf("Label(%+v) = %q, want %q", tc.l, got, tc.want)
		}
	}
}
