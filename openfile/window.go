package openfile

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

// Window shows addr in a NEW window of the user's default browser: where the browser's launch
// line is known and it takes --new-window (the Chromium and Firefox families do), it starts that
// browser with --new-window <addr>; in every other case (an unknown browser, Safari, a failed
// lookup, a failed launch) it opens addr the ordinary way, which may be a tab. It returns once
// the browser has started.
func Window(addr string) error {
	if err := checkAddr(addr); err != nil {
		return err
	}
	if argv, ok := defaultWindowCommand(addr); ok {
		cmd := exec.Command(argv[0], argv[1:]...)
		if err := cmd.Start(); err == nil {
			go cmd.Wait() // reap it; the window outlives this call
			return nil
		}
	}
	return OpenURL(addr)
}

// DefaultWindowCommand is the command Window would run for addr on this system, for display
// (ok is false when Window would use the ordinary opener).
func DefaultWindowCommand(addr string) ([]string, bool) { return defaultWindowCommand(addr) }

// defaultWindowCommand is the new-window command for this system's default browser.
func defaultWindowCommand(addr string) ([]string, bool) {
	switch runtime.GOOS {
	case "linux", "freebsd", "openbsd", "netbsd":
		if line := linuxDefaultBrowserExec(); line != "" {
			return WindowCommand(line, addr)
		}
	case "darwin":
		return MacWindowCommand(macDefaultBrowserID(), addr)
	case "windows":
		if line := windowsDefaultBrowserCommand(); line != "" {
			return WindowCommand(line, addr)
		}
	}
	return nil, false
}

// ---- pure parts ----------------------------------------------------------------------------

// newWindowName matches the executable names, flatpak app ids and snap names of the browsers
// known to take --new-window <url>: the Chromium family (Chrome, Chromium, Brave, Edge, Vivaldi,
// Opera, Thorium) and the Firefox family (Firefox, LibreWolf, Waterfox, Floorp, Zen). Safari and
// the rest are not in it and get the ordinary opener. "zen" must stand as a word of its own
// (zen, zen-browser, app.zen_browser.zen), so zenity and the like do not match.
var newWindowName = regexp.MustCompile(`(?i)(chrome|chromium|brave|msedge|microsoft-edge|com\.microsoft\.edge|vivaldi|opera|thorium|firefox|librewolf|waterfox|floorp|(^|[^a-z])zen([^a-z]|$))`)

// WindowCommand turns a browser's launch line (a .desktop Exec value, or a Windows
// shell\open\command) into the command that opens addr in a new window: the line with its file
// and URL placeholders removed, then --new-window <addr>. ok is false when the line is empty or
// the browser is not one that takes --new-window.
func WindowCommand(line, addr string) (argv []string, ok bool) {
	toks := splitLine(line)
	var out []string
	skipping := false // inside a flatpak "@@u ... @@" file-forwarding span
	for _, t := range toks {
		switch {
		case t == "@@" || t == "@@u" || t == "@@f":
			skipping = !skipping
			continue
		case skipping:
			continue
		// Windows registry lines end in the way a link is handed over: Edge --single-argument "%1",
		// Chrome -- "%1", Firefox -osint -url "%1". Those words must go with the placeholder.
		case isFieldCode(t), t == "--single-argument", t == "--", t == "-osint", t == "-url":
			continue
		}
		if t = stripFieldCodes(t); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 || !takesNewWindow(out) {
		return nil, false
	}
	return append(out, "--new-window", addr), true
}

// takesNewWindow: the launcher itself (first word), or, for "flatpak run <app-id>" and
// "snap run <name>", the app it runs, is a name in newWindowName. Options after that, such as
// flatpak's --command=/app/bin/chrome, are not looked at: they name no other browser.
func takesNewWindow(argv []string) bool {
	base := strings.TrimSuffix(strings.ToLower(filepath.Base(strings.ReplaceAll(argv[0], `\`, "/"))), ".exe")
	switch base {
	case "flatpak", "snap", "env":
		for _, a := range argv[1:] {
			if a == "run" || strings.HasPrefix(a, "-") || strings.Contains(a, "=") {
				continue
			}
			return newWindowName.MatchString(a) // the first word that names the app
		}
		return false
	}
	return newWindowName.MatchString(base)
}

func isFieldCode(t string) bool {
	return len(t) == 2 && t[0] == '%' && strings.ContainsRune("fFuUdDnNickvm123456789", rune(t[1]))
}

// stripFieldCodes removes %-codes glued into a word, and unescapes %%.
func stripFieldCodes(t string) string {
	if !strings.Contains(t, "%") {
		return t
	}
	var b strings.Builder
	for i := 0; i < len(t); i++ {
		if t[i] != '%' || i+1 == len(t) {
			b.WriteByte(t[i])
			continue
		}
		i++
		if t[i] == '%' {
			b.WriteByte('%')
		}
	}
	return b.String()
}

// splitLine splits a command line into words the way a .desktop Exec value is written:
// double quotes group, a backslash escapes inside them. Windows paths in quotes pass through,
// since a backslash there is only taken as an escape before a quote or a backslash.
func splitLine(s string) []string {
	var words []string
	var cur strings.Builder
	inWord, inQuote := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case inQuote && c == '\\' && i+1 < len(s) && (s[i+1] == '"' || s[i+1] == '\\'):
			i++
			cur.WriteByte(s[i])
		case c == '"':
			inQuote, inWord = !inQuote, true
		case !inQuote && (c == ' ' || c == '\t'):
			if inWord {
				words = append(words, cur.String())
				cur.Reset()
				inWord = false
			}
		default:
			cur.WriteByte(c)
			inWord = true
		}
	}
	if inWord {
		words = append(words, cur.String())
	}
	return words
}

// DesktopExec is the Exec value of the [Desktop Entry] group of a .desktop file ("" if none).
func DesktopExec(desktop string) string {
	in := false
	for _, ln := range strings.Split(desktop, "\n") {
		ln = strings.TrimSpace(ln)
		if strings.HasPrefix(ln, "[") {
			in = ln == "[Desktop Entry]"
			continue
		}
		if in && strings.HasPrefix(ln, "Exec=") {
			return strings.TrimSpace(strings.TrimPrefix(ln, "Exec="))
		}
	}
	return ""
}

// macBrowsers maps a macOS bundle id (lower case) to the app name `open -a` takes: the browsers
// above that can be named. Safari is not here, and so is opened the ordinary way.
var macBrowsers = map[string]string{
	"com.google.chrome":                   "Google Chrome",
	"com.google.chrome.beta":              "Google Chrome Beta",
	"org.chromium.chromium":               "Chromium",
	"com.microsoft.edgemac":               "Microsoft Edge",
	"com.brave.browser":                   "Brave Browser",
	"com.vivaldi.vivaldi":                 "Vivaldi",
	"com.operasoftware.opera":             "Opera",
	"org.mozilla.firefox":                 "Firefox",
	"org.mozilla.firefoxdeveloperedition": "Firefox Developer Edition",
	"io.gitlab.librewolf-community":       "LibreWolf",
	"net.waterfox.waterfox":               "Waterfox",
	"one.ablaze.floorp":                   "Floorp",
	"app.zen-browser.zen":                 "Zen",
}

// MacWindowCommand is the new-window command for a default browser given by its bundle id: open
// -na "<app>" --args --new-window <addr> ; ok is false for any browser not in macBrowsers.
func MacWindowCommand(bundleID, addr string) ([]string, bool) {
	app, ok := macBrowsers[strings.ToLower(strings.TrimSpace(bundleID))]
	if !ok {
		return nil, false
	}
	return []string{"open", "-na", app, "--args", "--new-window", addr}, true
}

// macHandlerID reads the https handler's bundle id out of `defaults read ... LSHandlers` text.
func macHandlerID(defaults string) string {
	id := ""
	for _, block := range strings.Split(defaults, "}") {
		if !strings.Contains(block, "LSHandlerURLScheme = https;") && !strings.Contains(block, `LSHandlerURLScheme = "https";`) {
			continue
		}
		for _, ln := range strings.Split(block, "\n") {
			ln = strings.TrimSpace(ln)
			if strings.HasPrefix(ln, "LSHandlerRoleAll") {
				if _, v, ok := strings.Cut(ln, "="); ok {
					id = strings.Trim(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(v), ";")), `"`)
				}
			}
		}
	}
	return id
}

// ---- system lookups ------------------------------------------------------------------------

// linuxDefaultBrowserExec: xdg-settings names the default browser's .desktop file; its Exec
// line launches it (for a Flatpak that is "flatpak run ... <app id>").
func linuxDefaultBrowserExec() string {
	out, err := exec.Command("xdg-settings", "get", "default-web-browser").Output()
	if err != nil {
		return ""
	}
	name := strings.TrimSpace(string(out))
	if name == "" || strings.ContainsAny(name, "/ ") {
		return ""
	}
	for _, dir := range desktopDirs() {
		if b, err := os.ReadFile(filepath.Join(dir, name)); err == nil {
			if line := DesktopExec(string(b)); line != "" {
				return line
			}
		}
	}
	return ""
}

func desktopDirs() []string {
	home, _ := os.UserHomeDir()
	data := os.Getenv("XDG_DATA_HOME")
	if data == "" && home != "" {
		data = filepath.Join(home, ".local", "share")
	}
	dirs := []string{filepath.Join(data, "applications")}
	if home != "" {
		dirs = append(dirs, filepath.Join(home, ".local", "share", "flatpak", "exports", "share", "applications"))
	}
	dirs = append(dirs, "/var/lib/flatpak/exports/share/applications")
	sys := os.Getenv("XDG_DATA_DIRS")
	if sys == "" {
		sys = "/usr/local/share:/usr/share"
	}
	for _, d := range strings.Split(sys, ":") {
		if d != "" {
			dirs = append(dirs, filepath.Join(d, "applications"))
		}
	}
	return dirs
}

func macDefaultBrowserID() string {
	out, err := exec.Command("defaults", "read", "com.apple.LaunchServices/com.apple.launchservices.secure", "LSHandlers").Output()
	if err != nil {
		return ""
	}
	return macHandlerID(string(out))
}
