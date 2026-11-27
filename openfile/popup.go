package openfile

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

// Size of the chat popup, in CSS pixels.
const (
	PopupWidth  = 420
	PopupHeight = 640
)

// Popup shows addr in a small window with no browser chrome: the user's default browser, when it
// is Chromium family, started in app mode (--app=<addr> --window-size=...) with its normal
// profile, so the page sees the same browser storage a full browser window sees. When the
// default browser cannot be found or is not Chromium family (Firefox has no app mode), it opens
// addr in the default browser the ordinary way. It returns once the browser has started.
func Popup(addr string) error {
	if argv, ok := defaultPopupCommand(addr); ok {
		cmd := exec.Command(argv[0], argv[1:]...)
		if err := cmd.Start(); err == nil {
			go cmd.Wait() // reap it; the window outlives this call
			return nil
		}
	}
	return OpenURL(addr)
}

// defaultPopupCommand is the popup command for this system's default browser.
func defaultPopupCommand(addr string) ([]string, bool) {
	switch runtime.GOOS {
	case "linux", "freebsd", "openbsd", "netbsd":
		if line := linuxDefaultBrowserExec(); line != "" {
			return PopupCommand(line, addr, PopupWidth, PopupHeight)
		}
	case "darwin":
		return MacPopupCommand(macDefaultBrowserID(), addr, PopupWidth, PopupHeight)
	case "windows":
		if line := windowsDefaultBrowserCommand(); line != "" {
			return PopupCommand(line, addr, PopupWidth, PopupHeight)
		}
	}
	return nil, false
}

// ---- pure parts ----------------------------------------------------------------------------

// chromiumName matches the executable names, flatpak app ids and snap names of the Chromium
// family: Chrome, Chromium, Brave, Edge, Vivaldi, Opera, Thorium. Firefox, Safari and the rest
// have no app mode and do not match.
var chromiumName = regexp.MustCompile(`(?i)(chrome|chromium|brave|msedge|microsoft-edge|com\.microsoft\.edge|vivaldi|opera|thorium)`)

// PopupCommand turns a browser's launch line (a .desktop Exec value, or a Windows shell\open\command)
// into the command that opens addr as a popup: the line with its file and URL placeholders
// removed, then --app=<addr> --window-size=<w>,<h>. ok is false when the line is empty or the
// browser is not Chromium family.
func PopupCommand(line, addr string, w, h int) (argv []string, ok bool) {
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
		case isFieldCode(t), t == "--single-argument":
			continue
		}
		if t = stripFieldCodes(t); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 || !isChromium(out) {
		return nil, false
	}
	return append(out, "--app="+addr, "--window-size="+strconv.Itoa(w)+","+strconv.Itoa(h)), true
}

// isChromium: the launcher itself (first word), or, for "flatpak run <app-id>" and
// "snap run <name>", the app it runs, is a Chromium-family name. Options after that, such as
// flatpak's --command=/app/bin/chrome, are not looked at: they name no other browser.
func isChromium(argv []string) bool {
	base := strings.TrimSuffix(strings.ToLower(filepath.Base(strings.ReplaceAll(argv[0], `\`, "/"))), ".exe")
	switch base {
	case "flatpak", "snap", "env":
		for _, a := range argv[1:] {
			if a == "run" || strings.HasPrefix(a, "-") || strings.Contains(a, "=") {
				continue
			}
			return chromiumName.MatchString(a) // the first word that names the app
		}
		return false
	}
	return chromiumName.MatchString(base)
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

// macBrowsers maps a macOS bundle id (lower case) to the app name `open -a` takes.
var macBrowsers = map[string]string{
	"com.google.chrome":       "Google Chrome",
	"com.google.chrome.beta":  "Google Chrome Beta",
	"org.chromium.chromium":   "Chromium",
	"com.microsoft.edgemac":   "Microsoft Edge",
	"com.brave.browser":       "Brave Browser",
	"com.vivaldi.vivaldi":     "Vivaldi",
	"com.operasoftware.opera": "Opera",
}

// MacPopupCommand is the popup command for a default browser given by its bundle id: open -na
// "<app>" --args --app=... ; ok is false for any browser that is not a known Chromium one.
func MacPopupCommand(bundleID, addr string, w, h int) ([]string, bool) {
	app, ok := macBrowsers[strings.ToLower(strings.TrimSpace(bundleID))]
	if !ok {
		return nil, false
	}
	return []string{"open", "-na", app, "--args", "--app=" + addr, "--window-size=" + strconv.Itoa(w) + "," + strconv.Itoa(h)}, true
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
