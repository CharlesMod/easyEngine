package autostart

import (
	"path/filepath"
	"strings"
)

// ---- Linux / XDG ----

// configHome is $XDG_CONFIG_HOME when it is absolute, else home/.config.
func configHome(xdg, home string) string {
	if filepath.IsAbs(xdg) {
		return xdg
	}
	return filepath.Join(home, ".config")
}

func desktopPath(configHome, name string) string {
	return filepath.Join(configHome, "autostart", name+".desktop")
}

const desktopReserved = " \t\n\"'\\><~|&;$*?#()`"

// desktopExec builds an Exec value per the Desktop Entry spec, including the
// second level of string escaping (every backslash doubled once more).
func desktopExec(argv []string) string {
	parts := make([]string, len(argv))
	for i, a := range argv {
		parts[i] = desktopArg(a)
	}
	return strings.Join(parts, " ")
}

func desktopArg(a string) string {
	if a == "" || strings.ContainsAny(a, desktopReserved) {
		var b strings.Builder
		b.WriteByte('"')
		for _, r := range a {
			switch r {
			case '"', '`', '$', '\\':
				b.WriteByte('\\')
			}
			b.WriteRune(r)
		}
		b.WriteByte('"')
		a = b.String()
	}
	a = strings.ReplaceAll(a, "%", "%%")
	a = strings.ReplaceAll(a, `\`, `\\`)
	a = strings.ReplaceAll(a, "\n", `\n`)
	a = strings.ReplaceAll(a, "\t", `\t`)
	a = strings.ReplaceAll(a, "\r", `\r`)
	return a
}

func desktopEntry(name, exe string, args []string) string {
	argv := append([]string{exe}, args...)
	return "[Desktop Entry]\n" +
		"Type=Application\n" +
		"Name=" + name + "\n" +
		"Exec=" + desktopExec(argv) + "\n" +
		"X-GNOME-Autostart-enabled=true\n" +
		"Terminal=false\n"
}

// desktopEnabled is false when the entry's main group says Hidden=true or
// X-GNOME-Autostart-enabled=false.
func desktopEnabled(contents string) bool {
	inMain := false
	for _, line := range strings.Split(contents, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") {
			inMain = line == "[Desktop Entry]"
			continue
		}
		if !inMain {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.ToLower(strings.TrimSpace(v))
		if k == "Hidden" && v == "true" {
			return false
		}
		if k == "X-GNOME-Autostart-enabled" && v == "false" {
			return false
		}
	}
	return true
}

// ---- macOS ----

// plistLabel is "com.easyengine."+name unless name already contains a dot.
func plistLabel(name string) string {
	if strings.Contains(name, ".") {
		return name
	}
	return "com.easyengine." + name
}

func plistPath(home, label string) string {
	return filepath.Join(home, "Library", "LaunchAgents", label+".plist")
}

var xmlEscaper = strings.NewReplacer(
	"&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;")

func launchAgentPlist(label, exe string, args []string) string {
	var b strings.Builder
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	b.WriteString(`<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">` + "\n")
	b.WriteString(`<plist version="1.0">` + "\n<dict>\n")
	b.WriteString("\t<key>Label</key>\n\t<string>" + xmlEscaper.Replace(label) + "</string>\n")
	b.WriteString("\t<key>ProgramArguments</key>\n\t<array>\n")
	for _, a := range append([]string{exe}, args...) {
		b.WriteString("\t\t<string>" + xmlEscaper.Replace(a) + "</string>\n")
	}
	b.WriteString("\t</array>\n")
	b.WriteString("\t<key>RunAtLoad</key>\n\t<true/>\n")
	b.WriteString("\t<key>ProcessType</key>\n\t<string>Interactive</string>\n")
	b.WriteString("</dict>\n</plist>\n")
	return b.String()
}

// ---- Windows ----

const runSubkey = `Software\Microsoft\Windows\CurrentVersion\Run`

// windowsCommandLine quotes the exe always and each arg by the
// CommandLineToArgvW rules.
func windowsCommandLine(exe string, args []string) string {
	parts := make([]string, 0, len(args)+1)
	parts = append(parts, winQuote(exe, true))
	for _, a := range args {
		parts = append(parts, winQuote(a, false))
	}
	return strings.Join(parts, " ")
}

func winQuote(s string, force bool) string {
	if !force && s != "" && !strings.ContainsAny(s, " \t\"") {
		return s
	}
	var b strings.Builder
	b.WriteByte('"')
	bs := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '\\':
			bs++
			continue
		case '"':
			b.WriteString(strings.Repeat(`\`, bs*2+1))
		default:
			b.WriteString(strings.Repeat(`\`, bs))
		}
		bs = 0
		b.WriteByte(c)
	}
	b.WriteString(strings.Repeat(`\`, bs*2))
	b.WriteByte('"')
	return b.String()
}
