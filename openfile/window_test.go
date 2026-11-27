package openfile

import (
	"reflect"
	"testing"
)

const u = "http://127.0.0.1:47480/chat/"

func TestWindowCommand(t *testing.T) {
	nw := []string{"--new-window", u}
	cases := []struct {
		name, line string
		want       []string // nil: not a browser known to take --new-window, fall back
	}{
		{"flatpak firefox", "/usr/bin/flatpak run --branch=stable --arch=x86_64 --command=firefox --file-forwarding org.mozilla.firefox @@u %u @@",
			append([]string{"/usr/bin/flatpak", "run", "--branch=stable", "--arch=x86_64", "--command=firefox", "--file-forwarding", "org.mozilla.firefox"}, nw...)},
		{"native firefox", "firefox %u", append([]string{"firefox"}, nw...)},
		{"flatpak chrome", "/usr/bin/flatpak run --branch=stable --arch=x86_64 --command=/app/bin/chrome --file-forwarding com.google.Chrome @@u %U @@",
			append([]string{"/usr/bin/flatpak", "run", "--branch=stable", "--arch=x86_64", "--command=/app/bin/chrome", "--file-forwarding", "com.google.Chrome"}, nw...)},
		{"flatpak librewolf", "flatpak run --command=librewolf io.gitlab.librewolf-community @@u %u @@",
			append([]string{"flatpak", "run", "--command=librewolf", "io.gitlab.librewolf-community"}, nw...)},
		{"flatpak zen", "flatpak run app.zen_browser.zen @@u %u @@", append([]string{"flatpak", "run", "app.zen_browser.zen"}, nw...)},
		{"waterfox and floorp", "waterfox %u", append([]string{"waterfox"}, nw...)},
		{"floorp", "/opt/floorp/floorp %u", append([]string{"/opt/floorp/floorp"}, nw...)},
		{"native chromium", "/usr/bin/chromium-browser %U", append([]string{"/usr/bin/chromium-browser"}, nw...)},
		{"quoted path", `"/opt/Google Chrome/google-chrome" --profile-directory=Default %U`,
			append([]string{"/opt/Google Chrome/google-chrome", "--profile-directory=Default"}, nw...)},
		{"windows chrome", `"C:\Program Files\Google\Chrome\Application\chrome.exe" -- "%1"`,
			append([]string{`C:\Program Files\Google\Chrome\Application\chrome.exe`}, nw...)},
		{"windows edge", `"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --single-argument %1`,
			append([]string{`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`}, nw...)},
		{"windows firefox", `"C:\Program Files\Mozilla Firefox\firefox.exe" -osint -url "%1"`,
			append([]string{`C:\Program Files\Mozilla Firefox\firefox.exe`}, nw...)},
		{"empty", "", nil},
		{"epiphany", "epiphany %U", nil},
		{"zenity is not zen", "zenity --info", nil},
		{"flatpak epiphany", "flatpak run --command=epiphany org.gnome.Epiphany @@u %U @@", nil},
	}
	for _, c := range cases {
		got, ok := WindowCommand(c.line, u)
		if c.want == nil {
			if ok || got != nil {
				t.Errorf("%s: want fallback, got %v", c.name, got)
			}
			continue
		}
		if !ok || !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s:\n got %q\nwant %q", c.name, got, c.want)
		}
	}
}

func TestDesktopExec(t *testing.T) {
	d := "[Desktop Entry]\nName=Chrome\nExec=/usr/bin/flatpak run com.google.Chrome @@u %U @@\n\n[Desktop Action new-window]\nExec=/usr/bin/other\n"
	if got := DesktopExec(d); got != "/usr/bin/flatpak run com.google.Chrome @@u %U @@" {
		t.Fatalf("got %q", got)
	}
	if DesktopExec("[Desktop Action x]\nExec=a\n") != "" {
		t.Fatal("only the main group counts")
	}
}

func TestMacWindowCommand(t *testing.T) {
	got, ok := MacWindowCommand("com.google.Chrome", u)
	want := []string{"open", "-na", "Google Chrome", "--args", "--new-window", u}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q", got)
	}
	got, ok = MacWindowCommand("org.mozilla.firefox", u)
	want = []string{"open", "-na", "Firefox", "--args", "--new-window", u}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("firefox: got %q", got)
	}
	if _, ok := MacWindowCommand("com.apple.Safari", u); ok {
		t.Fatal("safari is opened the ordinary way")
	}
	if _, ok := MacWindowCommand("", u); ok {
		t.Fatal("unknown default")
	}
	defaults := "(\n  {\n    LSHandlerContentType = \"public.html\";\n    LSHandlerRoleAll = \"com.apple.safari\";\n  },\n  {\n    LSHandlerRoleAll = \"com.brave.browser\";\n    LSHandlerURLScheme = https;\n  }\n)"
	if id := macHandlerID(defaults); id != "com.brave.browser" {
		t.Fatalf("handler %q", id)
	}
}
