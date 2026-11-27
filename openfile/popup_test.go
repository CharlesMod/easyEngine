package openfile

import (
	"reflect"
	"testing"
)

const u = "http://127.0.0.1:47480/chat/?mini=1"

func TestPopupCommand(t *testing.T) {
	app := []string{"--app=" + u, "--window-size=420,640"}
	cases := []struct {
		name, line string
		want       []string // nil: not Chromium family, fall back
	}{
		{"flatpak chrome", "/usr/bin/flatpak run --branch=stable --arch=x86_64 --command=/app/bin/chrome --file-forwarding com.google.Chrome @@u %U @@",
			append([]string{"/usr/bin/flatpak", "run", "--branch=stable", "--arch=x86_64", "--command=/app/bin/chrome", "--file-forwarding", "com.google.Chrome"}, app...)},
		{"flatpak brave", "flatpak run --branch=stable --arch=x86_64 --command=brave --file-forwarding com.brave.Browser @@u %U @@",
			append([]string{"flatpak", "run", "--branch=stable", "--arch=x86_64", "--command=brave", "--file-forwarding", "com.brave.Browser"}, app...)},
		{"flatpak firefox", "/usr/bin/flatpak run --branch=stable --arch=x86_64 --command=firefox --file-forwarding org.mozilla.firefox @@u %u @@", nil},
		{"native chromium", "/usr/bin/chromium-browser %U", append([]string{"/usr/bin/chromium-browser"}, app...)},
		{"quoted path", `"/opt/Google Chrome/google-chrome" --profile-directory=Default %U`,
			append([]string{"/opt/Google Chrome/google-chrome", "--profile-directory=Default"}, app...)},
		{"edge", "microsoft-edge-stable %U", append([]string{"microsoft-edge-stable"}, app...)},
		{"native firefox", "firefox %u", nil},
		{"windows chrome", `"C:\Program Files\Google\Chrome\Application\chrome.exe" --single-argument %1`,
			append([]string{`C:\Program Files\Google\Chrome\Application\chrome.exe`}, app...)},
		{"windows edge", `"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --single-argument %1`,
			append([]string{`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`}, app...)},
		{"windows firefox", `"C:\Program Files\Mozilla Firefox\firefox.exe" -osint -url "%1"`, nil},
		{"empty", "", nil},
		{"unknown", "epiphany %U", nil},
	}
	for _, c := range cases {
		got, ok := PopupCommand(c.line, u, PopupWidth, PopupHeight)
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

func TestMacPopupCommand(t *testing.T) {
	got, ok := MacPopupCommand("com.google.Chrome", u, 420, 640)
	want := []string{"open", "-na", "Google Chrome", "--args", "--app=" + u, "--window-size=420,640"}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q", got)
	}
	if _, ok := MacPopupCommand("org.mozilla.firefox", u, 420, 640); ok {
		t.Fatal("firefox has no app mode")
	}
	if _, ok := MacPopupCommand("", u, 420, 640); ok {
		t.Fatal("unknown default")
	}
	defaults := "(\n  {\n    LSHandlerContentType = \"public.html\";\n    LSHandlerRoleAll = \"com.apple.safari\";\n  },\n  {\n    LSHandlerRoleAll = \"com.brave.browser\";\n    LSHandlerURLScheme = https;\n  }\n)"
	if id := macHandlerID(defaults); id != "com.brave.browser" {
		t.Fatalf("handler %q", id)
	}
}
