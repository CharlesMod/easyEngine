package settings

import (
	"strings"
	"testing"
)

// The streamer key: on by default, a missing key is on, off is kept, anything else is refused
// with its line, and a file is always written with the key.
func TestStreamerKey(t *testing.T) {
	if Defaults().Streamer != "on" {
		t.Fatalf("default %q", Defaults().Streamer)
	}
	f, err := Parse([]byte(`{"mode":"local"}`))
	if err != nil || f.Streamer != "on" {
		t.Fatalf("missing key: %q %v", f.Streamer, err)
	}
	f, err = Parse([]byte(`{"streamer":"off"}`))
	if err != nil || f.Streamer != "off" {
		t.Fatalf("off: %q %v", f.Streamer, err)
	}
	_, err = Parse([]byte("{\n\"mode\":\"local\",\n\"streamer\":\"maybe\"\n}"))
	se, ok := err.(*Error)
	if !ok || se.Line != 3 || !strings.Contains(se.Msg, "streamer must be on or off") {
		t.Fatalf("bad value: %v", err)
	}
	if !strings.Contains(string(Encode(Defaults())), `"streamer": "on"`) {
		t.Fatalf("written file lacks the key:\n%s", Encode(Defaults()))
	}
}
