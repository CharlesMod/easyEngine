package run

import (
	"os"
	"strings"
	"testing"
)

// TestSaysListeningRealLogs reads llama-server's own logs: the newer build says it listens once,
// after the model loaded; the older one says it before loading and again after. The matcher
// finds exactly those lines and none of the others (the UI line "init_listene" among them).
func TestSaysListeningRealLogs(t *testing.T) {
	for file, want := range map[string][]string{
		"testdata/ready-b11279.txt": {"llama_server: listening on http://127.0.0.1:41745"},
		"testdata/ready-older.txt": {
			"main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 15",
			"main: server is listening on http://127.0.0.1:8080 - starting the main loop",
		},
	} {
		b, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for _, l := range strings.Split(string(b), "\n") {
			if saysListening(l) {
				got = append(got, l)
			}
		}
		if len(got) != len(want) {
			t.Fatalf("%s: matched %q, want %q", file, got, want)
		}
		for i := range want {
			if !strings.HasSuffix(got[i], want[i]) {
				t.Fatalf("%s: line %d is %q, want it to end %q", file, i, got[i], want[i])
			}
		}
	}
}
