package api

import (
	"errors"
	"strings"
	"testing"
)

func TestRunTestTable(t *testing.T) {
	ok := outcome{status: 200, body: `{"data":[{"id":"m"}]}`}
	chatOK := outcome{status: 200, body: `{"choices":[{"message":{"content":"ready"}}]}`, ms: 50}
	cases := []struct {
		name  string
		in    string
		model string
		send  func(method, url string) outcome
		light string
		code  string
	}{
		{"no url", "", "m", nil, "red", "no_url"},
		{"ok", "http://h/v1", "m", func(m, u string) outcome {
			if m == "GET" {
				return ok
			}
			return chatOK
		}, "green", "ok"},
		{"key rejected", "http://h/v1", "m", func(m, u string) outcome { return outcome{status: 401} }, "red", "key_rejected"},
		{"loading 503", "http://h/v1", "m", func(m, u string) outcome { return outcome{status: 503} }, "amber", "loading"},
		{"refused", "http://h/v1", "m", func(m, u string) outcome { return outcome{err: errors.New("connection refused")} }, "red", "nothing_listening"},
		{"model missing", "http://h/v1", "zzz", func(m, u string) outcome { return ok }, "red", "model_not_found"},
		{"not json", "http://h/v1", "m", func(m, u string) outcome { return outcome{status: 200, body: "<html>"} }, "red", "not_json"},
		{"reasoning text", "http://h/v1", "m", func(m, u string) outcome {
			if m == "GET" {
				return ok
			}
			return outcome{status: 200, body: `{"choices":[{"message":{"content":"x","reasoning_content":"thinking"}}]}`}
		}, "amber", "reasoning_text"},
		{"slow", "http://h/v1", "m", func(m, u string) outcome {
			if m == "GET" {
				return ok
			}
			c := chatOK
			c.ms = 9000
			return c
		}, "amber", "slow"},
		{"wrong base fixed by toggling /v1", "http://h", "m", func(m, u string) outcome {
			if strings.HasPrefix(u, "http://h/v1/") {
				if m == "GET" {
					return ok
				}
				return chatOK
			}
			return outcome{status: 404}
		}, "green", "ok_fixed_base"},
		{"wrong base everywhere", "http://h/v1", "m", func(m, u string) outcome { return outcome{status: 404} }, "red", "wrong_base"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			send := func(m, u, body, key string, to int) outcome { return c.send(m, u) }
			if c.send == nil {
				send = func(string, string, string, string, int) outcome { t.Fatal("sent"); return outcome{} }
			}
			r := RunTest(c.in, c.model, "", send)
			if r.Light != c.light || r.Code != c.code {
				t.Fatalf("got %s/%s (%s), want %s/%s", r.Light, r.Code, r.Message, c.light, c.code)
			}
		})
	}
}

func TestNormalizeBase(t *testing.T) {
	for in, want := range map[string]string{
		"  localhost:1234/v1/ ":        "http://localhost:1234/v1",
		"https://api.example.com/v1?x": "https://api.example.com/v1",
		"http://h:80//":                "http://h:80",
	} {
		if got := NormalizeBase(in); got != want {
			t.Errorf("NormalizeBase(%q) = %q, want %q", in, got, want)
		}
	}
}
