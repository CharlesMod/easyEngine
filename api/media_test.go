package api_test

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/CharlesMod/easyEngine/api"
	"github.com/CharlesMod/easyEngine/run"
)

// fromElsewhere is a caller from another machine as the master key (it needs no attach, and the
// check of media comes before the door): a listener whose callers all say they came through a forwarder.
func fromElsewhere(t *testing.T, s *run.Supervisor) func(method, path, body string, extra ...map[string]string) (int, string) {
	t.Helper()
	msrv := httptest.NewServer((&api.API{S: s, AnyHost: true, Tokens: testChecker{"tok-m": "master"}}).Handler())
	t.Cleanup(msrv.Close)
	return func(method, path, body string, extra ...map[string]string) (int, string) {
		h := map[string]string{"Authorization": "Bearer tok-m", "X-Forwarded-For": "10.9.8.6", "Content-Type": "application/json"}
		for _, x := range extra {
			for k, v := range x {
				h[k] = v
			}
		}
		c, b, _ := do(t, method, msrv.URL+path, body, h)
		return c, b
	}
}

// Every way media enters llama-server (docs/ENGINE.md 7g "Media from remote callers") is refused
// 400 images_not_offered to a caller from another machine, and nothing reaches the model server.
func TestMediaNotOfferedToRemote(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	post := fromElsewhere(t, s)
	before := upstreamBytes(t, e)
	cases := []struct{ name, path, body string }{
		{"chat image_url data URI", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`},
		{"chat image_url https", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}`},
		{"chat input_audio", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]}`},
		{"chat input_video", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"input_video","input_video":{"url":"https://x/v.mp4"}}]}]}`},
		{"chat video_url", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://x/v.mp4"}}]}]}`},
		{"anthropic image base64", "/v1/messages", `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}`},
		{"anthropic image url", "/v1/messages", `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}`},
		{"anthropic tool_result image", "/v1/messages", `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}]}`},
		{"responses input_image", "/v1/responses", `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://x/y.png"}]}]}`},
		{"completion multimodal_data", "/completion", `{"prompt":{"prompt_string":"<__media__> x","multimodal_data":["AAAA"]}}`},
		{"completion prompt array", "/completion", `{"prompt":["a",{"prompt_string":"<__media__> x","multimodal_data":["AAAA"]}]}`},
		{"bare marker in a prompt", "/completion", `{"prompt":"look <__media__> here"}`},
		{"random marker in a prompt", "/v1/completions", `{"prompt":"look <__media_k3j2h1__> here"}`},
		{"marker in a chat text part", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"text","text":"x <__media__>"}]}]}`},
		{"embeddings image_url item", "/v1/embeddings", `{"input":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"https://x/y.png"}}]}`},
		{"rerank image document", "/rerank", `{"query":"q","documents":[{"text":"t","image":"AAAA"}]}`},
		{"systemone images", "/v1/systemone", `{"images":["data:image/png;base64,AAAA"]}`},
		{"systemone files", "/v1/systemone", `{"files":["a.png"]}`},
	}
	for _, c := range cases {
		code, b := post("POST", c.path, c.body)
		if code != 400 || codeOf(b) != "images_not_offered" {
			t.Errorf("%s: %d %s", c.name, code, b)
		}
	}
	// the audio routes by path, with any body (or none), any method
	for _, c := range []struct{ method, path string }{
		{"POST", "/v1/audio/transcriptions"}, {"POST", "/audio/transcriptions"}, {"POST", "/v1//audio/transcriptions"},
		{"GET", "/v1/realtime"}, {"GET", "/realtime"}, {"POST", "/v1/realtime"},
	} {
		code, b := post(c.method, c.path, "")
		if code != 400 || codeOf(b) != "images_not_offered" {
			t.Errorf("%s %s: %d %s", c.method, c.path, code, b)
		}
	}
	// a multipart upload to any path, and a compressed body that could not be read
	mp := "--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF\r\n--b--\r\n"
	code, b := post("POST", "/v1/audio/transcriptions", mp, map[string]string{"Content-Type": "multipart/form-data; boundary=b"})
	if code != 400 || codeOf(b) != "images_not_offered" {
		t.Errorf("multipart transcription: %d %s", code, b)
	}
	code, b = post("POST", "/v1/chat/completions", mp, map[string]string{"Content-Type": "multipart/form-data; boundary=b"})
	if code != 400 || codeOf(b) != "images_not_offered" {
		t.Errorf("multipart chat: %d %s", code, b)
	}
	code, b = post("POST", "/v1/chat/completions", `{"messages":[]}`, map[string]string{"Content-Encoding": "gzip"})
	if code != 400 || codeOf(b) != "images_not_offered" {
		t.Errorf("gzip chat: %d %s", code, b)
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("a refused media request reached the model server")
	}
}

// A paired token is refused the same way on the paths its scope allows, with no attach needed.
func TestMediaNotOfferedToPairedToken(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	before := upstreamBytes(t, e)
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`, as(1))
	if c != 400 || codeOf(b) != "images_not_offered" {
		t.Fatalf("paired image: %d %s", c, b)
	}
	if upstreamBytes(t, e) != before {
		t.Fatal("a refused image reached the model server")
	}
}

// Text from a remote caller still passes, even with a tool schema that names a property "image".
func TestRemoteTextWithImageSchemaPasses(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 1)
	body := `{"messages":[{"role":"user","content":"draw"}],"tools":[{"type":"function","function":{"name":"show","parameters":{"type":"object","properties":{"image":{"type":"string"},"images":{"type":"array","items":{"type":"string"}}}}}}]}`
	c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", body, as(1))
	if c != 200 {
		t.Fatalf("remote text with a tool schema: %d %s", c, b)
	}
	if got := upstreamBytes(t, e); !strings.Contains(got, `"image":{"type":"string"}`) {
		t.Fatalf("the model server saw %s", got)
	}
}

// A local caller's image body reaches the model server byte for byte.
func TestLocalImageBodyUnchanged(t *testing.T) {
	e, _, srv := clientsEnv(t, true, nil)
	body := `{"messages":[{"role":"user","content":[{"type":"text","text":"what is it"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`
	c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", body, jsonCT)
	if c != 200 {
		t.Fatalf("local image: %d %s", c, b)
	}
	if got := upstreamBytes(t, e); got != body {
		t.Fatalf("the model server saw %s", got)
	}
	// a marker in a local prompt passes too
	c, _, _ = do(t, "POST", srv.URL+"/completion", `{"prompt":"a <__media__> b"}`, jsonCT)
	if c != 200 || upstreamBytes(t, e) != `{"prompt":"a <__media__> b"}` {
		t.Fatalf("local marker: %d %s", c, upstreamBytes(t, e))
	}
}

// A remote caller's chat body passes the allow list only with text content: a string, null, absent
// content, or parts that are exactly a type "text" and a text string.
func TestRemoteChatTextShapesPass(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	_, rs, _ := remoteSrv(t, s)
	growRemote(t, e, s, rs, 1, "a", 1)
	for name, body := range map[string]string{
		"string content": `{"messages":[{"role":"user","content":"hi"}]}`,
		"null content":   `{"messages":[{"role":"assistant","content":null,"tool_calls":[]}]}`,
		"absent content": `{"messages":[{"role":"assistant","tool_calls":[]}]}`,
		"text parts":     `{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"text":"b","type":"text"}]}]}`,
		"empty messages": `{"messages":[]}`,
	} {
		c, b, _ := do(t, "POST", rs.URL+"/v1/chat/completions", body, as(1))
		if c != 200 {
			t.Errorf("%s: %d %s", name, c, b)
			continue
		}
		// the engine adds its own id_slot: the messages arrive as sent
		want := body[strings.Index(body, `"messages"`) : len(body)-1]
		if got := upstreamBytes(t, e); !strings.Contains(got, want) {
			t.Errorf("%s: the model server saw %s", name, got)
		}
	}
}

// Anything else in a remote caller's chat body is refused, whatever it is called, and nothing is forwarded.
func TestRemoteChatNonTextRefused(t *testing.T) {
	e, s, _ := clientsEnv(t, true, nil)
	post := fromElsewhere(t, s)
	before := upstreamBytes(t, e)
	cases := map[string]string{
		"unknown part type":      `{"messages":[{"role":"user","content":[{"type":"future_media"}]}]}`,
		"unknown type with text": `{"messages":[{"role":"user","content":[{"type":"future_media","text":"x"}]}]}`,
		"extra key in text part": `{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`,
		"text not a string":      `{"messages":[{"role":"user","content":[{"type":"text","text":5}]}]}`,
		"text part without text": `{"messages":[{"role":"user","content":[{"type":"text"}]}]}`,
		"part not an object":     `{"messages":[{"role":"user","content":["x"]}]}`,
		"content object":         `{"messages":[{"role":"user","content":{"type":"text","text":"x"}}]}`,
		"content number":         `{"messages":[{"role":"user","content":7}]}`,
		"duplicate content":      `{"messages":[{"role":"user","content":"a","content":[{"type":"future_media"}]}]}`,
		"duplicate type":         `{"messages":[{"role":"user","content":[{"type":"future_media","type":"text","text":"x"}]}]}`,
		"duplicate messages":     `{"messages":[],"messages":[{"role":"user","content":[{"type":"future_media"}]}]}`,
		"messages an object":     `{"messages":{"role":"user","content":"x"}}`,
		"messages a string":      `{"messages":"x"}`,
		"message not an object":  `{"messages":["x"]}`,
	}
	for name, body := range cases {
		for _, p := range []string{"/v1/chat/completions", "/v1//chat/completions"} {
			c, b := post("POST", p, body)
			if c != 400 || codeOf(b) != "images_not_offered" {
				t.Errorf("%s %s: %d %s", name, p, c, b)
			}
		}
	}
	if got := upstreamBytes(t, e); got != before {
		t.Fatalf("a refused body reached the model server: %s", got)
	}
}

// The allow list is for remote callers: a local caller's unknown part type reaches the model server unchanged.
func TestLocalUnknownPartTypePasses(t *testing.T) {
	e, _, srv := clientsEnv(t, true, nil)
	body := `{"messages":[{"role":"user","content":[{"type":"future_media"}]}]}`
	c, b, _ := do(t, "POST", srv.URL+"/v1/chat/completions", body, jsonCT)
	if c != 200 {
		t.Fatalf("local unknown part: %d %s", c, b)
	}
	if got := upstreamBytes(t, e); got != body {
		t.Fatalf("the model server saw %s", got)
	}
}
