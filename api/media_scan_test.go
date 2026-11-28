package api

import (
	"strings"
	"testing"
)

func TestMediaIn(t *testing.T) {
	deep := strings.Repeat(`[`, maxJSONDepth+2) + strings.Repeat(`]`, maxJSONDepth+2)
	cases := []struct {
		name, body string
		want       bool
	}{
		{"plain chat", `{"messages":[{"role":"user","content":"hello"}]}`, false},
		{"text parts", `{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`, false},
		{"empty", ``, false},
		{"not json, no marker", `hello`, false},
		{"image_url data", `{"messages":[{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`, true},
		{"image_url https", `{"messages":[{"content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}`, true},
		{"type only", `{"content":[{"type":"image_url"}]}`, true},
		{"type case", `{"content":[{"type":" Input_Image "}]}`, true},
		{"input_audio", `{"messages":[{"content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]}`, true},
		{"input_video", `{"messages":[{"content":[{"type":"input_video","input_video":{"url":"x"}}]}]}`, true},
		{"video_url", `{"messages":[{"content":[{"type":"video_url","video_url":{"url":"x"}}]}]}`, true},
		{"anthropic base64", `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA"}}]}]}`, true},
		{"anthropic url", `{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://x/y.png"}}]}]}`, true},
		{"anthropic tool_result image", `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"type":"url","url":"https://x"}}]}]}]}`, true},
		{"responses input_image", `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://x/y.png"}]}]}`, true},
		{"responses input_file", `{"input":[{"role":"user","content":[{"type":"input_file","file_id":"f"}]}]}`, true},
		{"file part", `{"content":[{"type":"file","file":{"file_data":"x"}}]}`, true},
		{"document", `{"content":[{"type":"document","source":{}}]}`, true},
		{"media_marker part", `{"content":[{"type":"media_marker","text":"x"}]}`, true},
		{"completion multimodal_data", `{"prompt":{"prompt_string":"x","multimodal_data":["AAAA"]}}`, true},
		{"completion prompt array", `{"prompt":["a",{"prompt_string":"y","multimodal_data":["AAAA"]}]}`, true},
		{"multimodal_data empty list", `{"prompt":{"prompt_string":"y","multimodal_data":[]}}`, true},
		{"multimodal_data null", `{"prompt":{"prompt_string":"y","multimodal_data":null}}`, false},
		{"bare marker prompt", `{"prompt":"look <__media__> here"}`, true},
		{"random marker prompt", `{"prompt":"<__media_a1b2c3__>"}`, true},
		{"marker in chat text part", `{"messages":[{"content":[{"type":"text","text":"x <__media__>"}]}]}`, true},
		{"marker in a key", `{"<__media__>":1}`, true},
		{"embeddings image_url item", `{"input":[{"type":"text","text":"a"},{"image_url":{"url":"https://x"}}]}`, true},
		{"embeddings input image_url key", `{"input":{"image_url":"https://x"}}`, true},
		{"rerank image", `{"query":"q","documents":[{"text":"t","image":"AAAA"}]}`, true},
		{"systemone images", `{"images":["data:image/png;base64,AA"]}`, true},
		{"systemone files", `{"files":["x"]}`, true},
		{"systemone audio", `{"audio":"x"}`, true},
		{"video key", `{"video":"x"}`, true},
		{"data uri audio", `{"x":"data:audio/wav;base64,AA"}`, true},
		{"data uri video upper spaced", `{"x":"  DATA:VIDEO/mp4;base64,AA"}`, true},
		{"data uri not media", `{"x":"data:text/plain;base64,AA"}`, false},
		{"data uri mid string", `{"x":"see data:image/png here"}`, false},
		{"escaped key", `{"messages":[{"content":[{"type":"image_url"}]}]}`, true},
		{"duplicate key hides media", `{"image":"AAAA","image":null}`, true},
		{"duplicate type", `{"type":"image_url","type":"text"}`, true},
		{"null image key", `{"image":null,"messages":[]}`, false},
		{"tool schema property named image", `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"image":{"type":"string"},"images":{"type":"array"}}}}}]}`, false},
		{"tool schema with type image", `{"tools":[{"function":{"parameters":{"properties":{"p":{"type":"image"}}}}}]}`, false},
		{"functions schema", `{"functions":[{"parameters":{"properties":{"audio":{"type":"string"}}}}]}`, false},
		{"response_format schema", `{"response_format":{"type":"json_schema","json_schema":{"schema":{"properties":{"image":{"type":"string"}}}}}}`, false},
		{"grammar subtree", `{"grammar":{"image":"x"}}`, false},
		{"marker inside a skipped subtree", `{"tools":[{"function":{"description":"use <__media__>"}}]}`, true},
		{"data uri inside a skipped subtree", `{"tools":[{"function":{"description":"data:image/png;base64,AA"}}]}`, true},
		{"media next to a skipped subtree", `{"tools":[{"function":{}}],"messages":[{"content":[{"type":"image_url"}]}]}`, true},
		{"too deep, no marker", `{"a":` + deep + `}`, false},
		{"too deep, marker", `{"a":` + deep + `,"p":"<__media__>"}`, true},
		{"broken json, marker", `{"p":"<__media__>"`, true},
		{"trailing garbage, marker", `{"p":"x"} <__media__>`, true},
		{"trailing garbage", `{"p":"x"} y`, false},
	}
	for _, c := range cases {
		got, why := mediaIn([]byte(c.body))
		if got != c.want || (got && why == "") {
			t.Errorf("%s: mediaIn = %v %q, want %v", c.name, got, why, c.want)
		}
	}
}
