package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// This is a Go port of the mod's LlmProbe (Mod/Source/Bridge/LlmProbe.cs): the same
// steps (GET /models, then an 8-token chat) and the same green/amber/red rules.

type probeCode string

const (
	cOK             probeCode = "ok"
	cOKFixedBase    probeCode = "ok_fixed_base"
	cLoading        probeCode = "loading"
	cSlow           probeCode = "slow"
	cReasoning      probeCode = "reasoning_text"
	cNoURL          probeCode = "no_url"
	cBadURL         probeCode = "bad_url"
	cNothingListen  probeCode = "nothing_listening"
	cKeyRejected    probeCode = "key_rejected"
	cWrongBase      probeCode = "wrong_base"
	cModelNotFound  probeCode = "model_not_found"
	cTLS            probeCode = "tls"
	cNotJSON        probeCode = "not_json"
	cHTTPError      probeCode = "http_error"
	cTimeout        probeCode = "timeout"
	cOther          probeCode = "other"
	slowMs                    = 8000
	chatTimeoutMs             = 45000
	modelsTimeoutMs           = 15000
	healthTimeoutMs           = 8000
)

// TestResult is the answer of POST /engine/v1/test.
type TestResult struct {
	Light        string `json:"light"` // green | amber | red
	Message      string `json:"message"`
	LatencyMS    int64  `json:"latency_ms"`
	Model        string `json:"model"`
	Code         string `json:"code"`
	EffectiveURL string `json:"effective_base,omitempty"`
}

type outcome struct {
	status int
	body   string
	err    error
	ms     int64
}

type verdict struct {
	light string
	code  probeCode
	arg   string
}

func v(l string, c probeCode, arg string) *verdict { return &verdict{l, c, arg} }

type sender func(method, url, body, key string, timeoutMs int) outcome

// NormalizeBase trims, adds http://, drops query and trailing slashes and cuts a
// pasted ".../chat/completions" or ".../models" back to the base. "" if unusable.
func NormalizeBase(in string) string {
	s := strings.TrimSpace(in)
	if s == "" {
		return ""
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return ""
	}
	p := strings.TrimRight(u.Path, "/")
	for _, tail := range []string{"/chat/completions", "/completions", "/models"} {
		if strings.HasSuffix(strings.ToLower(p), tail) {
			p = strings.TrimRight(p[:len(p)-len(tail)], "/")
			break
		}
	}
	return u.Scheme + "://" + u.Host + p
}

func toggleV1(b string) string {
	if strings.HasSuffix(strings.ToLower(b), "/v1") {
		return b[:len(b)-3]
	}
	return b + "/v1"
}

func rootOf(b string) string {
	if strings.HasSuffix(strings.ToLower(b), "/v1") {
		return b[:len(b)-3]
	}
	return b
}

func classifyErr(err error) *verdict {
	var ne net.Error
	msg := err.Error()
	low := strings.ToLower(msg)
	switch {
	case strings.Contains(low, "tls") || strings.Contains(low, "certificate") || strings.Contains(low, "ssl"):
		return v("red", cTLS, "")
	case errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &ne) && ne.Timeout()):
		return v("amber", cTimeout, "")
	case strings.Contains(low, "refused") || strings.Contains(low, "no such host") || strings.Contains(low, "unreachable") || strings.Contains(low, "no route"):
		return v("red", cNothingListen, "")
	}
	return v("red", cOther, msg)
}

func tryJSON(body string) map[string]any {
	var m map[string]any
	if json.Unmarshal([]byte(body), &m) != nil {
		return nil
	}
	return m
}

func errMessage(body string) string {
	m := tryJSON(body)
	if m == nil {
		return ""
	}
	if e, ok := m["error"].(map[string]any); ok {
		if s, ok := e["message"].(string); ok {
			return s
		}
	}
	return ""
}

func classifyModels(o outcome, model string) *verdict {
	if o.err != nil {
		return classifyErr(o.err)
	}
	switch {
	case o.status == 401 || o.status == 403:
		return v("red", cKeyRejected, "")
	case o.status == 404:
		return v("red", cWrongBase, "")
	case o.status == 503:
		return v("amber", cLoading, "")
	case o.status < 200 || o.status >= 300:
		return v("red", cHTTPError, fmt.Sprint(o.status))
	}
	j := tryJSON(o.body)
	if j == nil {
		return v("red", cNotJSON, "")
	}
	data, ok := j["data"].([]any)
	if !ok {
		data, ok = j["models"].([]any)
	}
	if !ok {
		return v("red", cNotJSON, "")
	}
	if strings.TrimSpace(model) != "" && len(data) > 0 {
		found := false
		for _, e := range data {
			m, _ := e.(map[string]any)
			for _, k := range []string{"id", "model", "name"} {
				if id, ok := m[k].(string); ok {
					if strings.EqualFold(strings.TrimSpace(id), strings.TrimSpace(model)) {
						found = true
					}
					break
				}
			}
		}
		if !found {
			return v("red", cModelNotFound, strings.TrimSpace(model))
		}
	}
	return nil
}

func classifyChat(o outcome, model string) *verdict {
	if o.err != nil {
		return classifyErr(o.err)
	}
	switch {
	case o.status == 401 || o.status == 403:
		return v("red", cKeyRejected, "")
	case o.status == 404:
		if strings.Contains(strings.ToLower(errMessage(o.body)), "model") {
			return v("red", cModelNotFound, model)
		}
		return v("red", cWrongBase, "")
	case o.status == 503:
		return v("amber", cLoading, "")
	case o.status < 200 || o.status >= 300:
		msg := strings.ToLower(errMessage(o.body))
		if o.status == 400 && strings.Contains(msg, "model") && strings.Contains(msg, "not") {
			return v("red", cModelNotFound, model)
		}
		return v("red", cHTTPError, fmt.Sprint(o.status))
	}
	j := tryJSON(o.body)
	choices, _ := j["choices"].([]any)
	if len(choices) == 0 {
		return v("red", cNotJSON, "")
	}
	c0, _ := choices[0].(map[string]any)
	msg, _ := c0["message"].(map[string]any)
	content, _ := msg["content"].(string)
	reasoning, _ := msg["reasoning_content"].(string)
	if reasoning == "" {
		reasoning, _ = msg["reasoning"].(string)
	}
	if strings.TrimSpace(reasoning) != "" || strings.Contains(strings.ToLower(content), "<think") {
		return v("amber", cReasoning, "")
	}
	if o.ms > slowMs {
		return v("amber", cSlow, fmt.Sprint(o.ms))
	}
	return nil
}

func chatBody(model string) string {
	b, _ := json.Marshal(map[string]any{
		"model":                model,
		"messages":             []map[string]string{{"role": "user", "content": "Reply with one word: ready."}},
		"max_tokens":           8,
		"temperature":          0,
		"stream":               false,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
		"reasoning_effort":     "none",
	})
	return string(b)
}

func httpSender(c *http.Client) sender {
	return func(method, u, body, key string, timeoutMs int) outcome {
		var o outcome
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutMs)*time.Millisecond)
		defer cancel()
		var rd io.Reader
		if body != "" {
			rd = bytes.NewReader([]byte(body))
		}
		req, err := http.NewRequestWithContext(ctx, method, u, rd)
		if err != nil {
			o.err = err
			return o
		}
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		t0 := time.Now()
		resp, err := c.Do(req)
		if err != nil {
			o.err = err
			o.ms = time.Since(t0).Milliseconds()
			return o
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		o.status, o.body, o.ms = resp.StatusCode, string(b), time.Since(t0).Milliseconds()
		return o
	}
}

// RunTest runs the whole check against an OpenAI-compatible base URL.
func RunTest(baseInput, model, key string, send sender) TestResult {
	res := func(vd *verdict, base string, ms int64) TestResult {
		light, msg := describe(vd, model)
		return TestResult{Light: light, Message: msg, LatencyMS: ms, Model: model, Code: string(vd.code), EffectiveURL: base}
	}
	if strings.TrimSpace(baseInput) == "" {
		return res(v("red", cNoURL, ""), "", 0)
	}
	b := NormalizeBase(baseInput)
	if b == "" {
		return res(v("red", cBadURL, ""), "", 0)
	}
	fixed := false
	o := send("GET", b+"/models", "", key, modelsTimeoutMs)
	r := classifyModels(o, model)
	if r != nil && r.code == cWrongBase {
		alt := toggleV1(b)
		o2 := send("GET", alt+"/models", "", key, modelsTimeoutMs)
		r2 := classifyModels(o2, model)
		if r2 == nil || r2.code != cWrongBase {
			b, fixed, r = alt, true, r2
		} else {
			h := send("GET", rootOf(b)+"/health", "", key, healthTimeoutMs)
			if h.err == nil && h.status == 503 {
				return res(v("amber", cLoading, ""), b, 0)
			}
			sug := b + "/v1"
			if strings.HasSuffix(strings.ToLower(b), "/v1") {
				sug = rootOf(b)
			}
			return res(v("red", cWrongBase, sug), b, 0)
		}
	}
	if r != nil {
		return res(r, b, 0)
	}
	c := send("POST", b+"/chat/completions", chatBody(model), key, chatTimeoutMs)
	if cr := classifyChat(c, model); cr != nil {
		return res(cr, b, c.ms)
	}
	if fixed {
		return res(v("green", cOKFixedBase, b), b, c.ms)
	}
	return res(v("green", cOK, ""), b, c.ms)
}

func describe(vd *verdict, model string) (string, string) {
	var m string
	switch vd.code {
	case cOK:
		m = "Working: the model answered."
	case cOKFixedBase:
		m = "Working. The address needed a /v1 change: use " + vd.arg
	case cLoading:
		m = "The server is up but the model is still loading. Try again in a moment."
	case cSlow:
		m = "It answered, but slowly (" + vd.arg + " ms). Play will feel laggy."
	case cReasoning:
		m = "It answered, but with reasoning text mixed in. Turn thinking off for this model."
	case cNoURL:
		m = "Enter the endpoint address first."
	case cBadURL:
		m = "That is not a usable http(s) address."
	case cNothingListen:
		m = "Nothing is listening at that address."
	case cKeyRejected:
		m = "The server rejected the API key."
	case cWrongBase:
		m = "The address looks wrong (404). Try " + vd.arg
	case cModelNotFound:
		m = "The server does not know the model " + vd.arg + "."
	case cTLS:
		m = "The secure connection failed (certificate or TLS problem)."
	case cNotJSON:
		m = "The server answered, but not in the OpenAI format."
	case cHTTPError:
		m = "The server answered with HTTP " + vd.arg + "."
	case cTimeout:
		m = "No answer within the time limit."
	default:
		m = "Could not reach the server: " + vd.arg
	}
	return vd.light, m
}
