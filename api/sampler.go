package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"

	"hermesengine/catalog"
	"hermesengine/run"
)

// SamplerHeader names the sampler profile a request wants (rp | structured | official, or a
// name the game's profile adds). Missing or unknown means rp.
const SamplerHeader = "X-Hermes-Sampler"

// AppliedHeader is set on every proxied response: "<profile>;<mode>" when the request was
// subject to the policy, "none;<reason>" when it passed through untouched.
const AppliedHeader = "X-Hermes-Sampler-Applied"

const maxSamplerBody = 64 << 20

// completionPath is true for the endpoints that sample tokens.
func completionPath(p string) bool {
	switch p {
	case "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/completion":
		return true
	}
	return false
}

// RewriteSampler applies a policy to a JSON request body. It returns the new body, or the
// original when nothing changed. Only the sampler fields (catalog.SamplerFields) are touched;
// every other field keeps its exact bytes.
func RewriteSampler(body []byte, mode string, prof catalog.SamplerProfile) ([]byte, bool) {
	if mode == run.PolicyOff {
		return body, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil || obj == nil {
		return body, false
	}
	vals := prof.Fields()
	switch mode {
	case run.PolicyFill:
		for k, v := range vals {
			if _, ok := obj[k]; !ok {
				b, _ := json.Marshal(v)
				obj[k] = b
			}
		}
	default: // enforce
		for _, k := range catalog.SamplerFields {
			delete(obj, k)
		}
		for k, v := range vals {
			b, _ := json.Marshal(v)
			obj[k] = b
		}
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return body, false
	}
	return out, true
}

// applySampler runs the policy over r (a completion request) and sets the Applied header on w.
// It must run before the request is forwarded.
func (a *API) applySampler(w http.ResponseWriter, r *http.Request) {
	asked := strings.TrimSpace(r.Header.Get(SamplerHeader))
	name := strings.ToLower(asked)
	r.Header.Del(SamplerHeader) // the upstream (possibly the player's own server) does not need it
	info := a.S.Sampler()
	set := func(v string) { w.Header().Set(AppliedHeader, v) }
	if r.Method != http.MethodPost || !completionPath(r.URL.Path) {
		set("none;not-a-completion")
		return
	}
	if info.Policy == run.PolicyOff {
		set("none;off")
		return
	}
	if !info.Active {
		set("none;" + info.Policy + ";unknown-model")
		return
	}
	prof, ok := info.Profiles[name]
	if name == "" {
		name = catalog.ProfileRP
		prof, ok = info.Profiles[name]
	} else if !ok {
		a.S.Logf("sampler: unknown profile %q in %s, using rp", asked, SamplerHeader)
		name = catalog.ProfileRP
		prof, ok = info.Profiles[name]
	}
	if !ok { // no rp: fall back to official
		name = catalog.ProfileOfficial
		prof, ok = info.Profiles[name]
	}
	if !ok {
		set("none;" + info.Policy + ";no-profile")
		return
	}
	set(name + ";" + info.Policy)
	if r.Body == nil {
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxSamplerBody))
	r.Body.Close()
	if err != nil {
		r.Body = io.NopCloser(bytes.NewReader(nil))
		return // the upstream will report the broken request
	}
	if out, changed := RewriteSampler(body, info.Policy, prof); changed {
		body = out
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))
	r.TransferEncoding = nil
}
