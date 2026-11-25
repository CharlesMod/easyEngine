package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"
)

var transport = &http.Transport{
	Proxy:                 nil,
	MaxIdleConnsPerHost:   16,
	IdleConnTimeout:       60 * time.Second,
	ResponseHeaderTimeout: 0, // prefill of a long prompt can take minutes
}

func unavailable(w http.ResponseWriter, code int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	if code == 503 {
		w.Header().Set("Retry-After", "2")
	}
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": code, "type": typ, "message": msg}})
}

// proxy forwards every non-/engine path to llama-server (local) or, in external
// mode, /v1/* to the player's endpoint with its key.
func (a *API) proxy(w http.ResponseWriter, r *http.Request) {
	done := a.S.Begin() // count first, so a resize cannot slip in between Route and the forward
	defer done()
	rt := a.S.Route()
	switch rt.Kind {
	case "unavailable":
		unavailable(w, 503, "unavailable_error", rt.Message)
		return
	case "external":
		if rt.URL == "" {
			unavailable(w, 503, "unavailable_error", "No external endpoint configured")
			return
		}
		if r.URL.Path != "/v1" && !strings.HasPrefix(r.URL.Path, "/v1/") {
			unavailable(w, 404, "not_found_error", "Only /v1 is forwarded in external mode")
			return
		}
		base, err := url.Parse(rt.URL)
		if err != nil {
			unavailable(w, 502, "bad_gateway", "bad external URL")
			return
		}
		rest := strings.TrimPrefix(r.URL.Path, "/v1")
		rp := &httputil.ReverseProxy{
			Transport:     transport,
			FlushInterval: -1,
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.Out.URL.Scheme, pr.Out.URL.Host = base.Scheme, base.Host
				pr.Out.URL.Path = strings.TrimRight(base.Path, "/") + rest
				pr.Out.URL.RawPath = ""
				pr.Out.Host = base.Host
				if rt.Key != "" {
					pr.Out.Header.Set("Authorization", "Bearer "+rt.Key)
				}
			},
			ErrorHandler: badGateway,
		}
		rp.ServeHTTP(w, r)
	default:
		target, err := url.Parse(rt.URL)
		if err != nil {
			unavailable(w, 502, "bad_gateway", "bad server URL")
			return
		}
		rp := &httputil.ReverseProxy{
			Transport:     transport,
			FlushInterval: -1, // flush every write: SSE tokens must not sit in a buffer
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.SetURL(target)
				pr.Out.Host = target.Host
			},
			ErrorHandler: badGateway,
		}
		rp.ServeHTTP(w, r)
	}
}

func badGateway(w http.ResponseWriter, r *http.Request, err error) {
	if r.Context().Err() != nil {
		return // the client went away
	}
	unavailable(w, 502, "bad_gateway", fmt.Sprintf("upstream: %v", err))
}
