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

// yieldedReply answers while the card is yielded to another program: 503, a string code
// "yielded" (clients tell it from "loading" by that), and a longer Retry-After.
func yieldedReply(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "10")
	w.WriteHeader(503)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": "yielded", "type": "unavailable_error", "message": msg}})
}

// proxy forwards every non-/engine path to llama-server (local) or, in external
// mode, /v1/* to the player's endpoint with its key.
func (a *API) proxy(w http.ResponseWriter, r *http.Request) {
	done := a.S.Begin() // count first, so a resize cannot slip in between Route and the forward
	defer done()
	id := requestID(r)
	w.Header().Set(RequestIDHeader, id)
	rt := a.S.Route()
	switch rt.Kind {
	case "unavailable":
		if rt.Code == "yielded" {
			yieldedReply(w, rt.Message)
			return
		}
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
		if !a.applySampler(w, r) {
			return
		}
		rest := strings.TrimPrefix(r.URL.Path, "/v1")
		consumed := bearerConsumed(r.Context())
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
				} else if consumed {
					pr.Out.Header.Del("Authorization") // the engine's own token is not for the upstream
				}
			},
			ModifyResponse: dropCORS,
			ErrorHandler:   badGateway,
		}
		forward(w, r, rp, a.fwd(true, id))
	default:
		target, err := url.Parse(rt.URL)
		if err != nil {
			unavailable(w, 502, "bad_gateway", "bad server URL")
			return
		}
		if !a.applySampler(w, r) {
			return
		}
		rp := &httputil.ReverseProxy{
			Transport:     transport,
			FlushInterval: -1, // flush every write: SSE tokens must not sit in a buffer
			Rewrite: func(pr *httputil.ProxyRequest) {
				pr.SetURL(target)
				pr.Out.Host = target.Host
				pr.Out.Header.Del("Authorization") // a client's credentials never go to the local model server
			},
			ModifyResponse: dropCORS,
			ErrorHandler:   badGateway,
		}
		forward(w, r, rp, a.fwd(false, id))
	}
}

// heardWriter tells heard on every Write: the model server's reply bytes are flowing, so the
// watchdog for silence on the requests in flight is put back. It stays a Flusher (a stream must not
// sit in a buffer), and Unwrap lets http.ResponseController reach the real writer (flush, hijack).
type heardWriter struct {
	http.ResponseWriter
	heard func()
}

func (h *heardWriter) Write(b []byte) (int, error) {
	h.heard()
	return h.ResponseWriter.Write(b)
}

func (h *heardWriter) Flush() { http.NewResponseController(h.ResponseWriter).Flush() }

func (h *heardWriter) Unwrap() http.ResponseWriter { return h.ResponseWriter }

// hearing wraps w so each reply write counts as the model server being heard from.
func (a *API) hearing(w http.ResponseWriter) http.ResponseWriter {
	heard := a.heard
	if heard == nil {
		heard = a.S.Heard
	}
	return &heardWriter{ResponseWriter: w, heard: heard}
}

// dropCORS removes every Access-Control-* header the upstream sent: the model server reflects
// Origin, and the engine allows no cross-site use at all (the gate refuses it).
func dropCORS(resp *http.Response) error {
	for k := range resp.Header {
		if strings.HasPrefix(strings.ToLower(k), "access-control-") {
			delete(resp.Header, k)
		}
	}
	return nil
}

func badGateway(w http.ResponseWriter, r *http.Request, err error) {
	if r.Context().Err() != nil {
		return // the client went away
	}
	unavailable(w, 502, "bad_gateway", fmt.Sprintf("upstream: %v", err))
}
