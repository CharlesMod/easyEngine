package api

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"mime"
	"net/http"
	"path"
	"strconv"
	"strings"

	"github.com/CharlesMod/easyEngine/run"
)

// Resumable streams (llama-server b12651, tools/server/server-stream.cpp): a completion sent with
// an X-Conversation-Id header (matched without regard to case, :578-594) opens a session under
// that id, replacing and cancelling any session of the same id (:235-253, :597-604); GET
// /v1/stream?conv_id= replays it and DELETE /v1/stream?conv_id= cancels it (server.cpp:233-235,
// server-stream.cpp:456-470, :563-575); POST /v1/streams/lookup takes conversation_ids in its body
// and answers each conversation_id (:510-546). Any caller that knows an id could read or cancel
// another's stream, so the engine owns the name space: every id a request gives is rewritten to
// one that carries its owner (the attachment's key; else, for a caller of this machine, the
// commons; else the remote client's own), and the lookup's answer is mapped back. A remote caller
// is refused the stream paths outright (proxy.go). The lookup also matches a stored id that is the
// asked one then "::" (server-stream.cpp:533-536, "<id>::<model>"): only the part before the first
// "::" is owned, the rest is kept, so a lookup still finds an id's variants.

const convHeader = "X-Conversation-Id"

// streamPath is one of llama-server's resumable stream paths.
func streamPath(p string) bool {
	c := path.Clean("/" + p)
	return c == "/v1/stream" || c == "/v1/streams" || strings.HasPrefix(c, "/v1/stream/") || strings.HasPrefix(c, "/v1/streams/")
}

// convOwner is whose name space a request's conversation ids are in: its attachment's, else the
// commons for a caller of this machine, else the remote client's own (never the commons: it could
// replace and so cancel a local stream of the same id).
func convOwner(door run.ClientDoor, r *http.Request) string {
	switch {
	case door.Key != "":
		return door.Key
	case localCaller(r):
		return "commons"
	}
	return "remote:" + identity(r)
}

// convSep is what llama-server's lookup reads after an id ("<id>::<model>").
const convSep = "::"

// convID is a client's conversation id in the engine's name space: owner and the id's part before
// any "::" hashed, the rest kept as sent.
func convID(owner, id string) string {
	base, rest, has := strings.Cut(id, convSep)
	sum := sha256.Sum256([]byte(owner + "\x00" + base))
	out := "ee-" + hex.EncodeToString(sum[:16])
	if has {
		out += convSep + rest
	}
	return out
}

// ownConversations rewrites every conversation id r gives (the header, the stream paths' conv_id,
// the lookup's conversation_ids) into its owner's name space. It answers the lookup's map back
// (engine id -> the client's) when r is a lookup, and false after it has answered the client.
func (a *API) ownConversations(w http.ResponseWriter, r *http.Request, door run.ClientDoor) (map[string]string, bool) {
	owner := convOwner(door, r)
	if vs := r.Header.Values(convHeader); len(vs) > 0 {
		r.Header.Set(convHeader, convID(owner, vs[0]))
	}
	if !streamPath(r.URL.Path) {
		return nil, true
	}
	q := r.URL.Query()
	if vs, ok := q["conv_id"]; ok && len(vs) > 0 {
		q.Set("conv_id", convID(owner, vs[0]))
		r.URL.RawQuery = q.Encode()
	}
	if r.Method != http.MethodPost || path.Clean("/"+r.URL.Path) != "/v1/streams/lookup" {
		return nil, true
	}
	body, _ := io.ReadAll(r.Body) // read before the door (bufferedBody)
	r.Body.Close()
	obj, why := oneObject(body)
	if why != "" {
		badRequest(w, why)
		return nil, false
	}
	var ids []string
	if raw, ok := obj["conversation_ids"]; ok && json.Unmarshal(raw, &ids) != nil {
		badRequest(w, "conversation_ids must be a list of strings")
		return nil, false
	}
	back := map[string]string{} // the owned part -> the client's
	for i, id := range ids {
		ids[i] = convID(owner, id)
		h, _, _ := strings.Cut(ids[i], convSep)
		b, _, _ := strings.Cut(id, convSep)
		back[h] = b
	}
	if ids != nil {
		obj["conversation_ids"], _ = json.Marshal(ids)
	}
	out, err := marshalObject(obj)
	if err != nil {
		badRequest(w, "the request body must be exactly one JSON object")
		return nil, false
	}
	setBody(r, out)
	return back, true
}

// conversationsBack maps the engine's conversation ids in a lookup's JSON answer back to the
// client's; an id the client did not send is left out of its sight.
func conversationsBack(resp *http.Response, back map[string]string) error {
	mt, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mt != "application/json" || resp.Header.Get("Content-Encoding") != "" {
		return nil
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxSamplerBody+1))
	resp.Body.Close()
	if err != nil {
		return err
	}
	if len(b) > maxSamplerBody {
		return replyTooLarge{}
	}
	var v any
	if json.Unmarshal(b, &v) == nil {
		v = mapConv(v, back)
		if out, err := json.Marshal(v); err == nil {
			b = out
		}
	}
	resp.Body = io.NopCloser(bytes.NewReader(b))
	resp.ContentLength = int64(len(b))
	resp.Header.Set("Content-Length", strconv.Itoa(len(b)))
	return nil
}

func mapConv(v any, back map[string]string) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			if s, ok := e.(string); ok && k == "conversation_id" {
				h, rest, has := strings.Cut(s, convSep)
				c, ok := back[h]
				switch {
				case !ok:
					x[k] = "" // not one the client asked for: out of its sight
				case has:
					x[k] = c + convSep + rest
				default:
					x[k] = c
				}
				continue
			}
			x[k] = mapConv(e, back)
		}
	case []any:
		for i := range x {
			x[i] = mapConv(x[i], back)
		}
	}
	return v
}
