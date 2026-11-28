package api

import (
	"net/http"
	"path"
	"strings"
)

// What each llama-server path does with slots, read from the source of the pinned build
// (b12651, tools/server/server.cpp server_register_routes and server-context.cpp; the table is
// in docs/ENGINE.md section 7g). The door confines a request to its client's slots by this.
type pathKind int

const (
	// pathFree touches no slot (health, props, models, tokenize, the web UI, GET /slots).
	pathFree pathKind = iota
	// pathSlot runs on a slot and honours the body's id_slot (handle_completions_impl).
	pathSlot
	// pathSlotID is POST /slots/{id}?action=...: the id is in the path.
	pathSlotID
	// pathUnconfinable runs on a slot the server picks itself, or acts on every slot, or is a
	// multipart body the door cannot rewrite: it cannot be held to one client's slots.
	pathUnconfinable
)

// slotPaths honour id_slot: the JSON body goes through handle_completions_impl.
var slotPaths = map[string]bool{
	"/completion": true, "/completions": true, "/v1/completions": true,
	"/chat/completions": true, "/v1/chat/completions": true,
	"/v1/responses": true, "/responses": true,
	"/v1/messages": true, "/infill": true,
}

// unconfinablePaths take a slot the server picks (embeddings, rerank, speech, realtime,
// systemone), change every slot (lora adapters, props), act on another request by its id
// (control), read a multipart body (transcriptions), or dispatch to any of these (predict).
var unconfinablePaths = map[string]bool{
	"/embedding": true, "/embeddings": true, "/v1/embeddings": true,
	"/rerank": true, "/reranking": true, "/v1/rerank": true, "/v1/reranking": true,
	"/tts": true, "/v1/audio/speech": true, "/audio/speech": true,
	"/v1/audio/transcriptions": true, "/audio/transcriptions": true,
	"/v1/realtime": true, "/v1/systemone": true, "/v1/chat/completions/control": true,
	"/lora-adapters": true, "/props": true, "/predict": true,
}

// freePosts are POST paths that run no slot task.
var freePosts = map[string]bool{
	"/tokenize": true, "/detokenize": true, "/apply-template": true,
	"/chat/completions/input_tokens": true, "/v1/chat/completions/input_tokens": true,
	"/responses/input_tokens": true, "/v1/responses/input_tokens": true, "/v1/messages/count_tokens": true,
}

// classify is what a request's path does with slots. A GET reads (the realtime socket is the
// one GET that takes a slot); a POST, PUT, PATCH or DELETE the table does not know is taken as
// slot-touching and unconfinable: a new path in a newer build is held, not trusted.
func classify(method, p string) pathKind {
	p = path.Clean("/" + p)
	switch {
	case strings.HasPrefix(p, "/slots/") && method != http.MethodGet && method != http.MethodHead:
		return pathSlotID
	case method == http.MethodGet || method == http.MethodHead:
		if p == "/v1/realtime" {
			return pathUnconfinable
		}
		return pathFree
	case slotPaths[p]:
		return pathSlot
	case freePosts[p]:
		return pathFree
	case unconfinablePaths[p]:
		return pathUnconfinable
	}
	return pathUnconfinable
}
