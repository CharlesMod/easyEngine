package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"path"
	"strings"

	"github.com/CharlesMod/easyEngine/run"
)

// Media from a remote caller (docs/ENGINE.md section 7g, "Media from remote callers"): a caller
// from another machine may not put an image, audio, video or file into a request. llama-server
// fetches URLs it finds in a media part (any host the private-address check lets by) and decodes
// data: URIs and raw base64 in several shapes; the engine keeps all of it off the wire for remote
// callers rather than judge each. A local caller's body passes untouched.

// mediaPaths are the routes whose whole purpose is media: refused to a remote caller by path,
// body unread (multipart audio upload, websocket realtime audio).
func mediaPath(p string) bool {
	switch path.Clean("/" + p) {
	case "/v1/audio/transcriptions", "/audio/transcriptions", "/v1/realtime", "/realtime":
		return true
	}
	return false
}

// mediaRefusal reads what a remote request says before its body: the path, the content type, the
// content encoding. A reason, or "" for a request whose body must still be scanned.
func mediaRefusal(r *http.Request) string {
	if mediaPath(r.URL.Path) {
		return "this path takes audio"
	}
	if r.Method != http.MethodPost {
		return ""
	}
	ct := r.Header.Get("Content-Type")
	if mt, _, err := mime.ParseMediaType(ct); (err == nil && strings.HasPrefix(mt, "multipart/")) ||
		strings.Contains(strings.ToLower(ct), "multipart") {
		return "a multipart body carries files"
	}
	// llama-server's HTTP layer may inflate a compressed body: the engine could not read it
	for _, ce := range r.Header.Values("Content-Encoding") {
		if e := strings.ToLower(strings.TrimSpace(ce)); e != "" && e != "identity" {
			return "a compressed body cannot be read for media"
		}
	}
	return ""
}

// mediaAuditTag is the llama.cpp build whose source the media audit (docs/ENGINE.md section 7h,
// "Media from remote callers", the table of ways media enters a request) was done against.
// TestMediaAuditMatchesPin fails when the catalog's pinned build moves off it.
const mediaAuditTag = "b12651-easy1"

// textOnlyChatPath is a chat route a paired token may call: the body of a remote caller there must
// pass the allow list of textOnlyChat as well as the deny scan.
func textOnlyChatPath(p string) bool {
	switch path.Clean("/" + p) {
	case "/v1/chat/completions", "/chat/completions":
		return true
	}
	return false
}

// textOnlyChat is the allow list for a remote caller's chat body: every element of the top-level
// "messages" array is an object whose "content" is a string, null, absent, or an array of parts that
// each carry exactly "type":"text" and "text":<string> (strict: no other key, cache_control too). A key
// repeated in one of the objects read here is refused, since llama-server's parser may take the other
// copy. A body that is not one JSON object, or nests too deep, is left to the existing handling.
// Returns a reason, or "" when the body passes.
func textOnlyChat(body []byte) string {
	const only = "only text content parts are offered to callers from other machines"
	if len(bytes.TrimSpace(body)) == 0 || jsonDepth(body) > maxJSONDepth {
		return ""
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	root, err := readNode(dec)
	if err != nil || !root.obj {
		return ""
	}
	if _, err := dec.Token(); err != io.EOF {
		return ""
	}
	var msgs *mediaNode
	for i, k := range root.keys {
		if k == "messages" {
			if msgs != nil {
				return only + ": \"messages\" is given twice"
			}
			msgs = &root.items[i]
		}
	}
	if msgs == nil {
		return ""
	}
	if !isArrayNode(*msgs) {
		return only + ": \"messages\" is not an array"
	}
	for _, m := range msgs.items {
		if !m.obj {
			return only + ": a message is not an object"
		}
		var content *mediaNode
		for i, k := range m.keys {
			if k == "content" {
				if content != nil {
					return only + ": \"content\" is given twice"
				}
				content = &m.items[i]
			}
		}
		if content == nil || content.null || content.str != nil {
			continue
		}
		if !isArrayNode(*content) {
			return only + ": \"content\" is not a string or an array of parts"
		}
		for _, p := range content.items {
			if !p.obj || len(p.keys) != 2 {
				return only + ": a content part is not exactly a type and a text"
			}
			var typ, text *mediaNode
			for i, k := range p.keys {
				switch k {
				case "type":
					typ = &p.items[i]
				case "text":
					text = &p.items[i]
				}
			}
			if typ == nil || text == nil || typ.str == nil || *typ.str != "text" || text.str == nil {
				return only + ": a content part is not exactly a type and a text"
			}
		}
	}
	return ""
}

// isArrayNode tells an array node from a scalar or an object (a scalar has no items either).
func isArrayNode(n mediaNode) bool { return n.arr }

func mediaRefused(reason string) *run.ClientError {
	return &run.ClientError{Status: 400, Code: "images_not_offered",
		Msg: "images, audio, video and files are not offered to callers from other machines (" + reason + ")"}
}

// mediaTypes are the "type" strings of a content part (or an Anthropic/Responses block) that
// carries media or names a file.
var mediaTypes = map[string]bool{
	"image_url": true, "input_image": true, "input_audio": true, "input_video": true, "video_url": true,
	"image": true, "audio": true, "video": true, "file": true, "input_file": true, "media_marker": true,
	"document": true,
}

// mediaKeys are the keys under which a request hands llama-server media: a non-null value is refused.
var mediaKeys = map[string]bool{
	"image_url": true, "input_audio": true, "input_video": true, "video_url": true, "multimodal_data": true,
	"image_data": true, "image": true, "images": true, "audio": true, "video": true, "files": true,
}

// mediaSkipKeys are subtrees of tool and schema definitions: text the template renders, where a
// property may be called "image". Only a media marker or a media data URI still counts inside.
var mediaSkipKeys = map[string]bool{
	"tools": true, "functions": true, "response_format": true, "json_schema": true, "grammar": true,
}

// mediaNode is one JSON value as the scanner reads it: strings and nulls are told apart from
// other scalars, and an object keeps every pair in order (a repeated key too).
type mediaNode struct {
	str   *string
	null  bool
	items []mediaNode // array elements, or object values
	keys  []string    // object keys, parallel to items
	obj   bool
	arr   bool
}

func readNode(dec *json.Decoder) (mediaNode, error) {
	tok, err := dec.Token()
	if err != nil {
		return mediaNode{}, err
	}
	switch v := tok.(type) {
	case string:
		return mediaNode{str: &v}, nil
	case nil:
		return mediaNode{null: true}, nil
	case json.Delim:
		n := mediaNode{obj: v == '{', arr: v == '['}
		for dec.More() {
			if n.obj {
				kt, err := dec.Token()
				if err != nil {
					return n, err
				}
				k, _ := kt.(string)
				n.keys = append(n.keys, k)
			}
			c, err := readNode(dec)
			if err != nil {
				return n, err
			}
			n.items = append(n.items, c)
		}
		_, err := dec.Token() // the closing delimiter
		return n, err
	}
	return mediaNode{}, nil
}

// mediaIn reports whether a request body carries media, with a short reason. A body that is not
// one JSON value, or nests deeper than maxJSONDepth, is left to the existing handling (llama-server
// refuses it), except that a media marker anywhere in its bytes still counts.
func mediaIn(body []byte) (bool, string) {
	if len(bytes.TrimSpace(body)) == 0 {
		return false, ""
	}
	if jsonDepth(body) <= maxJSONDepth {
		dec := json.NewDecoder(bytes.NewReader(body))
		root, err := readNode(dec)
		if err == nil {
			if _, err = dec.Token(); err != io.EOF {
				err = errors.New("trailing data")
			} else {
				err = nil
			}
		}
		if err == nil {
			if r := scanNode(root, false); r != "" {
				return true, r
			}
			return false, ""
		}
	}
	if bytes.Contains(body, []byte("<__media")) {
		return true, "a media marker"
	}
	return false, ""
}

func scanText(s string) string {
	if strings.Contains(s, "<__media") {
		return "a media marker"
	}
	t := strings.TrimLeft(s, " \t\r\n")
	for _, p := range []string{"data:image/", "data:audio/", "data:video/"} {
		if len(t) >= len(p) && strings.EqualFold(t[:len(p)], p) {
			return "a " + p[5:len(p)-1] + " data URI"
		}
	}
	return ""
}

func scanNode(n mediaNode, skip bool) string {
	if n.str != nil {
		return scanText(*n.str)
	}
	if !n.obj {
		for _, c := range n.items {
			if r := scanNode(c, skip); r != "" {
				return r
			}
		}
		return ""
	}
	for i, c := range n.items {
		key := n.keys[i]
		if r := scanText(key); r != "" {
			return r
		}
		lk := strings.ToLower(strings.TrimSpace(key))
		if !skip {
			if mediaKeys[lk] && !c.null {
				return "a \"" + lk + "\" field"
			}
			if lk == "type" && c.str != nil {
				if t := strings.ToLower(strings.TrimSpace(*c.str)); mediaTypes[t] {
					return "a \"" + t + "\" part"
				}
			}
		}
		if r := scanNode(c, skip || mediaSkipKeys[lk]); r != "" {
			return r
		}
	}
	return ""
}
