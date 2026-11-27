// Package web embeds the setup page and the chat window page.
package web

import _ "embed"

//go:embed index.html
var Index []byte

// Chat is the page at /chat/: a slim bar and a same-origin frame holding llama-server's web UI.
//
//go:embed chat.html
var Chat []byte
