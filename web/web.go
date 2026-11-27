// Package web embeds the setup page and the chat window page.
package web

import _ "embed"

//go:embed index.html
var Index []byte

// Chat is the page at /chat/: a same-origin frame holding llama-server's web UI.
//
//go:embed chat.html
var Chat []byte

// Icon is the tab icon of the chat page, served at /chat/icon.png. It is the same picture as the
// tray's (tray/icons/tray.png); a test in package app keeps the two equal.
//
//go:embed icon.png
var Icon []byte
