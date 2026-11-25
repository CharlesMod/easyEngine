// Package web embeds the setup page.
package web

import _ "embed"

//go:embed index.html
var Index []byte
