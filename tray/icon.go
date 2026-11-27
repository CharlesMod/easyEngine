package tray

import _ "embed"

// The default icon, drawn by tray/internal/icongen (go run ./tray/internal/icongen).
var (
	//go:embed icons/tray.png
	iconPNG []byte
	//go:embed icons/tray.ico
	iconICO []byte
	//go:embed icons/template.png
	iconTemplate []byte
)

//go:generate go run ./internal/icongen -out icons

// DefaultIcon is the engine's neutral mark in every form the systems want.
func DefaultIcon() Icon { return Icon{PNG: iconPNG, ICO: iconICO, Template: iconTemplate} }
