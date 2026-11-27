module github.com/CharlesMod/easyEngine

go 1.27

require (
	github.com/ebitengine/purego v0.11.3
	github.com/godbus/dbus/v5 v5.2.4
	golang.org/x/net v0.35.0
	golang.org/x/sys v0.30.0
)

// The sizing core is a module of its own in core/ (docs/ENGINE.md section 13). A checkout builds
// it from that directory; another program requires it by its tag, core/vX.Y.Z.
require github.com/CharlesMod/easyEngine/core v0.1.4

replace github.com/CharlesMod/easyEngine/core => ./core
