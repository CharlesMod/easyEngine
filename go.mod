module hermesengine

go 1.27

require golang.org/x/sys v0.30.0

// The sizing core is a module of its own in core/ (docs/ENGINE.md section 13). A checkout builds
// it from that directory; another program requires it by its tag, core/vX.Y.Z.
require github.com/CharlesMod/easyEngine/core v0.1.0

replace github.com/CharlesMod/easyEngine/core => ./core
