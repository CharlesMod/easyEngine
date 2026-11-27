#!/bin/sh
# Cross-builds easy-engine for the three desktop targets into engine/dist/.
# Pure Go, no cgo: one toolchain builds all of them. Usage: ./build.sh [version]
set -eu
cd "$(dirname "$0")"
GO="${GO:-$(command -v go || echo "$HOME/.local/go/bin/go")}"
VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
GO="$GO" ./core-check.sh # the sizing core must still build at Go 1.23 on its own (ENGINE.md section 13)
OUT=dist
rm -rf "$OUT"
mkdir -p "$OUT"
for t in linux/amd64 windows/amd64 darwin/arm64 darwin/amd64; do
  os=${t%/*}; arch=${t#*/}
  ext=""; [ "$os" = windows ] && ext=.exe
  dir="$OUT/$os-$arch"; mkdir -p "$dir"
  echo "build $dir/easy-engine$ext"
  CGO_ENABLED=0 GOOS=$os GOARCH=$arch "$GO" build -trimpath -ldflags "-s -w -X main.version=$VERSION" -o "$dir/easy-engine$ext" ./cmd/easy-engine
done
# Layout is what the game mods expect: dist/<os>-<arch>/easy-engine[.exe] (ENGINE.md section 10).
( cd "$OUT" && sha256sum */easy-engine* > SHA256SUMS )
echo "done: $OUT/"
