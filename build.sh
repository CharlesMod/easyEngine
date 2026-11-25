#!/bin/sh
# Cross-builds hermes-engine for the three desktop targets into engine/dist/.
# Pure Go, no cgo: one toolchain builds all of them. Usage: ./build.sh [version]
set -eu
cd "$(dirname "$0")"
GO="${GO:-go}"
VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
OUT=dist
rm -rf "$OUT"
mkdir -p "$OUT"
for t in linux/amd64 windows/amd64 darwin/arm64 darwin/amd64; do
  os=${t%/*}; arch=${t#*/}
  ext=""; [ "$os" = windows ] && ext=.exe
  name="hermes-engine-$os-$arch$ext"
  echo "build $name"
  CGO_ENABLED=0 GOOS=$os GOARCH=$arch "$GO" build -trimpath -ldflags "-s -w -X main.version=$VERSION" -o "$OUT/$name" ./cmd/hermes-engine
done
( cd "$OUT" && sha256sum hermes-engine-* > SHA256SUMS )
echo "done: $OUT/"
