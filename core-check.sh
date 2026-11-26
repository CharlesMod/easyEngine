#!/bin/sh
# Checks the sizing core (core/) as another program sees it: its own module, go 1.23, no
# requirement, built, vetted and tested for the three desktop systems. The root's `go test ./...`
# never reaches core/ (a nested module is not part of the root's ./...), so this is its gate;
# build.sh runs it first. It never fetches a toolchain (GOTOOLCHAIN=local).
# vet's stdversion check reports any standard-library call newer than core/go.mod's go line, so a
# newer toolchain still refuses what Go 1.23 could not build; the core's own TestCoreStaysPure
# fails if that line is raised or a require is added. Usage: ./core-check.sh
set -eu
cd "$(dirname "$0")/core"
GO="${GO:-$(command -v go || echo "$HOME/.local/go/bin/go")}"
export GOTOOLCHAIN=local CGO_ENABLED=0
if [ "$(sed -n 's/^go //p' go.mod)" != 1.23 ]; then
  echo "core-check: core/go.mod must say go 1.23" >&2
  exit 1
fi
"$GO" test -count=1 ./...
for os in linux windows darwin; do
  echo "core-check: $os"
  GOOS=$os "$GO" build ./...
  GOOS=$os "$GO" vet ./...
done
echo "core-check: ok"
