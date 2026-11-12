#!/bin/sh
# Fetch model weights from upstream, verifying every file against manifest.json.
#
#   ./scripts/fetch-weights.sh                 # all six variants
#   ./scripts/fetch-weights.sh quarter-single  # just one
#   ./scripts/fetch-weights.sh --list          # names + sizes, no download
#
# Already-present files with a matching sha256 are skipped, so this is safe to
# re-run. A file whose hash does not match is deleted rather than kept: a
# corrupted weight file loads without error and quietly produces bad pixels.
#
# The default variant (quarter-single) is committed to the repo, so most users
# never need this script. The weights are not GPL — see weights/README.md — and
# they are not covered by the training-data licence either (LICENSE-DATA.md §2.2:
# the weights are unrestricted).
set -e

URL="https://raw.githubusercontent.com/aurelienpierreeng/ansel-denoise/master/models"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/weights"
MANIFEST="$DEST/manifest.json"

mkdir -p "$DEST"

# Always refresh the manifest: it is the authority for what "current" means, and an
# upstream revision bump changes the hash so a stale local file gets replaced.
curl -fsSL -o "$MANIFEST.tmp" "$URL/manifest.json"
mv "$MANIFEST.tmp" "$MANIFEST"

# Names and hashes out of the manifest, one "name sha256" per line. python3 because
# jq is not guaranteed to exist on a stock macOS.
parse() {
    python3 - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["models"]
for name, e in m.items():
    print(name, e["sha256"])
PY
}

if [ "$1" = "--list" ]; then
    parse | while read -r name sha; do
        printf '%-38s %s\n' "$name" "$(echo "$sha" | cut -c1-16)"
    done
    exit 0
fi

# Filter to the requested subset, if any.
filter() {
    if [ -z "$1" ]; then cat; else
        parse | grep -E "$1" || { echo "no variant matches '$1'" >&2; exit 1; }
    fi
}

parse | if [ -n "$1" ]; then grep -E "$1" || { echo "no variant matches '$1'" >&2; exit 1; }; else cat; fi |
while read -r name sha; do
    out="$DEST/$name"
    if [ -f "$out" ]; then
        have=$(shasum -a 256 "$out" | awk '{print $1}')
        if [ "$have" = "$sha" ]; then
            echo "ok       $name"
            continue
        fi
        echo "stale    $name (hash mismatch, re-downloading)"
        rm -f "$out"
    else
        echo "fetch    $name"
    fi
    curl -fsSL -o "$out" "$URL/$name"
    got=$(shasum -a 256 "$out" | awk '{print $1}')
    if [ "$got" != "$sha" ]; then
        echo "FAILED   $name — sha256 $got, expected $sha" >&2
        rm -f "$out"
        exit 1
    fi
    echo "verified $name"
done

echo
echo "weights in $DEST:"
ls -lh "$DEST"/*.anselnn 2>/dev/null | awk '{printf "  %-40s %s\n", $9, $5}'
