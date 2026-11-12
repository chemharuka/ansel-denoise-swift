#!/usr/bin/env bash
# Acceptance run for the DNG-in / DNG-out path (§389).
#
#   scripts/acceptance.sh [--skip-reverse]
#
# Prints a PASS/FAIL line per gate and the numbers behind it. Every gate that matters is run from
# the built binary, not from Swift test code, so this is the same path a host takes.
#
# The reverse gates (§3.3) mutate the source in place, rebuild, re-run, and restore. They are the
# reason the other gates can be trusted: a test that cannot fail is not a test, so each one shows
# the failure it is supposed to catch, and then shows the gate green again once restored.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
MODEL=${ANSEL_DENOISE_MODEL:-weights/denoise-quarter-single-v1.anselnn}
BIN=.build/debug/ansel-denoise-swift
FIX=Tests/Fixtures/small-rggb.dng
OUT=local/out
SKIP_REVERSE=0
[ "${1:-}" = "--skip-reverse" ] && SKIP_REVERSE=1

mkdir -p "$OUT"
PASS=0
FAIL=0

gate() {   # gate <id> <description> <exit status>
    if [ "$3" -eq 0 ]; then
        printf 'GATE %-3s PASS  %s\n' "$1" "$2"; PASS=$((PASS + 1))
    else
        printf 'GATE %-3s FAIL  %s\n' "$1" "$2"; FAIL=$((FAIL + 1))
    fi
}

build() {
    swift build 2>&1 | grep -E '^.*error:' && return 1
    return 0
}

echo "== build =="
build || { echo "build failed"; exit 1; }
echo "binary: $($BIN capabilities | tr -d '\n')"
echo

echo "== fixtures =="
python3 scripts/make-dng.py "$FIX" "$OUT/crop1024x1000.dng" --crop 1024x1000 >/dev/null
python3 scripts/make-dng.py "$FIX" "$OUT/bggr.dng" --cfa bggr >/dev/null
python3 scripts/make-dng.py "$FIX" "$OUT/phases.dng" --flat 2000 --phase-gains 1,1.1,1.2,1.3 >/dev/null
# A flat dark field with noise matched to the tool's own coefficients, for the energy gate.
python3 scripts/make-dng.py "$FIX" "$OUT/flatdark.dng" --crop 256x256 --flat 1463 --noise 1450 --seed 424242 >/dev/null
ls -l "$OUT"/*.dng | awk '{print "  " $5 "  " $9}'
echo

echo "== ① image in.dng --out out.dng --json =="
START=$(python3 -c 'import time; print(time.time())')
"$BIN" image "$MODEL" "$FIX" --gpu --json --out "$OUT/out.dng" > "$OUT/a1.json" 2> "$OUT/a1.err"
RC=$?
END=$(python3 -c 'import time; print(time.time())')
ELAPSED=$(python3 -c "print(f'{$END - $START:.2f}')")
head -3 "$OUT/a1.err"
python3 - "$OUT/a1.json" "$ELAPSED" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
keys = ['pieces','columns','rows','seams','singlePass','seconds','readSeconds','inferenceSeconds',
        'writeSeconds','backend','inputPath','outputPath','cfa','width','height','outputBytes',
        'clampedNegatives','clampedOverOne','finalMin','finalMax','sourceBelowBlack','peakBytes']
print('  json: ' + '  '.join(f'{k}={d[k]}' for k in keys))
print(f'  wall clock from the shell: {sys.argv[2]} s '
      f'(tool self-reports {d["seconds"]:.2f} s; read {d["readSeconds"]:.2f} + '
      f'infer {d["inferenceSeconds"]:.2f} + write {d["writeSeconds"]:.2f})')
# New fields the spec asks for must be there and sane.
missing = [k for k in ('inputPath','outputPath','readSeconds','writeSeconds','seconds') if k not in d]
if missing:
    raise SystemExit('missing JSON fields: ' + ', '.join(missing))
if not (d['readSeconds'] > 0 and d['writeSeconds'] > 0):
    raise SystemExit('readSeconds/writeSeconds must be positive')
PY
[ $? -eq 0 ] && [ $RC -eq 0 ]; gate ① "image model in.dng --out out.dng --json" $?
echo "  tiffdump IFD0 (first lines):"
tiffdump "$OUT/out.dng" 2>&1 | head -9 | sed 's/^/    /'
python3 - "$OUT/out.dng" <<'PY'
import struct, sys
d = open(sys.argv[1], 'rb').read()
assert d[:2] == b'II' and struct.unpack('<H', d[2:4])[0] == 42, 'not a TIFF'
n = struct.unpack('<H', d[8:10])[0]
assert n > 20, f'IFD0 has only {n} entries'
print(f'    IFD0: {n} entries, {len(d)} bytes — a legal DNG container')
PY
echo

echo "== ② geometry invariants (source vs output) =="
python3 scripts/dng-invariants.py "$FIX" "$OUT/out.dng"; gate ② "tags, CFA, levels, compression, photometric" $?
echo

echo "== ②/⑩ CFA phase is read from the file (bggr source) =="
"$BIN" image "$MODEL" "$OUT/bggr.dng" --gpu --json --out "$OUT/bggr-out.dng" > "$OUT/a2.json" 2>/dev/null
python3 - "$OUT/a2.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'  cfa={d["cfa"]} mosaicCfa={d["mosaicCfa"]} (source tag says bggr)')
raise SystemExit(0 if d['cfa'] == 'bggr' and d['mosaicCfa'] == 'bggr' else 1)
PY
gate ⑩a "bggr source is denoised as bggr" $?
python3 scripts/dng-invariants.py "$OUT/bggr.dng" "$OUT/bggr-out.dng" --phases \
    | tail -6 | sed 's/^/  /'
echo

echo "== ③ energy falls on a flat dark field =="
"$BIN" image "$MODEL" "$OUT/flatdark.dng" --gpu --json --out "$OUT/flatdark-out.dng" > "$OUT/a3.json" 2>/dev/null
python3 scripts/dng-invariants.py "$OUT/flatdark.dng" "$OUT/flatdark-out.dng" --flat 0,0,256 > "$OUT/a3.txt"; RC=$?
tail -6 "$OUT/a3.txt" | sed 's/^/  /'
gate ③ "adjacent-difference variance ratio < 0.5, |mean change| <= 5 %" $RC
echo

echo "== ⑤ 1024×1000 (not a multiple of 16) =="
"$BIN" image "$MODEL" "$OUT/crop1024x1000.dng" --gpu --json --out "$OUT/crop-out.dng" > "$OUT/a5.json" 2>/dev/null
python3 - "$OUT/a5.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'  tool reports {d["width"]}×{d["height"]}, {d["pieces"]} piece(s), limits 1024×1000')
raise SystemExit(0 if (d['width'], d['height']) == (1024, 1000) else 1)
PY
RC1=$?
python3 scripts/dng-invariants.py "$OUT/crop1024x1000.dng" "$OUT/crop-out.dng" > "$OUT/a5.txt"; RC2=$?
grep -E 'width|height|RESULT' "$OUT/a5.txt" | sed 's/^/  /'
[ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; gate ⑤ "1024×1000 round trip" $?
echo

echo "== ④ nothing negative reaches the writer =="
python3 - "$OUT/a1.json" "$OUT/crop.json" <<'PY'
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    print(f'  {p}: finalMin={d["finalMin"]} finalMax={d["finalMax"]} '
          f'clampedNegatives={d["clampedNegatives"]} sourceBelowBlack={d["sourceBelowBlack"]}')
    assert d['finalMin'] >= 0, f'{p}: a negative sample reached the writer'
PY
gate ④a "finalMin >= 0 in the CLI report" $?
# The Swift test is the real gate: it asserts the clamp had work to do *and* that the minimum is 0.
swift test --filter DNGPipelineTests/testWriterNeverSeesANegativeSample 2>&1 \
    | tee "$OUT/a4.log" | grep -E '④|Executed' | sed 's/^/  /'
grep -q 'Executed 1 test, with 0 failures' "$OUT/a4.log"; gate ④ "clamp asserted at the write boundary" $?
echo

echo "== ⑥ the float32 / tile / batch path still works =="
swift test 2>&1 | tee "$OUT/a6.log" | grep -E "Test Suite '[A-Za-z]+' (started|passed|failed)|Executed [0-9]+ test" | sed 's/^/  /'
echo "  grep -c 'passed (' = $(grep -c 'passed (' "$OUT/a6.log")"
echo "  grep -c 'failed (' = $(grep -c 'failed (' "$OUT/a6.log")"
grep -q "Executed 0 tests" "$OUT/a6.log" && { echo "  Executed 0 tests — the suite did not run"; RC=1; } || RC=0
FAILED_COUNT=$(grep -c 'failed (' "$OUT/a6.log")
[ $RC -eq 0 ] && [ "$FAILED_COUNT" -eq 0 ] && \
    grep -q "Test Suite 'DNGPipelineTests' started" "$OUT/a6.log" && \
    grep -q "Test Suite 'AnselDenoiseTests' started" "$OUT/a6.log" && \
    grep -q "Test Suite 'DenoisePlannerTests' started" "$OUT/a6.log"; gate ⑥ "all suites green" $?
echo

if [ $SKIP_REVERSE -eq 0 ] && [ -x scripts/reverse-gates.sh ]; then
    echo "== reverse gates (§3.3) =="
    scripts/reverse-gates.sh | sed 's/^/  /'
    RG=${PIPESTATUS[0]}
    gate '⑩⑪⑫⑬' "each mutation turns its gate red, and restoring turns it green" $RG
fi

echo
echo "== summary =="
echo "gates passed: $PASS   failed: $FAIL"
[ $FAIL -eq 0 ] || exit 1
