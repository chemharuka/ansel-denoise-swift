#!/usr/bin/env bash
# Reverse gates (§3.3): prove each acceptance check can actually fail.
#
# For each mutation: apply it, rebuild, run the gate that is supposed to catch it, expect red,
# restore, rebuild, re-run, expect green. Both sides of every number are printed, because "the gate
# went red" is only meaningful next to "and it goes green again when the bug is removed" — otherwise
# a gate that fails for an unrelated reason would look like proof.
#
# Each mutation is a text substitution held in an associative array below, applied with `patch`-style
# exact matching. If the text is not found the script aborts rather than reporting a vacuous pass:
# a mutation that silently did nothing would make the gate look green and prove the opposite.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=.build/debug/ansel-denoise-swift
MODEL=${ANSEL_DENOISE_MODEL:-weights/denoise-quarter-single-v1.anselnn}
FIX=Tests/Fixtures/small-rggb.dng
OUT=local/out
mkdir -p "$OUT"

FAILS=0
note() { printf '%s\n' "$*"; }
verdict() {   # verdict <label> <expected: red|green> <status>
    local want=$1 got
    [ "$2" -eq 0 ] && got=green || got=red
    if [ "$got" = "$1" ]; then
        note "    ok    expected $1, got $got — $3"
    else
        note "    BAD   expected $1, got $got — $3"
        FAILS=$((FAILS + 1))
    fi
}

apply_mutation() {   # apply_mutation <file> <old> <new> <label>
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import sys
path, old, new, label = sys.argv[1:5]
src = open(path).read()
if src.count(old) != 1:
    raise SystemExit(f'{label}: expected exactly one occurrence of the text to mutate, '
                     f'found {src.count(old)} — refusing to report a vacuous result')
open(path, 'w').write(src.replace(old, new))
PY
}

restore() { cp "$1.bak" "$1"; }

# ---------------------------------------------------------------- ⑩ hard-coded CFA
echo "⑩ CFA phase hard-coded to rggb (does the differential test catch it?)"
F=Sources/AnselDenoise/DNG/DenoiseDNGPipeline.swift
cp "$F" "$F.bak"

echo "  -- before: real code"
swift test --filter DNGPipelineTests/testCFATagChangesTheActualPixels 2>&1 \
    | grep -E '⑩|Executed' | sed 's/^/     /'
swift test --filter DNGPipelineTests/testCFATagChangesTheActualPixels 2>&1 \
    | grep -q 'Executed 1 test, with 0 failures'
verdict green $? "⑩-differential green before the mutation"

"$BIN" image "$MODEL" "$OUT/bggr.dng" --gpu --json --out "$OUT/r10-before.dng" > "$OUT/r10-before.json" 2>/dev/null
python3 - "$OUT/r10-before.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'     cfa={d["cfa"]} mosaicCfa={d["mosaicCfa"]} (must be bggr)')
raise SystemExit(0 if d['mosaicCfa'] == 'bggr' else 1)
PY
verdict green $? "② report keeps the file's phase"

apply_mutation "$F" \
    '        let mosaicCFA = request.cfaOverride ?? cfa.pattern' \
    '        let mosaicCFA = request.cfaOverride ?? "rggb"' \
    "⑩" || { note "  mutation could not be applied"; exit 1; }
swift build 2>&1 | grep -q 'error:' && { note "  build failed after mutation"; restore "$F"; exit 1; }

echo "  -- after: CFA hard-coded rggb"
"$BIN" image "$MODEL" "$OUT/bggr.dng" --gpu --json --out "$OUT/r10-after.dng" > "$OUT/r10-after.json" 2>/dev/null
python3 - "$OUT/r10-after.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f'     cfa={d["cfa"]} mosaicCfa={d["mosaicCfa"]} — the file says bggr, '
      f'the network was told {d["mosaicCfa"]}')
raise SystemExit(0 if d['mosaicCfa'] == 'bggr' else 1)
PY
verdict red $? "② report now contradicts the file"

swift test --filter DNGPipelineTests/testCFATagChangesTheActualPixels 2>&1 \
    | tee "$OUT/r10-after.log" | grep -E '⑩|XCTAssert|Executed' | sed 's/^/     /'
grep -q 'Executed 1 test, with 0 failures' "$OUT/r10-after.log"
verdict red $? "⑩ differential must go red: identical pixels, identical output"

# And the tag-only check still passes — which is exactly why the differential test has to exist.
python3 scripts/dng-invariants.py "$OUT/bggr.dng" "$OUT/r10-after.dng" > "$OUT/r10-tags.txt"
verdict green $? "tag-only check passes anyway (metadata is copied verbatim)"

restore "$F"; swift build 2>&1 | grep -q 'error:' && { note "  restore build failed"; exit 1; }
swift test --filter DNGPipelineTests/testCFATagChangesTheActualPixels 2>&1 \
    | grep -q 'Executed 1 test, with 0 failures'
verdict green $? "restored: differential green again"
echo

# ---------------------------------------------------------------- ⑪ no clamp
echo "⑪ [0,1] clamp removed at the write boundary (does ④ catch it?)"
F=Sources/AnselDenoise/AnselDenoiseImage.swift
cp "$F" "$F.bak"

swift test --filter DNGPipelineTests/testWriterNeverSeesANegativeSample 2>&1 | grep -E '④|Executed' | sed 's/^/     /'
swift test --filter DNGPipelineTests/testWriterNeverSeesANegativeSample 2>&1 \
    | grep -q 'Executed 1 test, with 0 failures'
verdict green $? "④ green before the mutation"

# Remove the clamp: keep the counters so the test's "the clamp had work to do" assertion still
# fires on the real quantity, but stop clamping the values.
apply_mutation "$F" \
    '            if v < 0 { negatives += 1; result[i] = 0 }
            else if v > 1 { overs += 1; result[i] = 1 }' \
    '            if v < 0 { negatives += 1 }
            else if v > 1 { overs += 1 }' \
    "⑪" || { note "  mutation could not be applied"; exit 1; }
swift build 2>&1 | grep -q 'error:' && { note "  build failed after mutation"; restore "$F"; exit 1; }

swift test --filter DNGPipelineTests/testWriterNeverSeesANegativeSample 2>&1 \
    | tee "$OUT/r11-after.log" | grep -E '④|XCTAssert|Executed' | sed 's/^/     /'
grep -q 'Executed 1 test, with 0 failures' "$OUT/r11-after.log"
verdict red $? "④ must go red"

restore "$F"; swift build 2>&1 | grep -q 'error:' && { note "  restore build failed"; exit 1; }
swift test --filter DNGPipelineTests/testWriterNeverSeesANegativeSample 2>&1 \
    | grep -q 'Executed 1 test, with 0 failures'
verdict green $? "restored: ④ green again"
echo

# ---------------------------------------------------------------- ⑫ centred padding
echo "⑫ right/bottom padding replaced by centred padding (does ②/⑤ catch it?)"
F=Sources/AnselDenoise/AnselDenoiseImage.swift
cp "$F" "$F.bak"

swift test --filter DNGPipelineTests/testCFAPhaseIsPreservedThroughPaddingAndCrop 2>&1 \
    | grep -E 'phase means|Executed' | sed 's/^/     /'
swift test --filter DNGPipelineTests/testCFAPhaseIsPreservedThroughPaddingAndCrop 2>&1 \
    | grep -q 'Executed 1 test, with 0 failures'
verdict green $? "phase test green before the mutation"

# Centre the pad: shift the piece in by half the padding before the forward pass, and shift the
# crop back by the same amount afterwards. This is the "obvious" fix that moves the mosaic.
apply_mutation "$F" \
    '                var input = buf
                if padW > 0 || padH > 0 {
                    var padded = [Float](repeating: 0, count: rw * rh)
                    for y in 0..<rh {
                        let sy = min(y, piece.sourceHeight - 1)      // replicate the last row
                        for x in 0..<rw {
                            let sx = min(x, piece.sourceWidth - 1)   // replicate the last column
                            padded[y * rw + x] = buf[sy * piece.sourceWidth + sx]
                        }
                    }
                    input = padded
                }' \
    '                var input = buf
                if padW > 0 || padH > 0 {
                    var padded = [Float](repeating: 0, count: rw * rh)
                    let centreY = padH / 2, centreX = padW / 2
                    for y in 0..<rh {
                        let sy = min(max(y - centreY, 0), piece.sourceHeight - 1)
                        for x in 0..<rw {
                            let sx = min(max(x - centreX, 0), piece.sourceWidth - 1)
                            padded[y * rw + x] = buf[sy * piece.sourceWidth + sx]
                        }
                    }
                    input = padded
                }' \
    "⑫" || { note "  mutation could not be applied"; exit 1; }
swift build 2>&1 | grep -q 'error:' && { note "  build failed after mutation"; restore "$F"; exit 1; }

swift test --filter DNGPipelineTests/testCFAPhaseIsPreservedThroughPaddingAndCrop 2>&1 \
    | tee "$OUT/r12-after.log" | grep -E 'phase means|XCTAssert|Executed' | sed 's/^/     /'
grep -q 'Executed 1 test, with 0 failures' "$OUT/r12-after.log"
verdict red $? "②/⑤ must go red"

restore "$F"; swift build 2>&1 | grep -q 'error:' && { note "  restore build failed"; exit 1; }
swift test --filter DNGPipelineTests/testCFAPhaseIsPreservedThroughPaddingAndCrop 2>&1 \
    | grep -q 'Executed 1 test, with 0 failures'
verdict green $? "restored: phase test green again"
echo

# ---------------------------------------------------------------- ⑬ skip the denoise
echo "⑬ skip inference, write the source mosaic back (does ③ catch it?)"
F=Sources/AnselDenoise/DNG/DenoiseDNGPipeline.swift
cp "$F" "$F.bak"

echo "  -- before: real code"
"$BIN" image "$MODEL" "$OUT/flatdark.dng" --gpu --out "$OUT/r13-before.dng" >/dev/null 2>&1
python3 scripts/dng-invariants.py "$OUT/flatdark.dng" "$OUT/r13-before.dng" --flat 0,0,256 \
    > "$OUT/r13-before.txt"; RC=$?
grep -E 'adjacent-difference|mean:|ratio < 0.5' "$OUT/r13-before.txt" | sed 's/^/     /'
verdict green $RC "③ green before the mutation"

apply_mutation "$F" \
    '        let denoised = try DenoiseImageRunner.denoiseMosaic(mosaic, width: width, height: height,
                                                            request: imageRequest,
                                                            model: request.model,
                                                            metal: request.metal,
                                                            progress: progress)' \
    '        let denoised = try DenoiseImageRunner.denoiseMosaic(mosaic.map { min(max($0, 0), 1) },
                                                            width: width, height: height,
                                                            request: DenoiseImageRequest(
                                                                input: request.input, output: request.output,
                                                                width: width, height: height,
                                                                a: request.a, b: request.b,
                                                                cfa: request.cfaOverride ?? cfa.pattern,
                                                                useGPU: false, strength: 0,
                                                                budgetBytes: budget),
                                                            model: request.model,
                                                            metal: request.metal,
                                                            progress: progress)' \
    "⑬" || { note "  mutation could not be applied"; exit 1; }
swift build 2>&1 | grep -q 'error:' && { note "  build failed after mutation"; restore "$F"; exit 1; }

"$BIN" image "$MODEL" "$OUT/flatdark.dng" --gpu --out "$OUT/r13-after.dng" >/dev/null 2>&1
python3 scripts/dng-invariants.py "$OUT/flatdark.dng" "$OUT/r13-after.dng" --flat 0,0,256 \
    > "$OUT/r13-after.txt"; RC=$?
grep -E 'adjacent-difference|mean:|ratio < 0.5' "$OUT/r13-after.txt" | sed 's/^/     /'
verdict red $RC "③ must go red — the output is the input"

restore "$F"; swift build 2>&1 | grep -q 'error:' && { note "  restore build failed"; exit 1; }
"$BIN" image "$MODEL" "$OUT/flatdark.dng" --gpu --out "$OUT/r13-restored.dng" >/dev/null 2>&1
python3 scripts/dng-invariants.py "$OUT/flatdark.dng" "$OUT/r13-restored.dng" --flat 0,0,256 \
    > "$OUT/r13-restored.txt"; RC=$?
grep -E 'adjacent-difference|ratio < 0.5' "$OUT/r13-restored.txt" | sed 's/^/     /'
verdict green $RC "restored: ③ green again"
echo

rm -f Sources/AnselDenoise/DNG/*.bak Sources/AnselDenoise/*.bak
echo "reverse-gate failures: $FAILS"
[ $FAILS -eq 0 ] || exit 1
