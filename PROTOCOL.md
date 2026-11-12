# Protocol

How a host process (Photons, or anything else MIT-licensed) drives this executable.

The point of the protocol is the licence boundary. The host never links this code; it starts
`ansel-denoise-swift` as a child process and reads one JSON object back.

**Two ways to move pixels, and the one to prefer.** Hand the tool a `.dng` and it reads the RAW,
denoises it, and writes a DNG — the host passes two paths and never sees a sample. That is what
`image` does when the extension is `.dng`, and it is why a 60 MP frame now takes 42 s instead of
350 s: 284 s of that was the host shuttling a mosaic through float32 files. The older
float32-mosaic form (`image` with `.f32`, plus `tile`/`batch`) still works and is still the right
call for experiments and benchmarks.

## Handshake

```
$ ansel-denoise-swift capabilities
{"tool":{"name":"ansel-denoise-swift","protocol":2},
 "features":["image","image-dng","tile","batch","bench","info","gpu"],
 "cfaPatterns":["rggb","bggr","grbg","gbrg"],
 "gpu":"Apple M1 Max",
 "inputLayout":"float32-planar-mosaic","outputLayout":"float32-planar-mosaic",
 "imageInputLayouts":["float32-planar-mosaic","dng-cfa"],
 "imageOutputLayouts":["float32-planar-mosaic","dng-cfa"]}
```

Check `protocol` (an integer) **and** the feature you need: `"image"` for the float32 form,
`"image-dng"` for DNG in/out. Treat a mismatch as "the feature is unavailable", not as a crash: the
tool is optional, and an older binary should cost a menu entry, not a launch failure. A host that
only understands protocol 1 keeps working through `inputLayout`/`outputLayout`.

Check the **exit status** as well as the JSON. Failures exit **2** with one line on stderr — never
a Swift backtrace (an uncaught error would `SIGTRAP`, i.e. exit 133, which is indistinguishable
from a crash over a pipe). A binary that cannot create a Metal device also exits non-zero rather
than silently falling back to CPU — a host that asked for `--gpu` must not get a 30× slowdown
without being told.

## Denoise a DNG — `image`, preferred

```
ansel-denoise-swift image MODEL.anselnn shot.dng --gpu --json --out shot-denoised.dng
```

The tool reads the DNG's mosaic, takes black level / white level / CFA phase **from the file**,
normalizes, plans for memory, denoises, de-normalizes, clamps to [0,1] and writes a DNG whose
metadata is the source's. Nothing about the picture's geometry is the host's business any more.

| Flag | Meaning |
|---|---|
| `--out` | Output DNG. **Required**, and must also end in `.dng`. |
| `--a`, `--b` | Per-channel Poisson-Gaussian coefficients, `Var = a·x + b` on [0,1]. Optional here — omitted, the tool uses `0.0004,0.0003,0.0004` / `3e-6,1e-6,1e-6`, the same defaults the host used to send. |
| `--sigma-scale` | Per-channel sigma calibration. Default `2.82,3.94,2.96`. |
| `--strength` | Blend toward the input: `out = in + s·(denoised − in)`. Default 0.85. |
| `--budget-gb` | Memory ceiling override. Omit to derive it from installed RAM (2/3, or 50 % below 12 GB). |
| `--split CxR` | Force a cut pattern. For measuring seams. |
| `--gpu` | Use Metal. Without it, CPU (roughly 30× slower). |
| `--json` | Machine-readable summary on stdout. Progress goes to stderr. |
| `--width`, `--height` | *Optional.* If given they are **checked** against the file; a mismatch is an error, never a silent preference for either. |
| `--cfa` | Override the CFA phase. Experimental — the file is authoritative. |

Response:

```json
{"pieces":1,"columns":1,"rows":1,"seams":0,"singlePass":true,
 "peakBytes":38717685760,"budgetBytes":45208020650,"wholeImageBytes":38717685760,
 "seconds":41.478,"readSeconds":2.247,"inferenceSeconds":38.278,"writeSeconds":0.857,
 "ms":41478.4,"backend":"gpu",
 "inputPath":"shot.dng","outputPath":"shot-denoised.dng",
 "cfa":"rggb","mosaicCfa":"rggb","pattern":"rggb",
 "width":9536,"height":6344,
 "blackLevels":[512,512,512,512],"whiteLevel":16383,
 "blackRepeatRows":2,"blackRepeatCols":2,
 "outputBytes":66788000,"tiles":950,
 "clampedNegatives":29486,"clampedOverOne":0,"finalMin":0,"finalMax":1,
 "sourceBelowBlack":29544,
 "inputLayout":"dng-cfa","outputLayout":"dng-cfa"}
```

The fields a host needs to decide anything:

| Field | Why it is reported |
|---|---|
| `pieces` / `seams` / `singlePass` | `seams == 0` means no grid artefact at all. |
| `seconds` | Tool's own total: read + inference + write. |
| `readSeconds` / `inferenceSeconds` / `writeSeconds` | The split, so a slow run can be attributed. On a 60 MP frame: 2.2 / 38.3 / 0.9 s — i.e. IO is 7 %, not 81 %. |
| `cfa` | Phase as tag 33422 states it. |
| `mosaicCfa` | Phase the network was actually told. Equals `cfa` unless the stored image is larger than its `ActiveArea` (then it is rotated) or `--cfa` overrode it. |
| `clampedNegatives` | Samples the network produced below 0, clamped before quantizing. Non-zero is normal (residual overshoot on clipped blacks). |
| `finalMin` | Minimum of the array handed to the writer. **Always ≥ 0**; a negative here would become a sample below the black level. |
| `sourceBelowBlack` | How many source samples were already below the black level. Tells "the clamp had nothing to do" apart from "the clamp ate signal". |
| `outputBytes` / `tiles` | The DNG that was written. |

### What the tool guarantees about the output DNG

Same width, same height, same `CFAPattern` (`33422`), same `BlackLevel`/`BlackLevelRepeatDim`,
same `WhiteLevel`, same `ActiveArea`/`DefaultCrop*`, same `OpcodeList2`/`3`, same colour matrices,
same XMP, same EXIF/GPS, and the preview bytes copied across **unchanged**. `Compression` is 7
(lossless JPEG) and `PhotometricInterpretation` is 32803 (CFA) — an uncompressed source therefore
comes back compressed, which is the one tag that changes on purpose.

Verified by `scripts/dng-invariants.py`, which decodes both files with its own SOF3 decoder rather
than the tool's — a shared bug in the reader would cancel out of a Swift-only comparison.

### What the tool checks, and refuses

| Input | Result |
|---|---|
| Non-DNG, or a DNG with no CFA image (already demosaiced) | exit 2, `只支持 DNG：…` / `… 里没有 CFA RAW 数据` |
| Truncated file | exit 2, `解码 RAW 失败：文件在 … 处被截断` |
| `--width/--height` disagreeing with the file | exit 2, both numbers in the message |
| `.dng` input with `.f32` output (or the reverse) | exit 2 — a format mismatch is a typo, not an intent |
| Compression other than 1 or 7, or a non-2×2 CFA | exit 2 — guessing a decode would look like a correct picture |

Nothing is written to `--out` on any of these paths.

## Denoise a float32 mosaic — `image`, legacy

```
ansel-denoise-swift image MODEL.anselnn mosaic.f32 \
    --width 9536 --height 6344 \
    --a "0.0004,0.0003,0.0004" --b "3e-6,1e-6,1e-6" \
    --sigma-scale "2.82,3.94,2.96" \
    --cfa rggb --gpu --strength 0.85 --json --out denoised.f32
```

Here the host owns normalization and DNG writing, exactly as before. Same JSON keys, minus the DNG
ones. Input and output are raw `Float32`, little-endian, `width × height` values, row-major.

`--width`/`--height` **need not** be a multiple of 16 — the tool pads internally and crops back.
See "Padding" below.

`seams` is the number of internal cuts that had to be feathered. **Zero means the frame went
through in one pass and carries no grid artefact at all** — the normal case, and the one worth
logging.

### Why one pass when possible

The network denoises differently near an input edge than in its interior: an edge sample has less
surround to reason about. Measured against a single-pass reference on a 1024² crop of real raw,
the mismatch is **21×** larger at 0–4 px from a border than at 24–48 px, and only reaches zero
past ~96 px. Feathering blends those differing answers; it cannot undo them.

One pass is also faster, which was not the expected result: 60 MP took **55.8 s** whole versus
**77.9 s** as 1426 tiles at 256². Small tiles spend about a third of their work on overlap that is
then blended away.

### Memory

0.64 GB per megapixel, measured (1 MP → 0.68 GB, 16.8 MP → 10.6 GB, 60.3 MP → 37.8 GB). The
ceiling is 50% of physical RAM below 12 GB — an 8 GB machine must keep 4 GB for the OS — and 2/3
from 12 GB up.

The split count is computed, not picked from a list: 100 MP at a 4 GB budget needs ~17 pieces,
which no 1/2/4/6/9 list covers, and overlap adds area so the count that fits is always higher than
`area / pieceBudget` suggests.

### Padding

The network halves resolution `depth` times, so both dimensions of a forward pass must be
multiples of `2^depth` (16 for the shipped models) or the bottleneck stops being integral and the
pass traps. Sensors are not so obliging: a Leica SL3 gives 9536×6344, and 6344 = 16·396 + 8.

The pad goes on the **right and bottom only**, and is cropped before the result leaves. Never
centred: a centred pad shifts the mosaic, and every pixel's colour is decided by its index parity,
so a shift swaps R for G — a checkerboard that no tag comparison would catch, because the CFA tag
is copied verbatim either way. One extra `mult` is added on each side even when the dimension
divides exactly: without it the outermost column is the last thing the network sees and its
residual overshoots there (measured on 9536×6344: 10 264 pixels at x ≥ 9523 came out negative).

### Clamping

The residual overshoots on clipped-black pixels — a pixel whose input is exactly 0 (below the
black level) can be predicted to go darker, giving a small negative. Measured on a 60 MP frame:
29 486 samples below 0 out of 60.5 M (0.049 %), against 29 544 that were already below the black
level in the source.

The result is clamped to `[0, 1]` **after** the `strength` blend, because the blend mixes back
toward an input that is not itself guaranteed to be in range (below-black samples normalize
negative). Left unclamped, a negative would denormalize to a sample below the black level, which a
writer would either wrap around to white or refuse. `finalMin` in the JSON report is that array's
minimum and is always ≥ 0.

### What the tool owns, and why

| Decision | Where | Why there |
|---|---|---|
| Whether to split | tool | A function of *this* implementation's memory profile |
| How many pieces | tool | Computed from the budget, accounting for overlap area |
| Where cuts fall | tool | It also feathers them; geometry and window must agree |
| Padding, CFA phase | tool | Needs `depth` and the CFA pattern, both of which it knows |
| `strength` blend, clamping | tool | Pure arithmetic on normalized values |
| Black/white levels, CFA phase | tool **from the DNG file** (DNG form) / host (`.f32` form) | The file is the only authority; a host that passes them separately can pass them wrong |
| Reading and writing the DNG | tool (DNG form) / host (`.f32` form) | The DNG path exists to stop the host doing it |
| Tiling for `tile`/`batch` | host | Those subcommands deliberately do nothing implicit |

## On `--sigma-scale` and `--strength`

Two different knobs, **not** interchangeable:

- `--sigma-scale` tells the network how much noise to expect. Larger → it removes more. This
  re-runs the network.
- `--strength` decides how much of the answer to keep. It does not change the inference at all;
  lowering it puts original noise back along with the texture.

The upstream Ansel default for sigma calibration is `(2·1.41, 2·1.97, 2·1.48)`. Those factors are
sensor/measurement constants, not model properties, so they do **not** change with variant — the
same three numbers apply to quarter, half and large.

## Lower-level subcommands

`tile` (one rectangle) and `batch` (many equal squares) do **not** pad, plan or feather — a caller
using them owns all of that. They exist for benchmarking and for reproducing a specific case.

```
ansel-denoise-swift tile  MODEL in.f32 --width 512 --height 512 --a ... --b ... --gpu --out out.f32
ansel-denoise-swift batch MODEL tiles.f32 --tile 256 --count 1426 --a ... --b ... --gpu --out out.f32
ansel-denoise-swift bench MODEL --compare-cpu --tiles 64,128,256,512
ansel-denoise-swift info  MODEL
```

Sizes passed to `tile`/`batch` **must** be multiples of `2^depth` (16 for the shipped models) or
they trap. `image` is the one that handles arbitrary dimensions.

## Models

Weights are not covered by this repository's GPL: upstream's `LICENSE-DATA.md` §2.2 says the
weights are the trainer's to use, ship or sell. The default variant is committed under `weights/`;
`scripts/fetch-weights.sh` verifies the others against upstream's `manifest.json` sha256.
