# Protocol

How a host process (Photons, or anything else MIT-licensed) drives this executable.

The point of the protocol is the licence boundary. The host never links this code; it starts
`ansel-denoise-swift` as a child process, hands it a file of float32 samples, and reads back a
file of float32 samples. Nothing of this program ends up inside the host's address space.

Prefer the **`image`** subcommand. Use the lower-level ones only for experiments.

## Handshake

```
$ ansel-denoise-swift capabilities
{"tool":{"name":"ansel-denoise-swift","protocol":1},
 "features":["image","tile","batch","bench","info","gpu"],
 "cfaPatterns":["rggb","bggr","grbg","gbrg"],
 "gpu":"Apple M1 Max",
 "inputLayout":"float32-planar-mosaic","outputLayout":"float32-planar-mosaic"}
```

Check `protocol` (an integer) **and** the presence of `"image"` in `features`. Treat a mismatch
as "the feature is unavailable", not as a crash: the tool is optional, and an older binary should
cost a menu entry, not a launch failure.

Check the **exit status** as well as the JSON. A binary that cannot create a Metal device exits
non-zero rather than silently falling back to CPU — a host that asked for `--gpu` must not get a
30× slowdown without being told.

## Denoise a whole image — `image`

```
ansel-denoise-swift image MODEL.anselnn mosaic.f32 \
    --width 9536 --height 6344 \
    --a "0.0004,0.0003,0.0004" --b "3e-6,1e-6,1e-6" \
    --sigma-scale "2.82,3.94,2.96" \
    --cfa rggb --gpu --strength 0.85 --json --out denoised.f32
```

| Flag | Meaning |
|---|---|
| `--width`, `--height` | Mosaic dimensions. **Need not be a multiple of 16** — the tool pads internally and crops the result. |
| `--a`, `--b` | Per-channel Poisson-Gaussian coefficients, `Var = a·x + b`, `x` normalized to [0,1]. Three comma-separated values, R G B. |
| `--sigma-scale` | Per-channel sigma calibration. Defaults to `2.82,3.94,2.96`. |
| `--cfa` | `rggb` (default), `bggr`, `grbg`, `gbrg`. CFA at the top-left sample. |
| `--strength` | Blend toward the input: `out = in + s·(denoised − in)`, `s` in [0,1]. Default 0.85. |
| `--budget-gb` | Memory ceiling override, in GB. Omit to derive it from installed RAM. |
| `--split CxR` | Force a cut pattern (e.g. `2x2`) instead of planning. For measuring seams. |
| `--gpu` | Use Metal. Without it, CPU (roughly 30× slower). |
| `--json` | Machine-readable summary on stdout. Progress goes to stderr. |
| `--out` | Output path. |

Input and output are both raw `Float32`, little-endian, `width × height` values, row-major — no
header, no dimensions, same shape in and out.

Response:

```json
{"pieces":1,"columns":1,"rows":1,"seams":0,"singlePass":true,
 "peakBytes":38717685760,"budgetBytes":45812984490,"wholeImageBytes":38717685760,
 "ms":55789.9,"backend":"gpu","cfa":"rggb","output":"denoised.f32"}
```

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

### What the tool owns, and why

| Decision | Where | Why there |
|---|---|---|
| Whether to split | tool | A function of *this* implementation's memory profile |
| How many pieces | tool | Computed from the budget, accounting for overlap area |
| Where cuts fall | tool | It also feathers them; geometry and window must agree |
| Padding, CFA phase | tool | Needs `depth` and the CFA pattern, both of which it knows |
| `strength` blend | tool | Pure arithmetic on normalized values |
| Black/white levels | **host** | The tool never sees them; values in and out share one scale |
| DNG writing | **host** | The tool knows nothing about DNG |

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
