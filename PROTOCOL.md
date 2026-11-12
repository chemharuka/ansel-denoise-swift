# Protocol

How a host process (Photons, or anything else MIT-licensed) drives this executable.

The point of the protocol is the licence boundary: the host never links this code. It starts
`ansel-denoise-swift` as a child process, hands it a file of float32 tiles, and reads back a file
of float32 tiles. Everything the network knows about is a buffer of numbers — no image semantics,
no DNG, no CFA beyond a four-letter string. Tiling, feathering, CFA phase and black/white levels
all live in the host.

## Handshake

```
$ ansel-denoise-swift capabilities
{"tool":{"name":"ansel-denoise-swift","protocol":1},
 "features":["tile","batch","bench","info","gpu"],
 "cfaPatterns":["rggb","bggr","grbg","gbrg"],
 "gpu":"Apple M1 Max",
 "inputLayout":"float32-planar-mosaic","outputLayout":"float32-planar-mosaic"}
```

`protocol` is the integer to check. The host should call this once at launch and treat a
mismatch as "the feature is unavailable", not as a crash: the tool is optional.

Check the **exit status** as well as the JSON. A binary that cannot create a Metal device exits
non-zero rather than silently falling back to CPU — a host that requested `--gpu` must not get
a 30× slowdown without being told.

## Denoise a batch

```
ansel-denoise-swift batch MODEL.anselnn tiles.f32 \
    --tile 256 --count 1426 \
    --a "0.0004,0.0003,0.0004" --b "3e-6,1e-6,1e-6" \
    --sigma-scale "2.82,3.94,2.96" \
    --cfa rggb --gpu --json --out denoised.f32
```

| Flag | Meaning |
|---|---|
| `--tile N` | Tile edge. **Must be a multiple of 16** (the net halves resolution 4 times). |
| `--count N` | Number of tiles in the input file. |
| `--a`, `--b` | Per-channel Poisson-Gaussian coefficients, `Var = a·x + b`, `x` normalized to [0,1]. Three comma-separated values, R G B. |
| `--sigma-scale` | Per-channel sigma calibration. Defaults to `2.82,3.94,2.96`. |
| `--cfa` | `rggb` (default), `bggr`, `grbg`, `gbrg`. CFA at the tile's top-left. |
| `--gpu` | Use Metal. Without it, CPU (roughly 30× slower at every tile size). |
| `--json` | Machine-readable summary on stdout. Progress goes to stderr. |
| `--out` | Output path. |

Input and output are both raw `Float32`, little-endian, `count × tile × tile` values in
tile-major order. Nothing else is in the file — no header, no dimensions.

Response on stdout with `--json`:

```json
{"tiles":1426,"tileSize":256,"ms":77900.0,"msPerTile":54.7,
 "output":"denoised.f32","cfa":"rggb","backend":"gpu"}
```

## On `--sigma-scale` and strength

Two different knobs, and they are **not** interchangeable:

- `--sigma-scale` tells the network how much noise to expect. Larger → the network removes more.
  This re-runs the network.
- `strength` is a **host-side** blend, `out = original + s·(denoised − original)`, default 0.85 in
  the Photons client. It does not appear in this CLI at all: it is applied by the host after
  reassembly, in the same normalized domain as the input, so it cannot disturb black/white
  levels. Lowering it puts original noise back along with the texture.

The upstream Ansel default for sigma calibration is `(2·1.41, 2·1.97, 2·1.48)`. Those two
factors are sensor/measurement constants, not model properties, so they do **not** change with
variant — the same three numbers apply to quarter, half and large.

## What the protocol deliberately does not do

- **No tiling.** The host decides tile size, overlap and edge handling. The tool processes
  exactly the tiles it is given.
- **No blending.** The host reassembles. A single-tile test will never reveal a seam bug.
- **No normalization.** Values go in and come out on the same scale; black/white levels are the
  host's business.
- **No model download.** Weights are fetched separately and verified against upstream's
  `manifest.json` sha256.
