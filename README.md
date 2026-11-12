# ansel-denoise-swift

A Swift + Metal rewrite of [aurelienpierreeng/ansel-denoise](https://github.com/aurelienpierreeng/ansel-denoise),
the network behind Ansel's `rawdenoiseai` module. It denoises a camera RAW **in the CFA domain**
and writes it back as a DNG.

**DNG in, DNG out.** The tool opens the RAW itself, denoises the mosaic and writes a new DNG. The
output is still a Bayer mosaic — never demosaiced — with the source file's own black level, white
level, CFA phase, geometry and metadata.

Used by [Photons](https://github.com/chemharuka/photons) as an external plugin: a child process,
never linked into the MIT-licensed app.

- Repository: <https://github.com/chemharuka/ansel-denoise-swift>
- License: **GPL-3.0-or-later** (`LICENSE`). ansel-denoise is © Aurélien Pierre; this port keeps
  the same license. `LICENSE-SCOPE.md` explains the process-boundary arrangement.

## What it does

| Input | Output |
|---|---|
| `shot.dng` — CFA DNG (photometric 32803, compression 1 or 7) | `shot-denoised.dng` — same size, same CFA pattern, same black/white levels, denoised mosaic |

Nothing else changes: preview bytes, EXIF/GPS, colour matrices and the crop rectangle are copied
across. Only the samples are touched, and the one tag that changes on purpose is `Compression`
(an uncompressed source comes back as lossless JPEG, tag 7).

Inputs it refuses instead of guessing: non-DNG files, DNGs with no CFA plane (already demosaiced),
truncated files, unknown compression, non-2×2 CFA. Such a file would otherwise come back as a
plausible-looking wrong picture. Exit status is 2 and nothing is written.

## Usage

```
ansel-denoise-swift image MODEL.anselnn shot.dng --gpu --json --out shot-denoised.dng
```

That is the whole host-facing path: one DNG in, one DNG out, one JSON object on stdout.

| Flag | Meaning |
|---|---|
| `--out PATH` | Output DNG. Required. |
| `--gpu` | Run on Metal (≈30× faster than CPU). |
| `--json` | One JSON summary on stdout; progress goes to stderr. |
| `--strength S` | `out = in + S·(denoised − in)`, in the normalized domain. Default 0.85. |
| `--a`, `--b` | Per-channel Poisson-Gaussian noise model, `Var = a·x + b` on [0,1]. Per-camera values. |
| `--sigma-scale` | Per-channel sigma calibration. Default `2.82,3.94,2.96`. |

The JSON summary carries what a host needs to decide anything: `seconds` split into
`readSeconds`/`inferenceSeconds`/`writeSeconds`, `tiles`, `pieces`/`seams`/`singlePass`,
`clampedNegatives`, `finalMin`, plus the CFA pattern and levels it used.

Other subcommands — `info`, `tile`, `batch`, `bench`, `capabilities`, `models` — exist for
experiments, benchmarking and host negotiation. Hosts should use `image`.

## Photons side

- Client: `Host/Photons/AnselDenoiseClient.swift` — **MIT**, because it only spawns the process
  and parses JSON. It is copied into Photons as
  `Sources/PhotonsExtensions/Denoise/AnselDenoiseClient.swift`; edit it here first and sync.
  Tiling, feathering and the strength blend are product logic and live on the Photons side, not here.
- Protocol: `PROTOCOL.md` (JSON fields, exit codes, what the output DNG is guaranteed to preserve).
- Install: `./scripts/install-to-photons.sh` (→ `/opt/homebrew/bin`), or `--in-app` for
  `Photons.app/Contents/Resources/bin`.
- In the app the feature is **right-click ▸ 模型降噪…** on a single DNG. The result lands beside the
  original as `<name>-denoised.dng` and is stacked with it. Missing tool or model greys the row out
  with the reason; nothing else in the app changes.

`Package.swift` makes this a standalone SPM package. Photons' `Vendor/ansel-denoise-swift/` is a mirror of
this repository, kept only so the app builds from one tree — it is never a SwiftPM dependency and
its copy of the client must stay byte-identical to `Host/`.

## Weights

`denoise-quarter-single-v1.anselnn` (1.9 MB, the default variant) is committed. The other five are
fetched and sha256-verified by `scripts/fetch-weights.sh`. Only `single` is implemented — the
`multi` variants need superpixel binning.

The weights are **not GPL**: upstream's data license places no restriction on the resulting weights.
See `weights/README.md`.

The default `quarter` model is the right one for most use: it is 17× faster than `large` and loses
only a few points of noise reduction, and it degrades far more gracefully when a camera's sigma is
estimated badly. Compare model sizes only after sigma calibration is right — with a mis-scaled sigma
the large model looks *worse*, because it follows the wrong instruction more faithfully.

## Build

```
swift build -c release
.build/release/ansel-denoise-swift capabilities
```

macOS 14+, no third-party dependencies. `swift test` runs the unit and DNG-pipeline suites;
`scripts/acceptance.sh` runs the gates against the built binary the way a host would.

## Notes

- `PORTING.md` — what had to be correct before the output matched the reference: sigma calibration,
  channels-last Metal kernels, the constant-input probe that catches silent GPU bugs.
- Model: 4-level UNet (`base=8`, 475 K parameters). Input is 5 planes at full mosaic resolution —
  `[mosaic, R one-hot, G one-hot, B one-hot, σ]` — and the network predicts a noise field, so
  `out = in − noise`. CFA layout and noise level are inputs, not baked-in parameters, which is why
  one weight set serves every profiled camera and ISO without retraining.
