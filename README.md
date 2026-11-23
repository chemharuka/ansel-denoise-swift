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
| `--strength S` | `out = in + S·(denoised − in)`, in the normalized domain. Default 0.85. `auto` derives it from the noise curve. |
| `--a`, `--b` | Per-channel Poisson-Gaussian noise model, `Var = a·x + b` on [0,1]. Per-camera values. |
| `--sigma-scale` | Per-channel sigma calibration. Default `2.82,3.94,2.96`; `1,1,1` with `--estimate-noise`. |
| `--estimate-noise` | Measure the curve from this file instead of trusting `--a`/`--b`. |
| `--taper q` | Give up `q` of the blend weight towards the highlights (see below). |
| `--taper-knee lo,hi` | Where that taper starts and ends, in linear RAW brightness. Default `0.05,0.25`. |
| `--shadow-taper q` | The mirror at the floor; default knees `0.002,0.02`. |
| `--look nlmf` | The whole conditioning Photons' nonlocal denoiser uses: measured curve, `auto` strength, both tapers. |
| `--dump-field P` | Write `P.i.f32` / `P.d.f32` / `P.field.json` — the normalized input, the final field, and what produced them. |
| `--dual-raw` | Keep both raws in one file: preview in IFD0, `SubIFDs = [original, denoised]`. Roughly 1.7× the file size. |

The JSON summary carries what a host needs to decide anything: `seconds` split into
`readSeconds`/`inferenceSeconds`/`writeSeconds`, `tiles`, `pieces`/`seams`/`singlePass`,
`clampedNegatives`, `finalMin`, plus the CFA pattern and levels it used.

`--dual-raw` writes one DNG containing two full-resolution CFA raws, in the shape a camera DNG
uses: **IFD0 is the preview page**, and `330 SubIFDs` lists the **original** un-denoised mosaic
(bit-identical to the source's samples) first and the **denoised** mosaic second, tagged
`270 ImageDescription = "denoised alternate: ansel-denoise strength=…"`.

The order matters and was measured, not assumed: `CIRAWFilter` resolves a multi-raw DNG by scanning
`SubIFDs` and taking the **first** entry whose `NewSubfileType` is 0 — IFD0 is only consulted when
there is none. So the original has to be `SubIFDs[0]`, and a file that put it in IFD0 would hand the
*denoised* layer to the system decoder. As written, `CIRAWFilter` renders the output identically to
the input (verified pixel for pixel), and promoting `SubIFDs[1]` renders exactly the denoised
picture.

To surface the denoised layer, follow `330`, or *promote* it (point `SubIFDs[0]` at the alternate and
set the count to 1); the alternate repeats the level/CFA/crop tags so a promoted copy is a
self-consistent raw. Profile, colour-matrix and opcode tags are deliberately **not** repeated — a
second copy makes ImageIO apply them twice and changes the rendering of the whole file. Off by
default; without the flag the output is byte-for-byte what earlier releases wrote.

`noise` measures a file without denoising it: the same decode, normalization and estimator
`image` uses, so the sigma it prints is the sigma a run would condition on.

```
ansel-denoise-swift noise shot.dng            # plane-by-plane a/b, sigma(0.05), derived strength
ansel-denoise-swift noise shot.dng --json
```

Other subcommands — `info`, `tile`, `batch`, `bench`, `capabilities`, `models` — exist for
experiments, benchmarking and host negotiation. Hosts should use `image`.

## Conditioning: the measured curve, and the blend that goes with it

`--a/--b` default to one camera's numbers. On a different sensor, or the same one at a different
ISO, they are wrong in a way nothing reports. The curve is a property of the file, so
`--estimate-noise` measures it: diagonal differences inside each CFA plane, MAD → sigma, 15th
percentile per brightness bin, `Var = a·x + b` by least squares, intercept doubled to compensate for
the tone curve's shadow lift. That estimator is the same one `nonlocal-matchfilter-swift` (Photons'
other denoise engine) uses, so both condition on the same sigma field for the same frame.

A measured curve is already in the mosaic domain, so the `2.82/3.94/2.96` profile calibration must
**not** be applied on top — that factor exists to lift profile values, which are fitted on
demosaiced pixels. `--look nlmf` sets `--sigma-scale 1,1,1` for this reason.

`--strength auto` maps green sigma at 5 % brightness onto 0.70–0.85 (log-linear, clamped). The
blend is then reduced per pixel towards the ends of the tonal range:

```
s(x) = s · (1 − q_hi·smoothstep(x, hiLo, hiHi)) · (1 − q_lo·(1 − smoothstep(x, loLo, loHi)))
```

`--look nlmf` uses `q_hi = 0.18` over 0.05–0.25 and `q_lo = 0.50` below 0.002 — the settings
Photons' nonlocal path was measured with. With no taper the factor is exactly 1, so the old
frame-wide blend is reproduced bit for bit.

**Measured parity** (Leica SL3 60 MP, quarter model, paired 64×64 blocks of the green plane,
high-frequency retention = MAD of first differences out/in):

| block brightness | ansel `--look nlmf` | nonlocal | ratio |
|---|---|---|---|
| 0.000–0.002 | 0.579 | 0.583 | 0.99 |
| 0.002–0.005 | 0.565 | 0.571 | 0.99 |
| 0.005–0.010 | 0.468 | 0.478 | 0.98 |
| 0.010–0.020 | 0.367 | 0.383 | 0.96 |
| 0.020–0.040 | 0.378 | 0.417 | 0.91 |
| 0.040–0.080 | 0.520 | 0.521 | 1.00 |
| 0.080–0.160 | 0.498 | 0.514 | 0.97 |
| 0.160–0.320 | 0.564 | 0.546 | 1.03 |

That is ISO 3200 (`L1030546`), the regime this module is used in: every bin within 9 %, most within
4 %, at the same blend weight and the same sigma. On a base-ISO frame the two engines still differ in
the bright bins — the ansel network has less gain there, so it keeps 20–30 % more high-frequency
content at full strength — and that difference is texture, not grain. Matching it would need a
per-brightness strength *inversion* of the network, which is a different feature, not a taper.

## Photons side

- Client: `Host/Photons/AnselDenoiseClient.swift` — **MIT**, because it only spawns the process
  and parses JSON. It is copied into Photons as
  `Sources/PhotonsExtensions/Denoise/AnselDenoiseClient.swift`; edit it here first and sync.
  The host passes two paths and one policy — `--look nlmf` by default, or `.classic` with pinned
  coefficients — and the tool owns the file, the tiling, the noise model and the blend.
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
