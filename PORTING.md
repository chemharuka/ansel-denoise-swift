# Porting notes

What it took to turn the upstream PyTorch model into Swift + Metal, and the mistakes worth not
repeating. Everything here was found by a failing test or a measured number, not by reading code.

## The model, in one paragraph

`denoise-quarter-single-v1.anselnn` is a 4-level UNet, `base=8`, 475,265 parameters, 1.9 MB.
Input is **5 planes at full mosaic resolution**: `[mosaic, R one-hot, G one-hot, B one-hot, σ]`.
Each layer is `[conv3x3 → GELU]×2`, downsampling is 2×2 stride 2, the decoder does 2× nearest
upsample → 1×1 conv → concat skip → `[conv3x3 → GELU]×2`. No normalization layers anywhere —
deliberately, because instance norm would make tiled output depend on tiling.

Output is a predicted *noise* field; the result is `out = in − noise`.

## Bugs that produced wrong pixels without failing

### 1. `conv` ignored `stride`

The reference numpy implementation read stride from the tensor layout and the Swift one did not.
Result: "down" produced 32→29 instead of 32→16. Shapes still looked plausible at first glance
because the decoder's `precondition` was later. Fix: explicit `stride` parameter and
`(Hp - kh) / stride + 1`.

### 2. Metal helper kernels used plane indexing on a channels-last buffer

The convolution kernels were written NHWC (each tap reads `C` contiguous floats). Three helper
kernels — `concat_channels`, `upsample2x_nearest`, `sub_planes` — indexed as if the buffer were
NCHW. All three silently produced garbage while CPU stayed correct.

**This is detectable in minutes with a constant-input probe.** Feed a 32² tile of 0.2:

```
CPU  mean=0.199862  min=0.190150  max=0.207067   ← plausible
GPU  mean=0.217140  min=-0.781539 max=2.174546   ← obviously wrong
```

On a real photograph the same bug is nearly invisible, because the output is meant to be smooth
and slightly different. Always probe with a constant before trusting a GPU path.

The fix in each case was `((y/2)*W + x/2)*C + c` (and `out[i*(aC+bC)+k]` for concat) rather than
`(c*H + y/2)*W + x/2`.

### 3. Truncated Huffman / SOS nibble (found in the DNG writer, same project)

Not this repo, but discovered in the same pipeline and worth recording: the SOS "selectors"
byte is `Td << 4 | Ta`. Writing `[componentID, tableID]` produces a file libraw tolerates (it
picks tables by component index) but Core Image refuses — resulting in an all-black image.

**A single decoder passing means nothing.** Every DNG this project writes is checked with both
Core Image and libraw.

## Numbers that surprised me

**σ calibration is worth more than model size.** Missing the `×2` lifting-Haar factor and the
per-channel demosaic medians: quarter drops 88%→78%, but *large* drops 93%→46%. Comparing model
sizes with wrong σ is meaningless — the big model looks worse precisely because it followed the
wrong instruction more faithfully.

**GPU speedup is flat, not shape-dependent.** 18.6× at 64², 35.1× at 128², then a plateau:
32.9× / 30.9× / 30.4× at 256² / 512² / 1024². So there is no reason to tune tile size for GPU
efficiency; choose it for quality (overlap) and memory instead.

**The full-strength result is not the most accurate.** On a synthetic ramp+noise input the
`strength = 1.0` output sits *further* from the truth than the 0.85 blend: the network
over-corrects, removing real signal along with the noise. That is the empirical case for
blending at 0.85 rather than shipping pure output — and it is why the client's default is 0.85.

## What is not ported

- **`multi` variants** (`denoise-{quarter,half,large}-multi-v1`) need superpixel binning and a
  coarse/fine pair with a guide plane. Only `single` is wired up. The other three `.anselnn`
  files load but the forward pass does not implement `arch == "unet-ms"`.
- **`gauss_*` / nmf weights** are a different project entirely (MIT, 15.3 M params). They output
  3-channel RGB, so they cannot write back to a mosaic DNG.
- **Noise profile measurement.** The host supplies `a`/`b`. Upstream has tooling for fitting them
  from a raw pair; it is not part of this component.

## Reproducing the verification

```bash
swift test -c release --scratch-path /tmp/ansel-rel

# against the independent numpy reference
python3 ref/ref_forward.py denoise-quarter-single-v1.anselnn probe.f32
# expect max|Δ| < 1e-5 (measured 1.9e-7)

# CPU vs GPU
$EXE tile MODEL in.f32 --width 512 --height 512 --a ... --b ... --out cpu.f32
$EXE tile MODEL in.f32 --width 512 --height 512 --a ... --b ... --gpu --out gpu.f32
# expect max|Δ| < 1e-5 (measured 4.2e-7)
```
