# Model weights

## Licence: these are **not** GPL

Upstream separates three things, and the weights fall outside all the restrictions:

| What | Licence | Covers |
|---|---|---|
| `Sources/` (this repo) | GPL-3.0-or-later | the code |
| `LICENSE-DATA.md` (ATDL-1.1) | training **data** | raw sensor tiles only |
| `.anselnn` weights | **none of the above** | not restricted |

`LICENSE-DATA.md` §2.2, verbatim:

> **The resulting weights are yours**: this license places no restriction on them — use, ship,
> sell or license them however you see fit, in open-source or commercial applications, related
> to Ansel or not.

So committing a weight file here creates no licensing problem, and no GPL obligation attaches to
it. (The GPL in this repo covers the *implementation* — the Swift port and its Metal kernels —
not the numbers the network runs on.)

## What is committed and what is fetched

`denoise-quarter-single-v1.anselnn` (1.9 MB) is committed so a fresh clone can run the tests and
denoise a frame without any network access. It is the default variant and the smallest.

The other five are **not** committed — 100 MB of binaries in git history is a permanent cost for
what is a single `curl`. Get them with:

```sh
./scripts/fetch-weights.sh                 # all six, sha256-verified
./scripts/fetch-weights.sh quarter-single  # one variant
./scripts/fetch-weights.sh --list          # names, no download
```

The script re-reads upstream's `manifest.json` each run, so an upstream revision bump replaces
stale local files automatically. A file that fails verification is **deleted**, not kept: a
corrupt weight file loads without error and quietly produces bad pixels.

## The six variants

| File | Size | Params | 256² GPU | σ reduction | Implemented here |
|---|---|---|---|---|---|
| `denoise-quarter-single-v1` | 1.9 MB | 0.48 M | 51 ms | 88% | ✅ default |
| `denoise-half-single-v1` | 7.6 MB | 1.90 M | 216 ms | 92% | ✅ |
| `denoise-large-single-v1` | 30 MB | 7.59 M | 882 ms | 93% | ✅ |
| `denoise-quarter-multi-v1` | 9.4 MB | — | — | — | ❌ needs superpixel binning |
| `denoise-half-multi-v1` | 15 MB | — | — | — | ❌ |
| `denoise-large-multi-v1` | 38 MB | — | — | — | ❌ |

The three `multi` files are committed nowhere and load nowhere: they use `arch == "unet-ms"`, a
coarse/fine pair with a guide plane that this port does not implement.

## Why quarter is the default beyond cost

It is the most robust to a wrong σ estimate. With the sigma calibration missing entirely, quarter
degrades 88%→78% while **large** degrades 93%→46% — the bigger model looks worse because it
followed the wrong instruction more faithfully. When comparing variants, first check that
σ is right, or the comparison is meaningless.
