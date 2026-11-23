//
//  NoiseEstimate.swift
//  AnselDenoise
//
//  Measure a sensor's Poisson-Gaussian noise curve from one image, and the blend laws that
//  belong with it.
//
//  Why the measured curve, and not the shipped constants
//  -----------------------------------------------------
//  `--a`/`--b` default to one full-frame sensor's numbers. Fed to a different camera — or to the
//  same camera at a different ISO — they are wrong in a way nothing reports: too high over-smooths
//  the shadows, too low leaves grain the module was supposed to remove. The curve is a property of
//  *this file*, so measure it here.
//
//  The method is the one the nonlocal match-filter denoiser uses (`NonlocalMatchFilter/NoiseEstimate.swift`,
//  MIT, Photons' other denoise path). It is reproduced here rather than reinvented so both engines
//  condition on the same sigma field for the same frame — that is what makes their results
//  comparable at all. Two choices decide whether the numbers are right:
//
//    1. Measure on diagonal differences inside one CFA plane. Neighbouring mosaic samples are
//       different colours, so their difference is dominated by the scene. Diagonals of a 2×2 cell
//       share a colour, so a difference there is noise plus a little scene structure.
//    2. Take the lower envelope of the cloud, not its mean. Texture only ever *adds* variance, so
//       the honest level at a given brightness is a low percentile of the estimates in that
//       brightness bin. A least-squares line through binned means over-estimates badly.
//
//  MAD → sigma is `MAD / 0.6745 / √2`: 0.6745 converts a MAD to a standard deviation, and √2 undoes
//  the doubling of variance that a difference of two independent samples introduces.
//
//  The read-noise floor is doubled (`interceptScale`). The model describes the sensor, but a
//  rendered picture lifts the shadows 2–4×, where residual grain is amplified on screen in a way
//  the linear model never sees. Doubling `b` multiplies sigma only where `b` dominates — dark
//  pixels. This is a *look* decision shared with the other denoiser, not physics; it is applied
//  here so both engines agree.
//

import Foundation

/// `Var = a·x + b` per CFA plane, in plane order R, G(red row), B, G(blue row), x on [0,1].
public struct NoiseEstimateModel: Sendable, Equatable {
    /// Shot-noise slope per plane.
    public let a: [Double]
    /// Read-noise variance per plane — already multiplied by `interceptScale`.
    public let b: [Double]
    /// Blocks that fed each plane's fit, for diagnostics.
    public let blocks: [Int]
    public let seconds: Double

    public init(a: [Double], b: [Double], blocks: [Int], seconds: Double) {
        self.a = a
        self.b = b
        self.blocks = blocks
        self.seconds = seconds
    }

    public func sigma(_ x: Double, plane: Int) -> Double {
        let v = a[plane] * x + b[plane]
        return v > 0 ? v.squareRoot() : 0
    }

    /// The conditioning sigma the blend law keys on: green plane, 5 % brightness.
    public var referenceSigma: Double { sigma(AnselStrength.reference, plane: 1) }

    /// Collapse the four CFA planes onto the network's three channels (R, G, B).
    ///
    /// Both green sites carry the same filter on the same silicon; they are separate planes above
    /// only because they sit at different positions in the 2×2 cell. Pooling them by block count
    /// removes most of the difference between two independent fits, and the network has one G
    /// plane, not two.
    public var channelA: [Double] {
        [a[0], pooledGreen(a[1], a[3]), a[2]]
    }

    public var channelB: [Double] {
        [b[0], pooledGreen(b[1], b[3]), b[2]]
    }

    private func pooledGreen(_ g1: Double, _ g2: Double) -> Double {
        let n1 = Double(blocks[1]), n2 = Double(blocks[3])
        guard n1 + n2 > 0 else { return 0.5 * (g1 + g2) }
        return (g1 * n1 + g2 * n2) / (n1 + n2)
    }
}

public enum NoiseEstimateError: Error, CustomStringConvertible {
    case tooFewBlocks(plane: Int, blocks: Int)

    public var description: String {
        switch self {
        case .tooFewBlocks(let p, let n):
            return "plane \(p) only yielded \(n) flat blocks; a fit on that would be guesswork"
        }
    }
}

public enum AnselNoiseEstimator {

    /// Block edge in plane pixels. 32 samples a side gives a stable MAD and rarely straddles a
    /// texture boundary.
    public static let block = 32
    /// Brightness bins across the histogram.
    public static let bins = 40
    /// Multiplier on the fitted intercept; see the file header.
    public static let interceptScale = 2.0
    /// Percentile taken inside each bin — the lower envelope.
    ///
    /// 15 rather than 10: raising it moves sigma up on every frame, which is the safe direction.
    /// An under-estimated sigma makes the network read noise as texture, and no blend weight puts
    /// that back; an over-estimated one only costs a little fine detail.
    public static let lowPercentile = 15.0

    /// CFA pattern → the four (row parity, column parity) sites, in R, G, B, G plane order.
    ///
    /// A 2×2 mosaic has one red site, two green and one blue; the two greens are kept apart because
    /// they sit in different rows and their fits are independent measurements of the same sensor.
    static func sites(pattern: String) throws -> [(dy: Int, dx: Int)] {
        let p = Array(pattern.lowercased())
        guard p.count == 4 else { throw NoiseEstimateError.tooFewBlocks(plane: 0, blocks: 0) }
        var red: (Int, Int)?, blue: (Int, Int)?
        var greens: [(Int, Int)] = []
        for (i, ch) in p.enumerated() {
            let site = (i / 2, i % 2)
            switch ch {
            case "r": red = site
            case "b": blue = site
            case "g": greens.append(site)
            default: throw NoiseEstimateError.tooFewBlocks(plane: 0, blocks: 0)
            }
        }
        guard let r = red, let b = blue, greens.count == 2 else {
            throw NoiseEstimateError.tooFewBlocks(plane: 0, blocks: 0)
        }
        return [r, greens[0], b, greens[1]]
    }

    /// Measure the whole frame.
    ///
    /// Cost on a 60 MP file is a few seconds against the minute the denoise itself takes, so the
    /// frame is not subsampled. `checkerboard` halves the blocks, alternating in both axes: unlike a
    /// coarser stride it thins every brightness level evenly, which is what the sigma²-on-mean
    /// regression needs.
    public static func measure(_ mosaic: [Float], width: Int, height: Int,
                               cfa: String, checkerboard: Bool = false) throws -> NoiseEstimateModel {
        let t0 = Date()
        let sites = try sites(pattern: cfa)
        var a = [Double](repeating: 0, count: 4)
        var b = [Double](repeating: 0, count: 4)
        var counts = [Int](repeating: 0, count: 4)

        for p in 0..<4 {
            let (dy, dx) = sites[p]
            let ph = (height - dy + 1) / 2
            let pw = (width - dx + 1) / 2
            guard ph > block, pw > block else {
                throw NoiseEstimateError.tooFewBlocks(plane: p, blocks: 0)
            }

            // Gather this site's samples into a contiguous plane so the inner loops can stride
            // without recomputing the mosaic offsets.
            var plane = [Float](repeating: 0, count: ph * pw)
            for y in 0..<ph {
                let src = (y * 2 + dy) * width + dx
                let dst = y * pw
                for x in 0..<pw { plane[dst + x] = mosaic[src + x * 2] }
            }

            let stats = blockStats(plane, width: pw, height: ph, checkerboard: checkerboard)
            guard stats.count >= bins else {
                throw NoiseEstimateError.tooFewBlocks(plane: p, blocks: stats.count)
            }
            let (aa, bb) = fit(stats)
            a[p] = aa
            // Doubled only where it dominates, i.e. in the shadows; see the file header.
            b[p] = bb * interceptScale
            counts[p] = stats.count
            plane = []
        }

        return NoiseEstimateModel(a: a, b: b, blocks: counts,
                                  seconds: Date().timeIntervalSince(t0))
    }

    /// Per-block (mean, sigma) pairs, skipping blocks with no measurable noise.
    static func blockStats(_ plane: [Float], width: Int, height: Int,
                           checkerboard: Bool = false) -> [(Double, Double)] {
        var out: [(Double, Double)] = []
        out.reserveCapacity((width / block) * (height / block))
        var by = 0
        for y0 in stride(from: 0, to: height - block, by: block) {
            var bx = 0
            defer { by += 1 }
            for x0 in stride(from: 0, to: width - block, by: block) {
                defer { bx += 1 }
                if checkerboard && (bx + by) % 2 != 0 { continue }

                var sum = 0.0
                var diffs: [Double] = []
                diffs.reserveCapacity(2 * (block - 1) * (block - 1))
                for y in y0..<(y0 + block) {
                    let row = y * width
                    for x in x0..<(x0 + block) { sum += Double(plane[row + x]) }
                }
                // Diagonals of the 2×2 cell: same colour on both ends, so the difference is noise.
                for y in y0..<(y0 + block - 1) {
                    let row = y * width
                    let next = row + width
                    for x in x0..<(x0 + block - 1) {
                        diffs.append(Double(plane[next + x + 1]) - Double(plane[row + x]))
                        diffs.append(Double(plane[next + x]) - Double(plane[row + x + 1]))
                    }
                }
                guard !diffs.isEmpty else { continue }
                diffs.sort()
                let median = percentileOfSorted(diffs, 0.5)
                var dev = [Double](repeating: 0, count: diffs.count)
                for (i, d) in diffs.enumerated() { dev[i] = abs(d - median) }
                dev.sort()
                let mad = percentileOfSorted(dev, 0.5)
                let sigma = mad / 0.6744897501960817 / 2.0.squareRoot()
                out.append((sum / Double(block * block), sigma))
            }
        }
        return out
    }

    /// Fit `Var = a·x + b` to the lower envelope of the cloud, with `a, b ≥ 0`.
    static func fit(_ pairs: [(Double, Double)]) -> (Double, Double) {
        guard pairs.count >= bins else { return (0, 0) }
        let sorted = pairs.sorted { $0.0 < $1.0 }
        let perBin = max(1, sorted.count / bins)

        var bx: [Double] = []
        var by: [Double] = []
        for start in stride(from: 0, to: sorted.count, by: perBin) {
            let slice = sorted[start..<min(start + perBin, sorted.count)]
            if slice.count < 20 { continue }
            var sigmas: [Double] = []
            var meanSum = 0.0
            for (m, s) in slice where s > 0 && s.isFinite {
                sigmas.append(s)
                meanSum += m
            }
            if sigmas.count < 10 { continue }
            sigmas.sort()
            let lo = percentileOfSorted(sigmas, lowPercentile / 100.0)
            bx.append(meanSum / Double(slice.count))
            by.append(lo * lo)          // variance
        }
        guard bx.count >= 3 else { return (0, 0) }

        // Ordinary least squares, then clamp: a negative intercept only means the dark end is
        // noisier than a straight line predicts, and a negative slope would make sigma fall with
        // brightness, which has no physical basis.
        var sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
        let n = Double(bx.count)
        for i in 0..<bx.count {
            sx += bx[i]; sy += by[i]
            sxx += bx[i] * bx[i]; sxy += bx[i] * by[i]
        }
        let denom = n * sxx - sx * sx
        guard abs(denom) > 1e-18 else { return (0, 0) }
        var a = (n * sxy - sx * sy) / denom
        var b = (sy - a * sx) / n
        if a < 0 { a = 0 }
        if b < 0 {
            // Pin the intercept to the darkest bins and refit the slope through it.
            b = by.prefix(2).min() ?? 0
            var num = 0.0, den = 0.0
            for i in 0..<bx.count {
                num += bx[i] * (by[i] - b)
                den += bx[i] * bx[i]
            }
            a = den > 1e-18 ? num / den : 0
        }
        return (a, b)
    }

    /// Value at `q` of an already-sorted array (linear interpolation).
    static func percentileOfSorted(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let pos = q * Double(sorted.count - 1)
        let lo = Int(pos.rounded(.down))
        let hi = min(lo + 1, sorted.count - 1)
        let t = pos - Double(lo)
        return sorted[lo] * (1 - t) + sorted[hi] * t
    }
}

/// How much of the network's answer to keep, as a function of how noisy the frame is.
///
/// One constant cannot serve every ISO: 1.0 plastifies fine texture, but a low value leaves a
/// high-ISO frame's grain half-removed, and that grain is what reads as detail. The anchors are the
/// same ones the nonlocal denoiser uses (a scan of 190 Leica SL3 frames, green plane, sigma at 5 %
/// brightness: 0.00355 at ISO 800 … 0.00680 at ISO 3200), so a frame gets the same force from
/// either engine.
public enum AnselStrength {

    /// Brightness at which sigma is compared.
    public static let reference = 0.05
    public static let sigmaLow = 0.00369
    public static let sigmaHigh = 0.00721
    public static let strengthLow = 0.70
    public static let strengthHigh = 0.85

    /// Linear in log sigma between the anchors, clamped outside them.
    ///
    /// Logs because shot noise grows with ISO multiplicatively and the eye judges grain by ratio,
    /// so a fixed sigma step should move the weight by a fixed amount. Keying on sigma rather than
    /// the ISO tag is deliberate: a bright ISO 3200 frame measures less noise than a dark ISO 800
    /// one and should be treated as the quieter frame it is.
    public static func strength(forSigma sigma: Double) -> Double {
        guard sigma > 0, sigma.isFinite else { return strengthLow }
        let t = (log(sigma) - log(sigmaLow)) / (log(sigmaHigh) - log(sigmaLow))
        return strengthLow + (strengthHigh - strengthLow) * min(max(t, 0), 1)
    }

    /// `strength` for a resolved `Var = a·x + b`, green plane.
    public static func strength(shot: [Double], read: [Double]) -> Double {
        guard shot.count > 1, read.count > 1 else { return strengthLow }
        let v = shot[1] * reference + read[1]
        return strength(forSigma: v > 0 ? v.squareRoot() : 0)
    }
}

/// How the blend weight is reduced towards the ends of the tonal range.
///
/// `strength` is one number per frame, which is as far as a frame-level noise comparison can
/// justify — but the detail a denoiser takes away is not spread evenly over the frame: it
/// concentrates where the signal is strong, and at the floor a pixel whose value is comparable to
/// its own noise carries no detail that can be separated from grain, yet the network still flattens
/// it. Both ends therefore give up a fraction of the weight:
///
///     s(x) = s · (1 − q_hi·smoothstep(x, hiLo, hiHi)) · (1 − q_lo·(1 − smoothstep(x, loLo, loHi)))
///
/// The smoothsteps are pinned to exactly 0 or 1 outside their knees, so a pixel below `highlightLow`
/// or above `shadowHigh` is blended with the unaltered `strength` — bit for bit, not approximately.
/// ``BlendTaper.identity`` therefore reproduces the historical frame-wide weight exactly.
public struct BlendTaper: Sendable, Equatable {
    /// Fraction of the weight given up in the highlights (0 keeps all of it).
    public var highlight: Double
    public var highlightLow: Double
    public var highlightHigh: Double
    /// Fraction given up at the floor.
    public var shadow: Double
    public var shadowLow: Double
    public var shadowHigh: Double

    public init(highlight: Double = 0, highlightLow: Double = 0.05, highlightHigh: Double = 0.25,
                shadow: Double = 0, shadowLow: Double = 0.002, shadowHigh: Double = 0.02) {
        self.highlight = highlight
        self.highlightLow = highlightLow
        self.highlightHigh = highlightHigh
        self.shadow = shadow
        self.shadowLow = shadowLow
        self.shadowHigh = shadowHigh
    }

    public static let identity = BlendTaper()

    /// The settings Photons uses for the nonlocal path: 0.18 given up over 0.05–0.25 of linear RAW
    /// brightness, and 0.50 at the floor below 0.002.
    ///
    /// The highlight knees are not mid-grey on purpose: a tone curve lifts the shadows 2–4×, so
    /// 0.05–0.25 here is already most of what the eye calls "bright". Measured on a dark museum
    /// frame, the 0.18–0.60 mapping acted on 0.2 % of the picture.
    public static let nlmf = BlendTaper(highlight: 0.18, highlightLow: 0.05, highlightHigh: 0.25,
                                        shadow: 0.50, shadowLow: 0.002, shadowHigh: 0.02)

    /// Multiplier on `strength` for an input value, in the normalized domain.
    @inline(__always)
    public func factor(for x: Float) -> Float {
        var f = 1.0
        if highlight > 0 {
            f *= 1 - highlight * Self.smoothstep(Double(x), highlightLow, highlightHigh)
        }
        if shadow > 0 {
            f *= 1 - shadow * (1 - Self.smoothstep(Double(x), shadowLow, shadowHigh))
        }
        return Float(f)
    }

    @inline(__always)
    static func smoothstep(_ x: Double, _ a: Double, _ b: Double) -> Double {
        guard b > a else { return x >= b ? 1 : 0 }
        var t = (x - a) / (b - a)
        t = t < 0 ? 0 : (t > 1 ? 1 : t)
        return t * t * (3 - 2 * t)
    }
}
