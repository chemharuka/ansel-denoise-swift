//
//  DetailGate.swift
//  AnselDenoise
//
//  564 — keep the fine band of the residual where the input has coherent texture.
//
//  A denoiser's residual `d − i` is not all noise. In the finest band (here about
//  1.5 plane pixels) texture and grain have the same amplitude, so a per-pixel
//  amplitude test cannot tell them apart. What separates them is *coherence*: grain
//  is uncorrelated from sample to sample, while a canvas weave or a painted edge
//  leaves a repeatable pattern. This file measures that coherence on the input,
//  turns it into a smooth per-pixel weight, and gives back that fraction of the
//  residual's fine band.
//
//  The algebra, on normalized input `i`, denoised `d` and blend weight `s`:
//
//      r  = d − i
//      f  = r − G(σf)(r)              fine band of the residual
//      fi = i − G(σf)(i)              same band on the input
//      env = sqrt(boxmean(fi², 33px)) squared fine-band envelope
//      C  = env / (floor(bin(x)) · c0)  self-normalized grain floor
//      w  = wMax·smoothstep(1.00, 1.40, C)
//      out = i + s·(r − w·f)
//
//  The floor is the 15th percentile of `env` inside a brightness bin, measured on
//  this frame (32 bins with edges at full-frame brightness quantiles). Bins follow
//  the local box-mean brightness, not the raw sample: grain would otherwise scatter
//  pixels into the bright bins and their p15 floor would sink. That makes
//  the gate independent of the noise model: a different `a`/`b` estimate moves
//  every `env` sample and its floor together, so `C` is unchanged. An earlier
//  version divided by `σ(x)·κ`, and two noise estimators that disagreed by 1.33×
//  on the measured frame closed the gate entirely.
//
//  `c0` puts the zero point of `C` back at 1.0. The floor is a p15 of the *same*
//  envelope distribution, so on a flat field `env/floor` has a median of
//  `p50/p15`, not one; with the 33 px envelope window measured here that ratio is
//  about 1.02. Left uncorrected, a knee at 1.00 opens on a blank wall. `c0` is the
//  median-to-p15 ratio of a synthetic unit white-noise field run through the same
//  fine band and box mean — a property of the filter, not of the frame or of the
//  noise model — measured once per operator and cached.
//
//  Compatibility boundary: `maxWeight == 0` runs the historical
//  `i + s·(d − i)` loop verbatim. The same is true pixel-by-pixel when `w` is
//  exactly zero, which is what a pure-noise window produces.
//

import Foundation
import Accelerate

/// Parameters of the coherence gate. `sigma` is in **plane** pixels; the caller
/// chooses the mosaic conversion (`planeToMosaic`) because a mosaic sample is half
/// a plane pixel in each direction.
public struct DetailGateOptions: Sendable, Equatable {
    /// Maximum fraction of the fine residual to preserve. `0` disables the gate.
    public var maxWeight: Double
    /// Coherence below this gets no detail back.
    public var kneeLow: Double
    /// Coherence above this gets the full `maxWeight`.
    public var kneeHigh: Double
    /// Gaussian sigma of the fine band, in plane pixels.
    public var sigma: Double

    public init(maxWeight: Double = 0.35,
                kneeLow: Double = 1.00,
                kneeHigh: Double = 1.40,
                sigma: Double = 1.5) {
        self.maxWeight = maxWeight
        self.kneeLow = kneeLow
        self.kneeHigh = kneeHigh
        self.sigma = sigma
    }

    public static let standard = DetailGateOptions()
    public static let off = DetailGateOptions(maxWeight: 0)
}

public enum DetailGate {
    /// One plane pixel spans two samples in a 2×2 CFA mosaic.
    public static let planeToMosaic = 2.0
    /// **Half** width of the envelope box, in plane pixels. The window is
    /// `2·16 + 1 = 33` plane pixels. (The first cut wrote `32` here and used it as
    /// a half-width too; 16 is the measured value and this constant now says
    /// explicitly which half of the window it is.)
    public static let boxPlanePixels = 16.0
    /// **Half** width of one weight-smoothing box, in weight-grid samples. The gate
    /// weight lives on the mosaic (one sample per photosite), and 564's measurements
    /// were taken against that field, so the constant is used as-is on the mosaic:
    /// 128 gives a 257-sample window. Tests that hand over a single plane use the
    /// same 128 on the plane grid. Two passes are chained.
    ///
    /// 564 §9 settled on this after the 33-px window still made the gate patchy:
    /// on L1040080 the standard deviation of 64-px block means was 0.313 with the
    /// narrow window. The gate's own weight range is only `0.35` now, so a wide,
    /// gentle field is what keeps the correction from reading as a patch.
    public static let weightSmoothingPlanePixels = 128.0
    /// How many times the weight field is smoothed before it is applied.
    public static let weightSmoothingPasses = 2
    /// Brightness bins for the grain floor.
    public static let floorBins = 32
    /// Percentile of the envelope that becomes the grain floor inside a bin.
    public static let floorPercentile: Float = 0.15
    /// A bin in which texture outnumbers flat area would otherwise set its own
    /// floor from texture, closing the gate exactly where the weave peaks. The
    /// grain floor is a smooth function of brightness, so each bin is capped at the
    /// running median of its neighbours; that removes a texture spike without
    /// flattening a genuine brightness trend.
    public static let floorTrendBins = 9

    /// Side of the synthetic field used to calibrate the envelope zero point.
    /// Large enough that the interior quantiles are stable, small enough that a
    /// one-off measurement costs a few milliseconds.
    private static let zeroPointFieldSide = 512
    /// Fixed seed; the calibration has to be identical in every process and build.
    private static let zeroPointSeed: UInt64 = 0x9E37_79B9_7F4A_7C15

    private struct ZeroPointKey: Hashable {
        let sigma: Float
        let boxHalf: Int
    }

    private static let zeroPointLock = NSLock()
    private static var zeroPointCache: [ZeroPointKey: Float] = [:]

    /// Resolution of the brightness histogram used to place the bin edges, and of
    /// each per-bin envelope histogram. Quantiles are read off histograms; nothing
    /// is sorted, so this stays O(n).
    private static let brightnessHistogramBins = 1024
    private static let envelopeHistogramBins = 256

    // MARK: - geometry / sigma

    /// 2×2 CFA pattern → colour index (0=R, 1=G, 2=B) for each of its four sites.
    public static func channelIndices(pattern: String) -> [Int] {
        let p = Array(pattern.lowercased())
        precondition(p.count == 4, "CFA pattern must be 4 characters, got '\(pattern)'")
        return p.map { ch in
            switch ch {
            case "r": return 0
            case "g": return 1
            case "b": return 2
            default: preconditionFailure("bad CFA character '\(ch)'")
            }
        }
    }

    /// Per-sample sigma from the existing `Var = a·x + b` model. The same clamp as
    /// `AnselDenoiseForward.makeInputPlanes` is used, so the network conditions on
    /// the same value at the same pixel. The gate itself no longer divides by this.
    @inline(__always)
    public static func sigma(value: Float, channel: Int,
                             a: [Float], b: [Float]) -> Float {
        let x = max(value, 0)
        let v = a[channel] * x + b[channel]
        return v > 0 ? v.squareRoot() : 0
    }

    /// Fill `sigma` (one value per mosaic sample) from the CFA pattern and noise model.
    public static func fillSigma(_ sigma: inout [Float],
                                 input: [Float], width: Int, height: Int,
                                 a: [Double], b: [Double], cfa: String) {
        let n = width * height
        precondition(sigma.count == n && input.count == n)
        precondition(a.count >= 3 && b.count >= 3)
        let quad = channelIndices(pattern: cfa)
        let af = a.prefix(3).map(Float.init)
        let bf = b.prefix(3).map(Float.init)
        for y in 0..<height {
            let row = y * width
            let q = (y & 1) * 2
            for x in 0..<width {
                let i = row + x
                sigma[i] = Self.sigma(value: input[i], channel: quad[q + (x & 1)],
                                      a: af, b: bf)
            }
        }
    }

    // MARK: - public blend

    /// Replace `output` (initially the denoised field `d`) with the gated blend.
    ///
    /// - Parameters:
    ///   - sigma: per-sample sigma from the noise model. Accepted for API
    ///     compatibility and consistency checks; the gate's floor is measured from
    ///     the frame and does not use it.
    ///   - strength: frame-wide blend weight.
    ///   - taper: the existing per-pixel reduction of `strength`.
    ///   - pixelScale: samples per plane pixel (2 for a CFA mosaic, 1 for tests that
    ///     hand over a single plane).
    /// - Returns: mean gate weight `w` over the whole field.
    @discardableResult
    public static func apply(input: [Float],
                             output: inout [Float],
                             width: Int, height: Int,
                             sigma: [Float],
                             strength: Float,
                             taper: BlendTaper = .identity,
                             options: DetailGateOptions = .standard,
                             pixelScale: Double = 1) -> Double {
        let n = width * height
        precondition(input.count == n && output.count == n && sigma.count == n)

        let maxWeight = Float(min(max(options.maxWeight, 0), 1))
        if maxWeight == 0 || strength == 0 || options.sigma <= 0 {
            legacyBlend(input: input, output: &output, strength: strength, taper: taper)
            return 0
        }

        let sigmaPixels = Float(options.sigma * pixelScale)
        let boxHalf = max(1, Int((boxPlanePixels * pixelScale).rounded()))

        var residual = [Float](repeating: 0, count: n)
        for i in 0..<n { residual[i] = output[i] - input[i] }

        var temp = [Float](repeating: 0, count: n)
        var coarse = [Float](repeating: 0, count: n)
        gaussianBlur(residual, into: &coarse, temp: &temp,
                     width: width, height: height, sigma: sigmaPixels)
        let coherence = coherenceField(input: input, width: width, height: height,
                                       sigmaPixels: sigmaPixels, boxHalf: boxHalf)
        let weights = weightField(coherence: coherence, options: options,
                                  maxWeight: maxWeight, pixelScale: pixelScale,
                                  width: width, height: height)
        return finish(input: input, output: &output, residual: residual, coarse: coarse,
                      weights: weights, strength: strength, taper: taper)
    }

    /// CFA-aware entry point. Coherence is measured on each colour plane at the
    /// requested plane-pixel scale; the residual's fine band is still filtered in
    /// the mosaic, where `1.5 plane px` is `3 mosaic px`. Measuring on the raw
    /// mosaic itself would count the 2×2 colour pattern as coherent detail, which
    /// gates every flat wall.
    @discardableResult
    public static func applyMosaic(input: [Float],
                                   output: inout [Float],
                                   width: Int, height: Int,
                                   sigma: [Float],
                                   cfa: String,
                                   strength: Float,
                                   taper: BlendTaper = .identity,
                                   options: DetailGateOptions = .standard) -> Double {
        applyMosaicDetailed(input: input, output: &output, width: width, height: height,
                            sigma: sigma, cfa: cfa, strength: strength, taper: taper,
                            options: options).meanWeight
    }

    /// CFA-aware entry point that also hands back the per-sample gate weight, so a
    /// diagnostics run can dump the field the blend actually used. The gate cost is
    /// the same as `applyMosaic`; the weight array already exists either way.
    @discardableResult
    public static func applyMosaicDetailed(input: [Float],
                                           output: inout [Float],
                                           width: Int, height: Int,
                                           sigma: [Float],
                                           cfa: String,
                                           strength: Float,
                                           taper: BlendTaper = .identity,
                                           options: DetailGateOptions = .standard)
        -> (meanWeight: Double, weights: [Float]) {
        let n = width * height
        precondition(input.count == n && output.count == n && sigma.count == n)

        let maxWeight = Float(min(max(options.maxWeight, 0), 1))
        if maxWeight == 0 || strength == 0 || options.sigma <= 0 {
            legacyBlend(input: input, output: &output, strength: strength, taper: taper)
            return (0, [Float](repeating: 0, count: n))
        }

        let sigmaMosaic = Float(options.sigma * planeToMosaic)
        let boxHalf = max(1, Int(boxPlanePixels.rounded()))

        var residual = [Float](repeating: 0, count: n)
        for i in 0..<n { residual[i] = output[i] - input[i] }

        var temp = [Float](repeating: 0, count: n)
        var coarse = [Float](repeating: 0, count: n)
        gaussianBlur(residual, into: &coarse, temp: &temp,
                     width: width, height: height, sigma: sigmaMosaic)
        let coherence = planeCoherenceField(input: input, width: width, height: height,
                                            cfa: cfa, sigma: Float(options.sigma),
                                            boxHalf: boxHalf)
        let weights = weightField(coherence: coherence, options: options,
                                  maxWeight: maxWeight, pixelScale: 1,
                                  width: width, height: height)
        let mean = finish(input: input, output: &output, residual: residual, coarse: coarse,
                          weights: weights, strength: strength, taper: taper)
        return (mean, weights)
    }

    private static func finish(input: [Float], output: inout [Float],
                               residual: [Float], coarse: [Float], weights: [Float],
                               strength: Float, taper: BlendTaper) -> Double {
        var sumWeight = 0.0
        for i in 0..<input.count {
            let w = weights[i]
            sumWeight += Double(w)

            let s = strength * taper.factor(for: input[i])
            if w == 0 {
                // Exactly the historical blend; this branch is what keeps pure noise
                // (and `detailGate = 0`) bit-for-bit on the old result.
                output[i] = input[i] + s * residual[i]
            } else {
                let fine = residual[i] - coarse[i]
                output[i] = input[i] + s * (coarse[i] + (1 - w) * fine)
            }
        }
        return sumWeight / Double(input.count)
    }

    /// Coherence -> weight, then two chained box passes over the finished field.
    /// The smoothing is a box convolution of a smoothstep of a box mean, so it cannot
    /// introduce a boundary; it only widens the knee at a hard edge. No epsilon is
    /// applied: zeroing a statistical tail would make the field discontinuous again.
    private static func weightField(coherence: [Float],
                                    options: DetailGateOptions, maxWeight: Float,
                                    pixelScale: Double, width: Int, height: Int) -> [Float] {
        let n = width * height
        let low = Float(options.kneeLow)
        let high = Float(options.kneeHigh)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = maxWeight * smoothstep(coherence[i], low, high)
        }
        let half = max(1, Int((weightSmoothingPlanePixels * pixelScale).rounded()))
        let window = 2 * half + 1
        var temp = [Float](repeating: 0, count: n)
        for _ in 0..<weightSmoothingPasses {
            boxMeanInPlace(&out, temp: &temp, width: width, height: height, size: window)
        }
        return out
    }

    /// The gate weight alone, for diagnostics and the continuity test.
    ///
    /// `sigma` is accepted for source compatibility and is not read: `C` is
    /// self-normalized, so two different noise models produce identical weights.
    public static func weights(input: [Float], width: Int, height: Int,
                               sigma: [Float],
                               options: DetailGateOptions = .standard,
                               pixelScale: Double = 1) -> [Float] {
        let n = width * height
        precondition(input.count == n && sigma.count == n)
        let maxWeight = Float(min(max(options.maxWeight, 0), 1))
        guard maxWeight > 0, options.sigma > 0 else {
            return [Float](repeating: 0, count: n)
        }
        let sigmaPixels = Float(options.sigma * pixelScale)
        let boxHalf = max(1, Int((boxPlanePixels * pixelScale).rounded()))
        let coherence = coherenceField(input: input, width: width, height: height,
                                       sigmaPixels: sigmaPixels, boxHalf: boxHalf)
        return weightField(coherence: coherence, options: options,
                           maxWeight: maxWeight, pixelScale: pixelScale,
                           width: width, height: height)
    }

    /// The historical `i + s·(d − i)` loop. Kept in one place so the fallback is
    /// literally the old arithmetic, not an algebraically equivalent rearrangement.
    public static func legacyBlend(input: [Float], output: inout [Float],
                                   strength: Float, taper: BlendTaper = .identity) {
        precondition(input.count == output.count)
        for i in 0..<input.count {
            let s = strength * taper.factor(for: input[i])
            output[i] = input[i] + s * (output[i] - input[i])
        }
    }

    // MARK: - filter pieces

    /// `C = env / (floor(bin) · c0)` for a single plane, where `env = sqrt(boxmean(fi²))`,
    /// `floor` is the 15th percentile of `env` among samples in the same brightness
    /// bin, and `c0` the envelope zero point (see `envelopeZeroPoint`).
    ///
    /// Bins are assigned from the box-mean brightness, not the raw sample. Grain moves
    /// a pixel across a brightness edge from sample to sample; with raw brightness the
    /// bright bins inherit a low-envelope tail and their p15 floor sinks, which opens
    /// the gate on flat walls. The local mean is the brightness the scene has there.
    private static func coherenceField(input: [Float], width: Int, height: Int,
                                       sigmaPixels: Float, boxHalf: Int) -> [Float] {
        let n = width * height
        var envelope = [Float](repeating: 0, count: n)
        var brightness = input
        var temp = [Float](repeating: 0, count: n)
        boxMeanInPlace(&brightness, temp: &temp, width: width, height: height,
                       size: 2 * boxHalf + 1)
        gaussianBlur(input, into: &envelope, temp: &temp,
                     width: width, height: height, sigma: sigmaPixels)
        for i in 0..<n {
            let fi = input[i] - envelope[i]
            envelope[i] = fi * fi
        }
        boxMeanInPlace(&envelope, temp: &temp, width: width, height: height,
                       size: 2 * boxHalf + 1)
        for i in 0..<n { envelope[i] = max(envelope[i], 0).squareRoot() }
        let radius = max(1, Int(ceil(3 * sigmaPixels)))
        let margin = envelopeMargin(boxHalf: boxHalf, radius: radius,
                                    width: width, height: height)
        let c0 = envelopeZeroPoint(sigmaPixels: sigmaPixels, boxHalf: boxHalf)
        return grainFloorCoherence(brightness: brightness, envelope: envelope,
                                   width: width, height: height, margin: margin,
                                   zeroPoint: c0)
    }

    /// `median(env) / p15(env)` on a synthetic unit white-noise field run through this
    /// exact fine band and box mean. The frame floor is itself a p15 of the envelope,
    /// so a flat patch would otherwise sit at `p50/p15` instead of 1; dividing by this
    /// ratio is what makes a knee at 1.00 keep a blank wall closed.
    ///
    /// Independent of the noise model by construction: the field is generated, not
    /// measured. Cached per `(sigmaPixels, boxHalf)`, so the cost is one filter pass
    /// per distinct operator, never per frame.
    public static func envelopeZeroPoint(sigmaPixels: Float, boxHalf: Int) -> Float {
        let key = ZeroPointKey(sigma: sigmaPixels, boxHalf: max(1, boxHalf))
        zeroPointLock.lock()
        let cached = zeroPointCache[key]
        zeroPointLock.unlock()
        if let cached { return cached }

        let value = measureEnvelopeZeroPoint(sigmaPixels: key.sigma, boxHalf: key.boxHalf)
        zeroPointLock.lock()
        zeroPointCache[key] = value
        zeroPointLock.unlock()
        return value
    }

    private static func measureEnvelopeZeroPoint(sigmaPixels: Float, boxHalf: Int) -> Float {
        let side = zeroPointFieldSide
        let n = side * side
        var rng = ZeroPointRNG(seed: zeroPointSeed)
        var field = [Float](repeating: 0, count: n)
        for i in 0..<n { field[i] = Float(rng.normal()) }

        var envelope = [Float](repeating: 0, count: n)
        var temp = [Float](repeating: 0, count: n)
        gaussianBlur(field, into: &envelope, temp: &temp,
                     width: side, height: side, sigma: sigmaPixels)
        for i in 0..<n {
            let fi = field[i] - envelope[i]
            envelope[i] = fi * fi
        }
        boxMeanInPlace(&envelope, temp: &temp, width: side, height: side,
                       size: 2 * boxHalf + 1)
        for i in 0..<n { envelope[i] = max(envelope[i], 0).squareRoot() }

        let radius = max(1, Int(ceil(3 * sigmaPixels)))
        let margin = envelopeMargin(boxHalf: boxHalf, radius: radius,
                                    width: side, height: side)
        var lo = Float.infinity, hi = -Float.infinity
        for y in margin..<(side - margin) {
            let row = y * side
            for x in margin..<(side - margin) {
                let v = envelope[row + x]
                if v < lo { lo = v }
                if v > hi { hi = v }
            }
        }
        guard hi > lo else { return 1 }
        let low = histogramPercentile(envelope, width: side, height: side, margin: margin,
                                      lo: lo, hi: hi, cells: brightnessHistogramBins,
                                      percentile: floorPercentile)
        let median = histogramPercentile(envelope, width: side, height: side, margin: margin,
                                         lo: lo, hi: hi, cells: brightnessHistogramBins,
                                         percentile: 0.5)
        return low > 0 && median > 0 ? median / low : 1
    }

    /// Same envelope and floor, but each 2×2 CFA site is filtered inside its own
    /// colour plane. This keeps the 2×2 colour pattern itself from reading as
    /// texture. The result is scattered back onto the mosaic, so every photosite
    /// sees the coherence of its own colour neighbourhood.
    public static func planeCoherenceField(input: [Float], width: Int, height: Int,
                                           cfa: String, sigma: Float,
                                           boxHalf: Int) -> [Float] {
        let n = width * height
        precondition(input.count == n)
        let p = Array(cfa.lowercased())
        precondition(p.count == 4, "CFA pattern must be 4 characters, got '\(cfa)'")

        var out = [Float](repeating: 0, count: n)
        for site in 0..<4 {
            let dy = site / 2, dx = site % 2
            let pw = (width - dx + 1) / 2
            let ph = (height - dy + 1) / 2
            guard pw > 0, ph > 0 else { continue }
            let planeN = pw * ph
            var plane = [Float](repeating: 0, count: planeN)
            for y in 0..<ph {
                let src = (y * 2 + dy) * width + dx
                let dst = y * pw
                for x in 0..<pw { plane[dst + x] = input[src + x * 2] }
            }

            let coherence = coherenceField(input: plane, width: pw, height: ph,
                                           sigmaPixels: sigma, boxHalf: boxHalf)

            for y in 0..<ph {
                let dst = (y * 2 + dy) * width + dx
                let src = y * pw
                for x in 0..<pw { out[dst + x * 2] = coherence[src + x] }
            }
        }
        return out
    }

    /// Divide the envelope by a grain floor measured *on this frame*, times the
    /// measured zero point `c0` (see `envelopeZeroPoint`).
    ///
    /// Brightness bins are placed at full-frame quantiles (32 bins, read off a
    /// 1024-cell histogram) so no bin is empty just because the scene has no
    /// mid-tones. Inside each bin the floor is the 15th percentile of the envelope,
    /// read off its own histogram. Both passes are O(n); nothing is sorted.
    ///
    /// The floor is measured away from the frame border by `margin` pixels. A box
    /// mean at the border replicates pixels, which lowers the envelope's variance
    /// there; on a small plane (a 2×2 CFA site is a quarter of the frame) that bias
    /// would drag the 15th percentile down and make a flat wall look coherent.
    private static func grainFloorCoherence(brightness: [Float], envelope: [Float],
                                            width: Int, height: Int,
                                            margin: Int, zeroPoint: Float) -> [Float] {
        let n = width * height
        precondition(brightness.count == n && envelope.count == n)
        let bins = floorBins

        let binOf = brightnessBins(brightness, bins: bins)
        let floors = binFloors(envelope, binOf: binOf, width: width, height: height,
                               margin: margin, bins: bins)
        let c0 = zeroPoint > 0 ? zeroPoint : 1
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let floor = floors[Int(binOf[i])] * c0
            out[i] = floor > 0 ? envelope[i] / floor : 0
        }
        return out
    }

    /// How many pixels to keep away from the border when measuring the floor.
    /// `0` when the plane is too small to have an interior.
    private static func envelopeMargin(boxHalf: Int, radius: Int,
                                       width: Int, height: Int) -> Int {
        let margin = boxHalf + radius
        return 2 * margin < min(width, height) ? margin : 0
    }

    /// Brightness-bin index per sample, with edges at full-frame quantiles.
    private static func brightnessBins(_ brightness: [Float], bins: Int) -> [UInt8] {
        let n = brightness.count
        var lo = brightness[0], hi = brightness[0]
        for v in brightness {
            if v < lo { lo = v }
            if v > hi { hi = v }
        }

        var binOf = [UInt8](repeating: 0, count: n)
        guard hi > lo else { return binOf }

        let cells = brightnessHistogramBins
        var histogram = [Int](repeating: 0, count: cells)
        let scale = Float(cells) / (hi - lo)
        for v in brightness {
            var index = Int((v - lo) * scale)
            if index < 0 { index = 0 } else if index >= cells { index = cells - 1 }
            histogram[index] += 1
        }

        var edges = [Float]()
        edges.reserveCapacity(bins - 1)
        var cumulative = 0
        var edge = 1
        for cell in 0..<cells {
            cumulative += histogram[cell]
            while edge < bins {
                let target = Double(edge) / Double(bins) * Double(n)
                if Double(cumulative) < target { break }
                edges.append(lo + (Float(cell) + 0.5) * (hi - lo) / Float(cells))
                edge += 1
            }
        }
        while edges.count < bins - 1 { edges.append(hi) }

        // One table from brightness-histogram cell to bin, so the per-pixel pass is
        // a multiply and an array read instead of a binary search.
        var cellToBin = [UInt8](repeating: 0, count: cells)
        var bin = 0
        for cell in 0..<cells {
            let v = lo + (Float(cell) + 0.5) * (hi - lo) / Float(cells)
            while bin < edges.count && edges[bin] <= v { bin += 1 }
            cellToBin[cell] = UInt8(min(bin, bins - 1))
        }

        for i in 0..<n {
            var index = Int((brightness[i] - lo) * scale)
            if index < 0 { index = 0 } else if index >= cells { index = cells - 1 }
            binOf[i] = cellToBin[index]
        }
        return binOf
    }

    /// 15th percentile of `envelope` inside each brightness bin, measured away from
    /// the border. Empty bins (and degenerate, constant bins) borrow the whole-frame
    /// floor, so a ratio always has a meaningful denominator.
    private static func binFloors(_ envelope: [Float], binOf: [UInt8],
                                  width: Int, height: Int, margin: Int,
                                  bins: Int) -> [Float] {
        let x0 = margin, x1 = width - margin
        let y0 = margin, y1 = height - margin
        let cells = envelopeHistogramBins

        var globalLo = Float.infinity, globalHi = -Float.infinity
        var lo = [Float](repeating: .infinity, count: bins)
        var hi = [Float](repeating: -.infinity, count: bins)
        var count = [Int](repeating: 0, count: bins)
        for y in y0..<y1 {
            let row = y * width
            for x in x0..<x1 {
                let i = row + x
                let v = envelope[i]
                if v < globalLo { globalLo = v }
                if v > globalHi { globalHi = v }
                let bin = Int(binOf[i])
                if v < lo[bin] { lo[bin] = v }
                if v > hi[bin] { hi[bin] = v }
                count[bin] += 1
            }
        }

        let globalFloor = { () -> Float in
            // Only needed when a bin is empty or degenerate. Brightness bins sit at
            // full-frame quantiles, so on a normal frame every bin has samples and
            // this whole-frame pass is skipped.
            histogramPercentile(envelope, width: width, height: height,
                                margin: margin, lo: globalLo, hi: globalHi,
                                cells: brightnessHistogramBins,
                                percentile: floorPercentile)
        }

        var histograms = [Int](repeating: 0, count: bins * cells)
        for y in y0..<y1 {
            let row = y * width
            for x in x0..<x1 {
                let i = row + x
                let bin = Int(binOf[i])
                guard hi[bin] > lo[bin] else { continue }
                let scale = Float(cells) / (hi[bin] - lo[bin])
                var index = Int((envelope[i] - lo[bin]) * scale)
                if index < 0 { index = 0 } else if index >= cells { index = cells - 1 }
                histograms[bin * cells + index] += 1
            }
        }

        var floors = [Float](repeating: 0, count: bins)
        var missing = false
        for bin in 0..<bins {
            guard count[bin] > 0, hi[bin] > lo[bin] else {
                missing = true
                continue
            }
            var cumulative = 0
            var chosen = cells - 1
            // A count-proportional target keeps the percentile exact when bin
            // occupancy is uneven; the global fallback uses the same rule.
            let binTarget = Double(floorPercentile) * Double(count[bin])
            for cell in 0..<cells {
                cumulative += histograms[bin * cells + cell]
                if Double(cumulative) >= binTarget {
                    chosen = cell
                    break
                }
            }
            floors[bin] = lo[bin] + (Float(chosen) + 0.5) * (hi[bin] - lo[bin]) / Float(cells)
        }
        if missing {
            let fallback = globalFloor()
            for bin in 0..<bins where !(count[bin] > 0 && hi[bin] > lo[bin]) {
                floors[bin] = fallback
            }
        }
        let trend = runningMedian(floors, width: floorTrendBins)
        for bin in 0..<bins where floors[bin] > trend[bin] {
            floors[bin] = trend[bin]
        }
        return floors
    }

    /// Running median with a shrinking window at the ends. `width` is expected to
    /// be odd; the upper middle element is used if a clipped window is even.
    private static func runningMedian(_ values: [Float], width: Int) -> [Float] {
        let half = max(0, width / 2)
        var out = values
        var window = [Float]()
        window.reserveCapacity(2 * half + 1)
        for i in 0..<values.count {
            let lo = max(0, i - half)
            let hi = min(values.count, i + half + 1)
            window.removeAll(keepingCapacity: true)
            window.append(contentsOf: values[lo..<hi])
            window.sort()
            out[i] = window[window.count / 2]
        }
        return out
    }

    /// Percentile read off a histogram over `[lo, hi]`, restricted to the interior.
    private static func histogramPercentile(_ values: [Float],
                                            width: Int, height: Int, margin: Int,
                                            lo: Float, hi: Float,
                                            cells: Int, percentile: Float) -> Float {
        guard hi > lo else { return max(lo, 0) }
        var histogram = [Int](repeating: 0, count: cells)
        let scale = Float(cells) / (hi - lo)
        var total = 0
        for y in margin..<(height - margin) {
            let row = y * width
            for x in margin..<(width - margin) {
                var index = Int((values[row + x] - lo) * scale)
                if index < 0 { index = 0 } else if index >= cells { index = cells - 1 }
                histogram[index] += 1
                total += 1
            }
        }
        guard total > 0 else { return max(lo, 0) }
        let target = Double(percentile) * Double(total)
        var cumulative = 0
        for cell in 0..<cells {
            cumulative += histogram[cell]
            if Double(cumulative) >= target {
                return lo + (Float(cell) + 0.5) * (hi - lo) / Float(cells)
            }
        }
        return hi
    }

    /// Separable Gaussian with edge replication. vImage is used for both passes so
    /// this stays O(N·radius) with vectorized inner loops even in a debug build.
    public static func gaussianBlur(_ src: [Float], into dst: inout [Float],
                                    temp: inout [Float],
                                    width: Int, height: Int, sigma: Float) {
        let n = width * height
        precondition(src.count == n && dst.count == n && temp.count == n)
        guard width > 0, height > 0, sigma > 0 else {
            dst = src
            return
        }
        let kernel = gaussianKernel(sigma: sigma)
        let radius = kernel.count / 2
        if radius == 0 {
            dst = src
            return
        }

        src.withUnsafeBufferPointer { sp in
            temp.withUnsafeMutableBufferPointer { tp in
                dst.withUnsafeMutableBufferPointer { dp in
                    var source = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: sp.baseAddress!),
                        height: vImagePixelCount(height),
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<Float>.stride)
                    var middle = vImage_Buffer(
                        data: tp.baseAddress!,
                        height: vImagePixelCount(height),
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<Float>.stride)
                    var dest = vImage_Buffer(
                        data: dp.baseAddress!,
                        height: vImagePixelCount(height),
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<Float>.stride)

                    kernel.withUnsafeBufferPointer { kp in
                        _ = vImageConvolve_PlanarF(
                            &source, &middle, nil, 0, 0,
                            kp.baseAddress!, 1, UInt32(kernel.count), 0,
                            vImage_Flags(kvImageEdgeExtend))
                        _ = vImageConvolve_PlanarF(
                            &middle, &dest, nil, 0, 0,
                            kp.baseAddress!, UInt32(kernel.count), 1, 0,
                            vImage_Flags(kvImageEdgeExtend))
                    }
                }
            }
        }
    }

    /// Symmetric, normalized, truncated at 3σ.
    public static func gaussianKernel(sigma: Float) -> [Float] {
        guard sigma > 0 else { return [1] }
        let radius = max(1, Int(ceil(3 * sigma)))
        var k = [Float](repeating: 0, count: 2 * radius + 1)
        let denom = 2 * sigma * sigma
        var sum: Float = 0
        for j in -radius...radius {
            let v = exp(-Float(j * j) / denom)
            k[j + radius] = v
            sum += v
        }
        for i in k.indices { k[i] /= sum }
        return k
    }

    /// 2-D box mean with edge replication, done as two O(N) sliding sums.
    /// `size` is the full window width.
    public static func boxMeanInPlace(_ values: inout [Float], temp: inout [Float],
                                      width: Int, height: Int, size: Int) {
        let n = width * height
        precondition(values.count == n && temp.count == n && width > 0 && height > 0)
        let s = max(1, size)
        let half = s / 2
        let inv = 1 / Float(s)

        values.withUnsafeBufferPointer { sp in
            temp.withUnsafeMutableBufferPointer { tp in
                for y in 0..<height {
                    let row = y * width
                    var sum: Float = 0
                    for k in 0..<s {
                        sum += sp[row + min(max(k - half, 0), width - 1)]
                    }
                    tp[row] = sum * inv
                    if width > 1 {
                        for x in 1..<width {
                            let add = min(x + s - 1 - half, width - 1)
                            let drop = max(x - 1 - half, 0)
                            sum += sp[row + add] - sp[row + drop]
                            tp[row + x] = sum * inv
                        }
                    }
                }
            }
        }

        temp.withUnsafeBufferPointer { sp in
            values.withUnsafeMutableBufferPointer { vp in
                // Column sums for one row at a time. Walking down a column (one
                // pixel per cache line) was the slow half of this routine; this
                // touches the temp buffer row-major while accumulating exactly the
                // same per-column sequence of adds and drops, so the result is
                // bit-for-bit what the column walk produced.
                var columnSum = [Float](repeating: 0, count: width)
                columnSum.withUnsafeMutableBufferPointer { cp in
                    for k in 0..<s {
                        let sy = min(max(k - half, 0), height - 1)
                        let row = sy * width
                        for x in 0..<width { cp[x] += sp[row + x] }
                    }
                    for y in 0..<height {
                        let row = y * width
                        for x in 0..<width { vp[row + x] = cp[x] * inv }
                        if y + 1 < height {
                            let add = min(y + s - half, height - 1)
                            let drop = max(y - half, 0)
                            let addRow = add * width
                            let dropRow = drop * width
                            for x in 0..<width {
                                cp[x] += sp[addRow + x] - sp[dropRow + x]
                            }
                        }
                    }
                }
            }
        }
    }

    @inline(__always)
    private static func smoothstep(_ x: Float, _ a: Float, _ b: Float) -> Float {
        guard b > a else { return x >= b ? 1 : 0 }
        var t = (x - a) / (b - a)
        t = t < 0 ? 0 : (t > 1 ? 1 : t)
        return t * t * (3 - 2 * t)
    }

    /// xorshift64* plus Box–Muller, fixed seed. Only used for the one-off
    /// calibration field, so it favours determinism over speed.
    private struct ZeroPointRNG {
        private var state: UInt64
        init(seed: UInt64) { state = seed }

        mutating func uniform() -> Double {
            state ^= state >> 12
            state ^= state << 25
            state ^= state >> 27
            let v = state &* 2685821657736338717
            return (Double(v >> 11) + 0.5) / Double(1 << 53)
        }

        mutating func normal() -> Double {
            let u1 = uniform(), u2 = uniform()
            return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
        }
    }
}
