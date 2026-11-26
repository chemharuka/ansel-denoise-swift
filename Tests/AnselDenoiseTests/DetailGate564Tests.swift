import XCTest
@testable import AnselDenoise

/// 564 — the coherence gate. The synthetic cases are deterministic: a canvas-like
/// 3.4 px sine in one window next to a pure-noise window, with the denoised field
/// modelled as "the old blend kept 40 % of the fine band".
///
/// The field is wide enough (1280 px) that the §9 weight smoothing — 128 samples
/// half-width, two passes (a 257-sample window) — stays local relative to the two
/// windows. The same 257-sample window runs on the real mosaic.
final class DetailGate564Tests: XCTestCase {

    // MARK: - deterministic noise

    struct RNG {
        var state: UInt64 = 0xC0FFEE123456789
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

    static let width = 1280
    static let height = 320
    static let noiseSigma: Float = 0.004

    // Texture lives on the left, noise on the right; both sample windows stay more
    // than one smoothing window away from the boundary at x = 384.
    static let textureX = 48..<288
    static let noiseX = 928..<1216
    static let sampleY = 96..<224

    /// Left half: 3.4 px sine + noise. Right half: noise only. `d` keeps 40 % of
    /// the sine's fine band, so the old blend is a plausible under-detailed result.
    static func makeCase(amplitude: Float = 0.012,
                         removal: Float = 0.6)
        -> (input: [Float], denoised: [Float], sigma: [Float]) {
        let w = width, h = height
        let n = w * h
        var rng = RNG()
        var input = [Float](repeating: 0.1, count: n)
        for y in 0..<h {
            for x in 0..<w {
                let i = y * w + x
                // The texture patch fades over its last 64 columns. A real canvas
                // region has an edge too; this keeps the synthetic from turning
                // that edge into a one-sample impulse that no window can smooth.
                let fade: Float
                if x < 320 {
                    fade = 1
                } else if x < 384 {
                    let t = Float(384 - x) / 64
                    fade = t * t * (3 - 2 * t)
                } else {
                    fade = 0
                }
                let weave = amplitude * fade * sin(2 * .pi * Float(x) / 3.4)
                input[i] = 0.1 + weave + noiseSigma * Float(rng.normal())
            }
        }

        var blurred = [Float](repeating: 0, count: n)
        var temp = [Float](repeating: 0, count: n)
        DetailGate.gaussianBlur(input, into: &blurred, temp: &temp,
                                width: w, height: h, sigma: 1.5)
        var denoised = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let fine = input[i] - blurred[i]
            denoised[i] = input[i] - removal * fine  // old fine-band retention = 1 − removal
        }
        let sigma = [Float](repeating: noiseSigma, count: n)
        return (input, denoised, sigma)
    }

    static func highpassRMS(_ values: [Float], width: Int, height: Int,
                            x0: Int, x1: Int, y0: Int, y1: Int) -> Double {
        var blurred = [Float](repeating: 0, count: values.count)
        var temp = [Float](repeating: 0, count: values.count)
        DetailGate.gaussianBlur(values, into: &blurred, temp: &temp,
                                width: width, height: height, sigma: 1.5)
        var sum = 0.0
        var count = 0
        for y in y0..<y1 {
            for x in x0..<x1 {
                let i = y * width + x
                let f = Double(values[i] - blurred[i])
                sum += f * f
                count += 1
            }
        }
        return (sum / Double(count)).squareRoot()
    }

    static func meanWeight(_ weights: [Float], width: Int,
                           x: Range<Int>, y: Range<Int>) -> Double {
        var sum = 0.0
        var count = 0
        for yy in y {
            for xx in x {
                sum += Double(weights[yy * width + xx])
                count += 1
            }
        }
        return sum / Double(count)
    }

    // MARK: - (a) texture retained, noise untouched

    func testFineTextureIsRetainedAndPureNoiseIsNot() {
        let c = Self.makeCase()
        let w = Self.width, h = Self.height
        var output = c.denoised
        let mean = DetailGate.apply(input: c.input, output: &output, width: w, height: h,
                                    sigma: c.sigma, strength: 1,
                                    options: .standard, pixelScale: 1)

        // `wMax` is 0.35 by design (§9: keep the correction under the visibility
        // threshold), so the first cut's "≥ 85 % of the input in the fine band"
        // is unreachable here: with this model the most any weight can return is
        // `1 − 0.6·(1 − 0.35) = 61 %`, and the measured frame's own R4 weave line
        // is 0.050σ against 0.071σ input (71 %). The assertions below are the
        // calibration that *does* apply: clearly more than the old blend, and the
        // noise half untouched. The 0.5 mean-weight bound from R1 is likewise out
        // of range: no pixel can exceed 0.35 now.
        XCTAssertGreaterThan(mean, 0.02)
        XCTAssertLessThanOrEqual(mean, DetailGateOptions.standard.maxWeight + 1e-6)

        let inputTexture = Self.highpassRMS(c.input, width: w, height: h,
                                            x0: Self.textureX.lowerBound,
                                            x1: Self.textureX.upperBound,
                                            y0: Self.sampleY.lowerBound,
                                            y1: Self.sampleY.upperBound)
        let outputTexture = Self.highpassRMS(output, width: w, height: h,
                                             x0: Self.textureX.lowerBound,
                                             x1: Self.textureX.upperBound,
                                             y0: Self.sampleY.lowerBound,
                                             y1: Self.sampleY.upperBound)
        let legacyTexture = Self.highpassRMS(c.denoised, width: w, height: h,
                                             x0: Self.textureX.lowerBound,
                                             x1: Self.textureX.upperBound,
                                             y0: Self.sampleY.lowerBound,
                                             y1: Self.sampleY.upperBound)
        let kept = outputTexture / inputTexture
        XCTAssertGreaterThanOrEqual(kept, 0.5,
                                    "fine sine kept \(kept * 100)% of the input")
        XCTAssertGreaterThanOrEqual(outputTexture / legacyTexture, 1.35,
                                    "the gate must return clearly more than the old blend")

        // The pure-noise half must stay on the old blend. A few pixels may cross
        // the low knee by chance; the mean difference has to remain sub-DN.
        var diff = 0.0
        var count = 0
        for y in Self.sampleY {
            for x in Self.noiseX {
                let i = y * w + x
                diff += abs(Double(output[i] - c.denoised[i]))
                count += 1
            }
        }
        XCTAssertLessThanOrEqual(diff / Double(count) * 65535, 1.0,
                                 "pure-noise mean error must be at most one DN")

        let textureMean = Self.meanWeight(
            DetailGate.weights(input: c.input, width: w, height: h,
                               sigma: c.sigma, options: .standard, pixelScale: 1),
            width: w, x: Self.textureX, y: Self.sampleY)
        XCTAssertGreaterThan(textureMean, 0.15,
                             "texture must open the gate; measured mean \(textureMean)")
    }

    func testPureNoiseMeanWeightIsSmall() {
        let c = Self.makeCase()
        let w = Self.width, h = Self.height
        let weights = DetailGate.weights(input: c.input, width: w, height: h,
                                         sigma: c.sigma, options: .standard, pixelScale: 1)
        let mean = Self.meanWeight(weights, width: w, x: Self.noiseX, y: Self.sampleY)
        XCTAssertLessThanOrEqual(mean, 0.05)
    }

    /// The gate's envelope used to be divided by `sigma(x)`, so an all-zero sigma
    /// (a caller that has no curve) closed it. It is now self-normalised; zero
    /// sigma must still return the same weights and the same fine band.
    func testZeroSigmaKeepsTheFineBand() {
        let c = Self.makeCase()
        let w = Self.width, h = Self.height
        let zero = [Float](repeating: 0, count: w * h)
        let baseline = DetailGate.weights(input: c.input, width: w, height: h,
                                          sigma: c.sigma, options: .standard, pixelScale: 1)
        let withZero = DetailGate.weights(input: c.input, width: w, height: h,
                                          sigma: zero, options: .standard, pixelScale: 1)
        XCTAssertTrue(baseline == withZero,
                      "the gate must not depend on the noise model")

        let inputTexture = Self.highpassRMS(c.input, width: w, height: h,
                                            x0: Self.textureX.lowerBound,
                                            x1: Self.textureX.upperBound,
                                            y0: Self.sampleY.lowerBound,
                                            y1: Self.sampleY.upperBound)
        var output = c.denoised
        DetailGate.apply(input: c.input, output: &output, width: w, height: h,
                         sigma: zero, strength: 1, options: .standard, pixelScale: 1)
        let outputTexture = Self.highpassRMS(output, width: w, height: h,
                                             x0: Self.textureX.lowerBound,
                                             x1: Self.textureX.upperBound,
                                             y0: Self.sampleY.lowerBound,
                                             y1: Self.sampleY.upperBound)
        XCTAssertGreaterThanOrEqual(outputTexture / inputTexture, 0.5)
    }

    // MARK: - (c) continuity

    func testWeightFieldHasNoBlockSeam() {
        // Texture amplitude comparable to the grain, as in the measured frame.
        let c = Self.makeCase(amplitude: 0.005, removal: 0.3)
        let w = Self.width, h = Self.height
        let weights = DetailGate.weights(input: c.input, width: w, height: h,
                                         sigma: c.sigma, options: .standard, pixelScale: 1)
        var maxStep: Float = 0
        var whereStep = (x: 0, y: 0)
        for y in 32..<(h - 32) {
            for x in 32..<(w - 32) {
                let i = y * w + x
                let sx = abs(weights[i + 1] - weights[i])
                let sy = abs(weights[i + w] - weights[i])
                if sx > maxStep { maxStep = sx; whereStep = (x, y) }
                if sy > maxStep { maxStep = sy; whereStep = (x, y) }
            }
        }
        if maxStep > 0.02 {
            let i = whereStep.y * w + whereStep.x
            print("maxStep \(maxStep) at \(whereStep), weights \(weights[i-2...i+2])")
        }
        XCTAssertLessThanOrEqual(maxStep, 0.02,
                                 "adjacent gate weights differ by \(maxStep)")
    }

    // MARK: - (d) the off switch is the old arithmetic

    func testGateZeroIsBitExactOldBlend() {
        var rng = RNG()
        let n = 99 * 77
        var input = [Float](repeating: 0, count: n)
        var denoised = [Float](repeating: 0, count: n)
        for i in 0..<n {
            input[i] = Float(rng.uniform())
            denoised[i] = input[i] + 0.01 * Float(rng.normal())
        }
        let sigma = [Float](repeating: 0.006, count: n)
        var reference = denoised
        DetailGate.legacyBlend(input: input, output: &reference, strength: 0.83,
                               taper: .nlmf)

        var gated = denoised
        let mean = DetailGate.apply(input: input, output: &gated, width: 99, height: 77,
                                    sigma: sigma, strength: 0.83, taper: .nlmf,
                                    options: .off, pixelScale: 1)
        XCTAssertEqual(mean, 0)
        XCTAssertEqual(gated, reference, "detailGate = 0 must be the historical loop")

        var mosaicGated = denoised
        let mosaicMean = DetailGate.applyMosaic(input: input, output: &mosaicGated,
                                                width: 99, height: 77, sigma: sigma,
                                                cfa: "rggb", strength: 0.83, taper: .nlmf,
                                                options: .off)
        XCTAssertEqual(mosaicMean, 0)
        XCTAssertEqual(mosaicGated, reference,
                       "detailGate = 0 on the mosaic path must be the historical loop")
    }

    /// A flat wall with different R/G/B levels must not look like texture. Measuring
    /// coherence on the raw mosaic would see the 2×2 colour pattern and gate it.
    func testMosaicColourPatternIsNotCoherence() {
        let w = 512, h = 256
        let n = w * h
        var rng = RNG()
        var input = [Float](repeating: 0, count: n)
        // rggb, with a deliberate per-channel offset so the pattern is visible.
        let offset: [Float] = [0, 0.01, 0.02, 0.01]
        for y in 0..<h {
            for x in 0..<w {
                let i = y * w + x
                let site = (y % 2) * 2 + (x % 2)
                input[i] = 0.1 + offset[site] + 0.003 * Float(rng.normal())
            }
        }
        let sigma = [Float](repeating: 0.003, count: n)
        var output = input
        let mean = DetailGate.applyMosaic(input: input, output: &output,
                                          width: w, height: h, sigma: sigma,
                                          cfa: "rggb", strength: 1,
                                          options: .standard)
        XCTAssertLessThanOrEqual(mean, 0.02)
        // With `output == input` the residual is zero, so this is the old blend; the
        // point of the test is the mean above: the 2×2 colour pattern must not open
        // the gate on a flat field.
        XCTAssertEqual(output, input)
    }

    // MARK: - the gate does not read the noise model

    /// The second cut divided the envelope by `sigma(x)·κ`. Two noise estimators
    /// that disagree at this brightness moved `C` enough to close the gate, so the
    /// floor is now measured from the frame. Changing `a`/`b` must move every sigma
    /// sample and leave every weight bit-for-bit identical.
    func testWeightsAreIndependentOfTheNoiseModel() {
        let c = Self.makeCase()
        let w = Self.width, h = Self.height
        let n = w * h
        let a: [Double] = [1.073e-3, 1.073e-3, 1.073e-3]
        let b: [Double] = [3.35e-5, 3.35e-5, 3.35e-5]
        func sigmaArray(scale: Double) -> [Float] {
            var sigma = [Float](repeating: 0, count: n)
            DetailGate.fillSigma(&sigma, input: c.input, width: w, height: h,
                                 a: a.map { $0 * scale }, b: b.map { $0 * scale },
                                 cfa: "rggb")
            return sigma
        }
        let baseline = sigmaArray(scale: 1)
        let high = sigmaArray(scale: 1.5)
        let low = sigmaArray(scale: 0.7)
        XCTAssertNotEqual(baseline, high)
        XCTAssertNotEqual(baseline, low)

        func weights(_ sigma: [Float]) -> [Float] {
            DetailGate.weights(input: c.input, width: w, height: h,
                               sigma: sigma, options: .standard, pixelScale: 1)
        }
        XCTAssertTrue(weights(baseline) == weights(high),
                      "the gate must not move when the noise model is scaled up")
        XCTAssertTrue(weights(baseline) == weights(low),
                      "the gate must not move when the noise model is scaled down")
    }

    // MARK: - the envelope zero point is calibrated, not assumed

    /// The frame floor is a p15 of the very envelope it divides, so a flat field
    /// would sit at `p50/p15` — about 1.02 for the 33 px window — and a knee at
    /// 1.00 would open on a blank wall. `c0` is measured on a synthetic unit-noise
    /// field through this operator and must be deterministic across calls.
    func testEnvelopeZeroPointIsMeasuredAndCached() {
        let c0 = DetailGate.envelopeZeroPoint(sigmaPixels: 1.5, boxHalf: 16)
        print("564 envelope zero point c0 = \(c0)")
        XCTAssertGreaterThan(c0, 1.0)
        XCTAssertLessThan(c0, 1.20)
        XCTAssertEqual(c0, DetailGate.envelopeZeroPoint(sigmaPixels: 1.5, boxHalf: 16),
                       "the calibration is a property of the operator and must not drift")
    }

    // MARK: - the diagnostics dump

    /// `--dump-gate`: float32, mosaic size, row-major, plus a `width height` sidecar.
    func testGateDumpWritesFloatFieldAndSidecar() throws {
        let w = 64, h = 32
        let weights = (0..<(w * h)).map { Float($0) / Float(w * h) }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("detail-gate-564-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let prefix = dir.appendingPathComponent("gate").path
        try DenoiseImageRunner.writeGateWeights(prefix: prefix, weights: weights,
                                                width: w, height: h)

        let raw = try Data(contentsOf: URL(fileURLWithPath: prefix + ".f32"))
        XCTAssertEqual(raw.count, w * h * MemoryLayout<Float>.size)
        let read = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        XCTAssertEqual(read, weights)

        let sidecar = try String(contentsOfFile: prefix + ".txt", encoding: .utf8)
        let parts = sidecar.split(whereSeparator: { $0 == " " || $0 == "\n" })
        XCTAssertEqual(parts.map(String.init), ["64", "32"])
    }

    // MARK: - filter primitives

    func testBoxMeanMatchesBruteForceWithReplicatedEdges() {
        var rng = RNG()
        let w = 17, h = 11, size = 6
        let n = w * h
        let values = (0..<n).map { _ in Float(rng.normal()) }
        var actual = values
        var temp = [Float](repeating: 0, count: n)
        DetailGate.boxMeanInPlace(&actual, temp: &temp, width: w, height: h, size: size)

        let half = size / 2
        for y in 0..<h {
            for x in 0..<w {
                var sum: Float = 0
                for dy in 0..<size {
                    for dx in 0..<size {
                        let sy = min(max(y + dy - half, 0), h - 1)
                        let sx = min(max(x + dx - half, 0), w - 1)
                        sum += values[sy * w + sx]
                    }
                }
                XCTAssertEqual(actual[y * w + x], sum / Float(size * size),
                               accuracy: 1e-6)
            }
        }
    }
}
