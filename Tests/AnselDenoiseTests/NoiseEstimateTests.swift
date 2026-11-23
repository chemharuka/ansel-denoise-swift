import XCTest
@testable import AnselDenoise

final class NoiseEstimateTests: XCTestCase {

    // MARK: - helpers

    /// Deterministic standard-normal sampler, so a failure can be reproduced.
    struct RNG {
        var state: UInt64
        mutating func next() -> Double {
            // xorshift64*, uniform in (0,1)
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27
            let v = state &* 2685821657736338717
            return (Double(v >> 11) + 0.5) / Double(1 << 53)
        }
        mutating func normal() -> Double {
            let u1 = next(), u2 = next()
            return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
        }
    }

    /// A mosaic that walks every brightness level, so the sigma²-on-mean regression has range.
    func makeSynthetic(width: Int, height: Int, a: [Double], b: [Double],
                       seed: UInt64 = 0x9E3779B97F4A7C15) -> [Float] {
        var rng = RNG(state: seed)
        let sites = try! AnselNoiseEstimator.sites(pattern: "rggb")
        // plane index per 2×2 site, in the estimator's R, G, B, G order
        var planeOfSite = [Int](repeating: 0, count: 4)
        for (p, s) in sites.enumerated() { planeOfSite[s.dy * 2 + s.dx] = p }

        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let p = planeOfSite[(y % 2) * 2 + x % 2]
                // smooth gradient along x, 0.002 … 0.25: every brightness level is
                // represented, and a block is nearly flat, so no scene edges sit inside
                // a block and inflate the MAD.
                let xv = 0.002 + 0.248 * (Double(x) / Double(width - 1))
                let varX = a[p] * xv + b[p]
                let v = xv + varX.squareRoot() * rng.normal()
                out[y * width + x] = Float(v)
            }
        }
        return out
    }

    // MARK: - estimator

    func testEstimatorRecoversSyntheticCurve() throws {
        let a = [8.0e-4, 8.0e-4, 8.0e-4, 8.0e-4]
        let b = [2.0e-6, 2.0e-6, 2.0e-6, 2.0e-6]
        let mosaic = makeSynthetic(width: 2048, height: 2048, a: a, b: b)
        let m = try AnselNoiseEstimator.measure(mosaic, width: 2048, height: 2048, cfa: "rggb")

        // What the blend keys on is sigma at 5 % brightness, and that is what has to be right.
        // The intercept on its own is *not* an unbiased estimator: taking the 15th percentile of a
        // finite-width brightness bin reads the bin's low-brightness edge, which pulls `b` down a
        // little. The reference implementation (nonlocal-matchfilter) fits the same way, so both
        // engines condition on the same numbers — that shared convention is the point.
        for p in 0..<4 {
            XCTAssertEqual(m.a[p], a[p], accuracy: a[p] * 0.15,
                           "plane \(p) slope: measured \(m.a[p])")
            let trueSigma = (a[p] * AnselStrength.reference + b[p]).squareRoot()
            let measured = m.sigma(AnselStrength.reference, plane: p)
            XCTAssertEqual(measured, trueSigma, accuracy: trueSigma * 0.15,
                           "plane \(p) sigma at 5 %: measured \(measured), true \(trueSigma)")
            // Reported after the doubling the blend law wants; never negative, never absurd.
            XCTAssertGreaterThanOrEqual(m.b[p], 0)
            XCTAssertLessThan(m.b[p], b[p] * AnselNoiseEstimator.interceptScale * 2)
            XCTAssertGreaterThan(m.blocks[p], 0)
        }
    }

    func testEstimatorThrowsWhenThereAreNotEnoughBlocks() {
        // 256² mosaic → plane of 128² → 4×4 blocks of 32: far below the 40 bins a fit needs.
        let mosaic = makeSynthetic(width: 256, height: 256,
                                  a: [8e-4, 8e-4, 8e-4, 8e-4], b: [2e-6, 2e-6, 2e-6, 2e-6])
        XCTAssertThrowsError(try AnselNoiseEstimator.measure(mosaic, width: 256, height: 256,
                                                             cfa: "rggb"))
    }

    func testCheckerboardHalvesTheBlocks() throws {
        let a = [6e-4, 6e-4, 6e-4, 6e-4], b = [2e-6, 2e-6, 2e-6, 2e-6]
        let mosaic = makeSynthetic(width: 2048, height: 2048, a: a, b: b)
        let full = try AnselNoiseEstimator.measure(mosaic, width: 2048, height: 2048, cfa: "rggb")
        let half = try AnselNoiseEstimator.measure(mosaic, width: 2048, height: 2048,
                                                   cfa: "rggb", checkerboard: true)
        for p in 0..<4 {
            XCTAssertEqual(half.blocks[p], full.blocks[p] / 2, accuracy: full.blocks[p] / 20)
        }
    }

    func testGreensArePooledByBlockCount() {
        let m = NoiseEstimateModel(a: [1e-4, 2e-4, 3e-4, 4e-4],
                                   b: [1e-6, 2e-6, 3e-6, 4e-6],
                                   blocks: [100, 300, 100, 100], seconds: 0)
        XCTAssertEqual(m.channelA[0], 1e-4)
        XCTAssertEqual(m.channelA[2], 3e-4)
        // weighted: (2e-4·300 + 4e-4·100) / 400
        XCTAssertEqual(m.channelA[1], 2.5e-4, accuracy: 1e-12)
        XCTAssertEqual(m.channelB[1], 2.5e-6, accuracy: 1e-15)
    }

    // MARK: - strength

    func testStrengthSaturatesAtTheAnchors() {
        XCTAssertEqual(AnselStrength.strength(forSigma: 0.001), AnselStrength.strengthLow)
        XCTAssertEqual(AnselStrength.strength(forSigma: 0.02), AnselStrength.strengthHigh)
        XCTAssertEqual(AnselStrength.strength(forSigma: AnselStrength.sigmaLow),
                       AnselStrength.strengthLow, accuracy: 1e-12)
        XCTAssertEqual(AnselStrength.strength(forSigma: AnselStrength.sigmaHigh),
                       AnselStrength.strengthHigh, accuracy: 1e-12)
        // geometric midpoint → arithmetic midpoint of the weights
        let mid = (AnselStrength.sigmaLow * AnselStrength.sigmaHigh).squareRoot()
        XCTAssertEqual(AnselStrength.strength(forSigma: mid),
                       0.5 * (AnselStrength.strengthLow + AnselStrength.strengthHigh),
                       accuracy: 1e-12)
        XCTAssertEqual(AnselStrength.strength(forSigma: 0), AnselStrength.strengthLow)
    }

    // MARK: - taper

    func testIdentityTaperIsBitExact() {
        let t = BlendTaper.identity
        for x in stride(from: Float(0), through: 1, by: 0.05) {
            XCTAssertEqual(t.factor(for: x), 1)
        }
    }

    func testHighlightTaperIsExactOutsideItsKnees() {
        let t = BlendTaper(highlight: 0.18, highlightLow: 0.05, highlightHigh: 0.25)
        let q = Float(t.highlight)
        XCTAssertEqual(t.factor(for: 0), 1)                       // below the low knee
        XCTAssertEqual(t.factor(for: Float(t.highlightLow)), 1)
        XCTAssertEqual(t.factor(for: Float(t.highlightHigh)), 1 - q, accuracy: 1e-6)
        XCTAssertEqual(t.factor(for: 0.9), 1 - q, accuracy: 1e-6)
        // monotone: never increases with brightness
        var last = t.factor(for: 0)
        for i in 1...100 {
            let f = t.factor(for: Float(i) / 100)
            XCTAssertLessThanOrEqual(f, last + 1e-7)
            last = f
        }
    }

    func testNlmfTaperKeepsTheMiddleOfTheRangeWhole() {
        // Compound taper: 0.5 given up at the floor, 0.18 in the highlights, and exactly 1
        // between the two knees — that gap is what keeps ordinary midtones untouched.
        let t = BlendTaper.nlmf
        XCTAssertEqual(t.factor(for: 0), 0.5, accuracy: 1e-6)
        XCTAssertEqual(t.factor(for: 0.03), 1, accuracy: 1e-6)
        XCTAssertEqual(t.factor(for: 0.05), 1, accuracy: 1e-6)
        XCTAssertEqual(t.factor(for: 0.5), 1 - 0.18, accuracy: 1e-6)
    }

    func testShadowTaperIsExactAboveItsHighKnee() {
        let t = BlendTaper(highlight: 0, shadow: 0.5, shadowLow: 0.002, shadowHigh: 0.02)
        XCTAssertEqual(t.factor(for: 0), 0.5)
        XCTAssertEqual(t.factor(for: 0.002), 0.5)
        XCTAssertEqual(t.factor(for: 0.02), 1)                    // bit for bit above the knee
        XCTAssertEqual(t.factor(for: 0.5), 1)
        // monotone increasing
        var last = t.factor(for: 0)
        for i in 0...100 {
            let f = t.factor(for: 0.03 * Float(i) / 100)
            XCTAssertGreaterThanOrEqual(f, last - 1e-7)
            last = f
        }
    }

    func testBothTapersMultiply() {
        let t = BlendTaper(highlight: 0.2, highlightLow: 0.05, highlightHigh: 0.25,
                           shadow: 0.5, shadowLow: 0.002, shadowHigh: 0.02)
        // At the floor both terms act: (1 - 0.5) * 1 = 0.5.
        XCTAssertEqual(t.factor(for: 0), 0.5, accuracy: 1e-6)
        // 0.01 sits between the shadow knees and below the highlight ones: only the shadow acts.
        let f = t.factor(for: 0.01)
        XCTAssertGreaterThan(f, 0.5)
        XCTAssertLessThan(f, 1)
        XCTAssertEqual(t.factor(for: 0.5), 0.8, accuracy: 1e-6)   // only the highlight acts
    }
}
