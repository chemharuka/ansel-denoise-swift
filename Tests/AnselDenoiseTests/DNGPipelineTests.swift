//
//  DNGPipelineTests.swift
//
//  The DNG-in / DNG-out path, checked against the invariants that decide whether the picture is
//  right — not just whether the program exited 0.
//
//  What is worth testing here, and why
//  -----------------------------------
//  A mosaic is not an image. Sample (0,0) is one specific colour and every other pixel's colour is
//  decided by its index parity, so a bug that shifts the grid by one pixel produces a file with
//  *every tag correct* and a picture that is a checkerboard. Nothing in the metadata would notice.
//  That is why these tests combine three kinds of check:
//
//    - tag-level (width, height, CFA, black, white, compression, photometric) — catches the
//      bookkeeping;
//    - pixel-level on a flat field (noise actually fell, level did not move) — catches "it wrote
//      something, but not the denoised image";
//    - phase-level (each CFA phase's mean) — catches the shift that tags cannot see.
//

import XCTest
@testable import AnselDenoise

final class DNGPipelineTests: XCTestCase {

    // MARK: - Fixtures

    /// The shipped quarter model, from this repository — not a personal path.
    static var modelURL: URL? {
        let here = URL(fileURLWithPath: #filePath)              // Tests/AnselDenoiseTests/…
        let root = here.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [
            ProcessInfo.processInfo.environment["ANSEL_DENOISE_MODEL"],
            root.appendingPathComponent("weights/denoise-quarter-single-v1.anselnn").path,
            "/Users/luyao/workspace/ansel-probe/denoise-quarter-single-v1.anselnn",
        ].compactMap { $0 }
        for c in candidates where FileManager.default.fileExists(atPath: c) {
            return URL(fileURLWithPath: c)
        }
        return nil
    }

    static var fixtureURL: URL? {
        let here = URL(fileURLWithPath: #filePath)
        let root = here.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = root.appendingPathComponent("Tests/Fixtures/small-rggb.dng")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    static var metal: MetalConv? { try? MetalConv() }

    /// A scratch directory that outlives one test method but not the run.
    func scratch(_ name: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ansel-dng-tests/\(name)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - ② geometry invariants, on the real fixture

    func testDNGToDNGCopiesEveryGeometryTag() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("geometry")
        let out = dir.appendingPathComponent("out.dng")

        let result = try DenoiseDNGPipeline.run(.init(
            model: model, metal: Self.metal,
            input: fixture, output: out,
            a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
            budgetBytes: 4_000_000_000))

        let src = try SourceDNG(url: fixture)
        let dst = try SourceDNG(url: out)

        func num(_ doc: SourceDNG, _ tag: UInt16) -> [Double] {
            (doc.raw[tag] ?? doc.ifd0[tag])?.numericValues ?? []
        }
        func scalar(_ doc: SourceDNG, _ tag: UInt16) -> Double? {
            num(doc, tag).first
        }

        // Size: the mosaic grid is the picture's geometry.
        XCTAssertEqual(result.width, 1024)
        XCTAssertEqual(result.height, 1024)
        XCTAssertEqual(scalar(dst, 256), scalar(src, 256), "width")
        XCTAssertEqual(scalar(dst, 257), scalar(src, 257), "height")

        // The four things a writer must never invent.
        XCTAssertEqual(num(dst, 33422), num(src, 33422), "CFAPattern")
        XCTAssertEqual(num(dst, 50713), num(src, 50713), "BlackLevelRepeatDim")
        XCTAssertEqual(num(dst, 50714), num(src, 50714), "BlackLevel")
        XCTAssertEqual(num(dst, 50717), num(src, 50717), "WhiteLevel")

        // Compression and photometric: the output must still be a CFA lossless-JPEG DNG.
        XCTAssertEqual(scalar(dst, 259), 7, "Compression must be 7 (lossless JPEG)")
        XCTAssertEqual(scalar(dst, 262), 32803, "PhotometricInterpretation must stay CFA")

        // Black level is 512 in this fixture: a file that lost it would render bright and grey.
        XCTAssertEqual(result.normalization.blackByPhase, [512, 512, 512, 512])
        XCTAssertEqual(result.normalization.white, 16383)
        XCTAssertEqual(result.fileCFA, "rggb")
        XCTAssertEqual(result.mosaicCFA, "rggb")

        // And it must be a real DNG file, not a header with nothing behind it.
        XCTAssertGreaterThan(result.outputBytes, 100_000)
        XCTAssertGreaterThan(result.tileCount, 0)
    }

    // MARK: - ②/⑩ CFA phase comes from the file

    /// A bggr file must go through as bggr. This is the test that fails the moment someone replaces
    /// the tag read with `let cfa = "rggb"`.
    func testCFAPhaseIsReadFromTheFileNotHardCoded() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("cfa")
        let bggr = dir.appendingPathComponent("bggr.dng")
        try Self.rewriteCFA(of: fixture, to: [2, 1, 1, 0], at: bggr)

        let out = dir.appendingPathComponent("bggr-out.dng")
        let result = try DenoiseDNGPipeline.run(.init(
            model: model, metal: Self.metal,
            input: bggr, output: out,
            a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
            budgetBytes: 4_000_000_000))

        XCTAssertEqual(result.fileCFA, "bggr", "the phase must come from tag 33422")
        XCTAssertEqual(result.mosaicCFA, "bggr")

        let dst = try SourceDNG(url: out)
        XCTAssertEqual((dst.raw[33422] ?? dst.ifd0[33422])?.numericValues, [2, 1, 1, 0],
                       "the CFA tag must survive the round trip unchanged")
    }

    /// The same pixels tagged `rggb` and `bggr` must denoise *differently*.
    ///
    /// This is the companion to the test above, and it exists because the obvious check does not
    /// work: a mutation that hard-codes `rggb` still satisfies "the CFA tag survives", since the
    /// writer copies the tag verbatim either way. What it cannot fake is the pixels. The network
    /// takes the CFA as a one-hot channel plane and a Poisson-Gaussian sigma per channel, so
    /// telling it `rggb` when the file says `bggr` changes its output — measured on the shipped
    /// fixture at 7.9 % of samples by more than 1e-4, max 1.29e-3.
    ///
    /// So: two files with identical samples and different tags must produce different output. A
    /// hard-coded phase collapses the difference to zero and this fails.
    func testCFATagChangesTheActualPixels() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("cfa-pixels")

        let rggbTagged = dir.appendingPathComponent("as-rggb.dng")
        let bggrTagged = dir.appendingPathComponent("as-bggr.dng")
        try Self.rewriteCFA(of: fixture, to: [0, 1, 1, 2], at: rggbTagged)
        try Self.rewriteCFA(of: fixture, to: [2, 1, 1, 0], at: bggrTagged)
        // Same bytes everywhere except tag 33422 — assert that, so the test cannot pass because the
        // two inputs drifted apart for some other reason.
        let a = try Data(contentsOf: rggbTagged), b = try Data(contentsOf: bggrTagged)
        XCTAssertEqual(a.count, b.count)
        let differing = zip(a, b).filter { $0 != $1 }.count
        // rggb 00010102 → bggr 02010100: the two G positions are unchanged, so R and B differ.
        XCTAssertEqual(differing, 2, "only the CFA bytes may differ (00010102 vs 02010100): got \(differing)")

        func denoise(_ input: URL, to name: String) throws -> DenoiseDNGResult {
            try DenoiseDNGPipeline.run(.init(
                model: model, metal: Self.metal,
                input: input, output: dir.appendingPathComponent(name),
                a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
                budgetBytes: 4_000_000_000))
        }
        let asRed = try denoise(rggbTagged, to: "out-rggb.dng")
        let asBlue = try denoise(bggrTagged, to: "out-bggr.dng")
        XCTAssertEqual(asRed.mosaicCFA, "rggb")
        XCTAssertEqual(asBlue.mosaicCFA, "bggr")

        let d1 = try Data(contentsOf: dir.appendingPathComponent("out-rggb.dng"))
        let d2 = try Data(contentsOf: dir.appendingPathComponent("out-bggr.dng"))
        let redPixels = try SourceDNG(url: dir.appendingPathComponent("out-rggb.dng"))
            .readMosaic(data: d1).samples
        let bluePixels = try SourceDNG(url: dir.appendingPathComponent("out-bggr.dng"))
            .readMosaic(data: d2).samples

        var changed = 0
        var maxDiff = 0
        for i in 0..<redPixels.count {
            let diff = abs(Int(redPixels[i]) - Int(bluePixels[i]))
            if diff > 0 { changed += 1 }
            maxDiff = max(maxDiff, diff)
        }
        let fraction = Double(changed) / Double(redPixels.count)
        print("⑩ CFA differential: \(changed)/\(redPixels.count) samples differ "
              + "(\(String(format: "%.1f", 100 * fraction)) %), max \(maxDiff)")

        XCTAssertGreaterThan(fraction, 0.001,
                             "the CFA phase must reach the network — identical samples under two "
                             + "different tags cannot produce identical output")
        XCTAssertGreaterThan(maxDiff, 1)
    }

    /// `ActiveArea` anchors the CFA tag, so a stored image larger than its active area needs the
    /// phase rotated. This is arithmetic, not file I/O, so it can be pinned directly.
    func testCFAPhaseRotatesWithTheActiveAreaOrigin() {
        XCTAssertEqual(DNGMetadata.rotated("rggb", row: 0, col: 0), "rggb")
        XCTAssertEqual(DNGMetadata.rotated("rggb", row: 0, col: 1), "grbg")
        XCTAssertEqual(DNGMetadata.rotated("rggb", row: 1, col: 0), "gbrg")
        XCTAssertEqual(DNGMetadata.rotated("rggb", row: 1, col: 1), "bggr")
        // R and B swap, G stays G — the invariant that makes a rotation (not a scramble) valid.
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let p = DNGMetadata.rotated("rggb", row: r, col: c)
            XCTAssertEqual(p.filter { $0 == "r" }.count, 1)
            XCTAssertEqual(p.filter { $0 == "b" }.count, 1)
            XCTAssertEqual(p.filter { $0 == "g" }.count, 2)
        }
    }

    // MARK: - ③ the noise actually falls

    /// A flat dark field: iid noise at the level the tool's own noise model expects, written to a
    /// DNG, denoised, read back. Adjacent-difference variance must drop below half and the level
    /// must not move more than 5 %.
    ///
    /// A synthetic field rather than the shipped fixture on purpose: the fixture is a low-contrast
    /// synthetic *scene* (no 256² patch in it is flat — the flattest has an adjacent-difference
    /// variance of 143 codes), so it cannot answer "did the noise go down on a flat area".
    func testFlatDarkPatchLosesAtLeastHalfItsAdjacentVariance() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("flatdark")

        // Level and noise chosen so the default coefficients describe them: x = 0.06 on [0,1],
        // black 512, white 16383. Per phase sigma = sqrt(a_eff·x + b_eff) with the same
        // sigma-scaled coefficients the CLI derives from `--a/--b`.
        let aRaw = [0.0004, 0.0003, 0.0004], bRaw = [3e-6, 1e-6, 1e-6]
        let scale = [2.82, 3.94, 2.96]
        let level = 512.0 + 0.06 * (16383.0 - 512.0)
        var samples = [UInt16](repeating: 0, count: 256 * 256)
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func gauss() -> Double {
            // Box–Muller on an LCG: no dependency, and reproducible run to run.
            func u() -> Double {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                return (Double(seed >> 11) / Double(1 << 53)) * 0.999 + 0.0005
            }
            return (-2 * log(u())).squareRoot() * cos(2 * .pi * u())
        }
        for y in 0..<256 {
            for x in 0..<256 {
                let phase = (y % 2) * 2 + (x % 2)
                let channel = phase == 0 ? 0 : (phase == 3 ? 2 : 1)
                let a = aRaw[channel] * scale[channel] * scale[channel]
                let b = bRaw[channel] * scale[channel] * scale[channel]
                let sigma = (a * 0.06 + b).squareRoot() * (16383.0 - 512.0)
                let v = level + gauss() * sigma
                samples[y * 256 + x] = UInt16(min(65535, max(0, v.rounded())))
            }
        }

        let src = try Self.makeDNG(samples: samples, width: 256, height: 256,
                                   like: Self.fixtureURL, in: dir, named: "flat.dng")
        let out = dir.appendingPathComponent("flat-out.dng")
        _ = try DenoiseDNGPipeline.run(.init(
            model: model, metal: Self.metal,
            input: src, output: out,
            a: aRaw.map { $0 * scale[0] * scale[0] },
            b: bRaw.map { $0 * scale[0] * scale[0] },
            budgetBytes: 4_000_000_000))

        let before = try SourceDNG(url: src).readMosaic(data: try Data(contentsOf: src)).samples
        let after = try SourceDNG(url: out).readMosaic(data: try Data(contentsOf: out)).samples

        func adjacentVariance(_ s: [UInt16], w: Int, h: Int) -> Double {
            var total = 0.0
            var n = 0
            for y in 0..<h {
                for x in 0..<(w - 1) {
                    let d = Double(s[y * w + x + 1]) - Double(s[y * w + x])
                    total += d * d
                    n += 1
                }
            }
            return total / Double(n)
        }
        func mean(_ s: [UInt16]) -> Double { Double(s.reduce(0) { $0 + Int($1) }) / Double(s.count) }

        let vb = adjacentVariance(before, w: 256, h: 256)
        let va = adjacentVariance(after, w: 256, h: 256)
        let mb = mean(before), ma = mean(after)
        print("③ flat dark 256²: adjacent variance \(vb) → \(va) (ratio \(va / vb)), "
              + "mean \(mb) → \(ma) (\(100 * (ma - mb) / mb) %)")

        XCTAssertGreaterThan(vb, 0, "the synthetic field must actually contain noise")
        XCTAssertLessThan(va / vb, 0.5, "denoising must at least halve the adjacent variance")
        XCTAssertLessThanOrEqual(abs(ma - mb) / mb, 0.05, "the level must not move more than 5 %")
    }

    // MARK: - ④ nothing negative reaches the writer

    /// `min` of the array handed to the writer must be ≥ 0 — and the test has to prove the clamp
    /// was *needed*, or it passes for the wrong reason. The input here carries below-black samples,
    /// which normalize negative; the strength blend then pulls those negatives back into the
    /// result, so a build with the clamp removed fails on `finalMin`.
    func testWriterNeverSeesANegativeSample() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)

        // 64² is enough: the artifact is local and shows up on the first pass.
        let w = 64, h = 64
        var mosaic = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            // A third of the samples are below the black level — the case that produced 10 264
            // negative pixels on the 60 MP frame.
            mosaic[i] = (i % 3 == 0) ? -0.004 : 0.05
        }

        let result = try DenoiseImageRunner.denoiseMosaic(
            mosaic, width: w, height: h,
            request: DenoiseImageRequest(input: URL(fileURLWithPath: "/dev/null"),
                                         output: URL(fileURLWithPath: "/dev/null"),
                                         width: w, height: h,
                                         a: [0.0032, 0.0047, 0.0035],
                                         b: [2.4e-5, 1.6e-5, 2.4e-5],
                                         cfa: "rggb", useGPU: Self.metal != nil,
                                         strength: 0.85, budgetBytes: 1_000_000_000),
            model: model, metal: Self.metal)

        print("④ clamped \(result.clampedNegatives) negative and \(result.clampedOverOne) "
              + "over-one samples; final range [\(result.finalMin), \(result.finalMax)]")

        XCTAssertGreaterThan(result.clampedNegatives, 0,
                             "the input has below-black samples, so the clamp must have had work to do")
        XCTAssertGreaterThanOrEqual(result.finalMin, 0, "no negative sample may reach the writer")
        XCTAssertEqual(result.mosaic.min() ?? 0, result.finalMin, accuracy: 1e-9)
        XCTAssertTrue(result.mosaic.allSatisfy { $0 >= 0 })
    }

    /// The blend runs after the first clamp, so it can re-introduce negatives. This pins the order.
    func testStrengthBlendCannotReintroduceNegatives() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let w = 32, h = 32
        // Every sample below the black level, and a strength of 0 so the output is the input.
        let mosaic = [Float](repeating: -0.01, count: w * h)
        let result = try DenoiseImageRunner.denoiseMosaic(
            mosaic, width: w, height: h,
            request: DenoiseImageRequest(input: URL(fileURLWithPath: "/dev/null"),
                                         output: URL(fileURLWithPath: "/dev/null"),
                                         width: w, height: h,
                                         a: [0.0032, 0.0047, 0.0035],
                                         b: [2.4e-5, 1.6e-5, 2.4e-5],
                                         cfa: "rggb", useGPU: Self.metal != nil,
                                         strength: 0, budgetBytes: 100_000_000),
            model: model, metal: Self.metal)
        XCTAssertGreaterThanOrEqual(result.finalMin, 0,
                                    "strength=0 returns the input, which is negative — must be clamped")
        XCTAssertEqual(result.clampedNegatives, w * h)
    }

    // MARK: - ⑤ a height that is not a multiple of 16

    /// 1000 is not a multiple of 16, which is what killed the first version of the padding. The
    /// output must come back 1024×1000 — same size, not the padded one, and not 1024×1008.
    func testHeightThatIsNotAMultipleOf16SurvivesRoundTrip() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("height1000")

        let full = try SourceDNG(url: fixture)
        let (samples, w, h) = try full.readMosaic(data: try Data(contentsOf: fixture))
        XCTAssertEqual(w, 1024)
        var cropped = [UInt16](repeating: 0, count: 1024 * 1000)
        for y in 0..<1000 {
            for x in 0..<1024 { cropped[y * 1024 + x] = samples[y * w + x] }
        }
        let src = try Self.makeDNG(samples: cropped, width: 1024, height: 1000,
                                   like: fixture, in: dir, named: "crop1024x1000.dng")

        let out = dir.appendingPathComponent("crop-out.dng")
        let result = try DenoiseDNGPipeline.run(.init(
            model: model, metal: Self.metal,
            input: src, output: out,
            a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
            budgetBytes: 4_000_000_000))

        XCTAssertEqual(result.width, 1024)
        XCTAssertEqual(result.height, 1000)

        let dst = try SourceDNG(url: out)
        XCTAssertEqual((dst.raw[256] ?? dst.ifd0[256])?.numericValues.first, 1024, "width in file")
        XCTAssertEqual((dst.raw[257] ?? dst.ifd0[257])?.numericValues.first, 1000, "height in file")
        XCTAssertEqual((dst.raw[33422] ?? dst.ifd0[33422])?.numericValues, [0, 1, 1, 2], "CFA")

        let back = try dst.readMosaic(data: try Data(contentsOf: out))
        XCTAssertEqual(back.width, 1024)
        XCTAssertEqual(back.height, 1000)
        XCTAssertEqual(back.samples.count, 1024 * 1000)
    }

    /// Cropping the right/bottom padding must not shift the mosaic. If it did, the CFA phase would
    /// flip and every R sample would come back as B — a checkerboard that no tag comparison sees.
    ///
    /// The check is an *assignment*, not a per-phase tolerance: each output phase is matched to the
    /// input phase it is closest to, and that mapping must be the identity. A tolerance alone is
    /// not enough, because the network legitimately pulls neighbouring phases toward each other
    /// (measured here: 2400 → 2563 on G1, 2800 → 2654 on G2, a 7 % convergence on a synthetic
    /// 2-px checkerboard it partly removes). That convergence is smaller than the gap between
    /// phases, so the assignment is stable under it and still flips under a shift.
    func testCFAPhaseIsPreservedThroughPaddingAndCrop() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("phase")

        // 90×90: neither dimension is a multiple of 16 (90 % 16 == 10, so 6 px of padding each
        // way, centre 3). Odd half-padding is deliberate — that is the case where a *centred* pad
        // moves the mosaic by an odd number of rows and columns and swaps R for B. Dimensions whose
        // half-padding is even (e.g. 1024×1000, centre 4) hide that bug completely, which is why
        // this test does not reuse them.
        let w = 90, h = 90
        var samples = [UInt16](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let phase = (y % 2) * 2 + (x % 2)
                samples[y * w + x] = UInt16(2000 + phase * 400)      // a distinct level per phase
            }
        }
        let src = try Self.makeDNG(samples: samples, width: w, height: h,
                                   like: Self.fixtureURL, in: dir, named: "phase.dng")
        let out = dir.appendingPathComponent("phase-out.dng")
        _ = try DenoiseDNGPipeline.run(.init(
            model: model, metal: Self.metal,
            input: src, output: out,
            a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
            strength: 0.85, budgetBytes: 1_000_000_000))

        let decoded = try SourceDNG(url: out).readMosaic(data: try Data(contentsOf: out))
        XCTAssertEqual(decoded.width, w)
        XCTAssertEqual(decoded.height, h)

        func phaseMeans(_ s: [UInt16]) -> [Double] {
            var totals = [Double](repeating: 0, count: 4)
            var counts = [Int](repeating: 0, count: 4)
            for y in 0..<h {
                for x in 0..<w {
                    let p = (y % 2) * 2 + (x % 2)
                    totals[p] += Double(s[y * w + x])
                    counts[p] += 1
                }
            }
            return (0..<4).map { totals[$0] / Double(max(counts[$0], 1)) }
        }

        let before = phaseMeans(samples), after = phaseMeans(decoded.samples)
        var mapping: [Int] = []
        var worstCrosstalk = 0.0
        for p in 0..<4 {
            let best = (0..<4).min { abs(after[p] - before[$0]) < abs(after[p] - before[$1]) }!
            mapping.append(best)
            let right = abs(after[p] - before[p])
            let wrong = (0..<4).filter { $0 != p }.map { abs(after[p] - before[$0]) }.min()!
            if wrong > 0 { worstCrosstalk = max(worstCrosstalk, 1 - right / wrong) }
        }
        print("phase means before \(before.map { Int($0) }) after \(after.map { Int($0) })")
        print("phase assignment \(mapping) (must be [0, 1, 2, 3]); "
              + String(format: "margin %.2f", worstCrosstalk))

        XCTAssertEqual(mapping, [0, 1, 2, 3],
                       "each output phase must be closest to its own input phase — "
                       + "a one-pixel shift reassigns them")
        XCTAssertGreaterThan(worstCrosstalk, 0.3,
                             "the assignment must be decided by a clear margin, not by noise")
    }

    // MARK: - ② a caller-supplied size that disagrees with the file

    func testDisagreeingWidthFlagIsAnErrorNotASilentChoice() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        guard let fixture = Self.fixtureURL else { throw XCTSkip("fixture not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("dimmismatch")
        let out = dir.appendingPathComponent("never.dng")

        XCTAssertThrowsError(try DenoiseDNGPipeline.run(.init(
            model: model, metal: nil,
            input: fixture, output: out,
            a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
            expectedWidth: 1024, expectedHeight: 768))) { error in
            guard case DenoiseDNGError.dimensionMismatch = error else {
                return XCTFail("expected dimensionMismatch, got \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: out.path))
    }

    // MARK: - ⑥ the float32 path still works

    /// The old contract: `image` on `.f32` reads and writes a raw float32 mosaic. Photons' current
    /// callers and `bench` depend on it, so the DNG work must not have changed it.
    func testFloat32PathIsUnchanged() throws {
        guard let modelURL = Self.modelURL else { throw XCTSkip("model not present") }
        let model = try AnselDenoiseModel(contentsOf: modelURL)
        let dir = try scratch("f32")
        let w = 64, h = 64
        var mosaic = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) { mosaic[i] = 0.02 + Float(i % 17) * 0.001 }
        let inURL = dir.appendingPathComponent("in.f32")
        try mosaic.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: inURL)
        let outURL = dir.appendingPathComponent("out.f32")

        let result = try DenoiseImageRunner.run(
            DenoiseImageRequest(input: inURL, output: outURL, width: w, height: h,
                                a: [0.0032, 0.0047, 0.0035], b: [2.4e-5, 1.6e-5, 2.4e-5],
                                cfa: "rggb", useGPU: Self.metal != nil),
            model: model, metal: Self.metal)

        let data = try Data(contentsOf: outURL)
        XCTAssertEqual(data.count, w * h * MemoryLayout<Float>.size,
                       "the float32 path must still write exactly width×height floats")
        var readBack = [Float](repeating: 0, count: w * h)
        readBack.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        XCTAssertTrue(readBack.allSatisfy { $0.isFinite })
        XCTAssertEqual(readBack, result.mosaic, "the file and the returned array must agree")
        // Not a copy of the input: the network must have done something.
        XCTAssertNotEqual(readBack, mosaic)
    }

    // MARK: - helpers

    /// Write a new DNG carrying these samples, taking every tag from `like`.
    ///
    /// `DNGRawWriter` copies `ActiveArea` and `DefaultCropSize` from the source, so a synthesised
    /// file of a different size would claim the source's geometry. Those two are rewritten here so
    /// the fixture is a *self-consistent* DNG — otherwise a later comparison would be checking two
    /// files that are both slightly wrong in the same way.
    static func makeDNG(samples: [UInt16], width: Int, height: Int,
                        like source: URL?, in dir: URL, named name: String) throws -> URL {
        guard let source else { throw XCTSkip("a template DNG is required to synthesise one") }
        let url = dir.appendingPathComponent(name)
        _ = try DNGRawWriter.write(samples: samples, width: width, height: height,
                                   from: source, to: url)
        try patchGeometryTags(at: url, width: width, height: height)
        return url
    }

    /// Rewrite `ActiveArea` (4 LONGs) and `DefaultCropSize` (2 SHORTs) to a new extent, in place.
    ///
    /// In place on purpose: both tags keep their type and count, so no offset in the file changes.
    /// Rebuilding the IFD to change a value's *size* would mean re-laying-out every tile.
    static func patchGeometryTags(at url: URL, width: Int, height: Int) throws {
        var data = try Data(contentsOf: url)
        func put32(_ at: Int, _ v: UInt32) {
            for i in 0..<4 { data[at + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
        }
        func put16(_ at: Int, _ v: UInt16) {
            data[at] = UInt8(v & 0xFF)
            data[at + 1] = UInt8(v >> 8)
        }
        if let p = payloadOffset(of: 50829, in: data) {
            put32(p + 0, 0)                       // top
            put32(p + 4, 0)                       // left
            put32(p + 8, UInt32(height))          // bottom
            put32(p + 12, UInt32(width))          // right
        }
        if let p = payloadOffset(of: 50720, in: data) {
            put16(p + 0, UInt16(truncatingIfNeeded: width))
            put16(p + 2, UInt16(truncatingIfNeeded: height))
        }
        try data.write(to: url)
    }

    /// Rewrite tag 33422 in place, so a fixture can serve as a different CFA without re-encoding.
    static func rewriteCFA(of source: URL, to values: [UInt8], at destination: URL) throws {
        var data = try Data(contentsOf: source)
        guard let offset = entryOffset(of: 33422, in: data) else {
            throw XCTSkip("33422 is not in IFD0; cannot rewrite it in place")
        }
        for (i, v) in values.enumerated() { data[offset + i] = v }
        try data.write(to: destination)
    }

    /// Where a tag's value lives: the entry's inline field, or the payload it points at.
    ///
    /// Both are "just bytes" — a 4-byte value sits in the entry, anything longer sits elsewhere and
    /// the entry holds a pointer. Since every tag this file patches keeps its type and count, the
    /// returned offset stays valid after a write.
    static func entryOffset(of tag: UInt16, in data: Data) -> Int? {
        guard let entry = findEntry(tag, in: data) else { return nil }
        return entry.size <= 4 ? entry.valueField : entry.payload
    }

    static func payloadOffset(of tag: UInt16, in data: Data) -> Int? {
        guard let entry = findEntry(tag, in: data) else { return nil }
        return entry.size <= 4 ? entry.valueField : entry.payload
    }

    struct EntryLocation {
        let type: UInt16
        let count: Int
        let size: Int
        let valueField: Int
        let payload: Int
    }

    static func findEntry(_ tag: UInt16, in data: Data) -> EntryLocation? {
        func u16(_ o: Int) -> Int { Int(data[o]) | (Int(data[o + 1]) << 8) }
        func u32(_ o: Int) -> Int {
            Int(data[o]) | (Int(data[o + 1]) << 8) | (Int(data[o + 2]) << 16) | (Int(data[o + 3]) << 24)
        }
        let ifd0 = u32(4)
        let count = u16(ifd0)
        for i in 0..<count {
            let base = ifd0 + 2 + 12 * i
            guard u16(base) == Int(tag) else { continue }
            let type = u16(base + 2)
            let n = u32(base + 4)
            let size = componentSize(UInt16(type)) * n
            return EntryLocation(type: UInt16(type), count: n, size: size,
                                 valueField: base + 8,
                                 payload: size <= 4 ? base + 8 : u32(base + 8))
        }
        return nil
    }

    static func componentSize(_ type: UInt16) -> Int {
        switch type {
        case 1, 2, 6, 7: return 1
        case 3, 8: return 2
        case 4, 9, 13: return 4
        case 5, 10, 12: return 8
        case 11: return 4
        default: return 1
        }
    }
}
