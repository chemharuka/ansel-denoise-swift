import XCTest
@testable import AnselDenoise

final class AnselDenoiseTests: XCTestCase {
    /// The real model, fetched from the ansel-denoise release manifest.
    /// Skip (not fail) when absent so the suite is usable without the download.
    static var modelURL: URL? {
        let candidates = [
            ProcessInfo.processInfo.environment["ANSEL_DENOISE_MODEL"],
            "/Users/luyao/workspace/ansel-probe/denoise-quarter-single-v1.anselnn",
        ].compactMap { $0 }
        for c in candidates where FileManager.default.fileExists(atPath: c) {
            return URL(fileURLWithPath: c)
        }
        return nil
    }

    func testHeaderParses() throws {
        guard let url = Self.modelURL else { throw XCTSkip("model not present") }
        let m = try AnselDenoiseModel(contentsOf: url)
        XCTAssertEqual(m.config.arch, "unet")
        XCTAssertEqual(m.config.depth, 4)
        XCTAssertEqual(m.config.inChannels, 5)
        XCTAssertEqual(m.config.outChannels, 1)
        // 54 tensors, 475_265 parameters for denoise-quarter-single-v1.
        XCTAssertEqual(m.tensorInfos.count, 54)
        XCTAssertEqual(m.parameterCount, 475_265)
    }

    func testRejectsForeignFile() {
        let junk = Data("NOTANSEL".utf8) + Data(repeating: 0, count: 64)
        XCTAssertThrowsError(try AnselDenoiseModel(data: junk))
    }

    func testGELUIsTheErfForm() {
        // nn.GELU() default: 0.5t(1+erf(t/√2))
        XCTAssertEqual(gelu(0), 0, accuracy: 1e-7)
        XCTAssertEqual(gelu(1), 0.8413447, accuracy: 1e-6)
        XCTAssertEqual(gelu(-1), -0.1586553, accuracy: 1e-6)
        // The tanh approximation would give 0.8411919 at t=1 — assert we are not it.
        XCTAssertGreaterThan(abs(gelu(1) - 0.8411919), 1e-4)
    }

    func testConvMatchesHandComputedTinyCase() {
        // 1×1×3×3 input, single 3×3 kernel of all ones, identity-ish weights.
        let input = PlanarImage(width: 3, height: 3, channels: 1, data: [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ])
        let w = AnselWeights(name: "t", shape: [1, 1, 3, 3],
                             values: [Float](repeating: 1, count: 9))
        let out = AnselForward.conv(input, w, nil, padding: (1, 1))
        // Zero-padded 3×3 sum at the centre = 1+2+3+4+5+6+7+8+9 = 45
        XCTAssertEqual(out.data[4], 45, accuracy: 1e-5)
        // Top-left corner only sees 2×2 of the input: 1+2+4+5 = 12
        XCTAssertEqual(out.data[0], 12, accuracy: 1e-5)
    }

    func testDownsampleHalvesSpatialSize() {
        // 2×2 stride-2 conv on an 8×8 map must give 4×4.
        let input = PlanarImage(width: 8, height: 8, channels: 2)
        let w = AnselWeights(name: "d", shape: [2, 2, 2, 2],
                             values: [Float](repeating: 0.25, count: 16))
        let out = AnselForward.downsample(input, w, nil)
        XCTAssertEqual(out.width, 4)
        XCTAssertEqual(out.height, 4)
        XCTAssertEqual(out.channels, 2)
    }

    func testUpsampleNearestDoublesSize() {
        let input = PlanarImage(width: 2, height: 2, channels: 1, data: [1, 2, 3, 4])
        let out = AnselForward.upsampleNearest(input, scale: 2)
        XCTAssertEqual(out.width, 4)
        XCTAssertEqual(out.height, 4)
        // Each source pixel becomes a 2×2 block.
        XCTAssertEqual(out.data[0], 1)
        XCTAssertEqual(out.data[1], 1)
        XCTAssertEqual(out.data[4], 1)
        XCTAssertEqual(out.data[5], 1)
        XCTAssertEqual(out.data[3], 2)
    }

    /// End-to-end shape and finiteness on the real model.
    func testFullForwardProducesFiniteOutputOfSameSize() throws {
        guard let url = Self.modelURL else { throw XCTSkip("model not present") }
        let m = try AnselDenoiseModel(contentsOf: url)
        let fwd = AnselDenoiseForward(model: m)

        let H = 32, W = 32
        var mosaic = [Float](repeating: 0, count: H * W)
        var seed: UInt64 = 12345
        for i in 0..<(H * W) {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            mosaic[i] = Float(seed >> 33) / Float(1 << 31) * 0.5
        }
        var color = [Int](repeating: 0, count: H * W)
        for y in 0..<H {
            for x in 0..<W {
                let even = (y % 2 == 0), evenCol = (x % 2 == 0)
                color[y * W + x] = even ? (evenCol ? 0 : 1) : (evenCol ? 1 : 2)
            }
        }
        let planes = AnselDenoiseForward.makeInputPlanes(
            mosaic: mosaic, colorIndex: color, width: W, height: H,
            a: [0.0012, 0.0008, 0.0016], b: [0.00008, 0.00006, 0.00010])
        // makeInputPlanes returns (5, plane-major); forward wants that layout already.
        let out = try fwd.denoise(planes)

        XCTAssertEqual(out.width, W)
        XCTAssertEqual(out.height, H)
        XCTAssertEqual(out.channels, 1)
        XCTAssertTrue(out.data.allSatisfy { $0.isFinite }, "output has non-finite values")
        // Denoising must not move values wildly away from the input.
        var maxAbsDiff: Float = 0
        for i in 0..<(H * W) {
            maxAbsDiff = max(maxAbsDiff, abs(out.data[i] - mosaic[i]))
        }
        XCTAssertLessThan(maxAbsDiff, 0.5, "output drifted too far from input")
    }
}
