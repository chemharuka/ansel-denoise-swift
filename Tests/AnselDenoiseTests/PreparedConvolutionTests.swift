//
//  PreparedConvolutionTests.swift
//  AnselDenoise
//
//  GPLv3 component — see LICENSE-SCOPE.md.
//
//  What these tests are for. `MetalConv.prepare` exists so a layer's weights are
//  uploaded to the GPU once instead of once per convolution call, and
//  `AnselDenoiseMetal` caches those prepared layers. Three distinct things can go
//  wrong, and each needs its own assertion:
//
//    1. the cached path computes something different from the one-shot path
//       (correctness — `testPreparedConvolutionEqualsUnpreparedOutput`);
//    2. preparing twice has side effects, so "prepared" is not a pure function of
//       the weights (`testPreparingTwiceIsIdempotent`);
//    3. nothing is actually cached, or the cache is global and two models share
//       their packed weights (`testPreparedLayerCountMatchesTheModelLayerCount`,
//       `testTwoDifferentModelsDoNotSharePreparedLayers`).
//
//  (1) alone would pass even if every `conv` call re-packed the weights, so it
//  cannot show that the optimisation took effect. (3) and (4) are the tests that
//  would fail if the cache were removed or made process-wide.
//

import XCTest
import Metal
@testable import AnselDenoise

final class PreparedConvolutionTests: XCTestCase {

    // MARK: - fixtures

    private func makeMetal() throws -> MetalConv {
        do { return try MetalConv() }
        catch { throw XCTSkip("Metal unavailable on this machine: \(error)") }
    }

    /// The default model is committed to the repo, so these tests need no download.
    /// `ANSEL_DENOISE_MODEL` overrides it; the fallback walks from this file to the
    /// package root so the test works regardless of the process's cwd.
    static var packagedModelURL: URL? {
        if let p = ProcessInfo.processInfo.environment["ANSEL_DENOISE_MODEL"],
           FileManager.default.fileExists(atPath: p) {
            return URL(fileURLWithPath: p)
        }
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()    // AnselDenoiseTests
            .deletingLastPathComponent()    // Tests
            .deletingLastPathComponent()    // package root
        let candidate = root.appendingPathComponent("weights/denoise-quarter-single-v1.anselnn")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    /// Deterministic pseudo-random floats in [-0.5, 0.5), so a failure reproduces.
    private static func pseudoRandom(count: Int, seed: UInt64) -> [Float] {
        var s = seed
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            out[i] = Float(s >> 33) / Float(1 << 31) - 0.5
        }
        return out
    }

    private func runConv(_ metal: MetalConv, input: MetalConv.Tensor,
                         prepared: MetalConv.PreparedConvolution,
                         kind: MetalConv.ConvKind) throws -> [Float] {
        guard let cmd = metal.queue.makeCommandBuffer() else {
            throw MetalConvError.commandFailed("command buffer")
        }
        let out = try metal.conv(input, prepared: prepared, kind: kind, cmd: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw MetalConvError.commandFailed(String(describing: err)) }
        return metal.read(out)
    }

    private func runUnpreparedConv(_ metal: MetalConv, input: MetalConv.Tensor,
                                   w: AnselWeights, b: [Float]?,
                                   kind: MetalConv.ConvKind) throws -> [Float] {
        guard let cmd = metal.queue.makeCommandBuffer() else {
            throw MetalConvError.commandFailed("command buffer")
        }
        let out = try metal.conv(input, w: w, b: b, kind: kind, cmd: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { throw MetalConvError.commandFailed(String(describing: err)) }
        return metal.read(out)
    }

    private func denoise(_ engine: AnselDenoiseMetal, tile: Int, seed: UInt64 = 7) throws -> [Float] {
        let mosaic = AnselBench.syntheticMosaic(tile: tile, seed: seed)
        return try engine.denoise(mosaic: mosaic, width: tile, height: tile, cfa: "rggb",
                                 a: [0.0004, 0.0003, 0.0004], b: [3e-6, 1e-6, 1e-6])
    }

    private func assertBitwiseClose(_ a: [Float], _ b: [Float],
                                    tolerance: Float, _ what: String,
                                    file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "\(what): element count differs", file: file, line: line)
        var maxDiff: Float = 0
        for i in 0..<min(a.count, b.count) {
            maxDiff = max(maxDiff, abs(a[i] - b[i]))
        }
        XCTAssertLessThan(maxDiff, tolerance,
                          "\(what): max |prepared - reference| = \(maxDiff)", file: file, line: line)
    }

    // MARK: - 1. the core invariant: prepared == unprepared

    /// Channels-last `(H, W, C)` → planar `(C, H, W)`, the layout the CPU reference uses.
    private func planar(fromChannelsLast v: [Float], width: Int, height: Int,
                        channels: Int) -> PlanarImage {
        var data = [Float](repeating: 0, count: v.count)
        let n = width * height
        for c in 0..<channels {
            for i in 0..<n { data[c * n + i] = v[i * channels + c] }
        }
        return PlanarImage(width: width, height: height, channels: channels, data: data)
    }

    private func channelsLast(from img: PlanarImage) -> [Float] {
        var out = [Float](repeating: 0, count: img.data.count)
        let n = img.width * img.height
        for c in 0..<img.channels {
            for i in 0..<n { out[i * img.channels + c] = img.data[c * n + i] }
        }
        return out
    }

    /// Independent CPU implementation of the same conv, used as an anchor.
    ///
    /// This is what makes the parity test meaningful: both the prepared and the
    /// one-shot GPU paths go through `prepare`, so comparing them to each other
    /// alone would still pass if `prepare` packed the wrong numbers. Comparing
    /// against this reference catches that.
    private func cpuConvReference(_ input: [Float], width: Int, height: Int, inC: Int,
                                  w: AnselWeights, b: [Float]?,
                                  kind: MetalConv.ConvKind) -> [Float] {
        let padding: (Int, Int)
        let stride: Int
        switch kind {
        case .same3x3: padding = (1, 1); stride = 1
        case .point1x1: padding = (0, 0); stride = 1
        case .down2x2: padding = (0, 0); stride = 2
        }
        let out = AnselForward.conv(planar(fromChannelsLast: input, width: width, height: height,
                                           channels: inC),
                                    w, b, padding: padding, stride: stride)
        return channelsLast(from: out)
    }

    /// Packing weights into a different buffer must not change a single output
    /// value — not for 3×3, not for the 1×1, not for the stride-2 downsample.
    func testPreparedConvolutionEqualsUnpreparedOutput() throws {
        let metal = try makeMetal()

        // (outC, inC, kh, kw) and the ConvKind that matches the kernel extent.
        let cases: [(shape: [Int], kind: MetalConv.ConvKind)] = [
            ([3, 4, 3, 3], .same3x3),
            ([5, 4, 1, 1], .point1x1),
            ([4, 4, 2, 2], .down2x2),
        ]

        for (idx, c) in cases.enumerated() {
            let inC = c.shape[1], outC = c.shape[0]
            let w = AnselWeights(name: "t\(idx)",
                                 shape: c.shape,
                                 values: Self.pseudoRandom(count: c.shape.reduce(1, *),
                                                           seed: UInt64(1000 + idx)))
            let b = Self.pseudoRandom(count: outC, seed: UInt64(2000 + idx))
            let inputValues = Self.pseudoRandom(count: 12 * 10 * inC, seed: UInt64(3000 + idx))
            let input = try metal.makeTensor(width: 12, height: 10, channels: inC,
                                             from: inputValues)

            let prepared = try metal.prepare(w: w, b: b)
            let unprepared = try runUnpreparedConv(metal, input: input, w: w, b: b, kind: c.kind)
            let fromPrepared = try runConv(metal, input: input, prepared: prepared, kind: c.kind)

            XCTAssertFalse(unprepared.isEmpty, "case \(idx) produced no output")
            assertBitwiseClose(fromPrepared, unprepared, tolerance: 1e-5,
                               "case \(idx) \(c.shape) prepared vs unprepared")

            // Anchor both against the CPU: catches a `prepare` that drops or
            // reorders weights while staying self-consistent.
            let reference = cpuConvReference(inputValues, width: 12, height: 10, inC: inC,
                                             w: w, b: b, kind: c.kind)
            assertBitwiseClose(fromPrepared, reference, tolerance: 1e-5,
                               "case \(idx) \(c.shape) prepared vs CPU reference")
        }
    }

    /// A bias-free layer must stay bias-free through the prepared path: a cached
    /// `hasBias` that leaked across layers would show up here and nowhere else.
    func testPreparedConvolutionWithoutBiasMatchesUnprepared() throws {
        let metal = try makeMetal()
        let w = AnselWeights(name: "nobias", shape: [2, 3, 3, 3],
                             values: Self.pseudoRandom(count: 2 * 3 * 3 * 3, seed: 11))
        let input = try metal.makeTensor(width: 6, height: 6, channels: 3,
                                         from: Self.pseudoRandom(count: 6 * 6 * 3, seed: 12))

        let prepared = try metal.prepare(w: w, b: nil)
        let unprepared = try runUnpreparedConv(metal, input: input, w: w, b: nil, kind: .same3x3)
        let fromPrepared = try runConv(metal, input: input, prepared: prepared, kind: .same3x3)
        assertBitwiseClose(fromPrepared, unprepared, tolerance: 1e-5, "bias-free layer")
        assertBitwiseClose(fromPrepared,
                           cpuConvReference(Self.pseudoRandom(count: 6 * 6 * 3, seed: 12),
                                            width: 6, height: 6, inC: 3,
                                            w: w, b: nil, kind: .same3x3),
                           tolerance: 1e-5, "bias-free layer vs CPU reference")
    }

    // MARK: - 2. preparing twice is a pure operation

    /// Two independent prepares of the same weights must both be usable and must
    /// agree. A prepare that mutated shared state would make the second call differ.
    func testPreparingTwiceIsIdempotent() throws {
        let metal = try makeMetal()
        let w = AnselWeights(name: "twice", shape: [4, 6, 3, 3],
                             values: Self.pseudoRandom(count: 4 * 6 * 3 * 3, seed: 21))
        let b = Self.pseudoRandom(count: 4, seed: 22)
        let input = try metal.makeTensor(width: 9, height: 9, channels: 6,
                                         from: Self.pseudoRandom(count: 9 * 9 * 6, seed: 23))

        let first = try metal.prepare(w: w, b: b)
        let second = try metal.prepare(w: w, b: b)

        let outFirst = try runConv(metal, input: input, prepared: first, kind: .same3x3)
        let outSecond = try runConv(metal, input: input, prepared: second, kind: .same3x3)
        assertBitwiseClose(outSecond, outFirst, tolerance: 1e-6, "second prepare")

        // Interleaving must not matter either: reuse `first` after `second` ran.
        let outFirstAgain = try runConv(metal, input: input, prepared: first, kind: .same3x3)
        assertBitwiseClose(outFirstAgain, outFirst, tolerance: 1e-6, "first prepare reused after second")

        // And both must match the one-shot path, so neither prepare is a no-op that
        // merely happens to agree with itself.
        let reference = try runUnpreparedConv(metal, input: input, w: w, b: b, kind: .same3x3)
        assertBitwiseClose(outSecond, reference, tolerance: 1e-5, "prepared vs unprepared")
    }

    // MARK: - 3. the cache actually caches (the discriminating test)

    /// One inference must pack every convolution layer exactly once, and a second
    /// inference must pack nothing at all. Without the cache this count would be
    /// zero; with a broken key it would exceed the layer count.
    func testPreparedLayerCountMatchesTheModelLayerCount() throws {
        let url = try XCTUnwrap(Self.packagedModelURL,
                                "default model weights/denoise-quarter-single-v1.anselnn missing")
        let model = try AnselDenoiseModel(contentsOf: url)
        let metal = try makeMetal()
        let engine = AnselDenoiseMetal(model: model, metal: metal)

        let depth = model.config.depth
        // enc .0 + .2, down, bottleneck .0 + .2, up, dec .0 + .2, head.
        let expected = 6 * depth + 3
        let convWeights = model.tensors.keys.filter { $0.hasSuffix(".weight") }.count
        XCTAssertEqual(convWeights, expected,
                       "model has \(convWeights) conv weights, the UNet layout implies \(expected)")

        XCTAssertEqual(engine.preparedLayerCount, 0,
                       "the cache must start empty — nothing is packed before inference")

        _ = try denoise(engine, tile: 32)
        XCTAssertEqual(engine.preparedLayerCount, expected,
                       "after one inference every conv layer must be packed exactly once")

        // Second pass over a *different* tile size: same layers, no new packing.
        _ = try denoise(engine, tile: 48, seed: 99)
        XCTAssertEqual(engine.preparedLayerCount, expected,
                       "a second inference must not pack anything again")

        // Third pass, to rule out "counts twice then settles".
        _ = try denoise(engine, tile: 16, seed: 123)
        XCTAssertEqual(engine.preparedLayerCount, expected,
                       "layer count must be stable across repeated inference")
    }

    // MARK: - 4. ownership is per instance, not per process

    /// Two models whose layers have identical names but different weights must not
    /// share packed buffers. A process-wide name-keyed cache would make model B
    /// silently compute model A's output, and this test would fail on the parity
    /// assertion below.
    func testTwoDifferentModelsDoNotSharePreparedLayers() throws {
        let url = try XCTUnwrap(Self.packagedModelURL,
                                "default model weights/denoise-quarter-single-v1.anselnn missing")

        // Model B is the committed model with the head rescaled: same tensor names,
        // same shapes, different numbers, so outputs must differ.
        let perturbed = try Self.rescaledModel(contentsOf: url,
                                               tensors: ["head.weight", "head.bias"],
                                               scale: 1.7)
        let modelA = try AnselDenoiseModel(contentsOf: url)
        let modelB = try AnselDenoiseModel(data: perturbed)

        let metal = try makeMetal()
        let engineA = AnselDenoiseMetal(model: modelA, metal: metal)
        // A second MetalConv, so the two engines share no GPU engine state either —
        // the only thing that could leak between them is a global cache.
        let engineB = AnselDenoiseMetal(model: modelB, metal: try makeMetal())

        let outA = try denoise(engineA, tile: 32, seed: 5)
        let outB = try denoise(engineB, tile: 32, seed: 5)
        let outAAgain = try denoise(engineA, tile: 32, seed: 5)

        let expected = 6 * modelA.config.depth + 3
        XCTAssertEqual(engineA.preparedLayerCount, expected, "model A packed its own layers")
        XCTAssertEqual(engineB.preparedLayerCount, expected, "model B packed its own layers")
        XCTAssertEqual(engineA.preparedLayerCount + engineB.preparedLayerCount, 2 * expected,
                       "neither engine stole the other's packed weights")

        // Different weights must produce different output...
        var maxAB: Float = 0
        for i in 0..<min(outA.count, outB.count) {
            maxAB = max(maxAB, abs(outA[i] - outB[i]))
        }
        XCTAssertGreaterThan(maxAB, 1e-3,
                             "models with different head weights produced the same output — "
                             + "the prepared weights are being shared")

        // ...and each engine must still be deterministic across runs.
        assertBitwiseClose(outAAgain, outA, tolerance: 1e-6, "model A re-run")

        // Model B's GPU output must match model B's own CPU forward, not model A's.
        let cpuB = try Self.cpuForward(model: modelB, tile: 32, seed: 5)
        var maxBvsCpu: Float = 0
        for i in 0..<min(outB.count, cpuB.count) {
            maxBvsCpu = max(maxBvsCpu, abs(outB[i] - cpuB[i]))
        }
        XCTAssertLessThan(maxBvsCpu, 1e-4,
                          "model B's GPU output follows model A's weights "
                          + "(GPU vs model-B CPU forward max diff = \(maxBvsCpu))")

        // Finally: the same input through a *fresh* single-model engine must equal
        // engine A's output, proving nothing was mutated by B's presence.
        let engineAFresh = AnselDenoiseMetal(model: modelA, metal: try makeMetal())
        assertBitwiseClose(try denoise(engineAFresh, tile: 32, seed: 5), outA,
                           tolerance: 1e-6, "fresh engine on model A")
    }

    // MARK: - helpers for model B

    /// Rewrite the payload of an `.anselnn` with `scale` applied to the named
    /// tensors. Shapes are untouched, so every header offset stays valid.
    static func rescaledModel(contentsOf url: URL, tensors names: Set<String>,
                              scale: Float) throws -> Data {
        var bytes = [UInt8](try Data(contentsOf: url))
        let headerLen = bytes.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
        }
        let payloadStart = 12 + Int(headerLen)
        let header = try JSONSerialization.jsonObject(with: Data(bytes[12..<payloadStart]))
        guard let root = header as? [String: Any],
              let tensors = root["tensors"] as? [[String: Any]] else {
            throw AnselModelError.badJSON("no tensors array")
        }

        var touched = 0
        for t in tensors {
            guard let name = t["name"] as? String, names.contains(name),
                  let offset = t["offset"] as? Int,
                  let size = t["size"] as? Int else { continue }
            let start = payloadStart + offset
            for i in stride(from: 0, to: size, by: 4) {
                var v = bytes.withUnsafeBytes { raw -> Float in
                    raw.loadUnaligned(fromByteOffset: start + i, as: Float.self)
                }
                v *= scale
                withUnsafeBytes(of: &v) { src in
                    for k in 0..<4 { bytes[start + i + k] = src[k] }
                }
            }
            touched += 1
        }
        guard touched == names.count else {
            throw AnselModelError.tensorMissing("rescaled \(touched) of \(names.count) tensors")
        }
        return Data(bytes)
    }

    static func cpuForward(model: AnselDenoiseModel, tile: Int, seed: UInt64) throws -> [Float] {
        let mosaic = AnselBench.syntheticMosaic(tile: tile, seed: seed)
        let color = AnselDenoiseForward.colorIndexMap(pattern: "rggb", width: tile, height: tile)
        let planes = AnselDenoiseForward.makeInputPlanes(
            mosaic: mosaic, colorIndex: color, width: tile, height: tile,
            a: [0.0004, 0.0003, 0.0004], b: [3e-6, 1e-6, 1e-6])
        return try AnselDenoiseForward(model: model).denoise(planes).data
    }
}
