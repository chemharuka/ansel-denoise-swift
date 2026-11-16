//
//  AnselDenoiseMetal.swift
//  AnselDenoise
//
//  UNet forward on the GPU, mirroring AnselDenoiseForward exactly.
//  GPLv3 component — see LICENSE-SCOPE.md.
//
//  Same graph, same weights, same residual: only the executor changes. Kept as a
//  separate file so the CPU path stays available for parity testing — a GPU kernel
//  that is fast and wrong is worse than a slow one that is right.
//

import Foundation
import Metal

public final class AnselDenoiseMetal {
    let model: AnselDenoiseModel
    let cfg: AnselModelConfig
    let metal: MetalConv

    /// Convolution weights packed once per layer, keyed by the layer's tensor name.
    ///
    /// Owned by this instance and this instance's model: two models with colliding
    /// layer names cannot see each other's packed weights, because there is no
    /// process-wide cache to collide in.
    private var preparedLayers: [String: MetalConv.PreparedConvolution] = [:]
    /// `preparedLayers` is mutable state reachable from any thread that calls
    /// `denoise`, so every access is behind this lock.
    private let preparedLock = NSLock()

    /// Number of layers whose weights are currently packed. Grows on the first
    /// inference and stays put after that.
    var preparedLayerCount: Int {
        preparedLock.lock()
        defer { preparedLock.unlock() }
        return preparedLayers.count
    }

    public init(model: AnselDenoiseModel, metal: MetalConv) {
        self.model = model
        self.cfg = model.config
        self.metal = metal
    }

    /// Convenience: create the Metal engine for an already-loaded model.
    public convenience init(model: AnselDenoiseModel) throws {
        try self.init(model: model, metal: try MetalConv())
    }

    public var sizeMultiple: Int { 1 << cfg.depth }

    /// Denoise one tile on the GPU.
    ///
    /// - Parameters:
    ///   - mosaic: normalized (black-subtracted, white-normalized) mosaic, row-major.
    ///   - cfa: 4-char row-major CFA string describing the tile's top-left 2×2.
    ///   - a,b: per-channel Poisson-Gaussian coefficients, `Var = a·x + b`.
    /// - Returns: denoised mosaic, row-major, same size as `mosaic`.
    public func denoise(mosaic: [Float], width: Int, height: Int,
                        cfa: String, a: [Double], b: [Double]) throws -> [Float] {
        precondition(mosaic.count == width * height)
        let inC = cfg.inChannels
        let color = AnselDenoiseForward.colorIndexMap(pattern: cfa, width: width, height: height)

        // Build the 5 input planes, channels-last: [mosaic, R, G, B, sigma].
        var planes = [Float](repeating: 0, count: width * height * inC)
        for i in 0..<(width * height) {
            let c = color[i]
            let v = mosaic[i]
            let base = i * inC
            planes[base] = v
            planes[base + 1] = (c == 0) ? 1 : 0
            planes[base + 2] = (c == 1) ? 1 : 0
            planes[base + 3] = (c == 2) ? 1 : 0
            let varEst = a[c] * max(Double(v), 0) + b[c]
            planes[base + 4] = Float((varEst > 1e-12 ? varEst : 1e-12).squareRoot())
        }

        guard let cmd = metal.queue.makeCommandBuffer() else {
            throw MetalConvError.commandFailed("command buffer")
        }

        var h = try metal.makeTensor(width: width, height: height, channels: inC, from: planes)
        let signal = try metal.makeTensor(width: width, height: height,
                                          channels: cfg.outChannels,
                                          from: channelSlice(planes, width: width,
                                                             height: height, inC: inC,
                                                             outC: cfg.outChannels))

        let depth = cfg.depth
        var skips: [MetalConv.Tensor] = []
        skips.reserveCapacity(depth)

        for i in 0..<depth {
            h = try block(h, "enc.\(i).0.weight", "enc.\(i).0.bias",
                          "enc.\(i).2.weight", "enc.\(i).2.bias", cmd: cmd)
            skips.append(h)
            h = try convolution(h, weight: "down.\(i).weight", bias: "down.\(i).bias",
                                kind: .down2x2, cmd: cmd)
        }

        h = try block(h, "bottleneck.0.weight", "bottleneck.0.bias",
                      "bottleneck.2.weight", "bottleneck.2.bias", cmd: cmd)

        for i in 0..<depth {
            h = try metal.upsampleNearest(h, cmd: cmd)
            h = try convolution(h, weight: "up.\(i).weight", bias: "up.\(i).bias",
                                kind: .point1x1, cmd: cmd)
            let skip = skips[depth - 1 - i]
            precondition(skip.width == h.width && skip.height == h.height,
                         "skip \(i) \(skip.width)x\(skip.height) vs decoder \(h.width)x\(h.height)")
            h = try metal.concat(skip, h, cmd: cmd)
            h = try block(h, "dec.\(i).0.weight", "dec.\(i).0.bias",
                          "dec.\(i).2.weight", "dec.\(i).2.bias", cmd: cmd)
        }

        let noise = try convolution(h, weight: "head.weight", bias: "head.bias",
                                    kind: .same3x3, cmd: cmd)

        // Residual: output = signal − predicted noise.
        let out = try metal.subtract(signal, noise, cmd: cmd)

        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw MetalConvError.commandFailed(String(describing: err))
        }
        return metal.read(out)
    }

    /// Extract the first `outC` planes from a channels-last buffer into planar.
    private func channelSlice(_ planes: [Float], width: Int, height: Int,
                              inC: Int, outC: Int) -> [Float] {
        let n = width * height
        var slice = [Float](repeating: 0, count: n * outC)
        for c in 0..<outC {
            for i in 0..<n {
                slice[c * n + i] = planes[i * inC + c]
            }
        }
        return slice
    }

    /// Packed weights for one layer, packing on first use.
    ///
    /// Double-checked under the lock: the pack itself happens outside it so a
    /// second thread that arrives mid-pack does not block on GPU buffer allocation
    /// it does not need.
    private func preparedLayer(weight: String, bias: String) throws -> MetalConv.PreparedConvolution {
        preparedLock.lock()
        let cached = preparedLayers[weight]
        preparedLock.unlock()
        if let cached { return cached }

        let prepared = try metal.prepare(w: try model.weights(weight),
                                         b: model.tensors[bias]?.values)

        preparedLock.lock()
        // A concurrent call may have packed the same layer while we were packing;
        // keep whichever landed first so every caller uses one set of buffers.
        let existing = preparedLayers[weight]
        if existing == nil { preparedLayers[weight] = prepared }
        let result = existing ?? prepared
        preparedLock.unlock()
        return result
    }

    /// Convolve using this model's packed weights for `weight`/`bias`.
    private func convolution(_ input: MetalConv.Tensor, weight: String, bias: String,
                             kind: MetalConv.ConvKind,
                             cmd: MTLCommandBuffer) throws -> MetalConv.Tensor {
        try metal.conv(input, prepared: try preparedLayer(weight: weight, bias: bias),
                       kind: kind, cmd: cmd)
    }

    private func block(_ x: MetalConv.Tensor,
                       _ w0: String, _ b0: String,
                       _ w2: String, _ b2: String,
                       cmd: MTLCommandBuffer) throws -> MetalConv.Tensor {
        var h = try convolution(x, weight: w0, bias: b0, kind: .same3x3, cmd: cmd)
        try metal.geluInPlace(h, cmd: cmd)
        h = try convolution(h, weight: w2, bias: b2, kind: .same3x3, cmd: cmd)
        try metal.geluInPlace(h, cmd: cmd)
        return h
    }
}
