//
//  AnselDenoiseForward.swift
//  AnselDenoise
//
//  The evaluator: 5-plane mosaic in, denoised mosaic out.
//  GPLv3 component — see LICENSE-SCOPE.md.
//
//  Operator vocabulary is deliberately minimal so a C/OpenCL port can match it
//  bit-for-bit (Ansel's own two executors agree to 1.79e-07):
//    conv3x3 / conv1x1 (no padding tricks, explicit zero pad), exact GELU,
//    nearest-neighbour 2x upsample, concat on the channel axis, residual subtract.
//  No normalization layers: a pixel's output must not depend on the rest of the
//  tile, otherwise tiling would change the result.
//

import Foundation

/// A planar image: `channels × height × width`, interleaved-free (channel-major).
///
/// Channel-major layout makes the 3×3 inner loop walk one plane at a time, which
/// is both cache-friendly and the exact order a C port would use.
public struct PlanarImage {
    public let width: Int
    public let height: Int
    public let channels: Int
    public var data: [Float]

    public init(width: Int, height: Int, channels: Int, data: [Float]) {
        precondition(data.count == width * height * channels)
        self.width = width
        self.height = height
        self.channels = channels
        self.data = data
    }

    public init(width: Int, height: Int, channels: Int) {
        self.init(width: width, height: height, channels: channels,
                  data: [Float](repeating: 0, count: width * height * channels))
    }

    @inline(__always)
    public func index(_ c: Int, _ y: Int, _ x: Int) -> Int {
        (c * height + y) * width + x
    }
}

/// Exact GELU, matching PyTorch's default (`nn.GELU()`), i.e. the erf form.
///
/// The tanh approximation differs by up to 2.3e-04 on the real activations of
/// this network — far above the 1.79e-07 parity budget, so it is not usable here.
@inline(__always)
func gelu(_ t: Float) -> Float {
    0.5 * t * (1.0 + erff(t * 0.7071067811865476))
}

enum AnselForward {
    /// 2-D convolution with explicit zero padding, channels-first.
    ///
    /// - Parameters:
    ///   - padding: `(kh/2, kw/2)` for "same" padding.
    static func conv(_ input: PlanarImage,
                     _ w: AnselWeights,
                     _ b: [Float]?,
                     padding: (Int, Int),
                     stride: Int = 1) -> PlanarImage {
        let (ph, pw) = padding
        let inC = input.channels
        let h = input.height
        let wdt = input.width
        let outC = w.outC
        let kh = w.kh, kw = w.kw
        precondition(w.inC == inC, "conv weight expects \(w.inC) channels, got \(inC)")

        let oh = (h + 2 * ph - kh) / stride + 1
        let ow = (wdt + 2 * pw - kw) / stride + 1
        var out = PlanarImage(width: ow, height: oh, channels: outC)

        input.data.withUnsafeBufferPointer { src in
            out.data.withUnsafeMutableBufferPointer { dst in
                w.values.withUnsafeBufferPointer { wv in
                    for oc in 0..<outC {
                        let bias = b?[oc] ?? 0
                        let wBase = oc * inC * kh * kw
                        let outPlane = oc * (oh * ow)
                        for y in 0..<oh {
                            for x in 0..<ow {
                                var acc: Float = bias
                                for ic in 0..<inC {
                                    let srcPlane = ic * (h * wdt)
                                    let wPlane = wBase + ic * kh * kw
                                    for ky in 0..<kh {
                                        let sy = y * stride + ky - ph
                                        if sy < 0 || sy >= h { continue }
                                        let rowBase = srcPlane + sy * wdt
                                        let wRow = wPlane + ky * kw
                                        for kx in 0..<kw {
                                            let sx = x * stride + kx - pw
                                            if sx < 0 || sx >= wdt { continue }
                                            acc += src[rowBase + sx] * wv[wRow + kx]
                                        }
                                    }
                                }
                                dst[outPlane + y * ow + x] = acc
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// `[conv3x3 → GELU] × 2`, the block every UNet level is built from.
    static func block(_ x: PlanarImage,
                      _ w0: AnselWeights, _ b0: [Float]?,
                      _ w2: AnselWeights, _ b2: [Float]?) -> PlanarImage {
        var h = conv(x, w0, b0, padding: (1, 1))
        for i in h.data.indices { h.data[i] = gelu(h.data[i]) }
        h = conv(h, w2, b2, padding: (1, 1))
        for i in h.data.indices { h.data[i] = gelu(h.data[i]) }
        return h
    }

    /// Learned 2×2 stride-2 downsampling (a plain stride-2 conv, no padding).
    static func downsample(_ x: PlanarImage, _ w: AnselWeights, _ b: [Float]?) -> PlanarImage {
        // PyTorch: nn.Conv2d(w, w, 2, stride=2) — 2×2 kernel, stride 2, no padding.
        conv(x, w, b, padding: (0, 0), stride: 2)
    }

    /// Nearest-neighbour 2× upsample.
    static func upsampleNearest(_ x: PlanarImage, scale: Int = 2) -> PlanarImage {
        let oh = x.height * scale, ow = x.width * scale
        var out = PlanarImage(width: ow, height: oh, channels: x.channels)
        x.data.withUnsafeBufferPointer { src in
            out.data.withUnsafeMutableBufferPointer { dst in
                for c in 0..<x.channels {
                    let sPlane = c * (x.height * x.width)
                    let dPlane = c * (oh * ow)
                    for y in 0..<oh {
                        let sy = y / scale
                        for xx in 0..<ow {
                            dst[dPlane + y * ow + xx] = src[sPlane + sy * x.width + xx / scale]
                        }
                    }
                }
            }
        }
        return out
    }

    /// Concatenate two planar images on the channel axis (skip connections).
    static func concat(_ a: PlanarImage, _ b: PlanarImage) -> PlanarImage {
        precondition(a.width == b.width && a.height == b.height)
        var out = PlanarImage(width: a.width, height: a.height, channels: a.channels + b.channels)
        let planeLen = a.width * a.height
        out.data.replaceSubrange(0..<(a.channels * planeLen), with: a.data)
        out.data.replaceSubrange((a.channels * planeLen)..<out.data.count, with: b.data)
        return out
    }
}

/// Runs the UNet described by an `.anselnn` file.
public final class AnselDenoiseForward {
    let model: AnselDenoiseModel
    let cfg: AnselModelConfig

    public init(model: AnselDenoiseModel) {
        self.model = model
        self.cfg = model.config
    }

    /// Minimum tile size: the UNet halves resolution `depth` times.
    public var sizeMultiple: Int { 1 << cfg.depth }

    /// Tile overlap implied by the measured receptive field.
    ///
    /// Ansel documents the trained network's impulse response decaying to ~1e-6 of
    /// peak at r=32 and to zero by 96, and pads tiles by 48. Using a smaller overlap
    /// shows up as faint seams that a single-tile test will never reveal.
    public var overlap: Int { 48 }

    /// Denoise one tile.
    ///
    /// - Parameter planes: exactly 5 planes in the order
    ///   `[mosaic, R one-hot, G one-hot, B one-hot, sigma]`, same width/height,
    ///   values already black-subtracted and white-normalised.
    /// - Returns: the denoised mosaic (one plane), same size.
    public func denoise(_ planes: PlanarImage) throws -> PlanarImage {
        precondition(planes.channels == cfg.inChannels,
                     "expected \(cfg.inChannels) input planes, got \(planes.channels)")

        let depth = cfg.depth
        let widths = (0...depth).map { cfg.base << $0 }   // base·2^i
        var h = planes
        var skips: [PlanarImage] = []
        skips.reserveCapacity(depth)

        // Encoder: block, then learned /2. The last level feeds the bottleneck.
        for i in 0..<depth {
            h = AnselConv.block(h,
                                   try model.weights("enc.\(i).0.weight"),
                                   model.tensors["enc.\(i).0.bias"]?.values,
                                   try model.weights("enc.\(i).2.weight"),
                                   model.tensors["enc.\(i).2.bias"]?.values)
            skips.append(h)
            h = AnselConv.conv(h,
                               try model.weights("down.\(i).weight"),
                               model.tensors["down.\(i).bias"]?.values,
                               padding: (0, 0), stride: 2)
        }

        h = AnselConv.block(h,
                            try model.weights("bottleneck.0.weight"),
                               model.tensors["bottleneck.0.bias"]?.values,
                               try model.weights("bottleneck.2.weight"),
                               model.tensors["bottleneck.2.bias"]?.values)

        // Decoder: 2× nearest up, 1×1 conv to halve width, concat the skip, block.
        for i in 0..<depth {
            h = AnselForward.upsampleNearest(h, scale: 2)
            h = AnselConv.conv(h,
                               try model.weights("up.\(i).weight"),
                                  model.tensors["up.\(i).bias"]?.values,
                                  padding: (0, 0))
            let skip = skips[depth - 1 - i]
            precondition(skip.width == h.width && skip.height == h.height,
                         "skip \(i) is \(skip.width)x\(skip.height), decoder is \(h.width)x\(h.height)")
            h = AnselForward.concat(skip, h)
            h = AnselConv.block(h,
                                   try model.weights("dec.\(i).0.weight"),
                                   model.tensors["dec.\(i).0.bias"]?.values,
                                   try model.weights("dec.\(i).2.weight"),
                                   model.tensors["dec.\(i).2.bias"]?.values)
        }

        let noise = AnselConv.conv(h,
                                   try model.weights("head.weight"),
                                      model.tensors["head.bias"]?.values,
                                      padding: (1, 1))

        // Residual: output = noisy mosaic − predicted noise (first out_channels planes).
        var out = PlanarImage(width: planes.width, height: planes.height,
                              channels: cfg.outChannels)
        let planeLen = planes.width * planes.height
        for c in 0..<cfg.outChannels {
            let base = c * planeLen
            for i in 0..<planeLen {
                out.data[base + i] = planes.data[base + i] - noise.data[base + i]
            }
        }
        return out
    }

    /// Build the 5 input planes for a CFA mosaic.
    ///
    /// - Parameters:
    ///   - mosaic: black-subtracted, white-normalised values, row-major.
    ///   - colorIndex: per-photosite channel index, 0=R, 1=G, 2=B (from the 2×2 CFA pattern).
    ///   - a,b: per-channel Poisson-Gaussian `(a, b)` such that `Var = a·x + b`.
    public static func makeInputPlanes(mosaic: [Float],
                                       colorIndex: [Int],
                                       width: Int, height: Int,
                                       a: [Double], b: [Double]) -> PlanarImage {
        let n = width * height
        precondition(mosaic.count == n && colorIndex.count == n)
        var planes = PlanarImage(width: width, height: height, channels: 5)
        for i in 0..<n {
            let c = colorIndex[i]
            planes.data[i] = mosaic[i]                       // plane 0: signal
            planes.data[1 * n + i] = (c == 0) ? 1 : 0         // plane 1: R one-hot
            planes.data[2 * n + i] = (c == 1) ? 1 : 0         // plane 2: G one-hot
            planes.data[3 * n + i] = (c == 2) ? 1 : 0         // plane 3: B one-hot
            // plane 4: sigma from the *observed* value — identical at train and test time.
            let v = max(Double(mosaic[i]), 0.0)
            let varEst = a[c] * v + b[c]
            planes.data[4 * n + i] = Float((varEst > 1e-12 ? varEst : 1e-12).squareRoot())
        }
        return planes
    }
}
