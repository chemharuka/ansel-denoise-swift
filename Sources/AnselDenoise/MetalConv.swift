//
//  MetalConv.swift
//  AnselDenoise
//
//  GPU convolution for the ansel-denoise UNet.
//  GPLv3 component — see LICENSE-SCOPE.md.
//
//  Why a hand-written kernel instead of MPS / MPSGraph: the network is five
//  straight 3×3 / 1×1 convolutions per level with no normalization, so the whole
//  thing is one generic "im2col-free direct conv" kernel plus a GELU epilogue.
//  MPSGraph would need a graph build per shape and measurably more launch overhead
//  at these tile sizes.
//
//  Layout: channels-last (H, W, C) as float32 in a shared MTLBuffer. Channels-last
//  is what makes the 3×3 kernel read contiguous `C`-element vectors per tap, which
//  vectorises cleanly across the threadgroup.
//
//  The MSL is compiled at runtime via `device.makeLibrary(source:)` on purpose:
//  the same reason Photons ships `.metal` files as copied resources — the Metal
//  toolchain component is a network download that is not always present, and
//  `makeLibrary(source:)` needs only the OS runtime.
//

import Foundation
import Metal

public enum MetalConvError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case shaderCompile(String)
    case pipeline(String)
    case bufferAlloc
    case commandFailed(String)

    public var description: String {
        switch self {
        case .noDevice:            return "no Metal device"
        case .noQueue:             return "no Metal command queue"
        case .shaderCompile(let s): return "shader compile failed: \(s)"
        case .pipeline(let s):     return "pipeline failed: \(s)"
        case .bufferAlloc:         return "buffer allocation failed"
        case .commandFailed(let s): return "command failed: \(s)"
        }
    }
}

/// GPU conv engine holding one compiled pipeline set and reusable buffers.
public final class MetalConv {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private let convSame3x3: MTLComputePipelineState      // 3×3 pad 1, stride 1
    private let convPoint1x1: MTLComputePipelineState     // 1×1 stride 1
    private let convDown2x2: MTLComputePipelineState      // 2×2 stride 2, pad 0
    private let geluPipe: MTLComputePipelineState
    private let addPipe: MTLComputePipelineState          // a - b
    private let concatPipe: MTLComputePipelineState       // channel concat
    private let upsamplePipe: MTLComputePipelineState     // nearest 2×

    /// Channels-last planar tensor on the GPU.
    public struct Tensor {
        public let buffer: MTLBuffer
        public let width: Int
        public let height: Int
        public let channels: Int
        public var count: Int { width * height * channels }
    }

    public init(device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalConvError.noDevice
        }
        self.device = dev
        guard let q = dev.makeCommandQueue() else { throw MetalConvError.noQueue }
        self.queue = q

        let src = MetalConv.shaderSource
        let lib: MTLLibrary
        do {
            lib = try dev.makeLibrary(source: src, options: nil)
        } catch {
            throw MetalConvError.shaderCompile(String(describing: error))
        }

        func pipe(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                throw MetalConvError.pipeline("function \(name) not found")
            }
            do { return try dev.makeComputePipelineState(function: fn) }
            catch { throw MetalConvError.pipeline("\(name): \(error)") }
        }

        self.convSame3x3 = try pipe("conv3x3_same")
        self.convPoint1x1 = try pipe("conv1x1")
        self.convDown2x2 = try pipe("conv2x2_stride2")
        self.geluPipe = try pipe("gelu_inplace")
        self.addPipe = try pipe("sub_planes")
        self.concatPipe = try pipe("concat_channels")
        self.upsamplePipe = try pipe("upsample2x_nearest")
    }

    // MARK: - buffer helpers

    public func makeTensor(width: Int, height: Int, channels: Int,
                           from values: [Float]? = nil) throws -> Tensor {
        let count = width * height * channels
        let bytes = count * MemoryLayout<Float>.size
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalConvError.bufferAlloc
        }
        if let v = values {
            precondition(v.count == count)
            v.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        } else {
            memset(buf.contents(), 0, bytes)
        }
        return Tensor(buffer: buf, width: width, height: height, channels: channels)
    }

    public func read(_ t: Tensor) -> [Float] {
        let n = t.width * t.height * t.channels
        return Array(UnsafeBufferPointer(start: t.buffer.contents().bindMemory(to: Float.self, capacity: n),
                                         count: n))
    }

    private func weightBuffer(_ w: AnselWeights, paddedInC: Int? = nil) throws -> (MTLBuffer, Int) {
        // Kernel reads weights in (outC, kh, kw, inC) order — the layout that makes
        // the inner product over inC contiguous.
        let outC = w.outC, inC = w.inC, kh = w.kh, kw = w.kw
        var packed = [Float](repeating: 0, count: outC * kh * kw * inC)
        var k = 0
        for oc in 0..<outC {
            for ky in 0..<kh {
                for kx in 0..<kw {
                    for ic in 0..<inC {
                        packed[k] = w.values[((oc * inC + ic) * kh + ky) * kw + kx]
                        k += 1
                    }
                }
            }
        }
        let bytes = packed.count * MemoryLayout<Float>.size
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalConvError.bufferAlloc
        }
        packed.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        return (buf, packed.count)
    }

    private func biasBuffer(_ b: [Float]?) throws -> (MTLBuffer?, Int) {
        guard let b else { return (nil, 0) }
        let bytes = b.count * MemoryLayout<Float>.size
        guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalConvError.bufferAlloc
        }
        b.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes) }
        return (buf, b.count)
    }

    // MARK: - ops

    public func conv(_ input: Tensor, w: AnselWeights, b: [Float]?,
                     kind: ConvKind, cmd: MTLCommandBuffer) throws -> Tensor {
        let (kh, kw, stride, pad) = (w.kh, w.kw, kind.stride, kind.pad)
        let outH = (input.height + 2 * pad - kh) / stride + 1
        let outW = (input.width + 2 * pad - kw) / stride + 1
        let outC = w.outC

        let out = try makeTensor(width: outW, height: outH, channels: outC)
        let (wb, _) = try weightBuffer(w)
        let (bb, _) = try biasBuffer(b)

        let pipe = kind == .same3x3 ? convSame3x3 : (kind == .point1x1 ? convPoint1x1 : convDown2x2)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalConvError.commandFailed("encoder")
        }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(input.buffer, offset: 0, index: 0)
        enc.setBuffer(wb, offset: 0, index: 1)
        if let bb { enc.setBuffer(bb, offset: 0, index: 2) }
        enc.setBuffer(out.buffer, offset: 0, index: 3)

        var dims = SIMD4<Int32>(Int32(input.width), Int32(input.height), Int32(input.channels),
                                Int32(outC))
        var dims2 = SIMD4<Int32>(Int32(outW), Int32(outH), Int32(kh), Int32(kw))
        var str = SIMD2<Int32>(Int32(stride), Int32(pad))
        enc.setBytes(&dims, length: MemoryLayout<SIMD4<Int32>>.size, index: 4)
        enc.setBytes(&dims2, length: MemoryLayout<SIMD4<Int32>>.size, index: 5)
        enc.setBytes(&str, length: MemoryLayout<SIMD2<Int32>>.size, index: 6)
        var hasBias = Int32(b == nil ? 0 : 1)
        enc.setBytes(&hasBias, length: MemoryLayout<Int32>.size, index: 7)

        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: (outW + 15) / 16, height: (outH + 15) / 16, depth: outC)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        return out
    }

    public func geluInPlace(_ t: Tensor, cmd: MTLCommandBuffer) throws {
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalConvError.commandFailed("encoder")
        }
        enc.setComputePipelineState(geluPipe)
        enc.setBuffer(t.buffer, offset: 0, index: 0)
        var n = Int32(t.count)
        enc.setBytes(&n, length: MemoryLayout<Int32>.size, index: 1)
        let tg = MTLSize(width: 64, height: 1, depth: 1)
        let grid = MTLSize(width: (t.count + 63) / 64, height: 1, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    public enum ConvKind {
        case same3x3, point1x1, down2x2
        var stride: Int { self == .down2x2 ? 2 : 1 }
        var pad: Int { self == .same3x3 ? 1 : 0 }
    }

    /// `a - b` on the first `b.channels` planes of `a` (residual output).
    public func subtract(_ a: Tensor, _ b: Tensor, cmd: MTLCommandBuffer) throws -> Tensor {
        let out = try makeTensor(width: a.width, height: a.height, channels: b.channels)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalConvError.commandFailed("encoder")
        }
        enc.setComputePipelineState(addPipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var dims = SIMD4<Int32>(Int32(a.width), Int32(a.height), Int32(b.channels),
                                Int32(a.channels))
        enc.setBytes(&dims, length: MemoryLayout<SIMD4<Int32>>.size, index: 3)
        let tg = MTLSize(width: 64, height: 1, depth: 1)
        let total = a.width * a.height * b.channels
        enc.dispatchThreadgroups(MTLSize(width: (total + 63) / 64, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
        return out
    }

    /// Channel-axis concat: out = [a, b].
    public func concat(_ a: Tensor, _ b: Tensor, cmd: MTLCommandBuffer) throws -> Tensor {
        precondition(a.width == b.width && a.height == b.height)
        let out = try makeTensor(width: a.width, height: a.height, channels: a.channels + b.channels)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalConvError.commandFailed("encoder")
        }
        enc.setComputePipelineState(concatPipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        var dims = SIMD4<Int32>(Int32(a.width), Int32(a.height), Int32(a.channels),
                                Int32(b.channels))
        enc.setBytes(&dims, length: MemoryLayout<SIMD4<Int32>>.size, index: 3)
        let tg = MTLSize(width: 64, height: 1, depth: 1)
        let total = a.width * a.height * (a.channels + b.channels)
        enc.dispatchThreadgroups(MTLSize(width: (total + 63) / 64, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
        return out
    }

    public func upsampleNearest(_ t: Tensor, cmd: MTLCommandBuffer) throws -> Tensor {
        let out = try makeTensor(width: t.width * 2, height: t.height * 2, channels: t.channels)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalConvError.commandFailed("encoder")
        }
        enc.setComputePipelineState(upsamplePipe)
        enc.setBuffer(t.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        var dims = SIMD4<Int32>(Int32(t.width), Int32(t.height), Int32(t.channels), 0)
        enc.setBytes(&dims, length: MemoryLayout<SIMD4<Int32>>.size, index: 2)
        let tg = MTLSize(width: 64, height: 1, depth: 1)
        let total = out.width * out.height * out.channels
        enc.dispatchThreadgroups(MTLSize(width: (total + 63) / 64, height: 1, depth: 1),
                                 threadsPerThreadgroup: tg)
        enc.endEncoding()
        return out
    }

    // MARK: - shader

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Metal has no scalar erf(); use precise:: namespace (available in MSL 2.x+)
    // and fall back to an Abramowitz-Stegun polynomial if unavailable.
    inline float erf_approx(float x) {
        // A&S 7.1.26, |error| < 1.5e-7 — well inside the 1e-5 parity budget.
        float s = (x < 0.0f) ? -1.0f : 1.0f;
        float ax = fabs(x);
        float t = 1.0f / (1.0f + 0.3275911f * ax);
        float y = t * (0.254829592f + t * (-0.284496736f
                  + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        return s * (1.0f - y * exp(-ax * ax));
    }

    inline float gelu_exact(float t) {
        return 0.5f * t * (1.0f + erf_approx(t * 0.7071067811865476f));
    }

    // Direct conv, channels-last. One thread per output pixel-channel.
    // in:  (H, W, Cin)   w: (Cout, kh, kw, Cin)   out: (OH, OW, Cout)
    kernel void conv_generic(
        device const float* in       [[buffer(0)]],
        device const float* w        [[buffer(1)]],
        device const float* bias     [[buffer(2)]],
        device float* out            [[buffer(3)]],
        constant int4& inDims        [[buffer(4)]],   // W, H, Cin, Cout
        constant int4& outDims       [[buffer(5)]],   // OW, OH, kh, kw
        constant int2& stridePad     [[buffer(6)]],   // stride, pad
        constant int& hasBias        [[buffer(7)]],
        uint3 tid [[thread_position_in_grid]])
    {
        int OW = outDims.x, OH = outDims.y, KH = outDims.z, KW = outDims.w;
        int W = inDims.x, H = inDims.y, Cin = inDims.z, Cout = inDims.w;
        int S = stridePad.x, P = stridePad.y;
        int ox = (int)tid.x, oy = (int)tid.y, oc = (int)tid.z;
        if (ox >= OW || oy >= OH || oc >= Cout) return;

        int KHW = KH * KW;
        float acc = hasBias ? bias[oc] : 0.0f;
        const device float* wp = w + oc * KHW * Cin;
        for (int ky = 0; ky < KH; ++ky) {
            int sy = oy * S + ky - P;
            if (sy < 0 || sy >= H) continue;
            for (int kx = 0; kx < KW; ++kx) {
                int sx = ox * S + kx - P;
                if (sx < 0 || sx >= W) continue;
                const device float* ip = in + ((sy * W) + sx) * Cin;
                const device float* wk = wp + (ky * KW + kx) * Cin;
                for (int ic = 0; ic < Cin; ++ic) {
                    acc += ip[ic] * wk[ic];
                }
            }
        }
        out[((oy * OW) + ox) * Cout + oc] = acc;
    }

    kernel void conv3x3_same(device const float* in [[buffer(0)]],
                             device const float* w [[buffer(1)]],
                             device const float* bias [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant int4& inDims [[buffer(4)]],
                             constant int4& outDims [[buffer(5)]],
                             constant int2& stridePad [[buffer(6)]],
                             constant int& hasBias [[buffer(7)]],
                             uint3 tid [[thread_position_in_grid]]) {
        int OW = outDims.x, OH = outDims.y, KH = outDims.z, KW = outDims.w;
        int W = inDims.x, H = inDims.y, Cin = inDims.z, Cout = inDims.w;
        int ox = (int)tid.x, oy = (int)tid.y, oc = (int)tid.z;
        if (ox >= OW || oy >= OH || oc >= Cout) return;
        float acc = hasBias ? bias[oc] : 0.0f;
        const device float* wp = w + oc * KH * KW * Cin;
        for (int ky = 0; ky < 3; ++ky) {
            int sy = oy + ky - 1;
            if (sy < 0 || sy >= H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int sx = ox + kx - 1;
                if (sx < 0 || sx >= W) continue;
                const device float* ip = in + ((sy * W) + sx) * Cin;
                const device float* wk = wp + (ky * 3 + kx) * Cin;
                for (int ic = 0; ic < Cin; ++ic) acc += ip[ic] * wk[ic];
            }
        }
        out[((oy * OW) + ox) * Cout + oc] = acc;
    }

    kernel void conv1x1(device const float* in [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device const float* bias [[buffer(2)]],
                        device float* out [[buffer(3)]],
                        constant int4& inDims [[buffer(4)]],
                        constant int4& outDims [[buffer(5)]],
                        constant int2& stridePad [[buffer(6)]],
                        constant int& hasBias [[buffer(7)]],
                        uint3 tid [[thread_position_in_grid]]) {
        int OW = outDims.x, OH = outDims.y;
        int W = inDims.x, H = inDims.y, Cin = inDims.z, Cout = inDims.w;
        int ox = (int)tid.x, oy = (int)tid.y, oc = (int)tid.z;
        if (ox >= OW || oy >= OH || oc >= Cout) return;
        const device float* ip = in + ((oy * W) + ox) * Cin;
        const device float* wk = w + oc * Cin;
        float acc = hasBias ? bias[oc] : 0.0f;
        for (int ic = 0; ic < Cin; ++ic) acc += ip[ic] * wk[ic];
        out[((oy * OW) + ox) * Cout + oc] = acc;
    }

    kernel void conv2x2_stride2(device const float* in [[buffer(0)]],
                                device const float* w [[buffer(1)]],
                                device const float* bias [[buffer(2)]],
                                device float* out [[buffer(3)]],
                                constant int4& inDims [[buffer(4)]],
                                constant int4& outDims [[buffer(5)]],
                                constant int2& stridePad [[buffer(6)]],
                                constant int& hasBias [[buffer(7)]],
                                uint3 tid [[thread_position_in_grid]]) {
        int OW = outDims.x, OH = outDims.y, KH = outDims.z, KW = outDims.w;
        int W = inDims.x, H = inDims.y, Cin = inDims.z, Cout = inDims.w;
        int ox = (int)tid.x, oy = (int)tid.y, oc = (int)tid.z;
        if (ox >= OW || oy >= OH || oc >= Cout) return;
        float acc = hasBias ? bias[oc] : 0.0f;
        const device float* wp = w + oc * KH * KW * Cin;
        for (int ky = 0; ky < 2; ++ky) {
            int sy = oy * 2 + ky;
            if (sy >= H) continue;
            for (int kx = 0; kx < 2; ++kx) {
                int sx = ox * 2 + kx;
                if (sx >= W) continue;
                const device float* ip = in + ((sy * W) + sx) * Cin;
                const device float* wk = wp + (ky * 2 + kx) * Cin;
                for (int ic = 0; ic < Cin; ++ic) acc += ip[ic] * wk[ic];
            }
        }
        out[((oy * OW) + ox) * Cout + oc] = acc;
    }

    kernel void gelu_inplace(device float* x [[buffer(0)]],
                             constant int& n [[buffer(1)]],
                             uint gid [[thread_position_in_grid]]) {
        if ((int)gid >= n) return;
        x[gid] = gelu_exact(x[gid]);
    }

    // a and b are both channels-last with outC channels here, so a straight
    // elementwise difference is correct — but only if both buffers really are the
    // same layout. (An earlier plane-major version assumed the caller had already
    // de-interleaved; it had not.)
    kernel void sub_planes(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant int4& dims [[buffer(3)]],   // W, H, outC, unused
                           uint gid [[thread_position_in_grid]]) {
        int W = dims.x, H = dims.y, outC = dims.z;
        int total = W * H * outC;
        if ((int)gid >= total) return;
        out[gid] = a[gid] - b[gid];
    }

    // Channels-last concat: element i of the output has aC + bC channels, of which
    // the first aC come from `a` and the rest from `b`. A planar (plane-major)
    // implementation here silently scrambles every decoder input.
    kernel void concat_channels(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant int4& dims [[buffer(3)]],  // W, H, aC, bC
                                uint gid [[thread_position_in_grid]]) {
        int W = dims.x, H = dims.y, aC = dims.z, bC = dims.w;
        int total = W * H * (aC + bC);
        if ((int)gid >= total) return;
        int cOut = aC + bC;
        int pixel = (int)gid / cOut;
        int c = (int)gid - pixel * cOut;
        out[gid] = (c < aC) ? a[pixel * aC + c] : b[pixel * bC + (c - aC)];
    }

    // Nearest 2x upsample, channels-last. The source index must be interleaved
    // (pixel * C + channel); a planar index here reads the wrong element for
    // every channel except the first and scrambles the whole decoder.
    kernel void upsample2x_nearest(device const float* in [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   constant int4& dims [[buffer(2)]],   // W, H, C, 0
                                   uint gid [[thread_position_in_grid]]) {
        int W = dims.x, H = dims.y, C = dims.z;
        int OW = W * 2, OH = H * 2;
        int total = OW * OH * C;
        if ((int)gid >= total) return;
        int c = (int)gid % C;
        int pix = (int)gid / C;
        int y = pix / OW, x = pix - y * OW;
        int sp = ((y / 2) * W + (x / 2)) * C + c;
        out[gid] = in[sp];
    }
    """
}
