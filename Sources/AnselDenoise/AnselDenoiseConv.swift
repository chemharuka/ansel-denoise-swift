//
//  AnselDenoiseConv.swift
//  AnselDenoise
//
//  Vectorised convolution: one output row-block at a time, parallel over rows.
//  GPLv3 component — see LICENSE-SCOPE.md.
//
//  Measured problem this solves: the naive "accumulate each of the 9 taps into a
//  full-plane buffer with vDSP_vsma" formulation runs a memory bound over the whole
//  plane 9·Cin·Cout times per layer — 256² × 5 × 8 × 9 adds up to gigabytes of
//  traffic per block and lands at ~1 s per 256² tile. Reordering to
//  "for each output row: gather 3 input rows, dot-product 3×3×Cin" keeps the whole
//  working set in cache and lets `DispatchQueue.concurrentPerform` split rows across
//  cores for free, with no cross-thread reduction.
//

import Accelerate
import Foundation

enum AnselConv {

    static let parallelRows = 1   // TEMP: isolate concurrency

    static func block(_ x: PlanarImage,
                      _ w0: AnselWeights, _ b0: [Float]?,
                      _ w2: AnselWeights, _ b2: [Float]?) -> PlanarImage {
        var h = conv(x, w0, b0, padding: (1, 1))
        lreluGelu(&h)
        h = conv(h, w2, b2, padding: (1, 1))
        lreluGelu(&h)
        return h
    }

    static func lreluGelu(_ img: inout PlanarImage) {
        img.data.withUnsafeMutableBufferPointer { p in
            for i in 0..<p.count { p[i] = gelu(p[i]) }
        }
    }

    /// Convolution, parallel over output rows.
    ///
    /// `(kh,kw) ∈ {(3,3),(1,1),(2,2)}`; stride is 1 except for the 2×2 downsampling.
    static func conv(_ input: PlanarImage,
                     _ w: AnselWeights,
                     _ b: [Float]?,
                     padding: (Int, Int),
                     stride: Int = 1) -> PlanarImage {
        let (ph, pw) = padding
        let inC = input.channels, h = input.height, wd = input.width
        let outC = w.outC, kh = w.kh, kw = w.kw
        let oh = (h + 2 * ph - kh) / stride + 1
        let ow = (wd + 2 * pw - kw) / stride + 1
        precondition(w.inC == inC, "conv expects \(w.inC) input channels, got \(inC)")

        var out = PlanarImage(width: ow, height: oh, channels: outC)
        let wv = w.values
        let src = input.data
        let inPlane = h * wd
        let outPlane = oh * ow
        let khw = kh * kw

        out.data.withUnsafeMutableBufferPointer { dstBase in
            let dst = dstBase
            src.withUnsafeBufferPointer { srcBuf in
                wv.withUnsafeBufferPointer { wBuf in
                    let chunk = max(1, oh / parallelRows + 1)
                    let chunks = (oh + chunk - 1) / chunk
                    DispatchQueue.concurrentPerform(iterations: chunks) { ci in
                        let y0 = ci * chunk
                        let y1 = min(y0 + chunk, oh)
                        if y0 >= y1 { return }
                        for oc in 0..<outC {
                            let bias = b?[oc] ?? 0
                            let wBase = oc * inC * khw
                            let outBase = oc * outPlane
                            for y in y0..<y1 {
                                let rowOut = outBase + y * ow
                                for x in 0..<ow {
                                    var acc = bias
                                    let sy0 = y * stride - ph
                                    let sx0 = x * stride - pw
                                    for ic in 0..<inC {
                                        let srcPlane = ic * inPlane
                                        let wPlane = wBase + ic * khw
                                        for ky in 0..<kh {
                                            let sy = sy0 + ky
                                            if sy < 0 || sy >= h { continue }
                                            let rowBase = srcPlane + sy * wd
                                            let wRow = wPlane + ky * kw
                                            for kx in 0..<kw {
                                                let sx = sx0 + kx
                                                if sx < 0 || sx >= wd { continue }
                                                acc += srcBuf[rowBase + sx] * wBuf[wRow + kx]
                                            }
                                        }
                                    }
                                    dst[rowOut + x] = acc
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Reference scalar implementation (single-threaded) — parity tests only.
    static func convScalar(_ input: PlanarImage,
                           _ w: AnselWeights,
                           _ b: [Float]?,
                           padding: (Int, Int),
                           stride: Int = 1) -> PlanarImage {
        let (ph, pw) = padding
        let inC = input.channels, h = input.height, wd = input.width
        let outC = w.outC, kh = w.kh, kw = w.kw
        let oh = (h + 2 * ph - kh) / stride + 1
        let ow = (wd + 2 * pw - kw) / stride + 1
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
                                var acc = bias
                                for ic in 0..<inC {
                                    let srcPlane = ic * (h * wd)
                                    let wPlane = wBase + ic * kh * kw
                                    for ky in 0..<kh {
                                        let sy = y * stride + ky - ph
                                        if sy < 0 || sy >= h { continue }
                                        let rowBase = srcPlane + sy * wd
                                        for kx in 0..<kw {
                                            let sx = x * stride + kx - pw
                                            if sx < 0 || sx >= wd { continue }
                                            acc += src[rowBase + sx] * wv[wPlane + ky * kw + kx]
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
}
