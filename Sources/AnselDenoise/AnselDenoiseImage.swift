//
//  AnselDenoiseImage.swift
//
//  The `image` subcommand's engine: one whole mosaic in, one whole denoised mosaic out.
//
//  This is the entry point a host should use. It owns everything about *fitting an image into
//  memory* — the memory budget, whether to split, where to cut, how to feather — so the host does
//  not have to model any of that.
//
//  Two front doors, one engine
//  ---------------------------
//  `run(_:...)` reads and writes `Float32` mosaics, which is what an older host still does.
//  `denoiseMosaic(...)` takes the array directly, which is what the DNG path needs: the tool reads
//  the DNG itself, so the mosaic never has to go to disk between decode and inference. Both go
//  through the same planning, padding, feathering and clamping code below, so the two paths cannot
//  drift apart in what they do to pixels.
//
//  Why the split logic lives here rather than in the host
//  ----------------------------------------------------
//  Splitting is a consequence of this implementation's memory use (0.64 GB/MP), which is a
//  property of *this* code. Putting the arithmetic in the host would mean the host hard-codes a
//  number that changes whenever these kernels change, and two programs would have to agree on it.
//

import Foundation

public struct DenoiseImageRequest: Sendable {
    public var input: URL
    public var output: URL
    public var width: Int
    public var height: Int
    /// Per-channel Poisson-Gaussian coefficients, `Var = a·x + b`, x normalized to [0,1].
    /// Callers pass these *already* multiplied by the sigma calibration, as the other
    /// subcommands do.
    public var a: [Double]
    public var b: [Double]
    public var sigmaScale: [Double]
    public var cfa: String
    public var useGPU: Bool
    /// Blend of the result against the input: `out = in + strength·(denoised − in)`, in [0,1].
    public var strength: Double
    /// Override for the memory ceiling, in bytes. `nil` derives it from installed RAM.
    public var budgetBytes: Int?
    /// Skip planning and use exactly this many columns × rows. `nil` picks the minimum that fits.
    public var forceColumns: Int?
    public var forceRows: Int?

    public init(input: URL, output: URL, width: Int, height: Int,
                a: [Double], b: [Double], sigmaScale: [Double] = [2.82, 3.94, 2.96],
                cfa: String = "rggb", useGPU: Bool = true, strength: Double = 0.85,
                budgetBytes: Int? = nil, forceColumns: Int? = nil, forceRows: Int? = nil) {
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.a = a
        self.b = b
        self.sigmaScale = sigmaScale
        self.cfa = cfa
        self.useGPU = useGPU
        self.strength = strength
        self.budgetBytes = budgetBytes
        self.forceColumns = forceColumns
        self.forceRows = forceRows
    }
}

public struct DenoiseImageResult: Sendable {
    public let plan: DenoisePlan
    public let milliseconds: Double
    public let backend: String
    public let pieces: Int
    /// Samples the result held that were below zero *before* the final clamp, and above one.
    /// Both are counted after the `strength` blend, i.e. on the array a writer would receive.
    public let clampedNegatives: Int
    public let clampedOverOne: Int
    /// Minimum and maximum of the values actually handed to the writer. Never negative after the
    /// clamp; a test asserting `finalMin >= 0` is what keeps the clamp from being deleted.
    public let finalMin: Float
    public let finalMax: Float
    /// The denoised mosaic, normalized. Same size as the input, row-major.
    ///
    /// `run(_:)` writes this to disk and drops it; the DNG path keeps it only long enough to
    /// quantize. Holding it here instead of re-reading the output file avoids a third pass over
    /// the image.
    public let mosaic: [Float]
}

public enum DenoiseImageError: Error, CustomStringConvertible {
    case badInput(String)
    case pieceFailed(Int, String)

    public var description: String {
        switch self {
        case .badInput(let m):       return "bad input: \(m)"
        case .pieceFailed(let i, let m): return "piece \(i) failed: \(m)"
        }
    }
}

public enum DenoiseImageRunner {

    /// File-to-file entry point: `Float32` mosaic in, `Float32` mosaic out.
    public static func run(_ request: DenoiseImageRequest,
                           model: AnselDenoiseModel,
                           metal: MetalConv?,
                           progress: ((String) -> Void)? = nil) throws -> DenoiseImageResult {

        let W = request.width, H = request.height
        guard W > 0, H > 0 else { throw DenoiseImageError.badInput("empty image") }
        let expected = W * H * MemoryLayout<Float>.size
        let inputData = try Data(contentsOf: request.input, options: .mappedIfSafe)
        guard inputData.count == expected else {
            throw DenoiseImageError.badInput(
                "\(request.input.lastPathComponent) is \(inputData.count) bytes, expected \(expected) for \(W)×\(H)")
        }
        var mosaic = [Float](repeating: 0, count: W * H)
        mosaic.withUnsafeMutableBytes { inputData.copyBytes(to: $0) }

        let result = try denoiseMosaic(mosaic, width: W, height: H,
                                       request: request, model: model, metal: metal,
                                       progress: progress)
        try result.mosaic.withUnsafeBufferPointer { Data(buffer: $0) }
            .write(to: request.output)
        return result
    }

    /// Memory-resident entry point. Plans, pads, runs the pieces, feathers, blends, clamps.
    ///
    /// Used by the DNG path, where the mosaic comes from a decoder rather than a file, and by
    /// `run(_:)` above. `budgetBytes` is taken literally: a caller that already holds the whole
    /// mosaic in memory should pass a budget that has that residency subtracted, or the plan will
    /// think there is more room than there is.
    public static func denoiseMosaic(_ mosaic: [Float],
                                     width: Int, height: Int,
                                     request: DenoiseImageRequest,
                                     model: AnselDenoiseModel,
                                     metal: MetalConv?,
                                     progress: ((String) -> Void)? = nil) throws -> DenoiseImageResult {
        let W = width, H = height
        guard W > 0, H > 0 else { throw DenoiseImageError.badInput("empty image") }
        guard mosaic.count == W * H else {
            throw DenoiseImageError.badInput("mosaic holds \(mosaic.count) samples, expected \(W * H) for \(W)×\(H)")
        }

        let budget = request.budgetBytes ?? DenoisePlanner.memoryLimit()
        let plan: DenoisePlan
        if let c = request.forceColumns, let r = request.forceRows {
            let pieces = DenoisePlanner.split(width: W, height: H, columns: c, rows: r,
                                              overlap: DenoisePlanner.cutOverlap)
            plan = DenoisePlan(columns: c, rows: r, pieces: pieces,
                               peakBytes: pieces.map { DenoisePlanner.bytes(forPixels: $0.sourcePixels) }.max() ?? 0,
                               wholeImageBytes: DenoisePlanner.bytes(forPixels: W * H),
                               budgetBytes: budget)
        } else {
            plan = DenoisePlanner.plan(width: W, height: H, budgetBytes: budget)
        }

        progress?("plan: \(plan.columns)×\(plan.rows) pieces, peak \(plan.peakBytes / 1_000_000_000) GB "
                  + "of \(budget / 1_000_000_000) GB budget")

        let gpu = metal.map { AnselDenoiseMetal(model: model, metal: $0) }
        let cpu = gpu == nil ? AnselDenoiseForward(model: model) : nil

        var acc = [Float](repeating: 0, count: W * H)
        var wsum = [Float](repeating: 0, count: W * H)
        var totalSeconds = 0.0

        for (index, piece) in plan.pieces.enumerated() {
            let t0 = Date()

            // Gather this piece plus its margin.
            var buf = [Float](repeating: 0, count: piece.sourcePixels)
            for row in 0..<piece.sourceHeight {
                let src = (piece.sourceY + row) * W + piece.sourceX
                let dst = row * piece.sourceWidth
                buf.replaceSubrange(dst..<(dst + piece.sourceWidth),
                                    with: mosaic[src..<(src + piece.sourceWidth)])
            }

            let out: [Float]
            do {
                // The network halves resolution `depth` times, so both dimensions must be a
                // multiple of 2^depth or the bottleneck stops being integral — the forward pass
                // traps on its own precondition. Real sensors are not so obliging: a Leica SL3
                // gives 9536×6344, and 6344 = 16·396 + 8.
                //
                // Pad on the right and bottom only, never centred: a centred pad shifts the mosaic
                // and can silently swap the CFA phase, turning R into G. Right/bottom padding
                // leaves the phase at (0,0) whatever the remainder is, and is cropped afterwards.
                //
                // One extra `mult` on each side is added even when the dimension already divides
                // evenly. Without it the outermost columns are the last thing the network sees on
                // that axis and its residual overshoots there — measured on a 9536×6344 frame,
                // 10 264 pixels at x ≥ 9523 came out negative while the input had none. The margin
                // is replicated, so it holds real signal, and it costs one row/column per side.
                let mult = 1 << model.config.depth
                let padW = mult - piece.sourceWidth % mult
                let padH = mult - piece.sourceHeight % mult
                let rw = piece.sourceWidth + padW
                let rh = piece.sourceHeight + padH

                var input = buf
                if padW > 0 || padH > 0 {
                    var padded = [Float](repeating: 0, count: rw * rh)
                    for y in 0..<rh {
                        let sy = min(y, piece.sourceHeight - 1)      // replicate the last row
                        for x in 0..<rw {
                            let sx = min(x, piece.sourceWidth - 1)   // replicate the last column
                            padded[y * rw + x] = buf[sy * piece.sourceWidth + sx]
                        }
                    }
                    input = padded
                }

                var resized: [Float]
                if let gpu {
                    resized = try gpu.denoise(mosaic: input, width: rw, height: rh,
                                              cfa: request.cfa, a: request.a, b: request.b)
                } else {
                    let color = AnselDenoiseForward.colorIndexMap(
                        pattern: request.cfa, width: rw, height: rh)
                    let planes = AnselDenoiseForward.makeInputPlanes(
                        mosaic: input, colorIndex: color, width: rw, height: rh,
                        a: request.a, b: request.b)
                    let den = try cpu!.denoise(planes)
                    resized = Array(den.data[0..<(rw * rh)])
                }

                if padW > 0 || padH > 0 {
                    out = crop(resized, fromWidth: rw,
                               toWidth: piece.sourceWidth, height: piece.sourceHeight)
                } else {
                    out = resized
                }
            } catch {
                throw DenoiseImageError.pieceFailed(index, "\(error)")
            }
            totalSeconds += Date().timeIntervalSince(t0)

            feather(piece: piece, processed: out, into: &acc, wsum: &wsum,
                    imageWidth: W, columns: plan.columns, rows: plan.rows,
                    column: index % plan.columns, row: index / plan.columns,
                    overlap: DenoisePlanner.cutOverlap)
            progress?("piece \(index + 1)/\(plan.pieces.count) done")
        }

        var result = [Float](repeating: 0, count: W * H)
        for i in 0..<acc.count where wsum[i] > 0 {
            // Clamp to the range the caller can represent. The network's residual occasionally
            // overshoots on clipped-black pixels — a pixel whose input is exactly 0 (below the
            // black level) can be predicted to go darker, giving a small negative. Measured on a
            // 9536×6344 frame: 0.018% of pixels, 94% of them with an input below 0.01, none of
            // them signal. Left unclamped they become negative samples in the output DNG, which a
            // writer will either wrap around to white or refuse.
            result[i] = min(max(acc[i] / wsum[i], 0), 1)
        }

        // Blend toward the input. Applied on normalized values, so black/white levels are the
        // caller's business and cannot drift here. Note the input is *not* guaranteed to be inside
        // [0,1]: samples below the black level normalize negative, and a DNG reader will hand those
        // over as they are. So the blend can pull values back out of range, and the clamp has to
        // run again afterwards — once, on the array that actually leaves here.
        let s = Float(min(max(request.strength, 0), 1))
        if s < 1 {
            for i in 0..<W * H where wsum[i] > 0 {
                result[i] = mosaic[i] + s * (result[i] - mosaic[i])
            }
        }

        var negatives = 0, overs = 0
        var lo = Float.infinity, hi = -Float.infinity
        for i in 0..<W * H {
            let v = result[i]
            if !v.isFinite { result[i] = 0; continue }
            if v < 0 { negatives += 1; result[i] = 0 }
            else if v > 1 { overs += 1; result[i] = 1 }
            let w = result[i]
            if w < lo { lo = w }
            if w > hi { hi = w }
        }
        if !lo.isFinite { lo = 0 }
        if !hi.isFinite { hi = 0 }

        return DenoiseImageResult(plan: plan,
                                  milliseconds: totalSeconds * 1000,
                                  backend: gpu != nil ? "gpu" : "cpu",
                                  pieces: plan.pieces.count,
                                  clampedNegatives: negatives,
                                  clampedOverOne: overs,
                                  finalMin: lo,
                                  finalMax: hi,
                                  mosaic: result)
    }

    /// Drop the padding added on the right and bottom before the forward pass.
    static func crop(_ src: [Float], fromWidth: Int, toWidth: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: toWidth * height)
        for y in 0..<height {
            let s = y * fromWidth
            let d = y * toWidth
            out.replaceSubrange(d..<(d + toWidth), with: src[s..<(s + toWidth)])
        }
        return out
    }

    /// Accumulate one piece, tapering only along sides that were cut (never the image border).
    ///
    /// The taper length equals the overlap the piece was given, so neighbouring weights sum to 1
    /// across the whole band. Two plausible alternatives that fail: a linear ramp (sums to <1,
    /// leaving dark bands) and multiplying a sin² taper into all four edges (corners get squared →
    /// weight 0, and 0/0 where four pieces meet).
    ///
    /// Feathering hides the *discontinuity* at a cut; it does not remove the underlying
    /// difference, because a pixel at a piece's edge genuinely saw less context. That is why the
    /// planner avoids cutting at all when the image fits in memory.
    static func feather(piece: DenoisePiece, processed: [Float],
                        into acc: inout [Float], wsum: inout [Float],
                        imageWidth: Int, columns: Int, rows: Int,
                        column: Int, row: Int, overlap: Int) {
        let taperX = min(overlap, piece.width / 2)
        let taperY = min(overlap, piece.height / 2)
        let insetY = piece.y - piece.sourceY
        let insetX = piece.x - piece.sourceX

        for py in 0..<piece.height {
            let wy = weight(at: py, length: piece.height,
                            nearLowCut: row > 0, nearHighCut: row < rows - 1, taper: taperY)
            if wy == 0 { continue }
            let srcRow = (py + insetY) * piece.sourceWidth + insetX
            let dstRow = (piece.y + py) * imageWidth + piece.x
            for px in 0..<piece.width {
                let wx = weight(at: px, length: piece.width,
                                nearLowCut: column > 0, nearHighCut: column < columns - 1,
                                taper: taperX)
                if wx == 0 { continue }
                let w = wx * wy
                acc[dstRow + px] += processed[srcRow + px] * w
                wsum[dstRow + px] += w
            }
        }
    }

    /// sin² rise on a cut side, cos² fall on the other, exactly 1 in the interior. Mirrored so
    /// neighbour A's falling half and neighbour B's rising half are complementary.
    @inline(__always)
    static func weight(at i: Int, length: Int,
                       nearLowCut: Bool, nearHighCut: Bool, taper: Int) -> Float {
        guard taper > 0 else { return 1 }
        if nearLowCut && i < taper {
            let t = (Float(i) + 0.5) / Float(taper) * .pi / 2
            let s = sin(t)
            return s * s
        }
        if nearHighCut && i >= length - taper {
            let t = (Float(length - 1 - i) + 0.5) / Float(taper) * .pi / 2
            let c = cos(t)
            return c * c
        }
        return 1
    }
}
