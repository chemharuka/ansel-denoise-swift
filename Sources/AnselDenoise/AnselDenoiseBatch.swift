//
//  AnselDenoiseBatch.swift
//  AnselDenoise
//
//  Batch driver: one process, many tiles, no per-tile spawn cost.
//  GPLv3 component — see LICENSE-SCOPE.md.
//
//  A 60 MP frame is ~380 tiles at 256². Measured: the inference itself is ~10 ms
//  per tile, but process spawn + reading/writing float files through Python costs
//  ~985 ms — 99% overhead. So the CLI takes a whole directory of tiles, or a
//  single packed mosaic, and returns the same shape in one run.
//

import Foundation

/// A pile of equally-sized tiles held in one flat buffer.
public struct TileBatch {
    public let tileSize: Int
    public let count: Int
    public let data: [Float]     // count × tileSize²

    public init(tileSize: Int, count: Int, data: [Float]) {
        precondition(data.count == count * tileSize * tileSize)
        self.tileSize = tileSize
        self.count = count
        self.data = data
    }
}

extension AnselDenoiseForward {
    /// Denoise many tiles in one call.
    ///
    /// - Parameters:
    ///   - batch: `count` tiles of `tileSize`², normalized mosaic values.
    ///   - cfa: 4-char row-major CFA string, e.g. "RGGB". Every tile must start on
    ///     the same CFA phase, which holds when tiles are taken at even multiples.
    ///   - a,b: per-channel `(a, b)` from `Var = a·x + b`.
    public func denoiseBatch(_ batch: TileBatch, cfa: String,
                             a: [Double], b: [Double]) throws -> [Float] {
        let n = batch.tileSize
        // Validate the tile size against the network's downsampling factor.
        let multiple = sizeMultiple
        precondition(n % multiple == 0,
                     "tile \(n) must be a multiple of \(multiple) for depth \(model.config.depth)")

        let color = AnselDenoiseForward.colorIndexMap(pattern: cfa, width: n, height: n)
        var out = [Float](repeating: 0, count: batch.data.count)
        let tileLen = n * n

        for t in 0..<batch.count {
            let base = t * tileLen
            let mosaic = Array(batch.data[base..<(base + tileLen)])
            let planes = AnselDenoiseForward.makeInputPlanes(
                mosaic: mosaic, colorIndex: color, width: n, height: n, a: a, b: b)
            let den = try denoise(planes)
            for i in 0..<tileLen { out[base + i] = den.data[i] }
        }
        return out
    }
}

extension AnselDenoiseForward {
    /// 2×2 CFA pattern string → per-photosite colour index (0=R, 1=G, 2=B).
    public static func colorIndexMap(pattern: String, width: Int, height: Int) -> [Int] {
        let p = Array(pattern.lowercased())
        precondition(p.count == 4, "CFA pattern must be 4 characters, got '\(pattern)'")
        func idx(_ ch: Character) -> Int {
            switch ch {
            case "r": return 0
            case "g": return 1
            case "b": return 2
            default: preconditionFailure("bad CFA character '\(ch)'")
            }
        }
        let quad = p.map(idx)
        var out = [Int](repeating: 0, count: width * height)
        for y in 0..<height {
            let r = (y % 2) * 2
            for x in 0..<width {
                out[y * width + x] = quad[r + (x % 2)]
            }
        }
        return out
    }
}
