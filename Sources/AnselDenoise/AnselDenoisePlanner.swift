//
//  AnselDenoisePlanner.swift
//
//  How to fit one image into memory: whole, or split into the fewest pieces that fit.
//  Pure arithmetic over sizes and the machine's RAM — no image or network knowledge.
//
//  Why this matters
//  ----------------
//  The network denoises differently near its input edges than in its interior, because an edge
//  pixel has less surround to reason about. Measured against a single-pass reference on a 1024²
//  crop of a real raw, the mismatch is 21× larger at 0–4 px from a border than at 24–48 px, and
//  only reaches zero past ~96 px. Feathering averages those differing answers; it cannot undo
//  them.
//
//  So the cheapest way to remove seams is to have fewer borders. Ideally none.
//  A single pass over 60 MP measured 55.6 s and 37.8 GB — both better than 1426 tiles at 256²
//  (77.9 s), because small tiles spend about a third of their work on overlap that is then
//  blended away.
//
//  Splitting is therefore a *memory* fallback, not a performance strategy: when the whole image
//  will not fit, cut it into as few rectangles as possible.
//

import Foundation

/// A rectangle of the input to hand to the network in one go.
public struct DenoisePiece: Sendable, Equatable {
    /// Region this piece owns in the output.
    public let x: Int, y: Int, width: Int, height: Int
    /// Region fed to the network: `width`/`height` grown by the overlap on internal sides.
    public let sourceX: Int, sourceY: Int, sourceWidth: Int, sourceHeight: Int

    public var sourcePixels: Int { sourceWidth * sourceHeight }
    public var outputPixels: Int { width * height }
}

public struct DenoisePlan: Sendable {
    public let columns: Int
    public let rows: Int
    public let pieces: [DenoisePiece]
    /// Bytes the largest single forward pass is expected to need.
    public let peakBytes: Int
    /// What the whole image in one pass would have needed.
    public let wholeImageBytes: Int
    public let budgetBytes: Int

    public var isSinglePass: Bool { pieces.count == 1 }
    /// Internal cuts that need feathering: one per internal grid line.
    public var seams: Int { max(0, columns - 1) + max(0, rows - 1) }
}

public enum DenoisePlanner {
    /// Memory per pixel for a forward pass, measured on the Metal implementation:
    /// 1 MP → 0.68 GB, 16.8 MP → 10.6 GB, 60.3 MP → 37.8 GB. All three give 0.63–0.64 GB/MP.
    public static let bytesPerPixel = 640.0

    /// Overlap at internal cuts, in pixels. Must be ≥ the ~96 px the network needs to recover
    /// full context, so neither side of a cut compares a starved edge against a saturated one.
    public static let cutOverlap = 128

    /// Fraction of physical RAM the tool may occupy.
    ///
    /// Small machines get a fixed reserve rather than a fraction: on 8 GB, 2/3 would be 5.3 GB and
    /// would starve the OS, so the ceiling is 50% (4 GB, leaving 4 GB). From 12 GB up the machine
    /// can afford the fraction, and 2/3 keeps more frames in a single pass.
    public static func memoryLimit(physicalMemory: UInt64 =
                                   ProcessInfo.processInfo.physicalMemory) -> Int {
        let gb = Double(physicalMemory) / 1e9
        let fraction = gb < 12 ? 0.5 : (2.0 / 3.0)
        return Int(Double(physicalMemory) * fraction)
    }

    public static func bytes(forPixels pixels: Int) -> Int {
        Int((Double(pixels) * bytesPerPixel).rounded(.up))
    }

    /// Choose a plan: a single pass if it fits, otherwise the fewest balanced cuts that do.
    ///
    /// The split is computed, not picked from a fixed list. Two reasons a list is wrong:
    ///   - the counts needed are arbitrary (100 MP at a 4 GB budget needs ~17 pieces — 1/2/4/6/9
    ///     never covers it cleanly), and
    ///   - overlap adds area, so the number of pieces that actually fits is always more than
    ///     `imageArea / pieceBudget` suggests: a piece owning 1.5 MP can be fed 1.8 MP once 128 px
    ///     of margin is added on its internal sides.
    /// Both are accounted for below: the search raises the cut count until the *fed* area of the
    /// worst piece fits, so it converges on the true minimum.
    public static func plan(width: Int, height: Int,
                            budgetBytes: Int,
                            overlap: Int = cutOverlap) -> DenoisePlan {
        let whole = bytes(forPixels: width * height)
        if whole <= budgetBytes {
            return DenoisePlan(
                columns: 1, rows: 1,
                pieces: [DenoisePiece(x: 0, y: 0, width: width, height: height,
                                      sourceX: 0, sourceY: 0,
                                      sourceWidth: width, sourceHeight: height)],
                peakBytes: whole, wholeImageBytes: whole, budgetBytes: budgetBytes)
        }

        let budgetPixels = max(1, Int(Double(budgetBytes) / bytesPerPixel))
        let limit = maxPieces(forPixels: width * height, budgetPixels: budgetPixels)

        var count = 2
        while count <= limit {
            for (c, r) in factorisations(of: count) {
                let pieces = split(width: width, height: height,
                                   columns: c, rows: r, overlap: overlap)
                let peak = pieces.map { bytes(forPixels: $0.sourcePixels) }.max() ?? whole
                if peak <= budgetBytes {
                    return DenoisePlan(columns: c, rows: r, pieces: pieces,
                                       peakBytes: peak, wholeImageBytes: whole,
                                       budgetBytes: budgetBytes)
                }
            }
            count += 1
        }

        // Nothing fits within the search bound: even the finest split leaves a piece over budget.
        // Return the finest attempted split so the caller can act on the reported `peakBytes`
        // rather than discovering an out-of-memory halfway through a long run.
        let r = max(2, limit)
        let pieces = split(width: width, height: height, columns: 1, rows: r, overlap: overlap)
        let peak = pieces.map { bytes(forPixels: $0.sourcePixels) }.max() ?? whole
        return DenoisePlan(columns: 1, rows: r, pieces: pieces,
                           peakBytes: peak, wholeImageBytes: whole, budgetBytes: budgetBytes)
    }

    /// Upper bound on pieces worth trying. Beyond one piece per budget-sized chunk (plus slack for
    /// overlap) nothing can improve, so the loop terminates even for absurd inputs.
    static func maxPieces(forPixels pixels: Int, budgetPixels: Int) -> Int {
        let nominal = pixels / max(budgetPixels, 1)
        return min(8192, max(16, nominal * 4))
    }

    // MARK: - geometry

    /// All (columns, rows) with `columns × rows == n`, walking outwards from square so pieces stay
    /// close to square and cut lines stay short.
    static func factorisations(of n: Int) -> [(Int, Int)] {
        var base: [(Int, Int)] = []
        var i = 1
        while i * i <= n {
            if n % i == 0 { base.append((i, n / i)) }
            i += 1
        }
        var woven: [(Int, Int)] = []
        for (a, b) in base {
            woven.append((a, b))
            if a != b { woven.append((b, a)) }
        }
        return woven
    }

    /// Cut `width × height` into `columns × rows` rectangles, growing each by `overlap` on every
    /// internal side. Cuts are evenly spaced; remainders go to the earlier rectangles so the last
    /// one is never a sliver.
    public static func split(width: Int, height: Int,
                             columns: Int, rows: Int, overlap: Int) -> [DenoisePiece] {
        let xs = spans(total: width, count: columns)
        let ys = spans(total: height, count: rows)
        var out: [DenoisePiece] = []
        out.reserveCapacity(columns * rows)
        for (ry, spanY) in ys.enumerated() {
            for (cx, spanX) in xs.enumerated() {
                let left = cx > 0 ? overlap : 0
                let right = cx < columns - 1 ? overlap : 0
                let top = ry > 0 ? overlap : 0
                let bottom = ry < rows - 1 ? overlap : 0
                let sx = max(0, spanX.lowerBound - left)
                let sy = max(0, spanY.lowerBound - top)
                let ex = min(width, spanX.upperBound + right)
                let ey = min(height, spanY.upperBound + bottom)
                out.append(DenoisePiece(x: spanX.lowerBound, y: spanY.lowerBound,
                                        width: spanX.count, height: spanY.count,
                                        sourceX: sx, sourceY: sy,
                                        sourceWidth: ex - sx, sourceHeight: ey - sy))
            }
        }
        return out
    }

    static func spans(total: Int, count: Int) -> [Range<Int>] {
        precondition(count > 0)
        let base = total / count
        let extra = total % count
        var out: [Range<Int>] = []
        var start = 0
        for i in 0..<count {
            let len = base + (i < extra ? 1 : 0)
            out.append(start..<(start + len))
            start += len
        }
        return out
    }
}
