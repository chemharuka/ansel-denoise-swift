//
//  DNGMetadata.swift
//
//  What a DNG file says about its own sample values: black level, white level and CFA phase.
//  Ported from Photons' `AnselDenoiseJob` (the tool is now the owner of that arithmetic, so this
//  copy is the authoritative one — the MIT side must not keep a second opinion).
//
//  Why this is a separate file from the reader: `SourceDNG` deliberately keeps every tag as raw
//  bytes so unknown tags survive a round trip. Decoding the handful of tags whose *numbers* we
//  actually need belongs next to the code that needs them, not inside the reader.
//

import Foundation

// MARK: - Numeric tag values

extension TIFFEntry {
    /// TIFF numeric payload → `Double`s.
    ///
    /// Only the types a DNG actually uses for level/pattern tags are decoded; an unknown type
    /// returns `[]` rather than a guess. A `BlackLevel` misread as 0 would shift the whole image,
    /// so "cannot read" must be a failure the caller can see, not a silent zero.
    var numericValues: [Double] {
        func u16(_ i: Int) -> Double {
            Double(UInt16(value[i]) | (UInt16(value[i + 1]) << 8))
        }
        func u32(_ i: Int) -> Double {
            Double(UInt32(value[i]) | (UInt32(value[i + 1]) << 8)
                   | (UInt32(value[i + 2]) << 16) | (UInt32(value[i + 3]) << 24))
        }
        var out: [Double] = []
        switch type {
        case 1:                                   // BYTE
            out = value.prefix(count).map(Double.init)
        case 3:                                   // SHORT
            var i = 0
            while i + 2 <= value.count { out.append(u16(i)); i += 2 }
        case 4:                                   // LONG
            var i = 0
            while i + 4 <= value.count { out.append(u32(i)); i += 4 }
        case 5, 10:                               // RATIONAL / SRATIONAL
            var i = 0
            while i + 8 <= value.count {
                let num = Double(u32(i)), den = Double(u32(i + 4))
                out.append(den == 0 ? 0 : num / den)
                i += 8
            }
        default:
            break
        }
        return out
    }
}

// MARK: - Normalization

/// One DNG's own black/white levels, and the two conversions across them.
///
/// `SourceDNGDecode` hands back the file's raw `UInt16` samples with the black level **not**
/// subtracted, and `DNGRawWriter` wants the same scale back. The network, in between, is specified
/// on `[0,1]` (`Var = a·x + b`), so exactly one subtraction and one division happen here, and the
/// inverse undoes them.
///
/// The round trip is identity for any sample in `[black, white]`: `normalize` stores a `Float`
/// (relative error ≤ 2⁻²⁴), and `denormalize` multiplies back in `Double` before rounding, so the
/// error stays far below half a code value.
public struct MosaicNormalization: Sendable, Equatable {
    /// Black level per CFA phase, row-major over `repeatRows × repeatCols`.
    public let blackByPhase: [Double]
    public let repeatRows: Int
    public let repeatCols: Int
    public let white: Double

    public init(blackByPhase: [Double], repeatRows: Int = 2, repeatCols: Int = 2, white: Double) {
        precondition(!blackByPhase.isEmpty, "black levels must not be empty")
        precondition(white > blackByPhase.max()!, "white level must exceed every black level")
        self.blackByPhase = blackByPhase
        self.repeatRows = max(1, repeatRows)
        self.repeatCols = max(1, repeatCols)
        self.white = white
    }

    public init(black: Double, white: Double) {
        self.init(blackByPhase: [black], repeatRows: 1, repeatCols: 1, white: white)
    }

    /// Mean black level — only for logging; the per-phase value is what the conversion uses.
    public var meanBlack: Double { blackByPhase.reduce(0, +) / Double(blackByPhase.count) }

    public func black(row: Int, col: Int) -> Double {
        let r = row % repeatRows, c = col % repeatCols
        return blackByPhase[min(r * repeatCols + c, blackByPhase.count - 1)]
    }

    /// Denominator of the conversion, per phase. `blackByPhase[0]` is *not* used for this: a 2×2
    /// black array whose four values differ (the usual case on a real sensor) has no single range,
    /// and subtracting one phase's black while dividing by another's range moves the level.
    public func range(row: Int, col: Int) -> Double { white - black(row: row, col: col) }

    @inline(__always)
    public func normalize(_ value: UInt16, row: Int, col: Int) -> Float {
        Float((Double(value) - black(row: row, col: col)) / range(row: row, col: col))
    }

    @inline(__always)
    public func denormalize(_ value: Float, row: Int, col: Int) -> UInt16 {
        guard value.isFinite else { return UInt16(black(row: row, col: col).rounded()) }
        let raw = Double(value) * range(row: row, col: col) + black(row: row, col: col)
        if raw <= 0 { return 0 }
        if raw >= Double(UInt16.max) { return UInt16.max }
        return UInt16(raw.rounded())
    }

    public func normalized(_ samples: [UInt16], width: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: samples.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                out[row + x] = normalize(samples[row + x], row: y, col: x)
            }
        }
        return out
    }

    /// Back to file samples. `Float` in, `UInt16` out, clamped to what a `UInt16` can hold.
    public func quantized(_ mosaic: [Float], width: Int, height: Int) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: mosaic.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                out[row + x] = denormalize(mosaic[row + x], row: y, col: x)
            }
        }
        return out
    }
}

// MARK: - Reading the tags that describe the samples

public enum DNGMetadataError: Error, CustomStringConvertible {
    case missingTag(String, UInt16)
    case badCFAPattern(String)
    case unsupportedCFASize(rows: Int, cols: Int)

    public var description: String {
        switch self {
        case .missingTag(let what, let tag):
            return "源文件缺少 \(what)（tag \(tag)），无法做黑/白电平归一化"
        case .badCFAPattern(let raw):
            return "无法解析 CFA 相位：\(raw)"
        case .unsupportedCFASize(let r, let c):
            return "不支持的 CFA 重复尺寸 \(r)×\(c)（只支持 2×2 Bayer）"
        }
    }
}

/// A CFA phase, plus where it came from.
///
/// `pattern` is the phase **at the mosaic's (0,0)**, which is what the network needs: it assumes
/// the first sample it sees is `pattern[0]`. `filePattern` is what `33422 CFAPattern` literally
/// says, which per the DNG spec is relative to the top-left of `ActiveArea`. When the stored
/// image is larger than the active area the two differ by the active area's origin, and using the
/// wrong one swaps R for G — a checkerboard that no metadata check would catch, because the output
/// copies the tag verbatim.
public struct DNGCfaInfo: Sendable, Equatable {
    public let pattern: String
    public let filePattern: String
    public let rotationRow: Int
    public let rotationCol: Int
    public var isRotated: Bool { rotationRow != 0 || rotationCol != 0 }
}

public enum DNGMetadata {

    /// `BlackLevel` / `WhiteLevel` / `BlackLevelRepeatDim`, all as the file states them.
    static func normalization(of doc: SourceDNG) throws -> MosaicNormalization {
        let blackEntry = doc.raw[50714] ?? doc.ifd0[50714]
        let whiteEntry = doc.raw[50717] ?? doc.ifd0[50717]
        let repeatEntry = doc.raw[50713] ?? doc.ifd0[50713]

        let blacks = blackEntry?.numericValues ?? []
        guard !blacks.isEmpty else { throw DNGMetadataError.missingTag("BlackLevel", 50714) }
        guard let white = (whiteEntry?.numericValues ?? []).first else {
            throw DNGMetadataError.missingTag("WhiteLevel", 50717)
        }
        let dims = repeatEntry?.numericValues ?? []
        let rows = dims.count >= 2 ? Int(dims[0]) : (blacks.count == 4 ? 2 : 1)
        let cols = dims.count >= 2 ? Int(dims[1]) : (blacks.count == 4 ? 2 : 1)
        guard white > (blacks.max() ?? 0) else {
            throw DNGMetadataError.missingTag("WhiteLevel > BlackLevel", 50717)
        }
        return MosaicNormalization(blackByPhase: blacks,
                                   repeatRows: max(1, rows), repeatCols: max(1, cols),
                                   white: white)
    }

    /// `33422 CFAPattern` → `"rggb"`, rotated into the mosaic's own frame when needed.
    ///
    /// Guessing is not allowed: a wrong phase turns the picture into a checkerboard and every
    /// numeric check still passes, so a missing tag is an error and not an assumed `rggb`.
    static func cfa(of doc: SourceDNG, width: Int, height: Int) throws -> DNGCfaInfo {
        let dim = (doc.raw[33421] ?? doc.ifd0[33421])?.numericValues ?? []
        let rows = dim.count >= 1 ? Int(dim[0]) : 2
        let cols = dim.count >= 2 ? Int(dim[1]) : 2
        guard rows > 0, cols > 0 else { throw DNGMetadataError.badCFAPattern("\(dim)") }
        guard rows == 2, cols == 2 else {
            throw DNGMetadataError.unsupportedCFASize(rows: rows, cols: cols)
        }

        let entry = doc.raw[33422] ?? doc.ifd0[33422]
        let values = (entry?.numericValues ?? []).map { Int($0.rounded()) }
        guard values.count == 4 else { throw DNGMetadataError.badCFAPattern("\(values)") }
        let letters: [Character] = values.map { v in
            switch v {
            case 0: return "r"
            case 1: return "g"
            case 2: return "b"
            default: return "?"
            }
        }
        guard !letters.contains("?") else { throw DNGMetadataError.badCFAPattern("\(values)") }
        let filePattern = String(letters)

        // `ActiveArea` = top, left, bottom, right. The tag's phase is anchored at its top-left, so
        // when the stored image is bigger than the active area (masked border pixels present) the
        // mosaic's own (0,0) sits at sensor origin and the phase has to be rotated by the active
        // area's origin. When the two rectangles coincide there is nothing to rotate — which is the
        // case for every file this tool has been run on so far.
        var rotationRow = 0, rotationCol = 0
        if let aa = activeArea(of: doc), (aa.width != width || aa.height != height) {
            rotationRow = aa.top % 2
            rotationCol = aa.left % 2
        }
        return DNGCfaInfo(pattern: rotated(filePattern, row: rotationRow, col: rotationCol),
                          filePattern: filePattern,
                          rotationRow: rotationRow, rotationCol: rotationCol)
    }

    /// `50829 ActiveArea` → origin plus extent, in pixels.
    static func activeArea(of doc: SourceDNG) -> (top: Int, left: Int, width: Int, height: Int)? {
        let v = (doc.raw[50829] ?? doc.ifd0[50829])?.numericValues ?? []
        guard v.count == 4 else { return nil }
        let top = Int(v[0]), left = Int(v[1]), bottom = Int(v[2]), right = Int(v[3])
        guard bottom > top, right > left else { return nil }
        return (top, left, right - left, bottom - top)
    }

    /// Rotate a 2×2 Bayer name by an origin offset, in the same row-major order the network reads.
    static func rotated(_ pattern: String, row: Int, col: Int) -> String {
        let p = Array(pattern.lowercased())
        guard p.count == 4 else { return pattern }
        let r = ((row % 2) + 2) % 2, c = ((col % 2) + 2) % 2
        func at(_ y: Int, _ x: Int) -> Character { p[((y + r) % 2) * 2 + ((x + c) % 2)] }
        return String([at(0, 0), at(0, 1), at(1, 0), at(1, 1)])
    }
}
