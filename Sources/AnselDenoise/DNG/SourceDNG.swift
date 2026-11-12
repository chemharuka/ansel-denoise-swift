import Foundation

/// Read side of the DNG writer: everything `DNGRawWriter` needs to know about an existing file,
/// and nothing else.
///
/// Deliberately not a general TIFF library. It reads IFD0, the RAW SubIFD, the EXIF and GPS
/// sub-IFDs and the preview page's strips — the five places a DNG keeps the metadata a denoise pass
/// must not disturb — and keeps every entry as raw bytes so unknown tags survive a round trip
/// byte-for-byte. That property is the point: a tag nobody wrote code for must still come out the
/// other side, which is only true if it is never decoded.
struct SourceDNG {
    /// IFD0, verbatim.
    var ifd0: [UInt16: TIFFEntry]
    /// The RAW image's IFD (the first SubIFD), verbatim.
    var raw: [UInt16: TIFFEntry]
    var exif: [UInt16: TIFFEntry]?
    var gps: [UInt16: TIFFEntry]?
    var preview: Preview?

    struct Preview {
        var entries: [UInt16: TIFFEntry]
        var chunks: [[UInt8]]
    }

    enum ReadError: Error, CustomStringConvertible {
        case notLittleEndianTIFF
        case truncated(String)
        case noRawSubIFD

        var description: String {
            switch self {
            case .notLittleEndianTIFF: return "只支持小端（II）TIFF/DNG"
            case .truncated(let what): return "文件在 \(what) 处被截断"
            case .noRawSubIFD: return "IFD0 没有 SubIFDs，无法定位 RAW 图像"
            }
        }
    }

    init(url: URL) throws {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        try self.init(data: data)
    }

    init(data: Data) throws {
        guard data.count > 8, data[0] == 0x49, data[1] == 0x49 else {
            throw ReadError.notLittleEndianTIFF
        }
        guard let magic = TIFFReader.u16(data, 2), magic == 42 else {
            throw ReadError.notLittleEndianTIFF
        }
        guard let ifd0Offset = TIFFReader.u32(data, 4) else {
            throw ReadError.truncated("IFD0 偏移")
        }
        ifd0 = try TIFFReader.readIFD(data, at: Int(ifd0Offset))

        // A capture DNG usually keeps its RAW in the first SubIFD; our own writer puts it in IFD0
        // (the preview becomes IFD1). Accept both, and prefer whichever actually carries the CFA
        // photometric (32803) — that is the one whose tags describe the samples.
        if let subIFDs = ifd0[330]?.value, subIFDs.count >= 4,
           let rawOffset = TIFFReader.u32(subIFDs, 0) {
            let candidate = try TIFFReader.readIFD(data, at: Int(rawOffset))
            if candidate[262]?.firstUInt32 == 32803 {
                raw = candidate
            } else if ifd0[262]?.firstUInt32 == 32803 {
                raw = ifd0
            } else {
                raw = candidate
            }
        } else if ifd0[262]?.firstUInt32 == 32803 {
            raw = ifd0
        } else {
            throw ReadError.noRawSubIFD
        }

        if let pointer = ifd0[34665]?.value, let offset = TIFFReader.u32(pointer, 0) {
            exif = try TIFFReader.readIFD(data, at: Int(offset))
        }
        if let pointer = ifd0[34853]?.value, let offset = TIFFReader.u32(pointer, 0) {
            gps = try TIFFReader.readIFD(data, at: Int(offset))
        }

        // The preview is whatever IFD carries strips (`273`/`279`) and is *not* the CFA image.
        // In a camera DNG that is IFD0; in a file this writer produced it is IFD1 (the RAW took IFD0).
        // No viewer should have to re-derive a thumbnail, so the bytes are copied across.
        var previewSource: [UInt16: TIFFEntry]? = nil
        if ifd0[262]?.firstUInt32 != 32803, ifd0[273] != nil {
            previewSource = ifd0
        } else if let nextEntry = TIFFReader.nextIFDOffset(data, after: Int(ifd0Offset)),
                  nextEntry != 0 {
            let ifd1 = try TIFFReader.readIFD(data, at: Int(nextEntry))
            if ifd1[273] != nil { previewSource = ifd1 }
        }

        if let entries = previewSource,
           let offsetsEntry = entries[273], let countsEntry = entries[279] {
            let n = min(offsetsEntry.count, countsEntry.count)
            var chunks: [[UInt8]] = []
            for i in 0..<n {
                guard let offset = TIFFReader.u32(offsetsEntry.value, i),
                      let count = TIFFReader.u32(countsEntry.value, i) else { break }
                let start = Int(offset), end = start + Int(count)
                guard end <= data.count else { throw ReadError.truncated("预览条带 \(i)") }
                chunks.append([UInt8](data[start..<end]))
            }
            if !chunks.isEmpty {
                preview = Preview(entries: entries, chunks: chunks)
            }
        }
    }
}

/// One IFD entry, with its value kept as raw bytes so unknown tags survive untouched.
struct TIFFEntry {
    var type: UInt16
    var count: Int
    var value: [UInt8]

    /// Bytes on disk for this entry's payload, or 0 when it fits inline.
    var inlineOrPayloadSize: Int { value.count }
}

enum TIFFReader {
    static func u16(_ data: Data, _ offset: Int) -> UInt16? {
        guard offset >= 0, offset + 2 <= data.count else { return nil }
        return UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }

    static func u32(_ data: Data, _ offset: Int) -> UInt32? {
        guard offset >= 0, offset + 4 <= data.count else { return nil }
        return UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
    }

    /// Reads the `index`-th little-endian UInt32 out of a raw tag payload (`bytes[index * 4...]`).
    static func u32(_ bytes: [UInt8], _ index: Int) -> UInt32? {
        let base = index * 4
        guard index >= 0, base + 4 <= bytes.count else { return nil }
        return UInt32(bytes[base])
            | (UInt32(bytes[base + 1]) << 8)
            | (UInt32(bytes[base + 2]) << 16)
            | (UInt32(bytes[base + 3]) << 24)
    }

    /// TIFF type → bytes per component.
    static func componentSize(_ type: UInt16) -> Int {
        switch type {
        case 1, 2, 6, 7: return 1          // BYTE, ASCII, SBYTE, UNDEFINED
        case 3, 8: return 2                // SHORT, SSHORT
        case 4, 9, 13: return 4            // LONG, SLONG, IFD
        case 5, 10: return 8               // RATIONAL, SRATIONAL
        case 11: return 4                  // FLOAT
        case 12: return 8                  // DOUBLE
        default: return 1
        }
    }

    /// Offset of the next IFD, or nil when the chain ends. Needed to find IFD1, where a file this
    /// writer produced keeps its preview (the RAW took IFD0).
    static func nextIFDOffset(_ data: Data, after ifdOffset: Int) -> UInt32? {
        guard let count = u16(data, ifdOffset) else { return nil }
        return u32(data, ifdOffset + 2 + 12 * Int(count))
    }

    static func readIFD(_ data: Data, at offset: Int) throws -> [UInt16: TIFFEntry] {
        guard let count = u16(data, offset) else { throw SourceDNG.ReadError.truncated("IFD 头") }
        var entries: [UInt16: TIFFEntry] = [:]
        entries.reserveCapacity(Int(count))
        for i in 0..<Int(count) {
            let base = offset + 2 + 12 * i
            guard let tag = u16(data, base),
                  let type = u16(data, base + 2),
                  let itemCount = u32(data, base + 4) else {
                throw SourceDNG.ReadError.truncated("IFD 条目 \(i)")
            }
            let size = componentSize(type) * Int(itemCount)
            if size <= 4 {
                let start = base + 8
                guard start + size <= data.count else {
                    throw SourceDNG.ReadError.truncated("IFD 内联值 \(tag)")
                }
                entries[tag] = TIFFEntry(type: type, count: Int(itemCount),
                                         value: [UInt8](data[start..<start + size]))
            } else {
                guard let payloadOffset = u32(data, base + 8) else {
                    throw SourceDNG.ReadError.truncated("IFD 指针 \(tag)")
                }
                let start = Int(payloadOffset)
                guard start >= 0, start + size <= data.count else {
                    throw SourceDNG.ReadError.truncated("tag \(tag) 的 payload")
                }
                entries[tag] = TIFFEntry(type: type, count: Int(itemCount),
                                         value: [UInt8](data[start..<start + size]))
            }
        }
        return entries
    }
}
