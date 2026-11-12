import Foundation

/// Writes a DNG whose **pixels are new and whose metadata is not**.
///
/// The one rule this type exists to enforce: *copy every tag, rewrite only what the file layout
/// forces you to rewrite.* A denoise pass changes sample values and nothing else, so anything that
/// describes geometry, colour or optics has to arrive at the far end untouched — in particular
///
///  - `50714 BlackLevel` / `50717 WhiteLevel` (the value range the samples live in),
///  - `50829 ActiveArea` / `50719` `50720` `DefaultCrop*` (what part of the sensor is the picture),
///  - `51022 OpcodeList2` (vignette/gain correction) and `51041 OpcodeList3`
///    (`WarpRectilinear`, the lens-distortion model),
///  - `50721`/`50722 ColorMatrix1/2`, `50964`/`50965 ForwardMatrix1/2`, `50728 AsShotNeutral`,
///  - `700 XMP`, `50740 DNGPrivateData`, the EXIF and GPS sub-IFDs, and the embedded preview.
///
/// Tags that **cannot** be copied, because they describe a layout that no longer exists:
///
/// | tag | why |
/// |---|---|
/// | `273` `279` StripOffsets/ByteCounts, `278` RowsPerStrip | strips are gone; this writes tiles |
/// | `322`–`325` TileWidth/Length/Offsets/ByteCounts | the tile grid and the compressed sizes are new |
/// | `254` `256` `257` `258` `259` `262` `277` `284` `339` | size / depth / compression / photometric describe *this* file |
/// | `330` SubIFDs | the source keeps its RAW in a SubIFD; here the RAW **is** IFD0 and the preview is IFD1 |
/// | `34665` `34853` ExifIFD/GPS | the data is copied verbatim, only the pointers move |
///
/// Keep the copy list honest by construction: `preserved` below starts from *all* source tags and
/// then subtracts `layoutTags`, so a tag nobody thought about is preserved by default rather than
/// dropped by omission.
///
/// Compression is `7` (lossless JPEG). Measured on a 60 MP frame with a 256×256 tile grid:
/// uncompressed 121 MB, `8` Deflate 86 MB, `7` SOF3 **57 MB** — and Core Image refuses the Deflate
/// file outright (`CIRAWFilter.nativeSize` comes back `0×0`, `outputImage` nil), so `7` is not just
/// smaller, it is the only compressed form the system decoder accepts.
public enum DNGRawWriter {

    public enum WriteError: Error, CustomStringConvertible {
        case notATIFF
        case missingTag(UInt16)
        case unexpectedLayout(String)
        case pixelCountMismatch(expected: Int, got: Int)

        public var description: String {
            switch self {
            case .notATIFF: return "源文件不是 TIFF/DNG（字节序标记缺失）"
            case .missingTag(let t): return "源文件缺少必需的 tag \(t)"
            case .unexpectedLayout(let s): return "源文件结构不符合预期：\(s)"
            case .pixelCountMismatch(let e, let n): return "像素数不符：期望 \(e)，收到 \(n)"
            }
        }
    }

    /// Tags that describe the byte layout of the pixels. Everything else is preserved.
    static let layoutTags: Set<UInt16> = [
        254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 339,
        322, 323, 324, 325, 330, 34665, 34853,
    ]

    public struct TileGrid: Sendable, Equatable {
        public var tileWidth: Int
        public var tileHeight: Int
        public init(tileWidth: Int = 256, tileHeight: Int = 256) {
            self.tileWidth = tileWidth
            self.tileHeight = tileHeight
        }
    }

    public struct Result: Sendable {
        public let byteCount: Int
        public let tileCount: Int
        public let sampledTiles: Int
        public let compressedFraction: Double
    }

    /// Rewrites `source` with the given sample values.
    ///
    /// - Parameters:
    ///   - samples: row-major mosaic, `width * height` entries, same CFA order as the source.
    ///   - grid: tile geometry. 256×256 matches what Adobe and Leica emit for this sensor.
    ///   - tableSampleTiles: how many tiles feed the Huffman statistics. 12 is plenty: the tables
    ///     only need the *shape* of the difference histogram, and more tiles cost time without
    ///     measurably shrinking the file.
    public static func write(
        samples: [UInt16],
        width: Int, height: Int,
        from source: URL,
        to destination: URL,
        grid: TileGrid = TileGrid(),
        tableSampleTiles: Int = 12
    ) throws -> Result {
        guard samples.count == width * height else {
            throw WriteError.pixelCountMismatch(expected: width * height, got: samples.count)
        }

        let source = try SourceDNG(url: source)

        // 1. Tile the mosaic. Edge tiles are padded by repeating the last row/column: the decoder
        //    crops them away with ActiveArea/DefaultCrop, so their contents never reach the picture,
        //    and repeating (rather than zero filling) keeps the difference histogram representative.
        let nx = (width + grid.tileWidth - 1) / grid.tileWidth
        let ny = (height + grid.tileHeight - 1) / grid.tileHeight
        var tiles: [[UInt16]] = []
        tiles.reserveCapacity(nx * ny)
        samples.withUnsafeBufferPointer { src in
            for ty in 0..<ny {
                for tx in 0..<nx {
                    var tile = [UInt16](repeating: 0, count: grid.tileWidth * grid.tileHeight)
                    tile.withUnsafeMutableBufferPointer { dst in
                        for y in 0..<grid.tileHeight {
                            let sy = min(ty * grid.tileHeight + y, height - 1)
                            let srcRow = sy * width
                            let dstRow = y * grid.tileWidth
                            for x in 0..<grid.tileWidth {
                                let sx = min(tx * grid.tileWidth + x, width - 1)
                                dst[dstRow + x] = src[srcRow + sx]
                            }
                        }
                    }
                    tiles.append(tile)
                }
            }
        }

        // 2. Huffman tables from a sample of the tiles, then encode every tile.
        let sampled = Array(tiles.prefix(max(1, tableSampleTiles)))
        let tables = RawLosslessJPEG.buildTables(
            sampled, width: grid.tileWidth, height: grid.tileHeight, components: 2)

        var blobs: [[UInt8]] = []
        blobs.reserveCapacity(tiles.count)
        for tile in tiles {
            blobs.append(tile.withUnsafeBufferPointer {
                RawLosslessJPEG.encode($0, width: grid.tileWidth, height: grid.tileHeight,
                                       components: 2, tables: tables)
            })
        }
        let compressedBytes = blobs.reduce(0) { $0 + $1.count }

        // 3. Assemble the file.
        var preserved = source.ifd0.filter { !layoutTags.contains($0.key) }
        for (tag, value) in source.raw where !layoutTags.contains(tag) {
            preserved[tag] = value          // the RAW's pixel-domain tags win over IFD0's
        }

        func put(_ tag: UInt16, _ type: UInt16, _ count: Int, _ payload: [UInt8]) {
            preserved[tag] = TIFFEntry(type: type, count: count, value: payload)
        }
        put(254, 4, 1, u32(0))                              // NewSubfileType: primary image
        put(256, 4, 1, u32(UInt32(width)))
        put(257, 4, 1, u32(UInt32(height)))
        put(258, 3, 1, u16(16))                             // BitsPerSample
        put(259, 3, 1, u16(7))                              // Compression: lossless JPEG
        put(262, 3, 1, u16(32803))                          // Photometric: CFA
        put(277, 3, 1, u16(1))                              // SamplesPerPixel
        put(284, 3, 1, u16(1))                              // PlanarConfiguration: chunky
        put(339, 3, 1, u16(1))                              // SampleFormat: unsigned
        put(322, 4, 1, u32(UInt32(grid.tileWidth)))
        put(323, 4, 1, u32(UInt32(grid.tileHeight)))
        put(324, 4, blobs.count, [UInt8](repeating: 0, count: 4 * blobs.count))   // filled below
        put(325, 4, blobs.count,
            blobs.flatMap { u32(UInt32($0.count)) })
        if source.exif != nil { put(34665, 4, 1, u32(0)) }
        if source.gps != nil { put(34853, 4, 1, u32(0)) }

        // IFD1 = the source's preview page, verbatim except its strip offsets/counts.
        var previewEntries: [UInt16: TIFFEntry] = [:]
        if let preview = source.preview {
            previewEntries = preview.entries.filter {
                ![273, 279, 330, 34665, 34853].contains($0.key)
            }
            previewEntries[254] = TIFFEntry(type: 4, count: 1, value: u32(1))
            previewEntries[273] = TIFFEntry(type: 4, count: preview.chunks.count,
                                            value: [UInt8](repeating: 0, count: 4 * preview.chunks.count))
            previewEntries[279] = TIFFEntry(type: 4, count: preview.chunks.count,
                                            value: preview.chunks.flatMap { u32(UInt32($0.count)) })
        }

        // Layout: header | IFD0 | payload0 | IFD1 | payload1 | EXIF | payloadE | GPS | payloadG |
        //         tiles | preview data
        let ifd0 = preserved.sorted { $0.key < $1.key }.map { (tag: $0.key, entry: $0.value) }
        let ifd1 = previewEntries.sorted { $0.key < $1.key }.map { (tag: $0.key, entry: $0.value) }
        let exif = (source.exif ?? [:]).sorted { $0.key < $1.key }.map { (tag: $0.key, entry: $0.value) }
        let gps = (source.gps ?? [:]).sorted { $0.key < $1.key }.map { (tag: $0.key, entry: $0.value) }

        let size0 = IFDLayout(headSize: 2 + 12 * ifd0.count + 4, payloadSize: payloadSize(ifd0))
        let size1 = IFDLayout(headSize: ifd1.isEmpty ? 0 : 2 + 12 * ifd1.count + 4,
                              payloadSize: payloadSize(ifd1))
        let sizeE = IFDLayout(headSize: exif.isEmpty ? 0 : 2 + 12 * exif.count + 4,
                              payloadSize: payloadSize(exif))
        let sizeG = IFDLayout(headSize: gps.isEmpty ? 0 : 2 + 12 * gps.count + 4,
                              payloadSize: payloadSize(gps))

        let offIFD1 = 8 + size0.total
        let offEXIF = offIFD1 + size1.total
        let offGPS = offEXIF + sizeE.total
        var offTiles = offGPS + sizeG.total
        offTiles += (4 - offTiles % 4) % 4

        var tileOffsets: [UInt32] = []
        var cursor = offTiles
        for blob in blobs {
            tileOffsets.append(UInt32(cursor))
            cursor += blob.count
            cursor += (4 - cursor % 4) % 4
        }
        var offPreview = cursor
        var previewOffsets: [UInt32] = []
        if let preview = source.preview {
            for chunk in preview.chunks {
                previewOffsets.append(UInt32(offPreview))
                offPreview += chunk.count
                offPreview += (4 - offPreview % 4) % 4
            }
        }

        // Back-fill the pointers that depend on the layout.
        var final0 = ifd0
        final0 = setEntry(final0, tag: 324, value: tileOffsets.flatMap { u32($0) })
        if source.exif != nil { final0 = setEntry(final0, tag: 34665, value: u32(UInt32(offEXIF))) }
        if source.gps != nil { final0 = setEntry(final0, tag: 34853, value: u32(UInt32(offGPS))) }
        var final1 = ifd1
        if !final1.isEmpty {
            final1 = setEntry(final1, tag: 273, value: previewOffsets.flatMap { u32($0) })
        }

        var out = Data()
        out.append(contentsOf: [0x49, 0x49])                 // "II"
        out.append(contentsOf: u16(42))
        out.append(contentsOf: u32(8))                       // IFD0 at byte 8
        // No padding between IFDs: `offIFD1`/`offEXIF`/`offGPS` are computed from head+payload sizes,
        // so an extra pad here shifts every later offset by 4 (this had the tiles 4 bytes out of
        // place, which the offset assertion below then caught).
        //
        // IFD0's "next IFD" pointer must point at IFD1 when there is one — leaving it 0 is how the
        // preview went missing: the preview page was written but nothing could reach it.
        appendIFD(&out, final0, payloadOffset: 8 + size0.headSize,
                  nextIFD: final1.isEmpty ? 0 : UInt32(offIFD1))
        if !final1.isEmpty {
            appendIFD(&out, final1, payloadOffset: offIFD1 + size1.headSize)
        }
        if !exif.isEmpty {
            appendIFD(&out, exif, payloadOffset: offEXIF + sizeE.headSize)
        }
        if !gps.isEmpty {
            appendIFD(&out, gps, payloadOffset: offGPS + sizeG.headSize)
        }
        while out.count < offTiles { out.append(0) }
        for (i, blob) in blobs.enumerated() {
            guard out.count == Int(tileOffsets[i]) else {
                throw WriteError.unexpectedLayout("瓦片 \(i) 偏移 \(out.count) ≠ 表内 \(tileOffsets[i])")
            }
            out.append(contentsOf: blob)
            pad4(&out)
        }
        if let preview = source.preview {
            for chunk in preview.chunks {
                out.append(contentsOf: chunk)
                pad4(&out)
            }
        }

        try out.write(to: destination)
        return Result(byteCount: out.count, tileCount: blobs.count,
                      sampledTiles: sampled.count,
                      compressedFraction: Double(compressedBytes) / Double(samples.count * 2))
    }

    // MARK: - little helpers

    static func u16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }
    static func u32(_ v: UInt32) -> [UInt8] {
        [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)]
    }

    static func pad4(_ out: inout Data) {
        while out.count % 4 != 0 { out.append(0) }
    }

    struct IFDLayout { let headSize: Int; let payloadSize: Int; var total: Int { headSize + payloadSize } }

    static func payloadSize(_ entries: [(tag: UInt16, entry: TIFFEntry)]) -> Int {
        var total = 0
        for e in entries where e.entry.value.count > 4 {
            total += e.entry.value.count
            if total % 2 != 0 { total += 1 }
        }
        return total
    }

    static func setEntry(_ entries: [(tag: UInt16, entry: TIFFEntry)],
                         tag: UInt16, value: [UInt8]) -> [(tag: UInt16, entry: TIFFEntry)] {
        entries.map { e in
            e.tag == tag ? (tag: tag, entry: TIFFEntry(type: e.entry.type,
                                                       count: e.entry.count, value: value)) : e
        }
    }

    static func appendIFD(_ out: inout Data, _ entries: [(tag: UInt16, entry: TIFFEntry)],
                          payloadOffset: Int, nextIFD: UInt32 = 0) {
        out.append(contentsOf: u16(UInt16(entries.count)))
        var payload = Data()
        for e in entries {
            out.append(contentsOf: u16(e.tag))
            out.append(contentsOf: u16(e.entry.type))
            out.append(contentsOf: u32(UInt32(e.entry.count)))
            if e.entry.value.count <= 4 {
                var inline = e.entry.value
                while inline.count < 4 { inline.append(0) }
                out.append(contentsOf: inline)
            } else {
                out.append(contentsOf: u32(UInt32(payloadOffset + payload.count)))
                payload.append(contentsOf: e.entry.value)
                if payload.count % 2 != 0 { payload.append(0) }
            }
        }
        out.append(contentsOf: u32(nextIFD))          // link to the next IFD (0 = last)
        out.append(payload)
    }
}
