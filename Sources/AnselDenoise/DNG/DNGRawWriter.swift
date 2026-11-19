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
/// | `330` SubIFDs | the source keeps its RAW in a SubIFD; in the single-raw file the RAW **is** IFD0 and the preview is IFD1. A dual-raw file takes the camera-DNG shape instead — preview in IFD0, both raws in `SubIFDs` |
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
///
/// Two raw layers, and which one a reader gets
/// -------------------------------------------
/// `write(primary:alternate:…)` writes two full-resolution CFA raws in the shape a camera DNG uses,
/// because that is the shape ImageIO actually resolves:
///
/// ```
/// IFD0        = preview page (NewSubfileType 1)      ← thumbnail; also owns EXIF/GPS
/// SubIFDs[0]  = original raw (NewSubfileType 0)      ← what a raw decoder renders
/// SubIFDs[1]  = denoised raw (NewSubfileType 0)
/// ```
///
/// Measured, not assumed: `CIRAWFilter` resolves a multi-raw DNG by scanning `SubIFDs` and taking
/// the **first** entry whose `NewSubfileType` is 0; IFD0 is only consulted when no such entry
/// exists. Two files identical apart from which layer sat in which slot both decoded the *first*
/// `SubIFDs` raw, so "IFD0 wins when it is a raw" is **not** the rule — a file that puts the
/// original in IFD0 and the denoised layer in `SubIFDs[0]` hands the system decoder the denoised
/// one. Hence: original first in `SubIFDs`, denoised second, preview in IFD0.
///
/// The alternate is reachable by following `330` (`SubIFDs[1]`), or by *promoting* it: rewrite
/// `SubIFDs[0]` to point at the alternate IFD and the `330` count to 1.
///
/// A single-raw write (`write(samples:…)`) keeps the shape this tool has always produced: the raw
/// **is** IFD0, the preview is IFD1, and there is no `330`. That keeps existing readers, tests and
/// `scripts/acceptance.sh` on exactly the bytes they already expect.
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

    /// Tags the alternate IFD repeats from the original because they describe **the samples**, not
    /// the optics or the look: value range, CFA phase, active area and crop.
    ///
    /// Deliberately excluded are the profile tags (50721/50722 matrices, 50728 AsShotNeutral,
    /// 50938/50939 hue-sat maps, 50981/50982 look table) and the opcode lists: those describe how to
    /// *develop* the file, and duplicating them makes ImageIO apply the correction twice — measured:
    /// a dual file whose two layers are identical then renders differently from its own source.
    static let sampleDescriptiveTags: Set<UInt16> = [
        50710, 50711, 50713, 50714, 50717, 50719, 50720, 50829, 33421, 33422,
    ]

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

    /// One full-resolution mosaic plus the description that identifies it in the file.
    ///
    /// `label` becomes `270 ImageDescription` of the alternate's IFD. Only the alternate gets one:
    /// the original's `270`, when the source had it, is copied verbatim like every other metadata
    /// tag.
    public struct Layer: Sendable {
        public var samples: [UInt16]
        public var label: String?

        public init(samples: [UInt16], label: String? = nil) {
            self.samples = samples
            self.label = label
        }
    }

    public struct Result: Sendable {
        public let byteCount: Int
        public let tileCount: Int
        public let sampledTiles: Int
        /// Compressed bytes of the **primary** layer over its uncompressed size.
        public let compressedFraction: Double
        /// Whether a second raw IFD was written (`SubIFDs` on IFD0).
        public let hasAlternate: Bool
        public let alternateByteCount: Int
        public let alternateTileCount: Int

        public init(byteCount: Int, tileCount: Int, sampledTiles: Int,
                    compressedFraction: Double,
                    hasAlternate: Bool = false,
                    alternateByteCount: Int = 0,
                    alternateTileCount: Int = 0) {
            self.byteCount = byteCount
            self.tileCount = tileCount
            self.sampledTiles = sampledTiles
            self.compressedFraction = compressedFraction
            self.hasAlternate = hasAlternate
            self.alternateByteCount = alternateByteCount
            self.alternateTileCount = alternateTileCount
        }
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
        try write(primary: Layer(samples: samples), alternate: nil,
                  width: width, height: height,
                  from: source, to: destination,
                  grid: grid, tableSampleTiles: tableSampleTiles)
    }

    /// Rewrites `source` with one or two full-resolution mosaics.
    ///
    /// - Parameters:
    ///   - primary: the original mosaic — `SubIFDs[0]` in a dual-raw file, IFD0 in a single-raw one.
    ///   - alternate: an optional second mosaic of the same size. `nil` produces exactly the
    ///     single-raw file `write(samples:…)` has always produced.
    ///
    /// Both layers are tiled and lossless-JPEG encoded independently; the alternate gets its own
    /// Huffman tables from its own first `tableSampleTiles` tiles.
    ///
    /// A dual-raw write needs a preview page to put in IFD0 (see the type's documentation for the
    /// measurement behind that layout). A source without one is refused rather than written in a
    /// shape whose default decode would be the *alternate* layer.
    public static func write(
        primary: Layer,
        alternate: Layer?,
        width: Int, height: Int,
        from source: URL,
        to destination: URL,
        grid: TileGrid = TileGrid(),
        tableSampleTiles: Int = 12
    ) throws -> Result {
        guard primary.samples.count == width * height else {
            throw WriteError.pixelCountMismatch(expected: width * height, got: primary.samples.count)
        }
        if let alternate, alternate.samples.count != width * height {
            throw WriteError.pixelCountMismatch(expected: width * height, got: alternate.samples.count)
        }

        let source = try SourceDNG(url: source)
        let dual = alternate != nil

        if dual && source.preview == nil {
            throw WriteError.unexpectedLayout(
                "源文件没有预览页，--dual-raw 无法把预览放进 IFD0"
                + "（没有预览时 SubIFDs[0] 会成为默认解码结果）")
        }

        // 1–2. Tile, build Huffman tables from a sample of the tiles, encode every tile. The two
        //      layers are encoded one after the other so the raw tile arrays never both exist.
        let primaryTiles = encode(primary.samples, width: width, height: height,
                                  grid: grid, tableSampleTiles: tableSampleTiles)
        let alternateTiles = alternate.map {
            encode($0.samples, width: width, height: height,
                   grid: grid, tableSampleTiles: tableSampleTiles)
        }

        // 3. Assemble the file.
        var preserved = source.ifd0.filter { !layoutTags.contains($0.key) }
        for (tag, value) in source.raw where !layoutTags.contains(tag) {
            preserved[tag] = value          // the RAW's pixel-domain tags win over IFD0's
        }

        // The IFD that carries the samples *and* the metadata describing them.
        var rawEntries = preserved
        func putRaw(_ tag: UInt16, _ type: UInt16, _ count: Int, _ payload: [UInt8]) {
            rawEntries[tag] = TIFFEntry(type: type, count: count, value: payload)
        }
        putRaw(254, 4, 1, u32(0))                           // NewSubfileType: primary image
        putRaw(256, 4, 1, u32(UInt32(width)))
        putRaw(257, 4, 1, u32(UInt32(height)))
        putRaw(258, 3, 1, u16(16))                          // BitsPerSample
        putRaw(259, 3, 1, u16(7))                           // Compression: lossless JPEG
        putRaw(262, 3, 1, u16(32803))                       // Photometric: CFA
        putRaw(277, 3, 1, u16(1))                           // SamplesPerPixel
        putRaw(284, 3, 1, u16(1))                           // PlanarConfiguration: chunky
        putRaw(339, 3, 1, u16(1))                           // SampleFormat: unsigned
        putRaw(322, 4, 1, u32(UInt32(grid.tileWidth)))
        putRaw(323, 4, 1, u32(UInt32(grid.tileHeight)))
        putRaw(324, 4, primaryTiles.blobs.count,
               [UInt8](repeating: 0, count: 4 * primaryTiles.blobs.count))   // filled below
        putRaw(325, 4, primaryTiles.blobs.count,
               primaryTiles.blobs.flatMap { u32(UInt32($0.count)) })
        // EXIF/GPS are duplicated data with one owner. Single-raw: that owner is IFD0, i.e. this
        // IFD. Dual-raw: it is the preview page in IFD0, so the raw's IFD must not repeat them.
        if !dual {
            if source.exif != nil { putRaw(34665, 4, 1, u32(0)) }
            if source.gps != nil { putRaw(34853, 4, 1, u32(0)) }
        }

        // The alternate IFD must be readable as a raw image on its own, because promoting it
        // (re-pointing `330`, or reordering `SubIFDs`) is the documented way to reach the denoised
        // layer. A bare pixel-domain IFD is not: with no `WhiteLevel`/`DefaultCropSize`/`CFAPattern`,
        // CIRAWFilter renders it at the sensor size with default levels — a darker, differently
        // cropped picture, not the denoised one.
        //
        // So it repeats the tags that describe **its own samples**: the value range, the CFA and the
        // crop. Profile and colour tags are deliberately *not* repeated: measured with CIRAWFilter,
        // duplicating `ProfileHueSatMapData`/`ProfileLookTableData` changes the rendering of the
        // whole file (the correction lands twice), and a dual-raw file whose pixels are identical to
        // its source has to render identically to it.
        var alternateEntries: [UInt16: TIFFEntry] = [:]
        if let alternateTiles, let alternate {
            var description = Array((alternate.label ?? "denoised alternate").utf8) + [0]
            if description.count > 4096 { description = Array(description[0..<4096]) }
            for tag: UInt16 in rawEntries.keys where sampleDescriptiveTags.contains(tag) {
                alternateEntries[tag] = rawEntries[tag]
            }
            alternateEntries[254] = TIFFEntry(type: 4, count: 1, value: u32(0))
            alternateEntries[256] = TIFFEntry(type: 4, count: 1, value: u32(UInt32(width)))
            alternateEntries[257] = TIFFEntry(type: 4, count: 1, value: u32(UInt32(height)))
            alternateEntries[258] = TIFFEntry(type: 3, count: 1, value: u16(16))
            alternateEntries[259] = TIFFEntry(type: 3, count: 1, value: u16(7))
            alternateEntries[262] = TIFFEntry(type: 3, count: 1, value: u16(32803))
            alternateEntries[270] = TIFFEntry(type: 2, count: description.count, value: description)
            alternateEntries[277] = TIFFEntry(type: 3, count: 1, value: u16(1))
            alternateEntries[284] = TIFFEntry(type: 3, count: 1, value: u16(1))
            alternateEntries[322] = TIFFEntry(type: 4, count: 1, value: u32(UInt32(grid.tileWidth)))
            alternateEntries[323] = TIFFEntry(type: 4, count: 1, value: u32(UInt32(grid.tileHeight)))
            // Its own tile table: same geometry, different compressed sizes per tile.
            alternateEntries[324] = TIFFEntry(type: 4, count: alternateTiles.blobs.count,
                                              value: [UInt8](repeating: 0,
                                                             count: 4 * alternateTiles.blobs.count))
            alternateEntries[325] = TIFFEntry(type: 4, count: alternateTiles.blobs.count,
                                              value: alternateTiles.blobs.flatMap { u32(UInt32($0.count)) })
            alternateEntries[339] = TIFFEntry(type: 3, count: 1, value: u16(1))
        }

        // The preview page: the source's, verbatim except its strip offsets/counts (and, in dual
        // mode, the structural pointers that hang off it).
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
        if dual {
            // Two offsets → 8 bytes, so the payload lands after the IFD's head rather than inline.
            previewEntries[330] = TIFFEntry(type: 4, count: 2, value: [UInt8](repeating: 0, count: 8))
            if source.exif != nil {
                previewEntries[34665] = TIFFEntry(type: 4, count: 1, value: u32(0))
            }
            if source.gps != nil {
                previewEntries[34853] = TIFFEntry(type: 4, count: 1, value: u32(0))
            }
        }

        // Page order in the file:
        //
        //   dual:   IFD0 = preview | SubIFDs[0] = original | SubIFDs[1] = alternate
        //   single: IFD0 = raw     | IFD1       = preview
        //
        // then EXIF, GPS, and finally the data (both layers' tiles, then the preview's chunks).
        func sorted(_ entries: [UInt16: TIFFEntry]) -> [(tag: UInt16, entry: TIFFEntry)] {
            entries.sorted { $0.key < $1.key }.map { (tag: $0.key, entry: $0.value) }
        }
        var pages: [[(tag: UInt16, entry: TIFFEntry)]]
        if dual {
            pages = [sorted(previewEntries), sorted(rawEntries), sorted(alternateEntries)]
        } else {
            pages = [sorted(rawEntries)]
            if !previewEntries.isEmpty { pages.append(sorted(previewEntries)) }
        }
        let primaryPage = dual ? 1 : 0
        let alternatePage = dual ? 2 : -1
        let chainPage = (!dual && pages.count > 1) ? 1 : -1      // IFD0.next → IFD1

        let exif = sorted(source.exif ?? [:])
        let gps = sorted(source.gps ?? [:])
        let sizeE = IFDLayout(headSize: exif.isEmpty ? 0 : 2 + 12 * exif.count + 4,
                              payloadSize: payloadSize(exif))
        let sizeG = IFDLayout(headSize: gps.isEmpty ? 0 : 2 + 12 * gps.count + 4,
                              payloadSize: payloadSize(gps))

        let layouts: [IFDLayout] = pages.map {
            IFDLayout(headSize: 2 + 12 * $0.count + 4, payloadSize: payloadSize($0))
        }
        var pageOffsets: [Int] = []
        var cursor = 8
        for layout in layouts {
            pageOffsets.append(cursor)
            cursor += layout.total
        }
        let offEXIF = cursor
        let offGPS = offEXIF + sizeE.total
        var offTiles = offGPS + sizeG.total
        offTiles += (4 - offTiles % 4) % 4

        var tileOffsets: [UInt32] = []
        cursor = offTiles
        for blob in primaryTiles.blobs {
            tileOffsets.append(UInt32(cursor))
            cursor += blob.count
            cursor += (4 - cursor % 4) % 4
        }
        var alternateOffsets: [UInt32] = []
        for blob in alternateTiles?.blobs ?? [] {
            alternateOffsets.append(UInt32(cursor))
            cursor += blob.count
            cursor += (4 - cursor % 4) % 4
        }
        var previewOffsets: [UInt32] = []
        if let preview = source.preview {
            for chunk in preview.chunks {
                previewOffsets.append(UInt32(cursor))
                cursor += chunk.count
                cursor += (4 - cursor % 4) % 4
            }
        }

        // Back-fill the pointers that depend on the layout.
        pages[primaryPage] = setEntry(pages[primaryPage], tag: 324,
                                      value: tileOffsets.flatMap { u32($0) })
        if dual {
            pages[alternatePage] = setEntry(pages[alternatePage], tag: 324,
                                            value: alternateOffsets.flatMap { u32($0) })
            // SubIFDs: original first — the first subfiletype-0 entry is what ImageIO decodes.
            pages[0] = setEntry(pages[0], tag: 330,
                                value: [UInt32(pageOffsets[1]), UInt32(pageOffsets[2])]
                                    .flatMap { u32($0) })
        }
        if source.exif != nil {
            pages[0] = setEntry(pages[0], tag: 34665, value: u32(UInt32(offEXIF)))
        }
        if source.gps != nil {
            pages[0] = setEntry(pages[0], tag: 34853, value: u32(UInt32(offGPS)))
        }
        if chainPage >= 0 {
            pages[chainPage] = setEntry(pages[chainPage], tag: 273,
                                        value: previewOffsets.flatMap { u32($0) })
        }

        var out = Data()
        out.append(contentsOf: [0x49, 0x49])                 // "II"
        out.append(contentsOf: u16(42))
        out.append(contentsOf: u32(8))                       // IFD0 at byte 8
        // No padding between pages: every offset above is computed from head+payload sizes, so an
        // extra pad here shifts everything after it (this had the tiles 4 bytes out of place, which
        // the offset assertions below then caught).
        //
        // IFD0's "next IFD" pointer must point at IFD1 when there is one — leaving it 0 is how the
        // preview went missing: the preview page was written but nothing could reach it. In the
        // dual-raw layout the preview *is* IFD0, so its chain ends (0) and the two raw pages hang
        // off `330` instead. SubIFDs terminate their own chain.
        for (i, page) in pages.enumerated() {
            guard out.count == pageOffsets[i] else {
                throw WriteError.unexpectedLayout("IFD \(i) 偏移 \(out.count) ≠ 计算值 \(pageOffsets[i])")
            }
            let next: UInt32 = (i == 0 && chainPage >= 0) ? UInt32(pageOffsets[chainPage]) : 0
            appendIFD(&out, page, payloadOffset: pageOffsets[i] + layouts[i].headSize, nextIFD: next)
        }
        if !exif.isEmpty {
            guard out.count == offEXIF else {
                throw WriteError.unexpectedLayout("EXIF 偏移 \(out.count) ≠ 计算值 \(offEXIF)")
            }
            appendIFD(&out, exif, payloadOffset: offEXIF + sizeE.headSize)
        }
        if !gps.isEmpty {
            guard out.count == offGPS else {
                throw WriteError.unexpectedLayout("GPS 偏移 \(out.count) ≠ 计算值 \(offGPS)")
            }
            appendIFD(&out, gps, payloadOffset: offGPS + sizeG.headSize)
        }
        while out.count < offTiles { out.append(0) }
        for (i, blob) in primaryTiles.blobs.enumerated() {
            guard out.count == Int(tileOffsets[i]) else {
                throw WriteError.unexpectedLayout("瓦片 \(i) 偏移 \(out.count) ≠ 表内 \(tileOffsets[i])")
            }
            out.append(contentsOf: blob)
            pad4(&out)
        }
        for (i, blob) in (alternateTiles?.blobs ?? []).enumerated() {
            guard out.count == Int(alternateOffsets[i]) else {
                throw WriteError.unexpectedLayout(
                    "备选瓦片 \(i) 偏移 \(out.count) ≠ 表内 \(alternateOffsets[i])")
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
        return Result(byteCount: out.count,
                      tileCount: primaryTiles.blobs.count,
                      sampledTiles: primaryTiles.sampledTiles,
                      compressedFraction: Double(primaryTiles.compressedBytes)
                          / Double(primary.samples.count * 2),
                      hasAlternate: alternateTiles != nil,
                      alternateByteCount: alternateTiles?.compressedBytes ?? 0,
                      alternateTileCount: alternateTiles?.blobs.count ?? 0)
    }

    // MARK: - tiling and lossless-JPEG encoding

    struct EncodedLayer {
        var blobs: [[UInt8]]
        var sampledTiles: Int
        var compressedBytes: Int
    }

    /// Tile one mosaic and encode every tile as its own SOF3 stream.
    ///
    /// Edge tiles are padded by repeating the last row/column: the decoder crops them away with
    /// ActiveArea/DefaultCrop, so their contents never reach the picture, and repeating (rather
    /// than zero filling) keeps the difference histogram representative. Each layer builds its own
    /// Huffman tables from its own first `tableSampleTiles` tiles — a denoised layer's difference
    /// distribution is narrower than the original's, and reusing the primary's tables would cost
    /// bytes on exactly the layer that is supposed to be the smaller one.
    static func encode(_ samples: [UInt16], width: Int, height: Int,
                       grid: TileGrid, tableSampleTiles: Int) -> EncodedLayer {
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
        return EncodedLayer(blobs: blobs, sampledTiles: sampled.count,
                            compressedBytes: blobs.reduce(0) { $0 + $1.count })
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
