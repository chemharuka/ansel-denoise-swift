import Foundation

/// Decodes a DNG's **mosaic** — the `PhotometricInterpretation = 32803` pixels, before demosaicing.
///
/// `CIRAWFilter` cannot give us this: it renders the *developed* picture, and every stage a denoise
/// pass must not touch (demosaic, white balance, tone) has already happened by the time it produces
/// pixels. So reading a RAW whose samples we intend to replace needs its own decoder.
///
/// Supported layouts, which is everything a capture DNG realistically is:
///
///  - `Compression = 1` — one uncompressed strip, or uncompressed tiles.
///  - `Compression = 7` — lossless-JPEG (SOF3) tiles, with 1 or 2 column-interleaved components,
///    `Ss = 1` (left-neighbour prediction) or `Ss = 7` unchanged. This is what Leica, Adobe and our
///    own writer emit.
///
/// Anything else throws. A wrong decode here would look like a correct picture, so guessing is the
/// one thing this must not do.
extension SourceDNG {

    public enum DecodeError: Error, CustomStringConvertible {
        case missingTileGeometry
        case unsupportedCompression(UInt16)
        case unsupportedPredictor(Int)
        case unsupportedComponents(Int)
        case truncatedTile(Int)
        case badHuffmanTable

        public var description: String {
            switch self {
            case .missingTileGeometry: return "既没有 Tile 也没有 Strip 的几何信息"
            case .unsupportedCompression(let c): return "不支持的压缩方式 \(c)（支持 1 和 7）"
            case .unsupportedPredictor(let s): return "不支持的 SOF3 预测器 Ss=\(s)（支持 1 和 7）"
            case .unsupportedComponents(let n): return "不支持的 SOF3 分量数 \(n)（支持 1 和 2）"
            case .truncatedTile(let i): return "瓦片 \(i) 的数据被截断"
            case .badHuffmanTable: return "DHT 损坏"
            }
        }
    }

    /// Returns the mosaic, row-major, `width * height` entries, and the pixel geometry.
    public func readMosaic(data: Data) throws -> (samples: [UInt16], width: Int, height: Int) {
        let width = try requiredUInt32(256, in: raw)
        let height = try requiredUInt32(257, in: raw)
        let compression = UInt16(truncatingIfNeeded: try requiredUInt32(259, in: raw))

        var samples = [UInt16](repeating: 0, count: width * height)

        if let tileWidthRaw = raw[322]?.firstUInt32, let tileHeightRaw = raw[323]?.firstUInt32,
           let tileOffsets = raw[324], let tileCounts = raw[325] {
            let tileWidth = Int(tileWidthRaw)
            let tileHeight = Int(tileHeightRaw)
            let nx = (width + tileWidth - 1) / tileWidth
            let ny = (height + tileHeight - 1) / tileHeight
            var index = 0
            for ty in 0..<ny {
                for tx in 0..<nx {
                    guard let offset = tileOffsets.uint32Array[safe: index],
                          let count = tileCounts.uint32Array[safe: index] else {
                        throw DecodeError.truncatedTile(index)
                    }
                    let start = Int(offset), end = start + Int(count)
                    guard end <= data.count else { throw DecodeError.truncatedTile(index) }
                    let blob = [UInt8](data[start..<end])

                    let pixels: [UInt16]
                    switch compression {
                    case 1:
                        pixels = blob.withUnsafeBytes { rawBuffer -> [UInt16] in
                            let bound = rawBuffer.bindMemory(to: UInt16.self)
                            return Array(bound.prefix(tileWidth * tileHeight))
                        }
                    case 7:
                        pixels = try SOF3.decode(blob, expectWidth: tileWidth, expectHeight: tileHeight)
                    default:
                        throw DecodeError.unsupportedCompression(compression)
                    }

                    for y in 0..<tileHeight {
                        let dy = ty * tileHeight + y
                        if dy >= height { break }
                        let dstRow = dy * width
                        let srcRow = y * tileWidth
                        let copyWidth = min(tileWidth, width - tx * tileWidth)
                        for x in 0..<copyWidth {
                            samples[dstRow + tx * tileWidth + x] = pixels[srcRow + x]
                        }
                    }
                    index += 1
                }
            }
            return (samples, width, height)
        }

        if let stripOffsets = raw[273], let stripCounts = raw[279] {
            guard compression == 1 else { throw DecodeError.unsupportedCompression(compression) }
            var cursor = 0
            for i in 0..<min(stripOffsets.count, stripCounts.count) {
                guard let offset = stripOffsets.uint32Array[safe: i],
                      let count = stripCounts.uint32Array[safe: i] else { break }
                let start = Int(offset)
                let available = min(Int(count), data.count - start)
                guard available > 0 else { throw DecodeError.truncatedTile(i) }
                let rows = available / 2 / width
                for r in 0..<rows where cursor < height {
                    let base = start + r * width * 2
                    for x in 0..<width {
                        let lo = data[base + x * 2], hi = data[base + x * 2 + 1]
                        samples[cursor * width + x] = UInt16(lo) | (UInt16(hi) << 8)
                    }
                    cursor += 1
                }
            }
            return (samples, width, height)
        }

        throw DecodeError.missingTileGeometry
    }

    private func requiredUInt32(_ tag: UInt16, in ifd: [UInt16: TIFFEntry]) throws -> Int {
        guard let entry = ifd[tag], let value = entry.firstUInt32 else {
            throw DNGRawWriter.WriteError.missingTag(tag)
        }
        return Int(value)
    }
}

extension TIFFEntry {
    /// First component as UInt32. Handles both LONG and SHORT so a tag's numeric width does not
    /// decide whether we can read it (Adobe writes some of these as SHORT, Leica as LONG).
    var firstUInt32: UInt32? {
        switch type {
        case 3 where value.count >= 2:
            return UInt32(value[0]) | (UInt32(value[1]) << 8)
        case 4 where value.count >= 4:
            return UInt32(value[0]) | (UInt32(value[1]) << 8)
                | (UInt32(value[2]) << 16) | (UInt32(value[3]) << 24)
        case 1 where value.count >= 1:
            return UInt32(value[0])
        default:
            return nil
        }
    }

    var uint32Array: [UInt32] {
        guard type == 4 else { return [] }
        var out: [UInt32] = []
        out.reserveCapacity(count)
        var i = 0
        while i + 4 <= value.count {
            out.append(UInt32(value[i]) | (UInt32(value[i + 1]) << 8)
                       | (UInt32(value[i + 2]) << 16) | (UInt32(value[i + 3]) << 24))
            i += 4
        }
        return out
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

/// SOF3 (lossless JPEG) decoder — the read half of `RawLosslessJPEG`, written from the spec rather
/// than by mirroring the encoder so a shared misunderstanding cannot cancel out.
enum SOF3 {
    enum Failure: Error, CustomStringConvertible {
        case noSOF3, noSOS, badTable
        var description: String {
            switch self {
            case .noSOF3: return "JPEG 流里没有 SOF3"
            case .noSOS: return "JPEG 流里没有 SOS"
            case .badTable: return "Huffman 表损坏"
            }
        }
    }

    static func decode(_ data: [UInt8], expectWidth: Int, expectHeight: Int) throws -> [UInt16] {
        var i = 2
        var precision = 16
        var components = 1
        var componentWidth = expectWidth
        var height = expectHeight
        var predictors = 1
        var tables: [[(length: Int, code: Int, symbol: Int)]] = []
        var sosStart = 0

        while i < data.count - 1 {
            guard data[i] == 0xFF else { i += 1; continue }
            let marker = data[i + 1]
            if marker == 0xD8 { i += 2; continue }
            guard i + 4 <= data.count else { break }
            let length = Int(data[i + 2]) << 8 | Int(data[i + 3])
            let body = i + 4
            switch marker {
            case 0xC3:
                precision = Int(data[body])
                height = Int(data[body + 1]) << 8 | Int(data[body + 2])
                componentWidth = Int(data[body + 3]) << 8 | Int(data[body + 4])
                components = Int(data[body + 5])
            case 0xC4:
                var p = body
                while p < i + 2 + length {
                    let isDC = (Int(data[p]) & 0xF0) == 0
                    let id = Int(data[p]) & 0x0F
                    var counts: [Int] = []
                    for l in 0..<16 { counts.append(Int(data[p + 1 + l])) }
                    let total = counts.reduce(0, +)
                    var symbols: [Int] = []
                    for s in 0..<total { symbols.append(Int(data[p + 17 + s])) }
                    p += 17 + total
                    guard isDC else { continue }
                    var code = 0
                    var symbolIndex = 0
                    var entries: [(Int, Int, Int)] = []
                    for l in 1...16 {
                        for _ in 0..<counts[l - 1] {
                            guard symbolIndex < symbols.count else { throw Failure.badTable }
                            entries.append((l, code, symbols[symbolIndex]))
                            symbolIndex += 1
                            code += 1
                        }
                        code <<= 1
                    }
                    while tables.count <= id { tables.append([]) }
                    tables[id] = entries
                }
            case 0xDA:
                var p = body
                let ns = Int(data[p]); p += 1
                var selectors: [Int] = []
                for _ in 0..<ns {
                    p += 1                                     // component id
                    selectors.append(Int(data[p]) >> 4)        // Td in the high nibble
                    p += 1
                }
                predictors = Int(data[p])                      // Ss
                sosStart = i + 2 + length
                // Re-order the table list so it is indexed by component, honouring the selector.
                tables = (0..<components).map { c in
                    let id = c < selectors.count ? selectors[c] : c
                    return id < tables.count ? tables[id] : []
                }
            default:
                break
            }
            if marker == 0xDA { break }
            i += 2 + length
        }
        guard sosStart > 0 else { throw Failure.noSOS }
        guard !tables.isEmpty else { throw Failure.noSOF3 }
        guard predictors == 1 || predictors == 7 else {
            throw SourceDNG.DecodeError.unsupportedPredictor(predictors)
        }
        guard components == 1 || components == 2 else {
            throw SourceDNG.DecodeError.unsupportedComponents(components)
        }

        var reader = BitReader(bytes: data, index: sosStart)
        var out = [UInt16](repeating: 0, count: componentWidth * components * height)
        let mask: Int32 = (1 << Int32(precision)) - 1
        let half: Int32 = 1 << (Int32(precision) - 1)
        let xc = componentWidth

        for y in 0..<height {
            let row = y * components * xc
            for x in 0..<xc {
                for c in 0..<components {
                    let table = tables[c]
                    var category = 0
                    var code = 0
                    search: for l in 1...16 {
                        code = (code << 1) | reader.bit()
                        for entry in table where entry.length == l && entry.code == code {
                            category = entry.symbol
                            break search
                        }
                    }
                    var d: Int32 = 0
                    if category > 0 {
                        let extra = Int32(reader.read(category))
                        let threshold = Int32(1) << Int32(category - 1)
                        d = extra >= threshold ? extra : extra - (Int32(1) << Int32(category)) + 1
                    }
                    let here = row + x * components + c
                    let predicted: Int32
                    if predictors == 1 {
                        if x > 0 {
                            predicted = Int32(out[here - components])
                        } else if y > 0 {
                            predicted = Int32(out[here - components * xc])
                        } else {
                            predicted = half
                        }
                    } else {
                        // Ss = 7: predict from the sample directly above.
                        predicted = y > 0 ? Int32(out[here - components * xc]) : half
                    }
                    out[here] = UInt16(truncatingIfNeeded: (predicted + d) & mask)
                }
            }
        }
        return out
    }

    struct BitReader {
        let bytes: [UInt8]
        var index: Int
        var acc: UInt32 = 0
        var bits = 0

        mutating func bit() -> Int {
            if bits == 0 {
                var byte = bytes[index]
                index += 1
                if byte == 0xFF {
                    index += 1                             // skip the stuffing byte
                }
                acc = UInt32(byte)
                bits = 8
            }
            bits -= 1
            return Int((acc >> UInt32(bits)) & 1)
        }

        mutating func read(_ n: Int) -> Int {
            var v = 0
            for _ in 0..<n { v = (v << 1) | bit() }
            return v
        }
    }
}
