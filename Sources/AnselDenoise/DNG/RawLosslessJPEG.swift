import Foundation

/// Lossless JPEG (SOF3) encoder — the compression DNG uses for CFA RAW when `Compression = 7`.
///
/// Why hand-rolled: ImageIO cannot *write* DNG at all (`com.adobe.raw-image` is read-only), and no
/// system or SwiftPM API produces a SOF3 stream. A 60 MP frame is 121 MB as uncompressed samples and
/// 57 MB as SOF3, so "write uncompressed and let the user deal with it" is not an option.
///
/// Three details are load-bearing, each learned the hard way:
///
/// 1. **The SOS selector byte is `Td << 4 | Ta`.** Writing `[componentID, tableID]` puts the table id
///    in the *AC* nibble, so the second component is decoded with table 0 and the entropy stream
///    desynchronises. libraw ignores the selector (it indexes tables by component) and therefore
///    decodes such a file correctly — Core Image honours it and returns an **all-black** image while
///    still reporting a successful render. Both decoders must be checked after any change here.
/// 2. **CFA data is split into two column-interleaved components.** Predicting within one component
///    then compares same-colour samples; a single component has to predict across the R→G→B colour
///    change. Measured on one 256×256 tile of denoised SL3 data: 82.3 % of raw with one component,
///    59.4 % with two.
/// 3. **Marker order matches what Adobe/Leica emit: `SOI → SOF3 → DHT… → SOS`**, with component ids
///    starting at 0.
public enum RawLosslessJPEG {

    /// One DC Huffman table, indexed by the JPEG "difference category" 0…16.
    public struct HuffmanTable: Sendable, Equatable {
        /// Canonical code per category, MSB-first, `length[category]` bits wide.
        public var code: [UInt32]
        /// Code length per category, 1…16. Zero means "category unused".
        public var length: [UInt8]

        public init(code: [UInt32], length: [UInt8]) {
            precondition(code.count == 17 && length.count == 17)
            self.code = code
            self.length = length
        }

        /// The symbol order a DHT segment must carry: grouped by code length, ascending symbol inside
        /// a group. The decoder rebuilds codes from this order, so it has to match `canonical()`.
        var dhtSymbols: [UInt8] {
            (1...16).flatMap { l in (0...16).filter { Int(length[$0]) == l }.map(UInt8.init) }
        }

        var dhtCounts: [UInt8] {
            (1...16).map { l in UInt8((0...16).filter { Int(length[$0]) == l }.count) }
        }
    }

    /// Encodes one tile/strip. `components` is 1 or 2 (see note 2 above); `width` is the *sample*
    /// width, so with two components each component row is `width / 2` samples.
    public static func encode(
        _ samples: UnsafeBufferPointer<UInt16>,
        width: Int, height: Int,
        components: Int = 2,
        tables: [HuffmanTable],
        precision: Int = 16
    ) -> [UInt8] {
        precondition(components == 1 || components == 2)
        precondition(tables.count == components)
        precondition(width % components == 0)
        precondition(samples.count >= width * height)

        let xc = width / components
        var writer = BitWriter()
        writer.bytes.reserveCapacity(width * height + 4096)

        // SOI
        writer.bytes += [0xFF, 0xD8]

        // SOF3
        writer.bytes += [0xFF, 0xC3]
        let soflen = UInt16(8 + 3 * components)
        writer.bytes += [UInt8(soflen >> 8), UInt8(soflen & 0xFF)]
        writer.bytes.append(UInt8(precision))
        writer.bytes += [UInt8((height >> 8) & 0xFF), UInt8(height & 0xFF)]
        writer.bytes += [UInt8((xc >> 8) & 0xFF), UInt8(xc & 0xFF)]
        writer.bytes.append(UInt8(components))
        for c in 0..<components { writer.bytes += [UInt8(c), 0x11, 0x00] }

        // DHT (one per component, table id == component id)
        for (tid, table) in tables.enumerated() {
            let counts = table.dhtCounts
            let symbols = table.dhtSymbols
            let len = UInt16(2 + 1 + 16 + symbols.count)
            writer.bytes += [0xFF, 0xC4, UInt8(len >> 8), UInt8(len & 0xFF), UInt8(tid)]
            writer.bytes += counts
            writer.bytes += symbols
        }

        // SOS — selector byte is Td in the high nibble (see note 1 above).
        writer.bytes += [0xFF, 0xDA]
        let soslen = UInt16(6 + 2 * components)
        writer.bytes += [UInt8(soslen >> 8), UInt8(soslen & 0xFF)]
        writer.bytes.append(UInt8(components))
        for c in 0..<components { writer.bytes += [UInt8(c), UInt8(c << 4)] }
        writer.bytes += [1, 0, 0]           // Ss = 1 (left neighbour), Se = 0, AhAl = 0

        let mask: Int32 = (1 << 16) - 1
        let half: Int32 = 1 << 15

        for y in 0..<height {
            let row = y * width
            for x in 0..<xc {
                for c in 0..<components {
                    let value = Int32(samples[row + x * components + c])
                    let predicted: Int32
                    if x > 0 {
                        predicted = Int32(samples[row + (x - 1) * components + c])
                    } else if y > 0 {
                        predicted = Int32(samples[row - width + c])   // row start: use the sample above
                    } else {
                        predicted = half                             // image start: mid-grey
                    }
                    var d = (value - predicted) & mask
                    if d >= half { d -= (1 << 16) }

                    let table = tables[c]
                    if d == 0 {
                        writer.write(table.code[0], Int(table.length[0]))
                        continue
                    }
                    let magnitude: Int32 = d > 0 ? d : -d
                    let category = Int(Int32.bitWidth - magnitude.leadingZeroBitCount)
                    writer.write(table.code[category], Int(table.length[category]))
                    // Negative differences are signed-magnitude shifted by (2^cat - 1), per JPEG.
                    let extra = d > 0 ? d : d + (1 << category) - 1
                    writer.write(UInt32(extra) & ((1 << category) - 1), category)
                }
            }
        }
        var finished = writer.finish()
        finished += [0xFF, 0xD9]
        return finished
    }

    // MARK: - Huffman tables from data

    /// Builds one table per component by measuring the difference-category histogram of a few tiles.
    ///
    /// Near-optimal rather than optimal on purpose: the code lengths come from each symbol's Shannon
    /// length with a Kraft repair pass, which lands within ~1 % of a package-merge table and keeps the
    /// implementation readable. Every category is forced to at least one occurrence so a rare
    /// difference in a tile that was not sampled still has a code.
    public static func buildTables(
        _ tiles: [[UInt16]], width: Int, height: Int,
        components: Int = 2, precision: Int = 16
    ) -> [HuffmanTable] {
        var histograms = [[Int]](repeating: [Int](repeating: 0, count: 17), count: components)
        let mask: Int32 = (1 << Int32(precision)) - 1
        let half: Int32 = 1 << (Int32(precision) - 1)
        let xc = width / components

        for tile in tiles {
            tile.withUnsafeBufferPointer { ptr in
                for y in 0..<height {
                    let row = y * width
                    for x in 0..<xc {
                        for c in 0..<components {
                            let value = Int32(ptr[row + x * components + c])
                            let predicted: Int32 = x > 0
                                ? Int32(ptr[row + (x - 1) * components + c])
                                : (y > 0 ? Int32(ptr[row - width + c]) : half)
                            var d = (value - predicted) & mask
                            if d >= half { d -= (1 << precision) }
                            let category = d == 0 ? 0 : Int(Int32.bitWidth - (d > 0 ? d : -d).leadingZeroBitCount)
                            histograms[c][category] += 1
                        }
                    }
                }
            }
        }
        return histograms.map { canonicalTable(fromHistogram: $0) }
    }

    /// Shannon lengths + Kraft repair → canonical codes, deterministic for a given histogram.
    static func canonicalTable(fromHistogram histogram: [Int]) -> HuffmanTable {
        var counts = histogram
        for c in 0...16 where counts[c] == 0 { counts[c] = 1 }      // every category gets a code

        let total = Double(counts.reduce(0, +))
        var lengths = (0...16).map { c -> Int in
            let p = Double(counts[c]) / total
            return min(16, max(1, Int((-log2(p)).rounded(.up))))
        }

        // Kraft: shrink the longest code until the prefix code is not oversubscribed.
        while lengths.reduce(0.0, { $0 + pow(2.0, -Double($1)) }) > 1.0 + 1e-12 {
            if let longest = lengths.indices.max(by: { lengths[$0] < lengths[$1] }), lengths[longest] > 1 {
                lengths[longest] -= 1
            } else {
                break
            }
        }

        var countPerLength = [Int](repeating: 0, count: 17)
        for l in lengths where l > 0 { countPerLength[l] += 1 }
        var nextCode = [Int](repeating: 0, count: 17)
        var code = 0
        for bits in 1...16 {
            code = (code + countPerLength[bits - 1]) << 1
            nextCode[bits] = code
        }

        var codes = [UInt32](repeating: 0, count: 17)
        var finalLengths = [UInt8](repeating: 0, count: 17)
        for c in 0...16 {
            codes[c] = UInt32(nextCode[lengths[c]])
            finalLengths[c] = UInt8(lengths[c])
            nextCode[lengths[c]] += 1
        }
        return HuffmanTable(code: codes, length: finalLengths)
    }

    // MARK: - Bit writer

    /// MSB-first bit packer with JPEG's byte stuffing (a `0xFF` in the stream gains a `0x00`).
    ///
    /// The output buffer lives *in* the writer and is returned by `finish()`. An earlier version
    /// copied a caller's array into a property (`init(into:)`) and then kept appending to the copy —
    /// the file came out with a valid header and no entropy data, i.e. 108 bytes instead of 1.3 KB,
    /// and every tile silently went missing. Keep one buffer, one owner.
    struct BitWriter {
        var bytes: [UInt8] = []
        private var acc: UInt32 = 0
        private var bits = 0

        init() {}

        mutating func write(_ code: UInt32, _ n: Int) {
            acc = (acc << UInt32(n)) | code
            bits += n
            while bits >= 8 {
                bits -= 8
                let byte = UInt8((acc >> UInt32(bits)) & 0xFF)
                bytes.append(byte)
                if byte == 0xFF { bytes.append(0x00) }
            }
        }

        mutating func finish() -> [UInt8] {
            if bits > 0 {
                let byte = UInt8((acc << UInt32(8 - bits)) & 0xFF)
                bytes.append(byte)
                if byte == 0xFF { bytes.append(0x00) }
                bits = 0
            }
            acc = 0
            return bytes
        }
    }
}
